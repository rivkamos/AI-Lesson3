"""
=================================================================
 טכניקות RAG מתקדמות — דוגמאות ללימוד, שלב אחר שלב
=================================================================

RAG בסיסי = הופכים את השאלה לוקטור -> מוצאים את הקטעים הקרובים ביותר ->
מכניסים אותם ל-LLM. זה עובד, אבל יש לזה נקודות חולשה. כל טכניקה כאן פותרת
נקודת חולשה אחת ספציפית. הקובץ מציג את כל השש, כל אחת בנפרד, ניתנת להרצה,
ומתועדת כדי שתוכלו להדגים לתלמידים אחת-אחת.

  1) הרחבת שאילתה (Query Expansion)   - ניסוח המשתמש קצר / מעורפל מדי
  2) ריבוי שאילתות (Multi-Query)       - ניסוח אחד מפספס קטעים רלוונטיים
  3) ניתוב (Routers)                   - שאלות שונות צריכות מקורות שונים
  4) דירוג מחדש (Reranking)            - "הכי קרוב" וקטורית != "הכי רלוונטי"
  5) אב-ילד (Parent-Child)             - קטעים קטנים מתאימים טוב אך חסרי הקשר
  6) דחיסת הקשר (Context Compression)  - בקטעים שהוחזרו יש הרבה רעש

הכול משתמש באותו סטאק קיים: LlamaIndex + Gemini.
אנחנו בונים אינדקס קטן אחד בזיכרון ממסמכי דוגמה, כך שהקובץ עצמאי לחלוטין
(לא צריך להעלות קבצים).

הרצה:
    set GEMINI_API_KEY=your_key
    python rag_advanced_techniques.py

טיפ לשיעור: הריצו סעיף אחד בכל פעם. כל פונקציית demo עצמאית, ורובן מדפיסות
פלט verbose=True כדי שהתלמידים יראו מה הטכניקה באמת עשתה (שאילתות שהורחבו,
החלטת ניתוב וכו').
=================================================================
"""

import os

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document,
)
from llama_index.core.node_parser import (
    SentenceSplitter,
    HierarchicalNodeParser,
    get_leaf_nodes,
)
from llama_index.core.retrievers import (
    QueryFusionRetriever,
    AutoMergingRetriever,
)
from llama_index.core.query_engine import (
    RetrieverQueryEngine,
    RouterQueryEngine,
    TransformQueryEngine,
)
from llama_index.core.tools import QueryEngineTool
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.postprocessor import (
    LLMRerank,
    SentenceEmbeddingOptimizer,
)
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.storage.docstore import SimpleDocumentStore

from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI


# =========================
# הגדרות Gemini / LLM (זהה ל-rag_new.py)
# =========================

Settings.llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=os.environ["GEMINI_API_KEY"],
)

Settings.embed_model = GoogleGenAIEmbedding(
    model="text-embedding-004",
    api_key=os.environ["GEMINI_API_KEY"],
)


# =========================
# מסמכי דוגמה (נושא: ועד בית)
# מעט ארוכים בכוונה, כדי שלאב-ילד ולדחיסה יהיה משהו אמיתי למזג / לקצץ.
# =========================

SAMPLE_TEXTS = [
    (
        "דמי תחזוקה. דמי התחזוקה החודשיים של הבניין הם 350 ש\"ח לדירה "
        "ומשולמים ב-1 לכל חודש. אפשר לשלם בהעברה בנקאית או דרך אפליקציית "
        "הבניין. תשלום באיחור אחרי ה-10 בחודש מוסיף קנס של 25 ש\"ח. בעלי "
        "דירה שמשכירים את הדירה נשארים אחראים לתשלום, ולא הדייר."
    ),
    (
        "קרן רזרבה. הבניין מחזיק קרן רזרבה לטווח ארוך לתיקונים גדולים כמו "
        "איטום גג, שיפוץ מעלית, וחידוש חזית. קרן הרזרבה נפרדת מדמי התחזוקה "
        "החודשיים ונבחנת פעם בשנה באסיפה הכללית. חשבונית 88231 כיסתה את "
        "תיקון המעלית במרץ ושולמה מתוך קרן הרזרבה."
    ),
    (
        "חניה ושטחים משותפים. לכל דירה מוקצה מקום חניה אחד בחניון התת-קרקעי. "
        "חניית אורחים מוגבלת לשעתיים. דירה 4B דיווחה על נזילת מים בתקרת "
        "החניון, שמקורה אותר בצינור ניקוז סדוק בקומה השנייה."
    ),
    (
        "אסיפות והחלטות. אסיפות כלליות של ועד הבית מתקיימות פעמיים בשנה. "
        "החלטות על הוצאה מקרן הרזרבה דורשות רוב קולות של בעלי הדירות "
        "הנוכחים. פרוטוקול כל אסיפה משותף באפליקציית הבניין תוך שבוע."
    ),
]

documents = [Document(text=t) for t in SAMPLE_TEXTS]


# אינדקס פשוט + מנוע שאילתות שעליו נשענות הטכניקות ה"בסיסיות".
base_nodes = SentenceSplitter(chunk_size=256).get_nodes_from_documents(documents)
base_index = VectorStoreIndex(base_nodes)


def banner(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


# =================================================================
# 1) הרחבת שאילתה (QUERY EXPANSION)
# -----------------------------------------------------------------
# הבעיה: שאילתה קצרה ("נזילה?") עם מעט מדי מילים מתאימה בצורה גרועה.
# הרעיון: מעשירים את השאילתה לפני החיפוש.
#
# כאן משתמשים ב-HyDE (Hypothetical Document Embeddings), צורה נפוצה של
# הרחבת שאילתה: ה-LLM כותב קודם תשובה היפותטית מומצאת לשאלה, ואנחנו מחפשים
# באמצעות הטקסט העשיר הזה במקום בשאילתה החשופה. פסקה מומצאת-אך-לעניין
# קרובה הרבה יותר לקטעים האמיתיים מאשר שאלה של 3 מילים.
# =================================================================

def demo_query_expansion():
    banner("1) הרחבת שאילתה (HyDE)")

    question = "נזילה במוסך?"

    # שלב 1: הטרנספורם שהופך את השאלה למסמך היפותטי.
    hyde = HyDEQueryTransform(include_original=True)

    # נציג לתלמידים את הטקסט המורחב שה-LLM ייצר:
    expanded = hyde.run(question)
    print("שאילתה מקורית :", question)
    print("מורחבת (HyDE):", expanded.custom_embedding_strs[0][:300], "...")

    # שלב 2: עוטפים מנוע שאילתות כלשהו כך שיחפש עם השאילתה המורחבת.
    base_qe = base_index.as_query_engine(similarity_top_k=3)
    hyde_qe = TransformQueryEngine(base_qe, query_transform=hyde)

    print("\nתשובה:", hyde_qe.query(question))


# =================================================================
# 2) ריבוי שאילתות (MULTI-QUERY RETRIEVAL)
# -----------------------------------------------------------------
# הבעיה: ניסוח יחיד של שאלה מוצא רק קטעים שמנוסחים כמוהו.
# הרעיון: מבקשים מה-LLM לייצר כמה ניסוחים שונים, מאחזרים לכל אחד,
#         ואז ממזגים את כל התוצאות (Reciprocal Rank Fusion).
#
# זה אותו QueryFusionRetriever מדוגמת ההיברידי, אבל כאן הטריק הוא
# num_queries > 1 (כמה וריאציות שאילתה שה-LLM מייצר) במקום שילוב של
# מאחזרים שונים.
# =================================================================

def demo_multi_query():
    banner("2) ריבוי שאילתות (MULTI-QUERY)")

    vector_retriever = base_index.as_retriever(similarity_top_k=3)

    multi_query_retriever = QueryFusionRetriever(
        [vector_retriever],
        similarity_top_k=4,
        num_queries=4,            # <-- ה-LLM כותב 4 וריאציות של השאלה
        mode="reciprocal_rerank",
        use_async=False,
        verbose=True,             # מדפיס את וריאציות השאילתה שנוצרו
    )

    print("\nשימו לב לוריאציות השאילתה שה-LLM ייצר למטה:\n")
    nodes = multi_query_retriever.retrieve("איך הדיירים משלמים?")

    print("\nתוצאות ממוזגות:")
    for n in nodes:
        print(f"  [{n.score:.4f}] {n.node.get_content()[:90]}...")


# =================================================================
# 3) ניתוב מתקדם ב-LLAMAINDEX (ROUTERS)
# -----------------------------------------------------------------
# הבעיה: אינדקס אחד זה בסדר, אבל במערכות אמיתיות יש כמה מקורות
#        (DB תשלומים, מסמכים משפטיים, שאלות נפוצות...). לשלוח כל שאלה לכולם
#        זה בזבזני ורועש.
# הרעיון: Router משתמש ב-LLM כדי לבחור איזה מנוע צריך לענות.
#
# בונים שני מנועי שאילתות מתמחים ונותנים ל-router לבחור.
# =================================================================

def demo_router():
    banner("3) מנוע שאילתות עם ניתוב (ROUTER)")

    # שני מנועים מתמחים על אותם נתונים (בחיים האמיתיים: נתונים שונים).
    payments_qe = base_index.as_query_engine(similarity_top_k=2)
    meetings_qe = base_index.as_query_engine(similarity_top_k=2)

    # כל מנוע נעטף כ-"כלי" עם תיאור. ה-router קורא את התיאורים האלה
    # כדי להחליט לאן לשלוח את השאלה.
    tools = [
        QueryEngineTool.from_defaults(
            query_engine=payments_qe,
            description="דמי ועד, תשלומים, קנסות, וקרן הרזרבה.",
        ),
        QueryEngineTool.from_defaults(
            query_engine=meetings_qe,
            description="אסיפות הוועד, הצבעות, וכללי קבלת החלטות.",
        ),
    ]

    router = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=tools,
        verbose=True,     # מדפיס איזה כלי ה-router בחר ולמה
    )

    print("\nש: מתי מתקיימות אסיפות הוועד?")
    print("תשובה:", router.query("מתי מתקיימות אסיפות הוועד?"))


# =================================================================
# 4) דירוג מחדש (RERANKING)
# -----------------------------------------------------------------
# הבעיה: "הכי קרוב" וקטורית זה לא אותו דבר כמו "הכי רלוונטי". ה-top-5
#        לפי מרחק האמבדינג כולל לעיתים קרובות החטאות-כמעט.
# הרעיון: מאחזרים יותר מדי (למשל top 10), ואז נותנים למודל חזק לתת ציון
#         מחדש לכל קטע מול השאלה ולשמור רק את הטובים באמת.
#
# משתמשים ב-LLMRerank: ה-LLM קורא כל מועמד ונותן ציון לרלוונטיות שלו.
# =================================================================

def demo_reranking():
    banner("4) דירוג מחדש (LLMRerank)")

    reranker = LLMRerank(
        top_n=3,              # שומרים את ה-3 הטובים אחרי הדירוג מחדש
        choice_batch_size=5,
    )

    # מאחזרים 8, ואז מדרגים מחדש עד ל-3.
    query_engine = base_index.as_query_engine(
        similarity_top_k=8,
        node_postprocessors=[reranker],
    )

    response = query_engine.query("מה קורה אם אני משלם את דמי הוועד באיחור?")
    print("\nתשובה:", response)
    print("\nמה נשמר אחרי הדירוג מחדש:")
    for n in response.source_nodes:
        print(f"  [{n.score:.4f}] {n.node.get_content()[:90]}...")


# =================================================================
# 5) אחזור אב-ילד (PARENT-CHILD) — מיזוג אוטומטי
# -----------------------------------------------------------------
# הבעיה: קטעים קטנים מתאימים בדיוק אך קצרים מדי כדי לענות טוב;
#        קטעים גדולים נותנים הקשר אך מתאימים פחות טוב.
# הרעיון: מאנדקסים קטעי-ילד קטנים להתאמה מדויקת, אבל כשכמה ילדים של
#         אותו אב מתאימים — מחזירים את האב הגדול יותר כדי שה-LLM יקבל
#         הקשר מלא.
#
# HierarchicalNodeParser בונה שכבות אב->ילד; AutoMergingRetriever מבצע
# אוטומטית את שלב "מיזוג הילדים בחזרה לאב".
# =================================================================

def demo_parent_child():
    banner("5) אחזור אב-ילד (מיזוג אוטומטי)")

    # שלב 1: פיצול להיררכיה: גדול (512) -> בינוני (256) -> קטן (128).
    # הקטע הקטן ביותר חייב להיות גדול מאורך המטא-דאטה, אחרת LlamaIndex
    # זורק "Metadata length is longer than chunk size".
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[512, 256, 128]
    )
    all_nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(all_nodes)   # הקטעים הקטנים ביותר

    # שלב 2: docstore מחזיק את כל הצמתים (אבות + ילדים); רק העלים הקטנים
    # נכנסים לאינדקס הווקטורי לצורך ההתאמה.
    docstore = SimpleDocumentStore()
    docstore.add_documents(all_nodes)
    storage_context = StorageContext.from_defaults(docstore=docstore)

    index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)

    # שלב 3: AutoMergingRetriever ממזג עלים בחזרה לאב שלהם
    # כשמספיק אחים מאותו אב מתאימים.
    base_retriever = index.as_retriever(similarity_top_k=6)
    retriever = AutoMergingRetriever(
        base_retriever,
        storage_context,
        verbose=True,    # מדפיס מתי הוא ממזג ילדים לאב
    )

    query_engine = RetrieverQueryEngine.from_args(retriever)
    print("\nתשובה:", query_engine.query("ספר לי על קרן הרזרבה."))


# =================================================================
# 6) דחיסת הקשר (CONTEXT COMPRESSION)
# -----------------------------------------------------------------
# הבעיה: בקטעים שהוחזרו מעורבות משפטים רלוונטיים עם מילוי מיותר.
#        המילוי הזה מבזבז טוקנים ויכול להסיח את ה-LLM.
# הרעיון: אחרי האחזור, מוחקים את המשפטים בתוך כל קטע שאינם רלוונטיים
#         לשאלה, ושומרים רק את השימושיים.
#
# SentenceEmbeddingOptimizer נותן ציון לכל משפט מול השאילתה ושומר את
# החלק העליון (כאן ה-50% העליונים).
# =================================================================

def demo_context_compression():
    banner("6) דחיסת הקשר (CONTEXT COMPRESSION)")

    optimizer = SentenceEmbeddingOptimizer(
        percentile_cutoff=0.5,   # שומרים בערך את ה-50% הרלוונטיים ביותר
    )

    query_engine = base_index.as_query_engine(
        similarity_top_k=3,
        node_postprocessors=[optimizer],
    )

    response = query_engine.query("כמה עולה קנס על איחור בתשלום?")
    print("\nתשובה:", response)
    print("\nההקשר הדחוס שנשלח בפועל ל-LLM:")
    for n in response.source_nodes:
        print("  -", n.node.get_content())


if __name__ == "__main__":
    # הריצו אחת בכל פעם בשיעור (הפכו את השאר להערה).
    demo_query_expansion()
    demo_multi_query()
    demo_router()
    demo_reranking()
    demo_parent_child()
    demo_context_compression()
