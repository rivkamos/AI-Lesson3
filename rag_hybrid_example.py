"""
=============================================================
 חיפוש היברידי (HYBRID SEARCH) — דוגמה ללימוד (דמו RAG)
=============================================================

מה זה חיפוש היברידי?
--------------------
RAG רגיל (מה ש-rag_new.py עושה) משתמש רק בחיפוש *צפוף* / סמנטי:
השאלה והמסמכים הופכים לוקטורי אמבדינג, ואנחנו מחזירים את הקטעים
שהוקטור שלהם הכי "קרוב" במשמעות.

זה מצוין למשמעות ("איך משלמים את דמי הוועד?" מתאים ל"הוראות תשלום
דמי התחזוקה החודשיים"), אבל זה *חלש* במילות מפתח מדויקות: קודי מוצר,
שמות, מזהים, מילים נדירות, מספרים.
דוגמה: חיפוש "דירה 4B" או "חשבונית 88231" — חיפוש סמנטי עלול לסטות,
בעוד חיפוש מילות מפתח פוגע בול.

חיפוש היברידי = צפוף (וקטורים) + דליל (מילות מפתח / BM25), ממוזגים יחד.

  1) מאחזר צפוף (Dense)   -> אמבדינגס, טוב במשמעות        (סמנטי)
  2) מאחזר דליל (Sparse)  -> ניקוד מילות מפתח BM25, טוב במונחים מדויקים
  3) מיזוג (Fusion)       -> ממזג את שתי רשימות התוצאות לדירוג אחד
                            (כאן: Reciprocal Rank Fusion, "RRF")

הקובץ עצמאי לחלוטין וניתן להרצה כדי שתראו את ההבדל. הוא משתמש בכמה מסמכי
דוגמה בזיכרון במקום בקבצים שהועלו, כך שלא צריך שלב העלאה.

הרצה:
    set GEMINI_API_KEY=your_key
    python rag_hybrid_example.py

(האמבדינגס עדיין קוראים ל-Gemini, אז צריך מפתח API. החצי של BM25 /
מילות המפתח רץ לגמרי לא מקוון.)

תלות נוספת (המאחזר הדליל):
    pip install llama-index-retrievers-bm25
=============================================================
"""

import os

from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI


# =========================
# הגדרות Gemini / LLM
# (אותו רעיון כמו ב-rag_new.py)
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
# 1) מסמכי דוגמה
# באפליקציה האמיתית אלה מגיעים מ-SimpleDirectoryReader (קבצים שהועלו).
# כאן הם מקודדים ידנית כדי שהדוגמה תהיה עצמאית.
# שימו לב ל"חשבונית 88231" ול"דירה 4B" — טוקנים מדויקים שמדגימים
# למה חיפוש מילות מפתח חשוב.
# =========================

SAMPLE_TEXTS = [
    "דמי התחזוקה החודשיים של הבניין הם 350 ש\"ח ומשולמים ב-1 לכל חודש.",
    "דיירים יכולים לשלם את דמי התחזוקה בהעברה בנקאית או דרך אפליקציית הבניין.",
    "חשבונית 88231 מכסה את תיקון המעלית שבוצע במרץ.",
    "עבודות איטום הגג מתוכננות לקיץ ומשולמות מתוך קרן הרזרבה.",
    "דירה 4B דיווחה על נזילת מים בתקרת החניון.",
    "אסיפות כלליות של ועד הבית מתקיימות פעמיים בשנה.",
]

documents = [Document(text=t) for t in SAMPLE_TEXTS]

# מפצלים את המסמכים לצמתים (קטעים). שני המאחזרים עובדים על אותם צמתים —
# זה חשוב: אנחנו רוצים שהם יחפשו בתוכן זהה, רק בשיטות ניקוד שונות.
nodes = SentenceSplitter(chunk_size=256).get_nodes_from_documents(documents)


# =========================
# 2) מאחזר צפוף (סמנטי / וקטורי)
# =========================

vector_index = VectorStoreIndex(nodes)

vector_retriever = vector_index.as_retriever(
    similarity_top_k=3,
)


# =========================
# 3) מאחזר דליל (מילות מפתח / BM25)
# BM25 הוא אלגוריתם קלאסי לדירוג לפי מילות מפתח (בלי אמבדינגס, בלי API).
# הוא מתגמל חפיפה מדויקת של מונחים בין השאילתה למסמך.
# =========================

bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=3,
)


# =========================
# 4) היברידי = מיזוג צפוף + דליל
# QueryFusionRetriever מריץ את שני המאחזרים וממזג את התוצאות שלהם.
# mode="reciprocal_rerank" == Reciprocal Rank Fusion (RRF): כל תוצאה
# מקבלת ניקוד של 1/(k + דירוג) בכל רשימה, ואז סוכמים. פריטים שמדורגים
# גבוה ב*אחת* מהשיטות עולים לראש — וזו כל המטרה של היברידי.
# =========================

hybrid_retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=3,
    num_queries=1,            # קבעו >1 כדי לתת ל-LLM לייצר גם וריאציות שאילתה
    mode="reciprocal_rerank",
    use_async=False,
)


def show(title, nodes_with_scores):
    """הדפסה מסודרת של תוצאות מאחזר כדי שתוכלו להשוות בין השיטות."""
    print(f"\n--- {title} ---")
    for n in nodes_with_scores:
        score = f"{n.score:.4f}" if n.score is not None else "n/a"
        print(f"  [{score}] {n.node.get_content()}")


def compare(question: str):
    """מריץ את שלושת המאחזרים על אותה שאלה ומדפיס את התוצאות."""
    print("\n" + "=" * 60)
    print(f"שאלה: {question}")
    print("=" * 60)

    show("צפוף בלבד (וקטורים סמנטיים)", vector_retriever.retrieve(question))
    show("דליל בלבד (מילות מפתח BM25)", bm25_retriever.retrieve(question))
    show("היברידי (צפוף + דליל, RRF)", hybrid_retriever.retrieve(question))


def ask_hybrid(question: str):
    """
    תשובת RAG מלאה באמצעות המאחזר ההיברידי.
    אותו מבנה כמו ask_question() ב-rag_new.py, רק עם אחזור היברידי.
    """
    query_engine = RetrieverQueryEngine.from_args(hybrid_retriever)
    response = query_engine.query(question)

    return {
        "answer": str(response),
        "sources": [
            n.node.get_content()
            for n in response.source_nodes
        ],
    }


if __name__ == "__main__":
    # א) שאלה סמנטית — המשמעות חשובה, המילים המדויקות שונות.
    #    שימו לב איך החיפוש הצפוף מצליח כאן.
    compare("איך אפשר לשלם את דמי הוועד?")

    # ב) שאלה עם מילת מפתח מדויקת — שימו לב ל"חשבונית 88231".
    #    שימו לב איך חיפוש BM25 / מילות מפתח פוגע בטוקן המדויק בעוד
    #    חיפוש סמנטי טהור עלול לדרג אותו נמוך יותר. ההיברידי מקבל
    #    את הטוב משני העולמות.
    compare("בשביל מה הייתה חשבונית 88231?")

    # ג) תשובה מלאה שנוצרת באמצעות אחזור היברידי.
    print("\n" + "=" * 60)
    print("תשובת RAG היברידית מלאה")
    print("=" * 60)
    result = ask_hybrid("מי דיווח על נזילת המים ואיפה?")
    print("\nתשובה:", result["answer"])
    print("\nמקורות:")
    for s in result["sources"]:
        print("  -", s)
