import os
import chromadb

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader,
    Settings
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

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

UPLOAD_FOLDER = "uploads"

# =========================
# הגדרות Gemini / LLM
# =========================

Settings.llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=os.environ["GEMINI_API_KEY"]
)

Settings.embed_model = GoogleGenAIEmbedding(
    model="text-embedding-004",
    api_key=os.environ["GEMINI_API_KEY"]
)

# =========================
# הקמת Chroma
# =========================

chroma_client = chromadb.PersistentClient(path="./chroma")

collection = chroma_client.get_or_create_collection(
    name="building_docs"
)

vector_store = ChromaVectorStore(
    chroma_collection=collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

# =========================
# אינדקס (נטען בעצלתיים / lazy)
# =========================

_index = None


def get_index():
    global _index

    if _index is None:
        _index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context
        )

    return _index


# =========================
# הוספת מסמך (מחליף את כל לוגיקת האינדוקס)
# =========================

def add_file(file_path: str):
    docs = SimpleDirectoryReader(
        input_files=[file_path]
    ).load_data()

    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context
    )

    global _index
    _index = index

    return True


# =========================
# שאילתת צ'אט / RAG (ask_question)
# =========================

def ask_question(question: str):

    index = get_index()

    query_engine = index.as_query_engine(
        similarity_top_k=5
    )

    response = query_engine.query(question)

    return {
        "answer": str(response),
        "sources": [
            {
                "source": n.metadata.get("file_name"),
                "page": n.metadata.get("page_label")
            }
            for n in response.source_nodes
        ]
    }


# =========================
# שאילתת RAG היברידית (צפוף + דליל)
# =========================
#
# RAG רגיל (ask_question למעלה) משתמש רק בחיפוש צפוף / וקטורי:
# הוא מתאים לפי משמעות באמצעות אמבדינגס. זה מפספס מילות מפתח מדויקות
# (מזהים, שמות, קודים, מספרים).
#
# חיפוש היברידי מוסיף מאחזר שני וממזג את השניים:
#   1) צפוף (DENSE)  -> מאחזר וקטורי מעל Chroma  (טוב במשמעות)
#   2) דליל (SPARSE) -> מאחזר מילות מפתח BM25     (טוב במונחים מדויקים)
#   3) מיזוג (FUSION) -> Reciprocal Rank Fusion (RRF) ממזג את שני הדירוגים
#
# הערה: Chroma שומר את האמבדינגס אבל לא את הטקסט הגולמי ש-BM25 צריך,
# לכן אנחנו קוראים את הטקסט בחזרה מהקבצים שהועלו כדי לבנות את BM25.

def _load_nodes_from_uploads():
    """טוען כל קובץ שהועלה ומפצל אותו לצמתי טקסט עבור BM25."""
    docs = SimpleDirectoryReader(
        input_dir=UPLOAD_FOLDER
    ).load_data()

    return SentenceSplitter(
        chunk_size=512
    ).get_nodes_from_documents(docs)


def ask_question_hybrid(question: str):

    index = get_index()

    # 1) צפוף (DENSE): מאחזר סמנטי / וקטורי (מתוך Chroma)
    vector_retriever = index.as_retriever(
        similarity_top_k=5
    )

    # 2) דליל (SPARSE): מאחזר מילות מפתח BM25 (מתוך הטקסט שהועלה)
    nodes = _load_nodes_from_uploads()

    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=5
    )

    # 3) מיזוג (FUSION): ממזג צפוף + דליל עם Reciprocal Rank Fusion
    hybrid_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=5,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False
    )

    query_engine = RetrieverQueryEngine.from_args(
        hybrid_retriever
    )

    response = query_engine.query(question)

    return {
        "answer": str(response),
        "mode": "hybrid (vector + BM25)",
        "sources": [
            {
                "source": n.metadata.get("file_name"),
                "page": n.metadata.get("page_label")
            }
            for n in response.source_nodes
        ]
    }


# =================================================================
# טכניקות RAG מתקדמות (פונקציה אחת = endpoint אחד)
# כל טכניקה פותרת נקודת חולשה אחת של RAG בסיסי. כולן רצות מעל אותו
# אינדקס Chroma, חוץ מאב-ילד שצריך היררכיית צמתים ולכן נבנה מחדש מתוך
# הקבצים שהועלו.
# =================================================================

def _format_sources(source_nodes):
    """פורמט מקורות משותף (אותו מבנה כמו ask_question)."""
    return [
        {
            "source": n.metadata.get("file_name"),
            "page": n.metadata.get("page_label"),
            "text": n.node.get_content()[:200],
        }
        for n in source_nodes
    ]


# -----------------------------------------------------------------
# 1) הרחבת שאילתה (QUERY EXPANSION) באמצעות HyDE
# הבעיה: שאילתה קצרה / מעורפלת עם מעט מדי מילים מתאימה בצורה גרועה.
# הרעיון: ה-LLM כותב קודם תשובה היפותטית, ואנחנו מחפשים עם הטקסט העשיר
#         הזה במקום עם השאלה החשופה.
# -----------------------------------------------------------------

def ask_query_expansion(question: str):
    #יוצר מנוע חיפוש רגיל על האינדקס,
    #  שמחזיר את 5 המסמכים הדומים ביותר לשאילתה.
    base_qe = get_index().as_query_engine(
        similarity_top_k=5
    )
# היא טכניקה שיוצרת תשובה היפותטית לשאלת המשתמש
#  ומשתמשת בה כדי לבצע חיפוש סמנטי
#  מדויק יותר אחר מסמכים רלוונטיים.
    hyde = HyDEQueryTransform(include_original=True)
    # הוא רכיב שמבצע טרנספורמציה
    #  (שינוי או הרחבה) של שאילתת המשתמש לפני החיפוש
    # , כדי לשפר את איכות שליפת המידע והתשובות
    qe = TransformQueryEngine(base_qe, query_transform=hyde)

    response = qe.query(question)

    return {
        "answer": str(response),
        "mode": "query expansion (HyDE)",
        "sources": _format_sources(response.source_nodes),
    }


# -----------------------------------------------------------------
# 2) ריבוי שאילתות (MULTI-QUERY RETRIEVAL)
# הבעיה: ניסוח אחד מוצא רק קטעים שמנוסחים כמוהו.
# הרעיון: ה-LLM מייצר כמה ניסוחים שונים, מאחזרים לכל אחד וממזגים את כל
#         התוצאות (Reciprocal Rank Fusion).
# -----------------------------------------------------------------

def ask_multi_query(question: str):
    vector_retriever = get_index().as_retriever(
        similarity_top_k=5
    )
#רכיב שתפקידו לבצע חיפוש
#  באמצעות כמה שאילתות ולאחד את התוצאות.
    multi_query_retriever = QueryFusionRetriever(
        [vector_retriever],
        similarity_top_k=5,
        num_queries=4,            # ה-LLM כותב 4 וריאציות של השאלה
        mode="reciprocal_rerank",
        use_async=False,
    )

    query_engine = RetrieverQueryEngine.from_args(
        multi_query_retriever
    )

    response = query_engine.query(question)

    return {
        "answer": str(response),
        "mode": "multi-query (4 variations, fused)",
        "sources": _format_sources(response.source_nodes),
    }


# -----------------------------------------------------------------
# 3) מנוע שאילתות עם ניתוב (ROUTER QUERY ENGINE)
# הבעיה: שאלות שונות שייכות למקורות שונים; לשלוח כל שאלה לכולם זה
#         בזבזני ורועש.
# הרעיון: ה-LLM קורא את התיאור של כל מנוע ובוחר את המתאים.
# (דמו: שני מנועים מעל אותו אינדקס — בייצור אלה היו מקורות נתונים שונים.)
# -----------------------------------------------------------------

def ask_router(question: str):
    index = get_index()

    payments_qe = index.as_query_engine(similarity_top_k=3)
    meetings_qe = index.as_query_engine(similarity_top_k=3)
    meetings_it = index.as_query_engine(similarity_top_k=3)

    tools = [
        QueryEngineTool.from_defaults(
            query_engine=payments_qe,
            description="דמי ועד, תשלומים, קנסות, וקרן הרזרבה.",
        ),
        QueryEngineTool.from_defaults(
            query_engine=meetings_qe,
            description="אסיפות הוועד, הצבעות, וכללי קבלת החלטות.",
        ),
        QueryEngineTool.from_defaults(
            query_engine=meetings_it,
            description="דיווחים, , וניהול אינטראקטיבי.",
        ),
    ]

    router = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=tools,
    )

    response = router.query(question)

    return {
        "answer": str(response),
        "mode": "router (LLM picks the engine)",
        "sources": _format_sources(response.source_nodes),
    }


# -----------------------------------------------------------------
# 4) דירוג מחדש (RERANKING)
# הבעיה: "הכי קרוב" וקטורית זה לא אותו דבר כמו "הכי רלוונטי".
# הרעיון: מאחזרים יותר מדי (top 8), ואז ה-LLM נותן ציון מחדש ושומר את ה-3 הטובים.
# -----------------------------------------------------------------

def ask_rerank(question: str):
    reranker = LLMRerank(
        top_n=3,
        choice_batch_size=5,
    )

    query_engine = get_index().as_query_engine(
        similarity_top_k=8,                  # מאחזרים יותר מדי
        node_postprocessors=[reranker],      # ואז מדרגים מחדש עד ל-3
    )

    response = query_engine.query(question)

    return {
        "answer": str(response),
        "mode": "reranking (over-fetch 8 -> best 3)",
        "sources": _format_sources(response.source_nodes),
    }


# -----------------------------------------------------------------
# 5) אחזור אב-ילד (PARENT-CHILD) — מיזוג אוטומטי
# הבעיה: קטעים קטנים מתאימים בדיוק אך חסרי הקשר.
# הרעיון: מאנדקסים ילדים קטנים להתאמה, אבל מחזירים את האב הגדול יותר
#         כשכמה ילדים של אותו אב מתאימים.
# צריך היררכיית צמתים + docstore, לכן בונים מחדש מהקבצים שהועלו
# (Chroma לבדו לא שומר את מבנה האב/ילד).
# -----------------------------------------------------------------

def ask_parent_child(question: str):
    docs = SimpleDirectoryReader(
        input_dir=UPLOAD_FOLDER
    ).load_data()

    # שכבות: גדול -> בינוני -> קטן
    # הערה: גודל הקטע הקטן ביותר חייב להיות גדול מאורך המטא-דאטה
    # (אחרת LlamaIndex זורק "Metadata length is longer than chunk size").
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[512, 256, 128]
    )
    all_nodes = node_parser.get_nodes_from_documents(docs)
    leaf_nodes = get_leaf_nodes(all_nodes)

    # ה-docstore מחזיק את כל הצמתים; רק העלים מאונדקסים/מוטמעים
    docstore = SimpleDocumentStore()
    docstore.add_documents(all_nodes)
    storage = StorageContext.from_defaults(docstore=docstore)

    index = VectorStoreIndex(leaf_nodes, storage_context=storage)

    base_retriever = index.as_retriever(similarity_top_k=6)
    retriever = AutoMergingRetriever(base_retriever, storage)

    query_engine = RetrieverQueryEngine.from_args(retriever)
    response = query_engine.query(question)

    return {
        "answer": str(response),
        "mode": "parent-child (auto-merging)",
        "sources": _format_sources(response.source_nodes),
    }


# -----------------------------------------------------------------
# 6) דחיסת הקשר (CONTEXT COMPRESSION)
# הבעיה: בקטעים שהוחזרו מעורבים משפטים רלוונטיים עם מילוי מיותר.
# הרעיון: אחרי האחזור, מוחקים את המשפטים הלא-רלוונטיים בתוך כל קטע,
#         ושומרים רק את הרלוונטיים ביותר (כאן ה-50% העליונים).
# -----------------------------------------------------------------

def ask_context_compression(question: str):
    optimizer = SentenceEmbeddingOptimizer(
        percentile_cutoff=0.5,
    )

    query_engine = get_index().as_query_engine(
        similarity_top_k=5,
        node_postprocessors=[optimizer],
    )

    response = query_engine.query(question)

    return {
        "answer": str(response),
        "mode": "context compression (keep top 50% sentences)",
        "sources": _format_sources(response.source_nodes),
    }