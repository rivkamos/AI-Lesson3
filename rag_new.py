import os
import chromadb

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader,
    Settings
)

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

# =========================
# Gemini / LLM config
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
# Chroma setup
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
# Index (lazy loaded)
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
# Add document (replace ALL your indexing logic)
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
# Chat / RAG query (replace ask_question)
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