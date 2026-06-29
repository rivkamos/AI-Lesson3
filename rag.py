from multiprocessing import context
from urllib import response

from ollama import embeddings, chat
import chromadb
import os
from pathlib import Path
from docx import Document
from pypdf import PdfReader
import re
import numpy as np
import uuid
from google import genai
from guardrails import Guard
from schemas import AnswerSchema

#client = genai.Client()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

os.makedirs("./chroma", exist_ok=True)

client_chroma = chromadb.PersistentClient(path="./chroma")

collection = client_chroma.get_or_create_collection(
    name="building_docs"
)


guard = Guard.from_pydantic(AnswerSchema, num_reasks=2)

def get_chroma_collection():
    return collection

def create_embedding_ollama(text: str):
    response = embeddings(
        model="bge-m3",
        prompt=text
    )

    return response["embedding"]

def create_embedding(text: str):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=f"passage: {text}"
    )
    print("=========Embedding Response===========")
    print(response)
    print("=====================================")
    return response.embeddings[0].values

def create_query_embedding_ollama(text: str):
    # Use instruction-tuned embedding for queries
    response = embeddings(
        model="bge-m3",
        prompt=f"Query: {text}"
    )
    return response["embedding"]

def create_query_embedding(text: str):
    return client.models.embed_content(
        model="gemini-embedding-001",
        contents=f"query: {text}"
    ).embeddings[0].values

def create_document_embedding_ollama(text: str):
    # Standard embedding for documents
    response = embeddings(
        model="bge-m3",
        prompt=text
    )
    return response["embedding"]

def create_document_embedding(text: str):
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=f"passage: {text}"
    )

    return response.embeddings[0].values

def create_embeddings_batch(texts):
    embeddings_list = []

    for text in texts:
        embedding = create_embedding(text)
        embedding = normalize_embedding(
            embedding
        ).tolist()

        embeddings_list.append(embedding)

    return embeddings_list

def normalize_embedding(embedding):
    return np.array(embedding) / np.linalg.norm(np.array(embedding))

def index_document(text: str, source: str):
    chunks = semantic_chunking(text)

    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):

        embedding = normalize_embedding(
            create_embedding(chunk)
        ).tolist()

        chunk_id = f"{source}::{i}::{uuid.uuid4()}"

        collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{
                "source": source,
                "chunk_index": i,
                "total_chunks": total_chunks,
                "char_length": len(chunk)
            }]
        )

def add_document(
    document_id: str,
    text: str,
    source: str,
    chunk_index: int,
    total_chunks: int
):
    embedding = normalize_embedding(
        create_embedding(text)
    ).tolist()

    collection.upsert(
        ids=[document_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[
            {
                "source": source,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "char_length": len(text)
            }
        ]
    )

def extract_txt(content: bytes):
    return content.decode("utf-8", errors="ignore")


def extract_docx(file_path: str):
    doc = Document(file_path)

    text = "\n".join([p.text for p in doc.paragraphs])

    return text

def extract_pdf(file_path: str):
    reader = PdfReader(file_path)

    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def get_file_type(filename: str):
    ext = Path(filename).suffix.lower()

    if ext == ".txt":
        return "txt"
    if ext == ".pdf":
        return "pdf"
    if ext == ".docx":
        return "docx"

    return None

def ask_question(question: str):

    if any(is_injection(question) ):
        raise ValueError("Potential prompt injection detected")

    query_embedding = normalize_embedding(
        create_query_embedding(question)
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"],
        #where={"user_id": user_id}  # Optional: Filter by user_id if needed"}
        #where_document={"$contains": "transformer"}
    )

    if not results["metadatas"][0]:
        return {
            "answer": "לא נמצא מידע במסמכים.",
            "sources": []
        }

    # לקחת את המסמך הכי רלוונטי
    best_source = results["metadatas"][0][0]["source"]

    full_doc = collection.get(
        where={"source": best_source},
        include=["documents", "metadatas"]
    )

    # מיון לפי סדר chunk
    sorted_pairs = sorted(
        zip(full_doc["documents"], full_doc["metadatas"]),
        key=lambda x: x[1]["chunk_index"]
    )

    context = "\n\n".join(doc for doc, _ in sorted_pairs)
    
    if not validate_context(context):
        return {
            "answer": "לא נמצא מידע מהימן במסמכים.",
            "sources": []
        }
    
    prompt = f"""
אתה עוזר ועד בית.

ענה רק על בסיס הטקסט.

אם אין תשובה: כתוב "לא נמצא מידע במסמכים".

טקסט:
{context}

שאלה:
{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

# return {
#     "answer": response.text,
#     "sources": full_doc["metadatas"]
# }

    return guard.parse({
        "answer": response.text,
        "sources_used": full_doc["metadatas"]
    })

def ask_question_ollama(question: str):

    query_embedding = normalize_embedding(
        create_query_embedding(question)
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas"]
    )

    if not results["metadatas"][0]:
        return {
            "answer": "לא נמצא מידע במסמכים.",
            "sources": []
        }

    # 🎯 לקחת source של התוצאה הכי רלוונטית
    best_source = results["metadatas"][0][0]["source"]

    # 🎯 להביא את כל המסמך
    full_doc = collection.get(
        where={"source": best_source},
        include=["documents", "metadatas"]
    )

    # 🔧 למיין לפי סדר אמיתי
    sorted_pairs = sorted(
        zip(full_doc["documents"], full_doc["metadatas"]),
        key=lambda x: x[1]["chunk_index"]
    )

    context = "\n\n".join([doc for doc, _ in sorted_pairs])

    prompt = f"""
אתה מערכת שליפת מידע בלבד.

כללים:
- אסור לכתוב תשובה מקורית
- אסור להוסיף מילים
- מותר רק להעתיק משפטים מהטקסט
- אם אין מידע → כתוב: לא נמצא מידע במסמכים

טקסט:
{context}

שאלה:
{question}

פלט (רק משפטים מהטקסט):
"""
    
    response = chat(
         model=" llama3.2:latest",#qwen2.5:3b
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    print("=========context===========")
    print(context)
    print("============================")
    print("=========Response===========")
    print(response)
    print("============================")

    return {
        "answer": response["message"]["content"],
        "sources": full_doc["metadatas"]
    }

def semantic_chunking(text: str):
    # Split by headers/sections first, then apply size limits
    sections = re.split(r'\n(?=[A-Zא-ת])', text)
    chunks = []
    for section in sections:
        if len(section) > 800:
            sub_chunks = split_to_chunks(section, max_chars=600, overlap=80)
            chunks.extend(sub_chunks)
        else:
            chunks.append(section)
    return chunks

def split_to_chunks(text: str):
    chunks = []

    sections = re.split(
        r'\n\s*(?=\d+\.)',
        text
    )

    for section in sections:
        section = section.strip()

        if section:
            chunks.append(section)

    return chunks

def is_injection(text: str) -> bool:
    blacklist = ["ignore previous instructions", "system prompt", "reveal"]
    return any(b in text.lower() for b in blacklist)

def validate_context(context: str) -> bool:
    if not context or len(context.strip()) < 10:
        return False

    blacklist = [
        "ignore previous instructions",
        "system prompt",
        "reveal prompt",
        "exe",
        "msi"
    ]

    return not any(b in context.lower() for b in blacklist)