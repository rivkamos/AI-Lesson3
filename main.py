import os
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from fastapi.responses import JSONResponse
from rag import add_document, ask_question, extract_docx, extract_pdf, extract_txt, get_chroma_collection, get_file_type, semantic_chunking, split_to_chunks

# ==================================================
# FastAPI Setup
# ==================================================

app = FastAPI(
    title="Building Committee AI",
    description="RAG System for Building Management",
    version="1.0.0"
)

# ==================================================
# CORS
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================================================
# Upload Folder
# ==================================================

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

# ==================================================
# Request Models
# ==================================================

class QuestionRequest(BaseModel):
    question: str

# ==================================================
# Health Check
# ==================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "message": "Backend is running"
    }

# ==================================================
# Upload PDF
# ==================================================


@app.post("/api/documents/upload")
async def upload_file(file: UploadFile = File(...)):

    try:
        file_type = get_file_type(file.filename)

        if not file_type:
            return {"error": "Unsupported file type"}

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)

        content = await file.read()

        with open(file_path, "wb") as f:
            f.write(content)

        # -------------------------
        # Extract text per type
        # -------------------------

        if file_type == "txt":
            text = extract_txt(content)

        elif file_type == "pdf":
            text = extract_pdf(file_path)

        elif file_type == "docx":
            text = extract_docx(file_path)

        else:
            return {"error": "Unknown type"}

        if not text.strip():
            return {"error": "No text extracted"}

        # -------------------------
        # Chunking
        # -------------------------

        chunks = semantic_chunking(text)
        # -------------------------
        # Save to Chroma
        # -------------------------

        for i, chunk in enumerate(chunks):
            add_document(
                document_id=str(uuid.uuid4()),
                text=chunk,
                source=file.filename,
                chunk_index=i,
                total_chunks=len(chunks)
            )

        return {
            "success": True,
            "file_type": file_type,
            "chunks": len(chunks)
        }

    except Exception as e:
        return {"error": str(e)}
# ==================================================
# debug chroma Endpoint
# ==================================================
@app.get("/api/debug/count")
def count():
    collection = get_chroma_collection() 
    return {
        "count": collection.count()
    }

# ==================================================
# chat Endpoint
# ==================================================
@app.post("/api/chat")
def chat(request: QuestionRequest):

    result = ask_question(
        request.question
    )

    return result

# ==================================================
# Root Endpoint
# ==================================================

@app.get("/")
def root():
    return {
        "application": "Building Committee AI",
        "version": "1.0.0",
        "status": "running"
    }