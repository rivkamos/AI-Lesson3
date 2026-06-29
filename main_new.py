import os

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from rag_new import add_file, ask_question

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app = FastAPI(
    title="Building Committee AI",
    description="RAG System for Building Management",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str


@app.post("/api/documents/upload")
async def upload_file(file: UploadFile = File(...)):

    path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    with open(path, "wb") as f:
        f.write(await file.read())

    add_file(path)

    return {
        "success": True
    }


@app.post("/api/chat")
def chat(request: QuestionRequest):

    return ask_question(request.question)