import os

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from rag_new import (
    add_file,
    ask_question,
    ask_question_hybrid,
    ask_query_expansion,
    ask_multi_query,
    ask_router,
    ask_rerank,
    ask_parent_child,
    ask_context_compression,
)

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


@app.post("/api/chat/hybrid")
def chat_hybrid(request: QuestionRequest):

    return ask_question_hybrid(request.question)


# =========================
# טכניקות RAG מתקדמות (endpoint אחד לכל טכניקה)
# =========================

@app.post("/api/chat/query-expansion")
def chat_query_expansion(request: QuestionRequest):

    return ask_query_expansion(request.question)


@app.post("/api/chat/multi-query")
def chat_multi_query(request: QuestionRequest):

    return ask_multi_query(request.question)


@app.post("/api/chat/router")
def chat_router(request: QuestionRequest):

    return ask_router(request.question)


@app.post("/api/chat/rerank")
def chat_rerank(request: QuestionRequest):

    return ask_rerank(request.question)


@app.post("/api/chat/parent-child")
def chat_parent_child(request: QuestionRequest):

    return ask_parent_child(request.question)


@app.post("/api/chat/context-compression")
def chat_context_compression(request: QuestionRequest):

    return ask_context_compression(request.question)