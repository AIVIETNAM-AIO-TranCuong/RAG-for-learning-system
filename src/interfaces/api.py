"""
FastAPI REST API Server -- Phase 4
Cung cap cac RESTful endpoints cho RAG Learning System.

Chay server:
  uvicorn src.interfaces.api:app --reload --port 8000
Tài liệu tương tác Swagger UI:
  http://localhost:8000/docs
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.schemas import RagAnswer, Summary, QuizSet, FlashcardSet
from src.db.store import get_store
from src.pipelines.indexing import index_document, SUPPORTED_EXTENSIONS
from src.pipelines.rag import answer
from src.pipelines.learning import summarize, generate_quiz, generate_flashcards

# ============================================================
# APP INITIALIZATION & CORS
# ============================================================

app = FastAPI(
    title="RAG Learning System API",
    description="REST API cho hệ thống hỏi đáp thông minh và hỗ trợ học tập (NotebookLM style).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST & RESPONSE SCHEMAS
# ============================================================

class AskRequest(BaseModel):
    question: str = Field(..., description="Câu hỏi cần trả lời")
    document_id: Optional[str] = Field(None, description="ID tài liệu (None = tìm trên toàn bộ tài liệu)")


class SummarizeRequest(BaseModel):
    document_id: str = Field(..., description="ID tài liệu cần tóm tắt")


class QuizRequest(BaseModel):
    document_id: str = Field(..., description="ID tài liệu để tạo quiz")
    topic: Optional[str] = Field(None, description="Chủ đề cụ thể (nếu có)")
    n_questions: int = Field(default=8, ge=1, le=20, description="Số lượng câu hỏi")


class FlashcardsRequest(BaseModel):
    document_id: str = Field(..., description="ID tài liệu để tạo flashcards")
    topic: Optional[str] = Field(None, description="Chủ đề cụ thể (nếu có)")
    n_cards: int = Field(default=15, ge=1, le=30, description="Số lượng thẻ ghi nhớ")


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    total_chunks: int
    message: str


class DeleteResponse(BaseModel):
    document_id: str
    message: str


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/", tags=["General"])
def root():
    """Thông tin chào mừng và trạng thái API."""
    return {
        "system": "RAG Learning System API",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs",
    }


@app.get("/health", tags=["General"])
def health_check():
    """Kiểm tra sức khỏe hệ thống tức thì."""
    return {
        "status": "healthy",
        "llm_backend": settings.llm_backend,
        "gemini_model": settings.gemini_model,
    }


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(..., description="Tệp tài liệu cần nạp vào hệ thống"),
    force_reindex: bool = Form(True, description="Tự động ghi đè và nạp lại nếu tài liệu đã tồn tại"),
):
    """
    Nạp tệp tài liệu vào hệ thống:
    Hỗ trợ: .pdf, .docx, .pptx, .txt, .md
    1. Lưu tệp tạm thời
    2. Chạy indexing pipeline (load, chunk, embed, lưu Qdrant)
    3. Trả về document_id và tổng số chunks
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Định dạng tệp '{ext}' không được hỗ trợ. Các định dạng hỗ trợ: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    temp_dir = tempfile.mkdtemp(prefix="rag_upload_")
    temp_path = Path(temp_dir) / file.filename

    try:
        # Lưu file tải lên vào thư mục tạm
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Chạy pipeline indexing đa định dạng
        doc_id = index_document(temp_path, force_reindex=force_reindex)

        # Lấy thông tin số chunks đã nạp
        store = get_store()
        docs = store.list_documents()
        chunk_count = 0
        for d in docs:
            if d.get("document_id") == doc_id:
                chunk_count = d.get("total_chunks", 0)
                break

        return UploadResponse(
            document_id=doc_id,
            filename=file.filename,
            total_chunks=chunk_count,
            message="Nạp tài liệu thành công.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi trong quá trình nạp tài liệu: {str(e)}",
        )
    finally:
        # Dọn dẹp thư mục tạm
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/documents", response_model=List[Dict[str, Any]], tags=["Documents"])
def list_documents():
    """Liệt kê danh sách tất cả các tài liệu đã nạp trong cơ sở tri thức."""
    try:
        store = get_store()
        return store.list_documents()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi lấy danh sách tài liệu: {str(e)}",
        )


@app.delete("/documents/{doc_id}", response_model=DeleteResponse, tags=["Documents"])
def delete_document(doc_id: str):
    """Xóa hoàn toàn một tài liệu và các vector chunks liên quan khỏi Qdrant."""
    try:
        store = get_store()
        store.delete_document(doc_id)
        return DeleteResponse(
            document_id=doc_id,
            message=f"Đã xóa thành công tài liệu có ID: {doc_id}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi xóa tài liệu {doc_id}: {str(e)}",
        )


@app.post("/ask", response_model=RagAnswer, tags=["RAG"])
def ask_question(req: AskRequest):
    """
    Hỏi đáp dựa trên tài liệu (RAG):
    Retrieve -> Rerank (CrossEncoder) -> Prompt (qa.j2) -> LLM -> RagAnswer (citations inline [S1], [S2]...)
    """
    try:
        if not req.question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Câu hỏi không được để trống.",
            )
        result = answer(query=req.question.strip(), document_id=req.document_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi trả lời câu hỏi: {str(e)}",
        )


@app.post("/summarize", response_model=Summary, tags=["Learning"])
def summarize_document(req: SummarizeRequest):
    """
    Tóm tắt tài liệu:
    Tự động chọn Single-pass (nếu <= batch_size chunks) hoặc Map-Reduce (nếu > batch_size chunks).
    """
    try:
        store = get_store()
        docs = [d["document_id"] for d in store.list_documents()]
        if req.document_id not in docs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Không tìm thấy tài liệu có ID: {req.document_id}",
            )

        result = summarize(document_id=req.document_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi tóm tắt tài liệu: {str(e)}",
        )


@app.post("/quiz", response_model=QuizSet, tags=["Learning"])
def create_quiz(req: QuizRequest):
    """
    Tạo bộ câu hỏi trắc nghiệm MCQ (A, B, C, D) từ tài liệu:
    Có thể chỉ định chủ đề cụ thể (topic) hoặc để None để tạo từ toàn bộ tài liệu.
    """
    try:
        result = generate_quiz(
            document_id=req.document_id,
            topic=req.topic,
            n_questions=req.n_questions,
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi tạo câu hỏi trắc nghiệm: {str(e)}",
        )


@app.post("/flashcards", response_model=FlashcardSet, tags=["Learning"])
def create_flashcards(req: FlashcardsRequest):
    """
    Tạo bộ thẻ ghi nhớ Flashcards (front / back) từ tài liệu:
    Hỗ trợ ôn tập và ghi nhớ các khái niệm cốt lõi.
    """
    try:
        result = generate_flashcards(
            document_id=req.document_id,
            topic=req.topic,
            n_cards=req.n_cards,
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi tạo flashcards: {str(e)}",
        )
