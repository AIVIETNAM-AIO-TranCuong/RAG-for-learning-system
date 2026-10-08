from typing import Optional, List
from pydantic import BaseModel


# ============================================================
# 1. CORE DATA SCHEMAS (Metadata & Retrieval)
# ============================================================

class ChunkMetadata(BaseModel):
    """Lưu trữ thông tin nguồn của một đoạn văn bản (chunk)."""
    document_id: str
    filename: str
    source: str
    page: int
    chunk_id: str
    section: Optional[str] = None


class RetrievedChunk(BaseModel):
    """Chunk văn bản được retrieve từ Qdrant kèm relevance score."""
    content: str
    metadata: ChunkMetadata
    score: float


class Citation(BaseModel):
    """Trích dẫn nguồn tương ứng với inline marker [S1], [S2], ..."""
    marker: str                 # "S1", "S2", ...
    chunk: RetrievedChunk


# ============================================================
# 2. RAG OUTPUT SCHEMAS
# ============================================================

class RagAnswer(BaseModel):
    """Câu trả lời RAG hoàn chỉnh với trích dẫn nguồn."""
    answer: str                        # Có inline markers [S1][S2]...
    citations: List[Citation]
    chunks: List[RetrievedChunk]       # Tất cả chunks đã dùng để generate


# ============================================================
# 3. LEARNING FEATURE SCHEMAS
# ============================================================

class Summary(BaseModel):
    """Bản tóm tắt tài liệu."""
    content: str                        # Nội dung tóm tắt
    document_id: str                    # ID tài liệu được tóm tắt
    total_chunks: int                   # Tổng số chunks đã xử lý
    method: str                         # "single" hoặc "map_reduce"


class QuizItem(BaseModel):
    """Một câu hỏi trắc nghiệm (MCQ)."""
    question: str
    options: List[str]                  # Danh sách 4 đáp án [A, B, C, D]
    answer: str                         # Đáp án đúng: "A" | "B" | "C" | "D"
    explanation: str                    # Giải thích vì sao đáp án đúng


class QuizSet(BaseModel):
    """Bộ câu hỏi trắc nghiệm từ tài liệu."""
    items: List[QuizItem]
    document_id: str
    topic: Optional[str] = None         # Chủ đề cụ thể (nếu có)


class Flashcard(BaseModel):
    """Một thẻ ghi nhớ (flashcard)."""
    front: str                          # Mặt trước: câu hỏi / khái niệm
    back: str                           # Mặt sau: đáp án / giải thích


class FlashcardSet(BaseModel):
    """Bộ flashcards từ tài liệu."""
    cards: List[Flashcard]
    document_id: str
    topic: Optional[str] = None

