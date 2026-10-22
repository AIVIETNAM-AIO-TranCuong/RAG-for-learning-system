# -*- coding: utf-8 -*-
"""
Pydantic Schemas cho hệ thống Đánh giá và Thực nghiệm RAG (RAGAS).
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class EvalSample(BaseModel):
    """Một mẫu câu hỏi kiểm thử kèm đáp án chuẩn vàng (Reference Ground Truth)."""
    id: str
    question: str
    reference_answer: str
    target_document: Optional[str] = None
    category: Optional[str] = None


class RagasScores(BaseModel):
    """4 chỉ số đánh giá RAGAS (thang điểm [0.0, 1.0])."""
    context_precision: float = Field(ge=0.0, le=1.0, description="Độ chính xác ngữ cảnh (ưu tiên chunk đúng)")
    context_recall: float = Field(ge=0.0, le=1.0, description="Độ bao phủ dữ kiện chuẩn")
    faithfulness: float = Field(ge=0.0, le=1.0, description="Độ trung thực của câu trả lời từ ngữ cảnh")
    answer_relevancy: float = Field(ge=0.0, le=1.0, description="Độ liên quan giữa câu hỏi và phản hồi")


class EvalResultItem(BaseModel):
    """Kết quả đánh giá chi tiết cho 1 câu hỏi."""
    sample_id: str
    question: str
    reference_answer: str
    generated_answer: str
    retrieved_contexts: List[str]
    scores: RagasScores


class ExperimentSummary(BaseModel):
    """Kết quả tổng hợp của 1 cấu hình thực nghiệm."""
    experiment_name: str
    config_name: str
    total_chunks: int
    n_samples: int
    avg_scores: RagasScores
    items: List[EvalResultItem] = []
