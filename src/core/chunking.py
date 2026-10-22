# -*- coding: utf-8 -*-
"""
Module Chunking đa phương pháp:
1. Recursive Chunking: Phân đoạn ký tự đệ quy theo kích thước và độ chồng lấp.
2. Semantic Chunking: Phân đoạn ngữ nghĩa dựa trên sự tương đồng embedding giữa các câu,
   hỗ trợ 3 phương pháp ngưỡng tự động:
     - 'interquartile': Ngưỡng Q3 + 1.5 * IQR (Phương pháp đề xuất từ thực nghiệm)
     - 'standard_deviation': Ngưỡng mean + k * std
     - 'percentile': Ngưỡng phân vị thứ p (mặc định 90%)
"""
import re
from typing import List, Optional, Literal, Dict, Any
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

ThresholdType = Literal["interquartile", "standard_deviation", "percentile"]


def split_text_into_sentences(text: str) -> List[str]:
    """
    Tách đoạn văn thành danh sách các câu đơn vị.
    Hỗ trợ cả dấu kết thúc câu tiếng Việt và tiếng Anh.
    """
    # Regex tách theo dấu chấm, chấm than, chấm hỏi, xuống dòng kép
    raw_sentences = re.split(r'(?<=[.?!;:\n])\s+', text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    if not sentences and text.strip():
        sentences = [text.strip()]
    return sentences


def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Tính khoảng cách cosine giữa 2 vector: 1 - cosine_similarity."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 1.0
    cosine_sim = np.dot(v1, v2) / (norm1 * norm2)
    return float(np.clip(1.0 - cosine_sim, 0.0, 2.0))


class SemanticChunker:
    """
    Phân đoạn văn bản dựa trên khoảng cách ngữ nghĩa giữa các câu liên tiếp.
    """

    def __init__(
        self,
        embeddings=None,
        threshold_type: ThresholdType = "interquartile",
        threshold_amount: Optional[float] = None,
        min_sentences_per_chunk: int = 1,
        max_chunk_chars: int = 2500,
    ):
        self.embeddings = embeddings
        self.threshold_type = threshold_type
        self.threshold_amount = threshold_amount
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.max_chunk_chars = max_chunk_chars

    def _get_embeddings(self):
        if self.embeddings is None:
            from src.db.store import get_store
            self.embeddings = get_store().embeddings
        return self.embeddings

    def calculate_breakpoint_threshold(self, distances: List[float]) -> float:
        """
        Tính giá trị ngưỡng khoảng cách để xác định điểm cắt chunk.
        """
        if not distances:
            return 0.5

        arr = np.array(distances)

        if self.threshold_type == "interquartile":
            # IQR: Q3 + 1.5 * (Q3 - Q1)
            q75, q25 = np.percentile(arr, [75, 25])
            iqr = q75 - q25
            multiplier = self.threshold_amount if self.threshold_amount is not None else 1.5
            return float(q75 + multiplier * iqr)

        elif self.threshold_type == "standard_deviation":
            # Mean + k * Std
            mean = np.mean(arr)
            std = np.std(arr)
            k = self.threshold_amount if self.threshold_amount is not None else 1.0
            return float(mean + k * std)

        elif self.threshold_type == "percentile":
            # Phân vị thứ p
            p = self.threshold_amount if self.threshold_amount is not None else 90.0
            return float(np.percentile(arr, p))

        else:
            raise ValueError(f"Không hỗ trợ threshold_type='{self.threshold_type}'")

    def split_text(self, text: str) -> List[str]:
        """Tách một chuỗi văn bản thuần thành danh sách các chunk ngữ nghĩa."""
        sentences = split_text_into_sentences(text)
        if len(sentences) <= self.min_sentences_per_chunk:
            return [text] if text.strip() else []

        embeddings_model = self._get_embeddings()
        vectors = embeddings_model.embed_documents(sentences)
        vectors_np = [np.array(v, dtype=np.float32) for v in vectors]

        # Tính khoảng cách giữa câu i và i+1
        distances = []
        for i in range(len(vectors_np) - 1):
            dist = cosine_distance(vectors_np[i], vectors_np[i + 1])
            distances.append(dist)

        threshold = self.calculate_breakpoint_threshold(distances)

        # Gom các câu thành chunks
        chunks = []
        current_sentences = [sentences[0]]
        current_len = len(sentences[0])

        for i, dist in enumerate(distances):
            next_sentence = sentences[i + 1]
            next_len = len(next_sentence)

            # Cắt nếu khoảng cách > threshold hoặc vượt kích thước tối đa
            is_breakpoint = dist > threshold
            is_oversized = (current_len + next_len + 1) > self.max_chunk_chars

            if (is_breakpoint or is_oversized) and len(current_sentences) >= self.min_sentences_per_chunk:
                chunks.append(" ".join(current_sentences).strip())
                current_sentences = [next_sentence]
                current_len = next_len
            else:
                current_sentences.append(next_sentence)
                current_len += next_len + 1

        if current_sentences:
            chunks.append(" ".join(current_sentences).strip())

        return [c for c in chunks if c.strip()]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Tách danh sách Document và giữ nguyên metadata tương ứng."""
        output_docs: List[Document] = []
        for doc in documents:
            text = doc.page_content.strip()
            if not text:
                continue
            text_chunks = self.split_text(text)
            for chunk_idx, chunk_text in enumerate(text_chunks):
                new_meta = dict(doc.metadata) if doc.metadata else {}
                new_meta["chunk_part"] = chunk_idx
                output_docs.append(Document(page_content=chunk_text, metadata=new_meta))
        return output_docs


def create_chunker(
    method: Literal["recursive", "semantic"] = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    semantic_threshold_type: ThresholdType = "interquartile",
    semantic_threshold_amount: Optional[float] = None,
    embeddings=None
):
    """
    Factory tạo bộ chunker tương ứng với cấu hình thực nghiệm.
    """
    if method == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    elif method == "semantic":
        return SemanticChunker(
            embeddings=embeddings,
            threshold_type=semantic_threshold_type,
            threshold_amount=semantic_threshold_amount
        )
    else:
        raise ValueError(f"Phương pháp chunking không hợp lệ: '{method}'")
