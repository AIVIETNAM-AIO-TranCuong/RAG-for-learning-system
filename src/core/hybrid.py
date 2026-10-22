# -*- coding: utf-8 -*-
"""
Module Hybrid Search: Kết hợp Dense Embedding và Sparse BM25.
Thuật toán hợp nhất: Reciprocal Rank Fusion (RRF).
"""
import re
from typing import List, Dict, Optional, Tuple
from rank_bm25 import BM25Okapi

from src.core.schemas import RetrievedChunk, ChunkMetadata
from src.db.store import QdrantStore, get_store


def tokenize_text(text: str) -> List[str]:
    """
    Tách từ cho BM25 hỗ trợ cả tiếng Việt và thuật ngữ kỹ thuật viết tắt (LoRA, DPO, RLHF, SFT).
    Chuyển về chữ thường và giữ lại các token từ ngữ/ký tự số hợp lệ.
    """
    if not text:
        return []
    # Tách theo từ hoặc chuỗi ký tự chữ/số
    tokens = re.findall(r'(?u)\b\w+\b', text.lower())
    return [t for t in tokens if len(t) > 1 or t.isdigit()]


class BM25Retriever:
    """
    Bộ truy xuất từ khóa thưa (Sparse Keyword Retriever) dựa trên BM25 Okapi.
    """

    def __init__(self, chunks: List[RetrievedChunk]):
        self.chunks = chunks
        self.corpus_tokens = [tokenize_text(c.content) for c in chunks]
        # Xử lý trường hợp corpus rỗng
        if not self.corpus_tokens or all(len(tokens) == 0 for tokens in self.corpus_tokens):
            self.bm25 = None
        else:
            self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """
        Tìm kiếm các chunk có điểm BM25 cao nhất theo câu truy vấn.
        """
        if not self.chunks or self.bm25 is None:
            return []

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return self.chunks[:top_k]

        scores = self.bm25.get_scores(query_tokens)
        scored_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in scored_indices:
            orig = self.chunks[idx]
            # Tạo bản sao chunk với điểm BM25
            results.append(RetrievedChunk(
                content=orig.content,
                metadata=orig.metadata,
                score=float(scores[idx])
            ))
        return results


def reciprocal_rank_fusion(
    dense_chunks: List[RetrievedChunk],
    sparse_chunks: List[RetrievedChunk],
    k: int = 60,
    top_k: int = 5
) -> List[RetrievedChunk]:
    """
    Thuật toán Reciprocal Rank Fusion (RRF) chuẩn công nghiệp:
      RRF_Score(d) = 1 / (k + rank_dense(d)) + 1 / (k + rank_sparse(d))
    
    Args:
        dense_chunks: Danh sách chunks từ Dense Embedding Vector Search
        sparse_chunks: Danh sách chunks từ Sparse BM25 Keyword Search
        k: Hằng số làm mịn rank (mặc định k = 60 theo Cormack et al. 2009)
        top_k: Số lượng chunks cuối cùng sau hợp nhất
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, RetrievedChunk] = {}

    def get_chunk_key(chunk: RetrievedChunk) -> str:
        if chunk.metadata and chunk.metadata.chunk_id:
            return chunk.metadata.chunk_id
        return hash(chunk.content[:80])

    # 1. Tính điểm từ Dense Channel
    for rank, chunk in enumerate(dense_chunks, 1):
        key = get_chunk_key(chunk)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (k + rank))
        chunk_map[key] = chunk

    # 2. Tính điểm từ Sparse BM25 Channel
    for rank, chunk in enumerate(sparse_chunks, 1):
        key = get_chunk_key(chunk)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (k + rank))
        if key not in chunk_map:
            chunk_map[key] = chunk

    # 3. Sắp xếp theo RRF score giảm dần
    sorted_keys = sorted(rrf_scores.keys(), key=lambda key: rrf_scores[key], reverse=True)[:top_k]

    fused_results: List[RetrievedChunk] = []
    for key in sorted_keys:
        orig = chunk_map[key]
        fused_results.append(RetrievedChunk(
            content=orig.content,
            metadata=orig.metadata,
            score=round(float(rrf_scores[key]), 6)
        ))

    return fused_results


# Bộ nhớ đệm BM25 Retriever theo document_id để tránh index lại mỗi câu query
_bm25_cache: Dict[str, BM25Retriever] = {}


def get_bm25_retriever_for_document(
    document_id: str,
    store: Optional[QdrantStore] = None
) -> Optional[BM25Retriever]:
    """
    Lấy hoặc khởi tạo BM25Retriever cho một tài liệu cụ thể.
    """
    active_store = store or get_store()
    cache_key = f"{active_store.collection_name}::{document_id}"

    if cache_key in _bm25_cache:
        return _bm25_cache[cache_key]

    # Đọc toàn bộ chunks của document_id từ Qdrant
    records = active_store.scroll_by_document(document_id)
    if not records:
        return None

    from src.pipelines.rag import _record_to_chunk
    chunks = [_record_to_chunk(r) for r in records]

    retriever = BM25Retriever(chunks)
    _bm25_cache[cache_key] = retriever
    return retriever
