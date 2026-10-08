"""
RAG Core Pipeline -- Phase 2

Cac ham chinh:
  retrieve()         - Semantic search Qdrant -> List[RetrievedChunk]
  rerank()           - Cross-Encoder reranking -> List[RetrievedChunk] (sorted)
  answer()           - End-to-end RAG: retrieve -> rerank -> prompt -> LLM -> RagAnswer
  fetch_all_chunks() - Lay toan bo chunks cua 1 doc (cho Phase 3 Summarization)
"""

import re
from pathlib import Path
from typing import List, Optional

from jinja2 import Environment, FileSystemLoader

from src.core.config import settings
from src.core.schemas import ChunkMetadata, RetrievedChunk, Citation, RagAnswer
from src.db.store import get_store
from src.llm import get_llm

# Path to Jinja2 templates directory
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# Lazy-loaded Jinja2 environment
_jinja_env: Optional[Environment] = None

# Lazy-loaded CrossEncoder reranker
_reranker = None


# ============================================================
# PRIVATE HELPERS
# ============================================================

def _get_jinja_env() -> Environment:
    """Initialize Jinja2 environment (singleton)."""
    global _jinja_env
    if _jinja_env is None:
        _jinja_env = Environment(
            loader=FileSystemLoader(str(_PROMPTS_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )
    return _jinja_env


def _get_reranker():
    """Load CrossEncoder reranker (singleton, lazy-loaded)."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("Can cai dat: pip install sentence-transformers")

        import torch
        rerank_dev = settings.reranker_device
        if rerank_dev == "cuda" and not torch.cuda.is_available():
            print("Notice: CUDA is not available. Falling back to CPU for reranker.")
            rerank_dev = "cpu"

        print(f"Loading reranker: {settings.reranker_model} on {rerank_dev}...")
        _reranker = CrossEncoder(
            settings.reranker_model,
            device=rerank_dev,
        )
        print(f"Reranker loaded: {settings.reranker_model}")
    return _reranker


def _scored_point_to_chunk(point) -> RetrievedChunk:
    """Chuyen doi Qdrant ScoredPoint thanh RetrievedChunk."""
    payload = point.payload or {}
    metadata = ChunkMetadata(
        document_id=payload.get("document_id", ""),
        filename=payload.get("filename", ""),
        source=payload.get("source", ""),
        page=payload.get("page", 0),
        chunk_id=payload.get("chunk_id", ""),
        section=payload.get("section"),
    )
    return RetrievedChunk(
        content=payload.get("content", ""),
        metadata=metadata,
        score=float(point.score),
    )


def _record_to_chunk(record) -> RetrievedChunk:
    """Chuyen doi Qdrant Record (scroll) thanh RetrievedChunk."""
    payload = record.payload or {}
    metadata = ChunkMetadata(
        document_id=payload.get("document_id", ""),
        filename=payload.get("filename", ""),
        source=payload.get("source", ""),
        page=payload.get("page", 0),
        chunk_id=payload.get("chunk_id", ""),
        section=payload.get("section"),
    )
    return RetrievedChunk(
        content=payload.get("content", ""),
        metadata=metadata,
        score=0.0,  # Scroll API khong tra ve score
    )


# ============================================================
# PUBLIC API
# ============================================================

def retrieve(
    query: str,
    top_k: int = None,
    document_id: Optional[str] = None,
) -> List[RetrievedChunk]:
    """
    Thuc hien semantic search tren Qdrant va tra ve cac chunks lien quan.

    Args:
        query: Cau hoi / noi dung can tim kiem
        top_k: So chunks tra ve (mac dinh: settings.initial_k)
        document_id: Chi tim trong tai lieu nay (None = tim tat ca)

    Returns:
        List[RetrievedChunk] sap xep theo score giam dan
    """
    if top_k is None:
        top_k = settings.initial_k

    store = get_store()
    query_vector = store.embed_text(query)
    scored_points = store.search(query_vector, top_k=top_k, document_id=document_id)

    chunks = [_scored_point_to_chunk(p) for p in scored_points]
    return chunks


def rerank(
    chunks: List[RetrievedChunk],
    query: str,
    rerank_k: int = None,
) -> List[RetrievedChunk]:
    """
    Rerank chunks bang Cross-Encoder de tang do chinh xac.

    Args:
        chunks: Danh sach chunks can rerank
        query: Cau hoi goc de so sanh
        rerank_k: So chunks giu lai sau rerank (mac dinh: settings.rerank_k)

    Returns:
        List[RetrievedChunk] da sap xep lai theo Cross-Encoder score, giu top rerank_k
    """
    if not chunks:
        return []

    if rerank_k is None:
        rerank_k = settings.rerank_k

    reranker = _get_reranker()

    # Chuan bi pairs (query, chunk_content) cho CrossEncoder
    pairs = [(query, chunk.content) for chunk in chunks]
    scores = reranker.predict(pairs)

    # Gan cross-encoder score vao chunks
    scored_chunks = [
        RetrievedChunk(content=c.content, metadata=c.metadata, score=float(s))
        for c, s in zip(chunks, scores)
    ]

    # Sort giam dan theo score, lay top rerank_k
    scored_chunks.sort(key=lambda x: x.score, reverse=True)
    return scored_chunks[:rerank_k]


def answer(
    query: str,
    document_id: Optional[str] = None,
) -> RagAnswer:
    """
    End-to-end RAG pipeline: retrieve -> rerank -> render prompt -> LLM -> RagAnswer.

    Args:
        query: Cau hoi cua nguoi dung
        document_id: Chi tim trong tai lieu nay (None = tim tat ca)

    Returns:
        RagAnswer voi answer text co inline markers [S1][S2]..., danh sach citations, va chunks
    """
    # 1. Retrieve nhieu chunks (initial_k) de co nhieu ung vien cho reranking
    initial_chunks = retrieve(query, top_k=settings.initial_k, document_id=document_id)

    if not initial_chunks:
        return RagAnswer(
            answer="Khong tim thay thong tin lien quan trong tai lieu.",
            citations=[],
            chunks=[],
        )

    # 2. Rerank de giu lai cac chunks chinh xac nhat
    final_chunks = rerank(initial_chunks, query, rerank_k=settings.rerank_k)

    # 3. Xay dung context cho Jinja2 template
    # Moi chunk duoc gan marker S1, S2, ...
    chunks_context = [
        {
            "marker": f"S{i + 1}",
            "content": chunk.content,
            "filename": chunk.metadata.filename,
            "page": chunk.metadata.page,
        }
        for i, chunk in enumerate(final_chunks)
    ]

    # 4. Render Jinja2 prompt
    env = _get_jinja_env()
    template = env.get_template("qa.j2")
    prompt = template.render(chunks=chunks_context, question=query)

    # 5. Goi LLM
    llm = get_llm()
    answer_text = llm.generate(prompt)

    # 6. Xay dung citations tu cac markers trong answer
    citations = []
    for i, chunk in enumerate(final_chunks):
        marker = f"S{i + 1}"
        # Chi them citation neu marker xuat hien trong answer
        if f"[{marker}]" in answer_text:
            citations.append(Citation(marker=marker, chunk=chunk))

    return RagAnswer(
        answer=answer_text,
        citations=citations,
        chunks=final_chunks,
    )


def fetch_all_chunks(document_id: str) -> List[RetrievedChunk]:
    """
    Lay toan bo chunks cua mot tai lieu (dung cho Summarization trong Phase 3).

    Args:
        document_id: ID tai lieu can lay chunks

    Returns:
        List[RetrievedChunk] sap xep theo so trang va chunk_id
    """
    store = get_store()
    records = store.scroll_by_document(document_id)

    chunks = [_record_to_chunk(r) for r in records]

    # Sap xep theo so trang de giu thu tu van ban
    chunks.sort(key=lambda c: (c.metadata.page, c.metadata.chunk_id))

    return chunks