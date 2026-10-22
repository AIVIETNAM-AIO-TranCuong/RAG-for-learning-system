import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""
Test suite cho Phase 2 -- RAG Core Pipeline
Chay: python test_rag_pipeline.py
Yeu cau: Da index it nhat 1 PDF vao Qdrant truoc khi chay test nay.
"""

import sys
import os
import warnings

# Tat cac warning khong can thiet khi chay test
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test 1: Kiem tra imports hoat dong dung."""
    print("\n[TEST 1] Kiem tra imports...")
    from src.core.config import settings
    from src.core.schemas import ChunkMetadata, RetrievedChunk, Citation, RagAnswer
    from src.db.store import get_store
    from src.pipelines.rag import retrieve, rerank, answer, fetch_all_chunks
    print(f"  - LLM_BACKEND: {settings.llm_backend}")
    print(f"  - GEMINI_MODEL: {settings.gemini_model}")
    print(f"  - INITIAL_K: {settings.initial_k}, RERANK_K: {settings.rerank_k}")
    print("  PASSED: All imports OK")
    return True

def test_schemas():
    """Test 2: Kiem tra Pydantic schemas."""
    print("\n[TEST 2] Kiem tra Pydantic schemas...")
    from src.core.schemas import ChunkMetadata, RetrievedChunk, Citation, RagAnswer

    meta = ChunkMetadata(
        document_id="test123",
        filename="test.pdf",
        source="/path/test.pdf",
        page=1,
        chunk_id="test123_0",
    )
    chunk = RetrievedChunk(content="Noi dung thu nghiem.", metadata=meta, score=0.95)
    citation = Citation(marker="S1", chunk=chunk)
    rag_ans = RagAnswer(answer="Cau tra loi [S1].", citations=[citation], chunks=[chunk])

    assert rag_ans.answer == "Cau tra loi [S1]."
    assert len(rag_ans.citations) == 1
    assert rag_ans.citations[0].marker == "S1"
    print("  PASSED: Schemas hoat dong dung")
    return True

def test_store_search():
    """Test 3: Kiem tra store.search() co the chay."""
    print("\n[TEST 3] Kiem tra store.search()...")
    from src.db.store import get_store
    store = get_store()
    
    # Kiem tra collection co documents khong
    docs = store.list_documents()
    if not docs:
        print("  SKIP: Chua co documents trong Qdrant. Hay index 1 PDF truoc.")
        return None

    print(f"  Tim thay {len(docs)} documents trong Qdrant")
    
    # Thu embed va search
    test_query = "What is this document about?"
    query_vector = store.embed_text(test_query)
    results = store.search(query_vector, top_k=3)
    
    assert isinstance(results, list), "search() phai tra ve list"
    print(f"  Tim thay {len(results)} ket qua")
    if results:
        print(f"  Top result score: {results[0].score:.4f}")
    print("  PASSED: store.search() hoat dong")
    return True

def test_retrieve():
    """Test 4: Kiem tra ham retrieve()."""
    print("\n[TEST 4] Kiem tra retrieve()...")
    from src.db.store import get_store
    from src.pipelines.rag import retrieve
    
    store = get_store()
    docs = store.list_documents()
    if not docs:
        print("  SKIP: Chua co documents. Hay index 1 PDF truoc.")
        return None

    chunks = retrieve("Noi dung chinh cua tai lieu la gi?", top_k=5)
    
    assert isinstance(chunks, list), "retrieve() phai tra ve list"
    assert len(chunks) <= 5, f"Phai tra ve <= 5 chunks, nhan duoc {len(chunks)}"
    
    if chunks:
        c = chunks[0]
        assert c.content, "Chunk phai co noi dung"
        assert c.metadata.filename, "Chunk phai co filename"
        assert c.score >= 0, "Score phai >= 0"
        print(f"  Top chunk: score={c.score:.4f}, page={c.metadata.page}, len={len(c.content)}")
    
    print(f"  PASSED: retrieve() tra ve {len(chunks)} chunks")
    return True

def test_rerank():
    """Test 5: Kiem tra ham rerank()."""
    print("\n[TEST 5] Kiem tra rerank()...")
    from src.core.schemas import ChunkMetadata, RetrievedChunk
    from src.pipelines.rag import rerank

    # Tao chunks gia
    def make_chunk(content, score):
        meta = ChunkMetadata(
            document_id="test", filename="test.pdf",
            source="/test.pdf", page=1, chunk_id=f"c_{score}"
        )
        return RetrievedChunk(content=content, metadata=meta, score=score)

    fake_chunks = [
        make_chunk("Python la ngon ngu lap trinh pho bien.", 0.7),
        make_chunk("Machine learning la mot nhanh cua AI.", 0.6),
        make_chunk("RAG la viet tat cua Retrieval-Augmented Generation.", 0.8),
        make_chunk("Qdrant la vector database.", 0.5),
        make_chunk("Embedding model bien van ban thanh vector so.", 0.9),
    ]

    query = "RAG la gi?"
    reranked = rerank(fake_chunks, query, rerank_k=3)
    
    assert len(reranked) <= 3, f"Phai tra ve <= 3 chunks, nhan duoc {len(reranked)}"
    # Kiem tra sap xep giam dan
    for i in range(len(reranked) - 1):
        assert reranked[i].score >= reranked[i+1].score, "Chunks phai duoc sap xep giam dan theo score"
    
    print(f"  PASSED: rerank() tra ve {len(reranked)} chunks, sap xep dung")
    for i, c in enumerate(reranked):
        print(f"    {i+1}. score={c.score:.4f}: {c.content[:60]}...")
    return True

def test_fetch_all_chunks():
    """Test 6: Kiem tra fetch_all_chunks()."""
    print("\n[TEST 6] Kiem tra fetch_all_chunks()...")
    from src.db.store import get_store
    from src.pipelines.rag import fetch_all_chunks

    store = get_store()
    docs = store.list_documents()
    if not docs:
        print("  SKIP: Chua co documents. Hay index 1 PDF truoc.")
        return None

    doc_id = docs[0]["document_id"]
    filename = docs[0]["filename"]
    total_chunks_expected = docs[0]["total_chunks"]

    chunks = fetch_all_chunks(doc_id)
    
    assert len(chunks) == total_chunks_expected, \
        f"Ky vong {total_chunks_expected} chunks, nhan duoc {len(chunks)}"
    print(f"  Tai lieu: {filename}")
    print(f"  Lay duoc {len(chunks)}/{total_chunks_expected} chunks")
    print(f"  PASSED: fetch_all_chunks() hoat dong dung")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 2 RAG PIPELINE -- TEST SUITE")
    print("=" * 60)

    results = {}
    tests = [
        ("imports", test_imports),
        ("schemas", test_schemas),
        ("store_search", test_store_search),
        ("retrieve", test_retrieve),
        ("rerank", test_rerank),
        ("fetch_all_chunks", test_fetch_all_chunks),
    ]

    for name, fn in tests:
        try:
            result = fn()
            results[name] = "PASSED" if result else "SKIPPED"
        except Exception as e:
            print(f"  FAILED: {e}")
            results[name] = "FAILED"

    print("\n" + "=" * 60)
    print("KET QUA:")
    for name, status in results.items():
        icon = "OK" if status == "PASSED" else ("--" if status == "SKIPPED" else "XX")
        print(f"  [{icon}] {name}: {status}")
    
    failed = [n for n, s in results.items() if s == "FAILED"]
    if failed:
        print(f"\nFAILED: {len(failed)} test(s) that bai: {failed}")
        sys.exit(1)
    else:
        print("\nTat ca tests PASSED hoac SKIPPED!")