import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""
Test suite cho Phase 4 -- Interfaces (FastAPI REST API & Typer CLI)
Chay: python test_interfaces.py
"""

import sys
import os
import warnings

# Dam bao in tieng Viet tren Windows khong bi loi UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Tat cac warning khong can thiet khi chay test
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from src.interfaces.api import app as fastapi_app
from src.interfaces.cli import app as cli_app

client = TestClient(fastapi_app)
runner = CliRunner()


# ============================================================
# HELPER
# ============================================================

def _get_test_document_id():
    """Lay document_id dau tien co san trong Qdrant."""
    from src.db.store import get_store
    store = get_store()
    docs = store.list_documents()
    if not docs:
        return None
    return docs[0]["document_id"]


# ============================================================
# FASTAPI TESTS
# ============================================================

def test_api_root():
    """Test 1: Kiem tra GET /"""
    print("\n[TEST 1] Kiem tra API GET /...")
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "version" in data
    print(f"  Response: {data}")
    print("  PASSED: GET / thanh cong")
    return True


def test_api_health():
    """Test 2: Kiem tra GET /health"""
    print("\n[TEST 2] Kiem tra API GET /health...")
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["qdrant_connected"] is True
    print(f"  Qdrant docs: {data.get('total_documents')}, Model: {data.get('gemini_model')}")
    print("  PASSED: GET /health thanh cong")
    return True


def test_api_documents():
    """Test 3: Kiem tra GET /documents"""
    print("\n[TEST 3] Kiem tra API GET /documents...")
    res = client.get("/documents")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    print(f"  So tai lieu tim thay: {len(data)}")
    if data:
        print(f"  Tai lieu mau: {data[0].get('filename')} (ID: {data[0].get('document_id')})")
    print("  PASSED: GET /documents thanh cong")
    return True


def test_api_ask():
    """Test 4: Kiem tra POST /ask (RAG Q&A)"""
    print("\n[TEST 4] Kiem tra API POST /ask...")
    doc_id = _get_test_document_id()
    payload = {
        "question": "Tom tat ngan gon thong tin ve tai lieu nay",
        "document_id": doc_id,
    }
    res = client.post("/ask", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert "citations" in data
    assert len(data["answer"]) > 0
    print(f"  Answer preview: {data['answer'][:120]}...")
    print(f"  Citations count: {len(data['citations'])}")
    print("  PASSED: POST /ask thanh cong")
    return True


def test_api_summarize():
    """Test 5: Kiem tra POST /summarize"""
    print("\n[TEST 5] Kiem tra API POST /summarize...")
    doc_id = _get_test_document_id()
    if not doc_id:
        print("  SKIP: Chua co tai lieu trong Qdrant.")
        return None

    payload = {"document_id": doc_id}
    res = client.post("/summarize", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "content" in data
    assert data["document_id"] == doc_id
    assert data["method"] in ("single", "map_reduce")
    print(f"  Method: {data['method']}, Chunks: {data['total_chunks']}")
    print(f"  Content length: {len(data['content'])} chars")
    print("  PASSED: POST /summarize thanh cong")
    return True


def test_api_quiz():
    """Test 6: Kiem tra POST /quiz"""
    print("\n[TEST 6] Kiem tra API POST /quiz...")
    doc_id = _get_test_document_id()
    if not doc_id:
        print("  SKIP: Chua co tai lieu trong Qdrant.")
        return None

    payload = {"document_id": doc_id, "n_questions": 2}
    res = client.post("/quiz", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "items" in data
    assert len(data["items"]) >= 1
    item = data["items"][0]
    assert len(item["options"]) == 4
    assert item["answer"] in ("A", "B", "C", "D")
    print(f"  Cau hoi tao duoc: {item['question']}")
    print(f"  Dap an: {item['answer']}")
    print("  PASSED: POST /quiz thanh cong")
    return True


def test_api_flashcards():
    """Test 7: Kiem tra POST /flashcards"""
    print("\n[TEST 7] Kiem tra API POST /flashcards...")
    doc_id = _get_test_document_id()
    if not doc_id:
        print("  SKIP: Chua co tai lieu trong Qdrant.")
        return None

    payload = {"document_id": doc_id, "n_cards": 3}
    res = client.post("/flashcards", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "cards" in data
    assert len(data["cards"]) >= 1
    card = data["cards"][0]
    assert "front" in card and "back" in card
    print(f"  The mau: {card['front']} -> {card['back'][:50]}...")
    print("  PASSED: POST /flashcards thanh cong")
    return True


# ============================================================
# TYPER CLI TESTS
# ============================================================

def test_cli_help():
    """Test 8: Kiem tra CLI --help"""
    print("\n[TEST 8] Kiem tra Typer CLI --help...")
    result = runner.invoke(cli_app, ["--help"])
    assert result.exit_code == 0
    assert "RAG Learning System CLI" in result.output
    print("  PASSED: CLI --help thanh cong")
    return True


def test_cli_list_docs():
    """Test 9: Kiem tra CLI list-docs"""
    print("\n[TEST 9] Kiem tra Typer CLI list-docs...")
    result = runner.invoke(cli_app, ["list-docs"])
    assert result.exit_code == 0
    print("  CLI Output preview:")
    for line in result.output.splitlines()[:6]:
        print(f"    {line}")
    print("  PASSED: CLI list-docs thanh cong")
    return True


# ============================================================
# MAIN RUNNER
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 4 INTERFACES -- TEST SUITE")
    print("=" * 60)

    results = {}
    tests = [
        ("api_root", test_api_root),
        ("api_health", test_api_health),
        ("api_documents", test_api_documents),
        ("api_ask", test_api_ask),
        ("api_summarize", test_api_summarize),
        ("api_quiz", test_api_quiz),
        ("api_flashcards", test_api_flashcards),
        ("cli_help", test_cli_help),
        ("cli_list_docs", test_cli_list_docs),
    ]

    for name, fn in tests:
        try:
            res = fn()
            results[name] = "PASSED" if res else "SKIPPED"
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = "FAILED"

    print("\n" + "=" * 60)
    print("KET QUA PHASE 4 INTERFACES:")
    for name, status in results.items():
        icon = "OK" if status == "PASSED" else ("--" if status == "SKIPPED" else "XX")
        print(f"  [{icon}] {name}: {status}")

    failed = [n for n, s in results.items() if s == "FAILED"]
    if failed:
        print(f"\nFAILED: {len(failed)} test(s) that bai: {failed}")
        sys.exit(1)
    else:
        print("\nTat ca Phase 4 tests PASSED hoac SKIPPED!")
