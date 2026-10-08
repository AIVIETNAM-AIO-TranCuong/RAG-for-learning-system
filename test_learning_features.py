"""
Test suite cho Phase 3 -- Learning Features (Summarize, Quiz, Flashcards)
Chay: python test_learning_features.py
Yeu cau: Da index it nhat 1 PDF vao Qdrant va cau hinh GEMINI_API_KEY trong .env.
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


# ============================================================
# NHOM 1: TESTS OFFLINE (Khong goi LLM)
# ============================================================

def test_imports():
    """Test 1: Kiem tra import tat ca modules Phase 3."""
    print("\n[TEST 1] Kiem tra imports...")
    from src.core.config import settings
    from src.core.schemas import Summary, QuizItem, QuizSet, Flashcard, FlashcardSet
    from src.pipelines.learning import (
        summarize, generate_quiz, generate_flashcards,
        _parse_json, _validate_quiz_items, _validate_flashcard_items
    )
    print(f"  - SUMMARIZE_BATCH_SIZE: {settings.summarize_batch_size}")
    print(f"  - N_QUIZ_QUESTIONS: {settings.n_quiz_questions}")
    print(f"  - N_FLASHCARDS: {settings.n_flashcards}")
    print("  PASSED: All imports OK")
    return True


def test_schemas():
    """Test 2: Kiem tra cac Pydantic schemas moi."""
    print("\n[TEST 2] Kiem tra Pydantic schemas...")
    from src.core.schemas import Summary, QuizItem, QuizSet, Flashcard, FlashcardSet

    s = Summary(content="Noi dung tom tat", document_id="doc1", total_chunks=5, method="single")
    assert s.method == "single"
    assert s.total_chunks == 5

    qi = QuizItem(
        question="Cau hoi?",
        options=["A. 1", "B. 2", "C. 3", "D. 4"],
        answer="A",
        explanation="Vi 1 dung"
    )
    qs = QuizSet(items=[qi], document_id="doc1", topic="Math")
    assert len(qs.items) == 1
    assert qs.items[0].answer == "A"

    fc = Flashcard(front="Khai niem A", back="Dinh nghia A")
    fcs = FlashcardSet(cards=[fc], document_id="doc1")
    assert len(fcs.cards) == 1
    assert fcs.cards[0].front == "Khai niem A"

    print("  PASSED: Schemas hoat dong dung")
    return True


def test_parse_json():
    """Test 3: Kiem tra ham _parse_json xu ly cac dinh dang markdown."""
    print("\n[TEST 3] Kiem tra _parse_json()...")
    from src.pipelines.learning import _parse_json

    # Case 1: JSON thuan
    raw1 = '[{"key": "value"}]'
    res1 = _parse_json(raw1)
    assert res1 == [{"key": "value"}]

    # Case 2: JSON trong ```json ... ```
    raw2 = '```json\n[{"key": "value"}]\n```'
    res2 = _parse_json(raw2)
    assert res2 == [{"key": "value"}]

    # Case 3: JSON co loi dan truoc va sau
    raw3 = 'Day la ket qua:\n```json\n[{"a": 1}, {"b": 2}]\n```\nHy vong ban thich.'
    res3 = _parse_json(raw3)
    assert len(res3) == 2

    # Case 4: JSON khong co tu khoa json trong fence
    raw4 = '```\n{"test": 123}\n```'
    res4 = _parse_json(raw4)
    assert res4["test"] == 123

    print("  PASSED: _parse_json() xu ly dung cac edge cases")
    return True


def test_validate_quiz():
    """Test 4: Kiem tra loc va validate quiz items."""
    print("\n[TEST 4] Kiem tra _validate_quiz_items()...")
    from src.pipelines.learning import _validate_quiz_items

    raw_items = [
        # Item hop le
        {
            "question": "Thu do Viet Nam la gi?",
            "options": ["A. Ha Noi", "B. Da Nang", "C. HCM", "D. Hue"],
            "answer": "A",
            "explanation": "Ha Noi la thu do."
        },
        # Item answer co them chu (vi du: "A. Ha Noi")
        {
            "question": "Mat troi moc o huong nao?",
            "options": ["A. Dong", "B. Tay", "C. Nam", "D. Bac"],
            "answer": "A. Dong",
            "explanation": "Huong Dong."
        },
        # Item thieu options (chi co 3) -> Phai bi loai
        {
            "question": "Cau hoi loi options?",
            "options": ["A. 1", "B. 2", "C. 3"],
            "answer": "A",
            "explanation": "Thieu options"
        },
        # Item trung cau hoi voi item 1 -> Phai bi loai
        {
            "question": "Thu do Viet Nam la gi?",
            "options": ["A. Ha Noi", "B. Hai Phong", "C. Can Tho", "D. Nha Trang"],
            "answer": "A",
            "explanation": "Trung cau hoi"
        },
        # Item answer khong hop le -> Phai bi loai
        {
            "question": "Cau hoi loi answer?",
            "options": ["A. 1", "B. 2", "C. 3", "D. 4"],
            "answer": "X",
            "explanation": "Answer sai"
        }
    ]

    valid = _validate_quiz_items(raw_items)
    assert len(valid) == 2, f"Ky vong giu lai 2 items hop le, thuc te duoc {len(valid)}"
    assert valid[0].answer == "A"
    assert valid[1].answer == "A"

    print("  PASSED: _validate_quiz_items() loc chuan xac")
    return True


def test_validate_flashcard():
    """Test 5: Kiem tra loc va validate flashcards."""
    print("\n[TEST 5] Kiem tra _validate_flashcard_items()...")
    from src.pipelines.learning import _validate_flashcard_items

    raw_cards = [
        {"front": "AI", "back": "Tri tue nhan tao"},
        {"front": "ML", "back": "Hoc may"},
        {"front": "AI", "back": "Dinh nghia trung"},  # Trung front -> Loai
        {"front": "", "back": "Front rong"},          # Rong -> Loai
        {"front": "Thieu back"}                       # Thieu field -> Loai
    ]

    valid = _validate_flashcard_items(raw_cards)
    assert len(valid) == 2, f"Ky vong giu lai 2 cards, thuc te duoc {len(valid)}"
    assert valid[0].front == "AI"
    assert valid[1].front == "ML"

    print("  PASSED: _validate_flashcard_items() loc chuan xac")
    return True


# ============================================================
# NHOM 2: TESTS ONLINE (Goi Gemini LLM that)
# ============================================================

def _get_test_document_id():
    """Helper: Lay document_id tu Qdrant de test."""
    from src.db.store import get_store
    store = get_store()
    docs = store.list_documents()
    if not docs:
        return None
    return docs[0]["document_id"]


def test_summarize():
    """Test 6: Kiem tra tinh nang Summarize voi LLM that."""
    print("\n[TEST 6] Kiem tra summarize() voi LLM...")
    from src.pipelines.learning import summarize

    doc_id = _get_test_document_id()
    if not doc_id:
        print("  SKIP: Chua co document trong Qdrant. Hay index 1 PDF truoc.")
        return None

    print(f"  Dang tom tat tai lieu doc_id={doc_id[:12]}...")
    summary = summarize(doc_id)

    assert summary.content, "Noi dung tom tat khong duoc rong"
    assert summary.document_id == doc_id
    assert summary.total_chunks > 0
    assert summary.method in ("single", "map_reduce")

    print(f"  Phuong phap: {summary.method}, Chunks: {summary.total_chunks}")
    print(f"  Do dai tom tat: {len(summary.content)} ky tu")
    print(f"  Trich doan: {summary.content[:150]}...")
    print("  PASSED: summarize() thanh cong")
    return True


def test_generate_quiz():
    """Test 7: Kiem tra tinh nang Generate Quiz voi LLM that."""
    print("\n[TEST 7] Kiem tra generate_quiz() voi LLM...")
    from src.pipelines.learning import generate_quiz

    doc_id = _get_test_document_id()
    if not doc_id:
        print("  SKIP: Chua co document trong Qdrant. Hay index 1 PDF truoc.")
        return None

    print(f"  Dang tao 3 cau hoi quiz tu doc_id={doc_id[:12]}...")
    quiz_set = generate_quiz(doc_id, n_questions=3)

    assert len(quiz_set.items) >= 1, "Phai tao duoc it nhat 1 cau hoi quiz"
    q = quiz_set.items[0]
    assert len(q.options) == 4, "Cau hoi phai co 4 options"
    assert q.answer in ("A", "B", "C", "D"), "Answer phai la A, B, C hoac D"

    print(f"  Tao thanh cong {len(quiz_set.items)} cau hoi:")
    for i, item in enumerate(quiz_set.items):
        print(f"    Cau {i+1}: {item.question} (Dap an: {item.answer})")
    print("  PASSED: generate_quiz() thanh cong")
    return True


def test_generate_flashcards():
    """Test 8: Kiem tra tinh nang Generate Flashcards voi LLM that."""
    print("\n[TEST 8] Kiem tra generate_flashcards() voi LLM...")
    from src.pipelines.learning import generate_flashcards

    doc_id = _get_test_document_id()
    if not doc_id:
        print("  SKIP: Chua co document trong Qdrant. Hay index 1 PDF truoc.")
        return None

    print(f"  Dang tao 5 flashcards tu doc_id={doc_id[:12]}...")
    flashcard_set = generate_flashcards(doc_id, n_cards=5)

    assert len(flashcard_set.cards) >= 1, "Phai tao duoc it nhat 1 flashcard"
    card = flashcard_set.cards[0]
    assert card.front, "Mat truoc khong duoc rong"
    assert card.back, "Mat sau khong duoc rong"

    print(f"  Tao thanh cong {len(flashcard_set.cards)} flashcards:")
    for i, c in enumerate(flashcard_set.cards):
        print(f"    The {i+1}: {c.front} -> {c.back[:50]}...")
    print("  PASSED: generate_flashcards() thanh cong")
    return True


# ============================================================
# MAIN RUNNER
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 3 LEARNING FEATURES -- TEST SUITE")
    print("=" * 60)

    results = {}
    tests = [
        ("imports", test_imports),
        ("schemas", test_schemas),
        ("parse_json", test_parse_json),
        ("validate_quiz", test_validate_quiz),
        ("validate_flashcard", test_validate_flashcard),
        ("summarize", test_summarize),
        ("generate_quiz", test_generate_quiz),
        ("generate_flashcards", test_generate_flashcards),
    ]

    for name, fn in tests:
        try:
            result = fn()
            results[name] = "PASSED" if result else "SKIPPED"
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = "FAILED"

    print("\n" + "=" * 60)
    print("KET QUA PHASE 3:")
    for name, status in results.items():
        icon = "OK" if status == "PASSED" else ("--" if status == "SKIPPED" else "XX")
        print(f"  [{icon}] {name}: {status}")

    failed = [n for n, s in results.items() if s == "FAILED"]
    if failed:
        print(f"\nFAILED: {len(failed)} test(s) that bai: {failed}")
        sys.exit(1)
    else:
        print("\nTat ca Phase 3 tests PASSED hoac SKIPPED!")
