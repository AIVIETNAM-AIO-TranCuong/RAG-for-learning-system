"""
Learning Features Pipeline -- Phase 3

Cac ham chinh:
  summarize()           - Tom tat tai lieu (single hoac map-reduce)
  generate_quiz()       - Tao bo cau hoi trac nghiem MCQ
  generate_flashcards() - Tao bo flashcards
"""

import json
import re
from pathlib import Path
from typing import List, Optional, Any

from jinja2 import Environment, FileSystemLoader

from src.core.config import settings
from src.core.schemas import (
    RetrievedChunk, Summary, QuizItem, QuizSet, Flashcard, FlashcardSet
)
from src.pipelines.rag import fetch_all_chunks, retrieve
from src.llm import get_llm

# Path to Jinja2 templates directory
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_jinja_env: Optional[Environment] = None


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


# ============================================================
# PRIVATE HELPERS
# ============================================================

def _parse_json(raw_text: str) -> Any:
    """
    Parse JSON tu LLM response, tu dong strip markdown code fences va text thua.
    
    LLM thuong tra ve:
      ```json
      [...]
      ```
    Ham nay xu ly ca truong hop co code fences hoac co loi mo dau / ket thuc.
    """
    text = raw_text.strip()
    
    # 1. Tim block code ```json ... ``` hoac ``` ... ```
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if match:
        text = match.group(1).strip()
    else:
        # 2. Fallback: Tim doan JSON bat dau bang [ hoac { den ] hoac }
        match_bracket = re.search(r'(\[[\s\S]*\]|\{[\s\S]*\})', text)
        if match_bracket:
            text = match_bracket.group(1).strip()
    
    return json.loads(text)


def _validate_quiz_items(raw_items: list) -> List[QuizItem]:
    """
    Loc va validate danh sach quiz items tu LLM output.
    - Loai bo items thieu field bat buoc
    - Loai bo items bi trung cau hoi
    - Chuan hoa va dam bao answer la A/B/C/D
    """
    valid = []
    seen_questions = set()
    
    if not isinstance(raw_items, list):
        return []
    
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        
        # Kiem tra cac field bat buoc
        if not all(k in item for k in ("question", "options", "answer", "explanation")):
            continue
        
        # Kiem tra options co dung 4 lua chon
        options = item["options"]
        if not isinstance(options, list) or len(options) != 4:
            continue
        
        # Chuan hoa answer (vi du: "A", "A.", "A. Lua chon 1" -> "A")
        ans_raw = str(item["answer"]).strip().upper()
        if not ans_raw:
            continue
        first_char = ans_raw[0]
        if first_char not in ("A", "B", "C", "D"):
            continue
        
        # Kiem tra trung lap cau hoi
        q_text = str(item["question"]).strip()
        q_normalized = q_text.lower()
        if not q_text or q_normalized in seen_questions:
            continue
        seen_questions.add(q_normalized)
        
        valid.append(QuizItem(
            question=q_text,
            options=[str(opt) for opt in options],
            answer=first_char,
            explanation=str(item["explanation"]).strip(),
        ))
    
    return valid


def _validate_flashcard_items(raw_items: list) -> List[Flashcard]:
    """
    Loc va validate danh sach flashcard items tu LLM output.
    - Loai bo items thieu front/back
    - Loai bo items bi trung front
    """
    valid = []
    seen_fronts = set()
    
    if not isinstance(raw_items, list):
        return []
    
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        
        if not all(k in item for k in ("front", "back")):
            continue
        
        front_text = str(item["front"]).strip()
        back_text = str(item["back"]).strip()
        
        if not front_text or not back_text:
            continue
        
        front_normalized = front_text.lower()
        if front_normalized in seen_fronts:
            continue
        seen_fronts.add(front_normalized)
        
        valid.append(Flashcard(
            front=front_text,
            back=back_text,
        ))
    
    return valid


def _chunks_to_context(chunks: List[RetrievedChunk]) -> List[dict]:
    """Chuyen doi chunks thanh context dicts cho Jinja2."""
    return [
        {
            "content": c.content,
            "page": c.metadata.page,
            "filename": c.metadata.filename,
        }
        for c in chunks
    ]


# ============================================================
# PUBLIC API
# ============================================================

def summarize(document_id: str) -> Summary:
    """
    Tom tat tai lieu. Tu dong chon strategy:
    - Neu <= batch_size chunks: dung summary_single.j2
    - Neu > batch_size chunks: dung Map-Reduce (summary_map.j2 + summary_reduce.j2)
    
    Args:
        document_id: ID tai lieu can tom tat
        
    Returns:
        Summary object chua noi dung tom tat va metadata
    """
    chunks = fetch_all_chunks(document_id)
    if not chunks:
        return Summary(
            content="Khong tim thay noi dung tai lieu.",
            document_id=document_id,
            total_chunks=0,
            method="single",
        )
    
    batch_size = settings.summarize_batch_size
    env = _get_jinja_env()
    llm = get_llm()
    
    if len(chunks) <= batch_size:
        # === SINGLE SUMMARY ===
        template = env.get_template("summary_single.j2")
        context = _chunks_to_context(chunks)
        prompt = template.render(chunks=context)
        summary_text = llm.generate(prompt)
        
        return Summary(
            content=summary_text.strip(),
            document_id=document_id,
            total_chunks=len(chunks),
            method="single",
        )
    else:
        # === MAP-REDUCE ===
        # MAP: Chia batch va tom tat tung batch
        batches = [chunks[i:i+batch_size] for i in range(0, len(chunks), batch_size)]
        map_template = env.get_template("summary_map.j2")
        
        partial_summaries = []
        for idx, batch in enumerate(batches):
            context = _chunks_to_context(batch)
            prompt = map_template.render(
                chunks=context,
                batch_index=idx + 1,
                total_batches=len(batches),
            )
            partial = llm.generate(prompt)
            partial_summaries.append(partial.strip())
            print(f"  [Map] Hoan thanh batch {idx + 1}/{len(batches)}")
        
        # REDUCE: Gop cac tom tat con thanh tom tat tong
        reduce_template = env.get_template("summary_reduce.j2")
        reduce_prompt = reduce_template.render(summaries=partial_summaries)
        final_summary = llm.generate(reduce_prompt)
        
        return Summary(
            content=final_summary.strip(),
            document_id=document_id,
            total_chunks=len(chunks),
            method="map_reduce",
        )


def generate_quiz(
    document_id: str,
    topic: Optional[str] = None,
    n_questions: Optional[int] = None,
) -> QuizSet:
    """
    Tao bo cau hoi trac nghiem MCQ tu tai lieu.
    
    Args:
        document_id: ID tai lieu
        topic: Chu de cu the (None = toan bo tai lieu)
        n_questions: So cau hoi (mac dinh: settings.n_quiz_questions)
        
    Returns:
        QuizSet object chua danh sach QuizItem
    """
    if n_questions is None:
        n_questions = settings.n_quiz_questions
    
    # Lay chunks: neu co topic thi retrieve theo topic, khong thi lay tat ca
    if topic:
        chunks = retrieve(topic, top_k=settings.initial_k, document_id=document_id)
    else:
        chunks = fetch_all_chunks(document_id)
    
    if not chunks:
        return QuizSet(items=[], document_id=document_id, topic=topic)
    
    # Render prompt
    env = _get_jinja_env()
    template = env.get_template("quiz.j2")
    context = _chunks_to_context(chunks)
    prompt = template.render(chunks=context, n_questions=n_questions, topic=topic)
    
    # Goi LLM
    llm = get_llm()
    raw_response = llm.generate(prompt)
    
    # Parse JSON + validate
    try:
        raw_items = _parse_json(raw_response)
    except Exception as e:
        print(f"Warning: Loi parse JSON quiz tu LLM: {e}")
        return QuizSet(items=[], document_id=document_id, topic=topic)
    
    valid_items = _validate_quiz_items(raw_items)
    
    return QuizSet(
        items=valid_items[:n_questions],
        document_id=document_id,
        topic=topic,
    )


def generate_flashcards(
    document_id: str,
    topic: Optional[str] = None,
    n_cards: Optional[int] = None,
) -> FlashcardSet:
    """
    Tao bo flashcards tu tai lieu.
    
    Args:
        document_id: ID tai lieu
        topic: Chu de cu the (None = toan bo tai lieu)
        n_cards: So flashcards (mac dinh: settings.n_flashcards)
        
    Returns:
        FlashcardSet object chua danh sach Flashcard
    """
    if n_cards is None:
        n_cards = settings.n_flashcards
    
    # Lay chunks: neu co topic thi retrieve theo topic, khong thi lay tat ca
    if topic:
        chunks = retrieve(topic, top_k=settings.initial_k, document_id=document_id)
    else:
        chunks = fetch_all_chunks(document_id)
    
    if not chunks:
        return FlashcardSet(cards=[], document_id=document_id, topic=topic)
    
    # Render prompt
    env = _get_jinja_env()
    template = env.get_template("flashcard.j2")
    context = _chunks_to_context(chunks)
    prompt = template.render(chunks=context, n_cards=n_cards, topic=topic)
    
    # Goi LLM
    llm = get_llm()
    raw_response = llm.generate(prompt)
    
    # Parse JSON + validate
    try:
        raw_items = _parse_json(raw_response)
    except Exception as e:
        print(f"Warning: Loi parse JSON flashcard tu LLM: {e}")
        return FlashcardSet(cards=[], document_id=document_id, topic=topic)
    
    valid_cards = _validate_flashcard_items(raw_items)
    
    return FlashcardSet(
        cards=valid_cards[:n_cards],
        document_id=document_id,
        topic=topic,
    )
