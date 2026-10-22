# -*- coding: utf-8 -*-
"""
Kịch bản Thực nghiệm 1: So sánh các phương pháp Phân đoạn văn bản (Chunking Methods)
Cấu hình đánh giá:
  - Recursive: 1500/200, 1000/150 (Baseline), 800/100, 500/50
  - Semantic: Interquartile, Std_dev, Percentile
Đo lường:
  - Số lượng chunks sinh ra
  - 4 chỉ số RAGAS: Faithfulness, Answer Relevancy, Context Precision, Context Recall
"""
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add repo root to sys.path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.core.config import settings
from src.core.eval_schemas import EvalSample, RagasScores, EvalResultItem, ExperimentSummary
from src.pipelines.indexing import index_document
from src.pipelines.rag import answer
from src.pipelines.evaluation import RagasEvaluator
from src.db.store import get_store

EXPERIMENT_CONFIGS = [
    {
        "name": "Recursive S:1500/O:200",
        "method": "recursive",
        "chunk_size": 1500,
        "chunk_overlap": 200,
    },
    {
        "name": "Recursive S:1000/O:150 (Baseline)",
        "method": "recursive",
        "chunk_size": 1000,
        "chunk_overlap": 150,
    },
    {
        "name": "Recursive S:800/O:100",
        "method": "recursive",
        "chunk_size": 800,
        "chunk_overlap": 100,
    },
    {
        "name": "Recursive S:500/O:50",
        "method": "recursive",
        "chunk_size": 500,
        "chunk_overlap": 50,
    },
    {
        "name": "Semantic (Interquartile)",
        "method": "semantic",
        "semantic_threshold_type": "interquartile",
    },
    {
        "name": "Semantic (Std_dev)",
        "method": "semantic",
        "semantic_threshold_type": "standard_deviation",
    },
    {
        "name": "Semantic (Percentile)",
        "method": "semantic",
        "semantic_threshold_type": "percentile",
    },
]


def load_testset(path: Path) -> List[EvalSample]:
    """Tải bộ testset chuẩn vàng từ tệp JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [EvalSample(**item) for item in data]


def run_experiment_1(
    sample_pdf_path: Path,
    testset_path: Path,
    max_samples: int = 5,
    output_dir: Path = REPO_ROOT / "experiments"
):
    """Chạy toàn diện Thực nghiệm 1 trên 7 cấu hình."""
    output_dir.mkdir(parents=True, exist_ok=True)
    testset = load_testset(testset_path)[:max_samples]
    print(f"\n{'='*70}")
    print(f"BẮT ĐẦU THỰC NGHIỆM 1: SO SÁNH 7 CẤU HÌNH CHUNKING")
    print(f"Tài liệu thử nghiệm: {sample_pdf_path.name}")
    print(f"Số lượng mẫu đánh giá: {len(testset)}")
    print(f"{'='*70}\n")

    evaluator = RagasEvaluator()
    store = get_store()
    all_summaries: List[Dict[str, Any]] = []

    for cfg in EXPERIMENT_CONFIGS:
        cfg_name = cfg["name"]
        method = cfg["method"]
        print(f"\n--- Đang chạy cấu hình: {cfg_name} ---")

        # 1. Index tài liệu theo cấu hình (force_reindex=False để tái sử dụng chunks nếu đã index)
        if method == "semantic":
            doc_id = index_document(
                sample_pdf_path,
                force_reindex=False,
                chunking_method="semantic",
                semantic_threshold_type=cfg["semantic_threshold_type"]
            )
        else:
            doc_id = index_document(
                sample_pdf_path,
                force_reindex=False,
                chunking_method="recursive",
                chunk_size=cfg["chunk_size"],
                chunk_overlap=cfg["chunk_overlap"]
            )

        # 2. Đếm số chunks sinh ra trong Qdrant
        records = store.scroll_by_document(doc_id)
        total_chunks = len(records)
        print(f"-> Tổng số chunks đã tạo trong Qdrant: {total_chunks}", flush=True)

        # 3. Đánh giá từng mẫu câu hỏi
        scores_list: List[RagasScores] = []
        result_items: List[Dict[str, Any]] = []

        for idx, sample in enumerate(testset, 1):
            print(f"  [{idx}/{len(testset)}] Query: '{sample.question[:50]}...'", flush=True)
            rag_res = answer(query=sample.question, document_id=doc_id, use_reranker=False, top_k=5)
            retrieved_contexts = [c.content for c in rag_res.chunks]

            scores = evaluator.evaluate_sample(
                question=sample.question,
                reference_answer=sample.reference_answer,
                generated_answer=rag_res.answer,
                retrieved_contexts=retrieved_contexts
            )
            scores_list.append(scores)
            result_items.append({
                "sample_id": sample.id,
                "question": sample.question,
                "scores": scores.model_dump()
            })
            print(f"      Score -> Faithfulness={scores.faithfulness:.2f}, AnswerRel={scores.answer_relevancy:.2f}, ContextPrec={scores.context_precision:.2f}, ContextRecall={scores.context_recall:.2f}", flush=True)
            import time
            time.sleep(1.0)

        # 4. Tính điểm trung bình
        avg_scores = RagasScores(
            context_precision=round(sum(s.context_precision for s in scores_list) / len(scores_list), 4),
            context_recall=round(sum(s.context_recall for s in scores_list) / len(scores_list), 4),
            faithfulness=round(sum(s.faithfulness for s in scores_list) / len(scores_list), 4),
            answer_relevancy=round(sum(s.answer_relevancy for s in scores_list) / len(scores_list), 4),
        )

        print(f"==> Kết quả Ragas {cfg_name}: Faithfulness={avg_scores.faithfulness}, "
              f"AnswerRel={avg_scores.answer_relevancy}, "
              f"ContextPrec={avg_scores.context_precision}, "
              f"ContextRecall={avg_scores.context_recall}\n", flush=True)

        all_summaries.append({
            "config_name": cfg_name,
            "method": method,
            "total_chunks": total_chunks,
            "scores": avg_scores.model_dump(),
            "items": result_items
        })

    # Lưu kết quả ra file JSON
    out_json = output_dir / "results_exp1_chunking.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    print(f"\n[HOÀN THÀNH THỰC NGHIỆM 1] Kết quả đã lưu vào: {out_json}")
    return all_summaries


if __name__ == "__main__":
    test_pdf = REPO_ROOT / "data" / "pdfs" / "[Description]-LLMs-Fine-tuning.pdf"
    if not test_pdf.exists():
        pdfs = list((REPO_ROOT / "data" / "pdfs").glob("*.pdf"))
        if pdfs:
            test_pdf = pdfs[0]
        else:
            print("Không tìm thấy tệp PDF trong data/pdfs/")
            sys.exit(1)

    testset_file = REPO_ROOT / "data" / "evaluation" / "testset_qa_gt.json"
    run_experiment_1(test_pdf, testset_file, max_samples=3)
