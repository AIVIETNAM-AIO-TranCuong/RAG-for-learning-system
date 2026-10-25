# -*- coding: utf-8 -*-
"""
Kịch bản Thực nghiệm 2: Tái xếp hạng bằng Cross-Encoder (Reranking Ablation Study)
Cấu hình đánh giá:
  1. Baseline: Recursive (1000/150) - Bi-Encoder trực tiếp top-5 chunks
  2. So sánh: Semantic (Interquartile) - Bi-Encoder trực tiếp top-5 chunks
  3. Đề xuất (Ours): Recursive (1000/150) + BGE-Reranker-v2-m3 (Two-Stage: initial_k=15 -> rerank_k=5)
Đo lường:
  - 4 chỉ số RAGAS
  - Độ tăng trưởng Delta (Ours vs Baseline)
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

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.core.config import settings
from src.core.eval_schemas import EvalSample, RagasScores
from src.pipelines.indexing import index_document
from src.pipelines.rag import answer
from src.pipelines.evaluation import RagasEvaluator
from src.db.store import get_store


def load_testset(path: Path) -> List[EvalSample]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [EvalSample(**item) for item in data]


def run_experiment_2(
    sample_pdf_path: Path,
    testset_path: Path,
    max_samples: int = 5,
    output_dir: Path = REPO_ROOT / "experiments"
):
    """Chạy Thực nghiệm 2: So sánh tác động của Cross-Encoder Reranker."""
    output_dir.mkdir(parents=True, exist_ok=True)
    testset = load_testset(testset_path)[:max_samples]
    print(f"\n{'='*70}")
    print(f"BẮT ĐẦU THỰC NGHIỆM 2: ABLATION STUDY CROSS-ENCODER RERANKER")
    print(f"Tài liệu thử nghiệm: {sample_pdf_path.name}")
    print(f"Số lượng mẫu đánh giá: {len(testset)}")
    print(f"{'='*70}\n")

    evaluator = RagasEvaluator()
    store = get_store()

    # 1. Đảm bảo 2 phiên bản tài liệu đã được index vào Qdrant
    print("-> Indexing Baseline: Recursive 1000/150...")
    doc_id_rec = index_document(
        sample_pdf_path,
        force_reindex=False,
        chunking_method="recursive",
        chunk_size=1000,
        chunk_overlap=150
    )
    rec_chunks_count = len(store.scroll_by_document(doc_id_rec))

    print("-> Indexing Semantic: Interquartile...")
    doc_id_sem = index_document(
        sample_pdf_path,
        force_reindex=False,
        chunking_method="semantic",
        semantic_threshold_type="interquartile"
    )
    sem_chunks_count = len(store.scroll_by_document(doc_id_sem))

    strategies = [
        {
            "name": "Recursive (1000/150) Baseline",
            "doc_id": doc_id_rec,
            "use_reranker": False,
            "total_chunks": rec_chunks_count,
            "top_k": 5
        },
        {
            "name": "Semantic (Interquartile)",
            "doc_id": doc_id_sem,
            "use_reranker": False,
            "total_chunks": sem_chunks_count,
            "top_k": 5
        },
        {
            "name": "Recursive + Reranker (Ours)",
            "doc_id": doc_id_rec,
            "use_reranker": True,
            "total_chunks": rec_chunks_count,
            "initial_k": 15,
            "top_k": 5
        }
    ]

    all_results: List[Dict[str, Any]] = []

    for strat in strategies:
        name = strat["name"]
        print(f"\n--- Đang đánh giá chiến lược: {name} ---", flush=True)
        scores_list: List[RagasScores] = []

        for idx, sample in enumerate(testset, 1):
            print(f"  [{idx}/{len(testset)}] Query: '{sample.question[:50]}...'", flush=True)
            rag_res = answer(
                query=sample.question,
                document_id=strat["doc_id"],
                use_reranker=strat["use_reranker"],
                initial_k=strat.get("initial_k", 15),
                top_k=strat["top_k"]
            )
            contexts = [c.content for c in rag_res.chunks]
            scores = evaluator.evaluate_sample(
                question=sample.question,
                reference_answer=sample.reference_answer,
                generated_answer=rag_res.answer,
                retrieved_contexts=contexts
            )
            scores_list.append(scores)
            print(f"      Score -> Faithfulness={scores.faithfulness:.2f}, AnswerRel={scores.answer_relevancy:.2f}, ContextPrec={scores.context_precision:.2f}, ContextRecall={scores.context_recall:.2f}", flush=True)
            import time
            time.sleep(1.0)

        avg = RagasScores(
            context_precision=round(sum(s.context_precision for s in scores_list) / len(scores_list), 4),
            context_recall=round(sum(s.context_recall for s in scores_list) / len(scores_list), 4),
            faithfulness=round(sum(s.faithfulness for s in scores_list) / len(scores_list), 4),
            answer_relevancy=round(sum(s.answer_relevancy for s in scores_list) / len(scores_list), 4),
        )

        print(f"==> Kết quả trung bình {name}: Faithfulness={avg.faithfulness}, AnswerRel={avg.answer_relevancy}, "
              f"ContextPrec={avg.context_precision}, ContextRecall={avg.context_recall}\n", flush=True)

        all_results.append({
            "strategy": name,
            "chunks": strat["total_chunks"],
            "use_reranker": strat["use_reranker"],
            "scores": avg.model_dump()
        })

    # Tính delta cải thiện giữa Ours và Baseline
    base_scores = all_results[0]["scores"]
    ours_scores = all_results[2]["scores"]
    delta = {
        "delta_faithfulness": round(ours_scores["faithfulness"] - base_scores["faithfulness"], 4),
        "delta_answer_relevancy": round(ours_scores["answer_relevancy"] - base_scores["answer_relevancy"], 4),
        "delta_context_precision": round(ours_scores["context_precision"] - base_scores["context_precision"], 4),
        "delta_context_recall": round(ours_scores["context_recall"] - base_scores["context_recall"], 4),
    }

    output_payload = {
        "results": all_results,
        "delta_improvement_vs_baseline": delta
    }

    out_json = output_dir / "results_exp2_reranking.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print(f"\n[HOÀN THÀNH THỰC NGHIỆM 2] Đã tính Delta: {delta}")
    print(f"Kết quả lưu vào: {out_json}")
    return output_payload


if __name__ == "__main__":
    test_pdf = REPO_ROOT / "data" / "pdfs" / "[Project]-Finance-Analytics-eDA.pdf"
    if not test_pdf.exists():
        pdfs = list((REPO_ROOT / "data" / "pdfs").glob("*.pdf"))
        if pdfs:
            test_pdf = pdfs[0]
        else:
            print("Không tìm thấy tệp PDF trong data/pdfs/")
            sys.exit(1)

    testset_file = REPO_ROOT / "data" / "evaluation" / "testset_qa_gt.json"
    run_experiment_2(test_pdf, testset_file, max_samples=3)
