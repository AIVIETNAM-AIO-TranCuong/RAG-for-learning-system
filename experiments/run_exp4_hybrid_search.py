# -*- coding: utf-8 -*-
"""
Kịch bản Thực nghiệm 4: Đánh giá Tác động của Hybrid Search (Dense + Sparse BM25 + RRF)
So sánh 4 chiến lược:
  1. Dense-Only Baseline (Multilingual-E5-large)
  2. Sparse-Only (BM25 Okapi)
  3. Hybrid Search (Dense E5 + Sparse BM25 với Reciprocal Rank Fusion)
  4. Hybrid Search + Cross-Encoder Reranker (Cấu hình toàn diện SOTA)

Mục tiêu cốt lõi:
  - Khắc phục hiện tượng Zero-Recall ở câu hỏi chứa từ khóa viết tắt kỹ thuật (DPO vs RLHF)
  - Tối ưu hóa Context Recall và Context Precision toàn diện
"""
import os
import sys
import time
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
from src.core.eval_schemas import EvalSample, RagasScores
from src.db.store import get_store
from src.pipelines.indexing import index_document
from src.pipelines.rag import answer, retrieve, rerank, _get_jinja_env
from src.pipelines.evaluation import RagasEvaluator
from src.core.hybrid import get_bm25_retriever_for_document
from src.llm import get_llm


def load_testset(path: Path) -> List[EvalSample]:
    """Tải bộ testset chuẩn vàng từ tệp JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [EvalSample(**item) for item in data]


def run_experiment_4(
    sample_pdf_path: Path,
    testset_path: Path,
    max_samples: int = 3,
    output_dir: Path = REPO_ROOT / "experiments"
):
    """Chạy Thực nghiệm 4: So sánh 4 chiến lược truy xuất (Dense vs BM25 vs Hybrid vs Hybrid+Reranker)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    testset = load_testset(testset_path)[:max_samples]

    print(f"\n{'='*75}", flush=True)
    print("BẮT ĐẦU THỰC NGHIỆM 4: ĐÁNH GIÁ HYBRID SEARCH (DENSE + BM25 + RRF)", flush=True)
    print(f"Tài liệu kiểm chuẩn: {sample_pdf_path.name}", flush=True)
    print(f"Số lượng mẫu đánh giá: {len(testset)} câu hỏi", flush=True)
    print(f"{'='*75}\n", flush=True)

    evaluator = RagasEvaluator()
    store = get_store()

    # Index tài liệu nếu chưa có
    doc_id = index_document(
        sample_pdf_path,
        force_reindex=False,
        chunking_method="recursive",
        chunk_size=1000,
        chunk_overlap=150,
        store=store
    )
    total_chunks = len(store.scroll_by_document(doc_id))
    print(f"-> Tổng số chunks trong tài liệu: {total_chunks}\n", flush=True)

    bm25_retriever = get_bm25_retriever_for_document(doc_id, store=store)

    strategies = [
        {
            "id": "dense_only",
            "name": "1. Dense-Only (Multilingual-E5)",
            "use_hybrid": False,
            "use_reranker": False,
            "mode": "dense"
        },
        {
            "id": "sparse_only",
            "name": "2. Sparse-Only (BM25 Okapi)",
            "use_hybrid": False,
            "use_reranker": False,
            "mode": "sparse"
        },
        {
            "id": "hybrid_search",
            "name": "3. Hybrid Search (Dense + BM25 RRF)",
            "use_hybrid": True,
            "use_reranker": False,
            "mode": "hybrid"
        },
        {
            "id": "hybrid_reranker",
            "name": "4. Hybrid + Cross-Encoder Reranker (Ours SOTA)",
            "use_hybrid": True,
            "use_reranker": True,
            "mode": "hybrid_rerank"
        }
    ]

    all_results: List[Dict[str, Any]] = []

    for strat in strategies:
        strat_name = strat["name"]
        strat_mode = strat["mode"]

        print(f"\n=======================================================", flush=True)
        print(f"-> ĐANG ĐÁNH GIÁ CHIẾN LƯỢC: {strat_name}", flush=True)
        print(f"=======================================================", flush=True)

        scores_list: List[RagasScores] = []
        q3_scores = None

        for idx, sample in enumerate(testset, 1):
            print(f"  [{idx}/{len(testset)}] Query: '{sample.question[:55]}...'", flush=True)

            if strat_mode == "sparse":
                # Sparse BM25 Only
                retrieved_chunks = bm25_retriever.search(sample.question, top_k=5)
                # Tạo câu trả lời bằng Jinja2 prompt + LLM
                chunks_ctx = [{"marker": f"S{i+1}", "content": c.content, "filename": c.metadata.filename, "page": c.metadata.page} for i, c in enumerate(retrieved_chunks)]
                prompt = _get_jinja_env().get_template("qa.j2").render(chunks=chunks_ctx, question=sample.question)
                generated_ans = get_llm().generate(prompt)
                contexts = [c.content for c in retrieved_chunks]
            else:
                rag_res = answer(
                    query=sample.question,
                    document_id=doc_id,
                    use_reranker=strat["use_reranker"],
                    use_hybrid=strat["use_hybrid"],
                    initial_k=15 if strat["use_reranker"] else 5,
                    top_k=5,
                    store=store
                )
                generated_ans = rag_res.answer
                contexts = [c.content for c in rag_res.chunks]

            scores = evaluator.evaluate_sample(
                question=sample.question,
                reference_answer=sample.reference_answer,
                generated_answer=generated_ans,
                retrieved_contexts=contexts
            )
            scores_list.append(scores)

            if idx == 3:  # Câu hỏi 3: DPO vs RLHF
                q3_scores = scores.model_dump()

            print(f"      Score -> Faithfulness={scores.faithfulness:.2f}, AnswerRel={scores.answer_relevancy:.2f}, "
                  f"ContextPrec={scores.context_precision:.2f}, ContextRecall={scores.context_recall:.2f}", flush=True)

            time.sleep(3.0)

        avg = RagasScores(
            context_precision=round(sum(s.context_precision for s in scores_list) / len(scores_list), 4),
            context_recall=round(sum(s.context_recall for s in scores_list) / len(scores_list), 4),
            faithfulness=round(sum(s.faithfulness for s in scores_list) / len(scores_list), 4),
            answer_relevancy=round(sum(s.answer_relevancy for s in scores_list) / len(scores_list), 4),
        )

        print(f"\n==> KẾT QUẢ TRUNG BÌNH {strat_name}:", flush=True)
        print(f"    Faithfulness:     {avg.faithfulness}", flush=True)
        print(f"    Answer Relevancy: {avg.answer_relevancy}", flush=True)
        print(f"    Context Precision:{avg.context_precision}", flush=True)
        print(f"    Context Recall:   {avg.context_recall}", flush=True)
        print(f"    Điểm Câu hỏi 3 (DPO/RLHF) -> Precision: {q3_scores['context_precision']}, Recall: {q3_scores['context_recall']}\n", flush=True)

        all_results.append({
            "strategy": strat_name,
            "mode": strat_mode,
            "scores": avg.model_dump(),
            "query_3_scores": q3_scores
        })

    # Tính delta so với Dense-Only Baseline
    base_scores = all_results[0]["scores"]
    hybrid_scores = all_results[2]["scores"]
    sota_scores = all_results[3]["scores"]

    delta_hybrid = {
        "delta_faithfulness": round(hybrid_scores["faithfulness"] - base_scores["faithfulness"], 4),
        "delta_context_precision": round(hybrid_scores["context_precision"] - base_scores["context_precision"], 4),
        "delta_context_recall": round(hybrid_scores["context_recall"] - base_scores["context_recall"], 4),
    }

    delta_sota = {
        "delta_faithfulness": round(sota_scores["faithfulness"] - base_scores["faithfulness"], 4),
        "delta_context_precision": round(sota_scores["context_precision"] - base_scores["context_precision"], 4),
        "delta_context_recall": round(sota_scores["context_recall"] - base_scores["context_recall"], 4),
    }

    output_payload = {
        "results": all_results,
        "delta_hybrid_vs_dense": delta_hybrid,
        "delta_sota_vs_dense": delta_sota
    }

    out_json = output_dir / "results_exp4_hybrid.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print(f"\n[HOÀN THÀNH THỰC NGHIỆM 4] Kết quả đã lưu vào: {out_json}", flush=True)
    return output_payload


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
    run_experiment_4(test_pdf, testset_file, max_samples=3)
