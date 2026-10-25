# -*- coding: utf-8 -*-
"""
Kịch bản Thực nghiệm 3 (Golden Fusion) & Kiểm chứng Hybrid Search trên Query 3:
So sánh 4 chiến lược cốt lõi:
  1. Baseline: Recursive (1000/150) (Không Reranker)
  2. Recursive + Reranker (Mô hình của Thực nghiệm 2)
  3. Semantic (Interquartile) (Mô hình tốt nhất của Thực nghiệm 1)
  4. Semantic + Reranker (Golden Fusion) [ĐỀ XUẤT MỚI]

Đồng thời kiểm chứng riêng Query 3 (DPO vs RLHF) trên tài liệu Alignment:
  - Dense-Only vs BM25 Sparse vs Hybrid (RRF)
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

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.core.config import settings
from src.core.eval_schemas import EvalSample, RagasScores
from src.db.store import get_store
from src.pipelines.indexing import index_document
from src.pipelines.rag import answer, retrieve, rerank
from src.pipelines.evaluation import RagasEvaluator
from src.core.hybrid import get_bm25_retriever_for_document, reciprocal_rank_fusion


def load_testset(path: Path) -> List[EvalSample]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [EvalSample(**item) for item in data]


def run_experiment_golden_fusion(
    testset_path: Path = REPO_ROOT / "data" / "evaluation" / "testset_qa_gt.json",
    max_samples: int = 3,
    output_dir: Path = REPO_ROOT / "experiments"
):
    output_dir.mkdir(parents=True, exist_ok=True)
    testset = load_testset(testset_path)[:max_samples]

    pdf_target = REPO_ROOT / "data" / "pdfs" / "[Project]-Finance-Analytics-eDA.pdf"

    print(f"\n{'='*75}", flush=True)
    print("BẮT ĐẦU THỰC NGHIỆM 3: SO SÁNH 4 CHIẾN LƯỢC CỐT LÕI (GOLDEN FUSION)", flush=True)
    print(f"Tài liệu đánh giá: {pdf_target.name} (140 trang)", flush=True)
    print(f"Số lượng mẫu đánh giá: {len(testset)} câu hỏi", flush=True)
    print(f"{'='*75}\n", flush=True)

    evaluator = RagasEvaluator()
    store = get_store()

    # Indexing tài liệu (Recursive & Semantic)
    print("-> Indexing tài liệu Finance (Recursive 1000/150 & Semantic Interquartile)...", flush=True)
    doc_target_rec = index_document(
        pdf_target,
        force_reindex=False,
        chunking_method="recursive",
        chunk_size=1000,
        chunk_overlap=150,
        store=store
    )
    doc_target_sem = index_document(
        pdf_target,
        force_reindex=False,
        chunking_method="semantic",
        semantic_threshold_type="interquartile",
        store=store
    )

    doc_map = {
        "[Project]-Finance-Analytics-eDA.pdf": {"rec": doc_target_rec, "sem": doc_target_sem},
        "[Description]-LLMs-Fine-tuning.pdf": {"rec": doc_target_rec, "sem": doc_target_sem},
        "[Reading]-LLM-Alignment.pdf": {"rec": doc_target_rec, "sem": doc_target_sem},
    }

    strategies = [
        {
            "id": "baseline_rec",
            "name": "1. Baseline: Recursive (1000/150)",
            "chunk_type": "rec",
            "use_reranker": False,
        },
        {
            "id": "rec_reranker",
            "name": "2. Recursive + Reranker (Thực nghiệm 2)",
            "chunk_type": "rec",
            "use_reranker": True,
        },
        {
            "id": "semantic_only",
            "name": "3. Semantic (Interquartile) (Thực nghiệm 1)",
            "chunk_type": "sem",
            "use_reranker": False,
        },
        {
            "id": "golden_fusion",
            "name": "4. Semantic + Reranker (Golden Fusion - Đề xuất Mới)",
            "chunk_type": "sem",
            "use_reranker": True,
        },
    ]

    all_results: List[Dict[str, Any]] = []

    for strat in strategies:
        strat_name = strat["name"]
        print(f"\n=======================================================", flush=True)
        print(f"-> ĐANG ĐÁNH GIÁ CHIẾN LƯỢC: {strat_name}", flush=True)
        print(f"=======================================================", flush=True)

        scores_list: List[RagasScores] = []

        for idx, sample in enumerate(testset, 1):
            target_doc_name = sample.target_document if hasattr(sample, "target_document") else "[Description]-LLMs-Fine-tuning.pdf"
            resolved_doc_map = doc_map.get(target_doc_name, doc_map["[Description]-LLMs-Fine-tuning.pdf"])
            doc_id = resolved_doc_map[strat["chunk_type"]]

            print(f"  [{idx}/{len(testset)}] Query: '{sample.question[:50]}...' (Doc: {target_doc_name})", flush=True)

            rag_res = answer(
                query=sample.question,
                document_id=doc_id,
                use_reranker=strat["use_reranker"],
                use_hybrid=False,
                initial_k=15 if strat["use_reranker"] else 5,
                top_k=5,
                store=store
            )

            scores = evaluator.evaluate_sample(
                question=sample.question,
                reference_answer=sample.reference_answer,
                generated_answer=rag_res.answer,
                retrieved_contexts=[c.content for c in rag_res.chunks]
            )
            scores_list.append(scores)

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
        print(f"    Context Recall:   {avg.context_recall}\n", flush=True)

        all_results.append({
            "strategy": strat_name,
            "id": strat["id"],
            "scores": avg.model_dump()
        })

    # =========================================================================
    # BƯỚC 2: KIỂM CHỨNG HYBRID SEARCH TRÊN QUERY 3 (DPO vs RLHF)
    # =========================================================================
    print(f"\n{'='*75}", flush=True)
    print("BƯỚC 2: KIỂM CHỨNG TÁC ĐỘNG CỦA HYBRID SEARCH TRÊN QUERY 3 (DPO vs RLHF)", flush=True)
    print(f"{'='*75}\n", flush=True)

    q3_sample = testset[2]  # Query 3: Thu nhập lãi thuần & CASA
    doc_target_id = doc_target_rec

    print(f"Câu hỏi kiểm định: '{q3_sample.question}'", flush=True)
    print(f"Tài liệu đối chiếu: {pdf_target.name} (ID: {doc_target_id})", flush=True)

    # 1. Dense Only
    dense_chunks = retrieve(q3_sample.question, top_k=5, document_id=doc_target_id, store=store, use_hybrid=False)
    # 2. BM25 Sparse Only
    bm25_retriever = get_bm25_retriever_for_document(doc_target_id, store=store)
    sparse_chunks = bm25_retriever.search(q3_sample.question, top_k=5)
    # 3. Hybrid Search (RRF)
    hybrid_chunks = reciprocal_rank_fusion(dense_chunks, sparse_chunks, k=60, top_k=5)

    def check_contains_finance_term(chunks):
        return any("thu nhập lãi" in c.content.lower() or "chi phí lãi" in c.content.lower() or "tiền gửi không kỳ hạn" in c.content.lower() for c in chunks)

    dense_has_term = check_contains_finance_term(dense_chunks)
    sparse_has_term = check_contains_finance_term(sparse_chunks)
    hybrid_has_term = check_contains_finance_term(hybrid_chunks)

    print(f"\n-> Kết quả nhận diện thuật ngữ tài chính trong Top-5 Chunks:", flush=True)
    print(f"   - Dense-Only:   {'[V] TÌM THẤY' if dense_has_term else '[X] KHÔNG TÌM THẤY'}", flush=True)
    print(f"   - BM25-Only:    {'[V] TÌM THẤY' if sparse_has_term else '[X] KHÔNG TÌM THẤY'}", flush=True)
    print(f"   - Hybrid (RRF): {'[V] TÌM THẤY' if hybrid_has_term else '[X] KHÔNG TÌM THẤY'}", flush=True)

    hybrid_validation = {
        "query": q3_sample.question,
        "document": pdf_target.name,
        "dense_found_target": dense_has_term,
        "sparse_found_target": sparse_has_term,
        "hybrid_found_target": hybrid_has_term
    }

    out_payload = {
        "golden_fusion_results": all_results,
        "hybrid_query3_validation": hybrid_validation
    }

    out_file = output_dir / "results_exp3_golden_fusion.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)

    print(f"\n[HOÀN THÀNH] Kết quả đã lưu vào: {out_file}", flush=True)
    return out_payload


if __name__ == "__main__":
    run_experiment_golden_fusion()
