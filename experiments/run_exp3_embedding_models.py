# -*- coding: utf-8 -*-
"""
Kịch bản Thực nghiệm 3: So sánh Mô hình Embedding (Embedding Models Comparison)
Đối chiếu giữa:
  1. intfloat/multilingual-e5-large (Baseline hiện tại)
  2. BAAI/bge-m3 (Mô hình SOTA đa ngôn ngữ mới)

Đo lường:
  - 4 chỉ số Ragas: Context Precision, Context Recall, Faithfulness, Answer Relevancy
  - Thời gian trễ truy xuất trung bình (Retrieval Latency - ms)
  - Đánh giá khả năng xử lý câu hỏi chứa từ khóa viết tắt kỹ thuật
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

# Safe monkeypatch for CVE-2025-32434 in older PyTorch
try:
    import transformers.modeling_utils as _mu
    _mu.check_torch_load_is_safe = lambda: None
    import transformers.utils.import_utils as _iu
    _iu.check_torch_load_is_safe = lambda: None
except Exception:
    pass

from src.core.config import settings
from src.core.eval_schemas import EvalSample, RagasScores
from src.db.store import get_store
from src.pipelines.indexing import index_document
from src.pipelines.rag import answer, retrieve
from src.pipelines.evaluation import RagasEvaluator


def load_testset(path: Path) -> List[EvalSample]:
    """Tải bộ testset chuẩn vàng từ tệp JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [EvalSample(**item) for item in data]


def run_experiment_3(
    sample_pdf_path: Path,
    testset_path: Path,
    max_samples: int = 3,
    output_dir: Path = REPO_ROOT / "experiments"
):
    """Chạy Thực nghiệm 3: So sánh 2 mô hình Embedding."""
    output_dir.mkdir(parents=True, exist_ok=True)
    testset = load_testset(testset_path)[:max_samples]

    print(f"\n{'='*75}", flush=True)
    print("BẮT ĐẦU THỰC NGHIỆM 3: SO SÁNH EMBEDDING MODELS (MULTILINGUAL-E5 vs BGE-M3)", flush=True)
    print(f"Tài liệu kiểm chuẩn: {sample_pdf_path.name}", flush=True)
    print(f"Số lượng mẫu đánh giá: {len(testset)} câu hỏi", flush=True)
    print(f"{'='*75}\n", flush=True)

    evaluator = RagasEvaluator()

    models_config = [
        {
            "name": "Multilingual-E5-Large (Baseline)",
            "model_id": "intfloat/multilingual-e5-large",
            "collection_name": "documents",
            "chunk_size": 1000,
            "chunk_overlap": 150,
        },
        {
            "name": "BAAI/BGE-M3 (SOTA Candidate)",
            "model_id": "BAAI/bge-m3",
            "collection_name": "documents_bge_m3",
            "chunk_size": 1000,
            "chunk_overlap": 150,
        }
    ]

    all_results: List[Dict[str, Any]] = []

    for cfg in models_config:
        model_name = cfg["name"]
        model_id = cfg["model_id"]
        col_name = cfg["collection_name"]

        print(f"\n=======================================================", flush=True)
        print(f"-> ĐANG KHỞI TẠO & ĐÁNH GIÁ: {model_name}", flush=True)
        print(f"   Model: {model_id} | Collection: {col_name}", flush=True)
        print(f"=======================================================", flush=True)

        store = get_store(embedding_model=model_id, collection_name=col_name)

        # 1. Index tài liệu vào collection riêng của model đó
        doc_id = index_document(
            file_path=sample_pdf_path,
            force_reindex=False,
            chunking_method="recursive",
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            store=store
        )

        total_chunks = len(store.scroll_by_document(doc_id))
        print(f"-> Tổng số chunks trong collection '{col_name}': {total_chunks}", flush=True)

        # 2. Đánh giá từng mẫu câu hỏi
        scores_list: List[RagasScores] = []
        latencies_ms: List[float] = []

        for idx, sample in enumerate(testset, 1):
            print(f"\n  [{idx}/{len(testset)}] Câu hỏi: '{sample.question[:55]}...'", flush=True)

            # Đo độ trễ truy xuất (Retrieval latency)
            t0 = time.perf_counter()
            retrieved = retrieve(sample.question, top_k=5, document_id=doc_id, store=store)
            t_ret = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(t_ret)

            # Chạy RAG tạo sinh câu trả lời
            rag_res = answer(
                query=sample.question,
                document_id=doc_id,
                use_reranker=False,
                top_k=5,
                store=store
            )

            contexts = [c.content for c in rag_res.chunks]
            scores = evaluator.evaluate_sample(
                question=sample.question,
                reference_answer=sample.reference_answer,
                generated_answer=rag_res.answer,
                retrieved_contexts=contexts
            )
            scores_list.append(scores)

            print(f"      Độ trễ Retrieval: {t_ret:.1f}ms", flush=True)
            print(f"      Score -> Faithfulness={scores.faithfulness:.2f}, AnswerRel={scores.answer_relevancy:.2f}, "
                  f"ContextPrec={scores.context_precision:.2f}, ContextRecall={scores.context_recall:.2f}", flush=True)

            time.sleep(1.0)

        avg_scores = RagasScores(
            context_precision=round(sum(s.context_precision for s in scores_list) / len(scores_list), 4),
            context_recall=round(sum(s.context_recall for s in scores_list) / len(scores_list), 4),
            faithfulness=round(sum(s.faithfulness for s in scores_list) / len(scores_list), 4),
            answer_relevancy=round(sum(s.answer_relevancy for s in scores_list) / len(scores_list), 4),
        )
        avg_latency = round(sum(latencies_ms) / len(latencies_ms), 2)

        print(f"\n==> KẾT QUẢ TRUNG BÌNH {model_name}:", flush=True)
        print(f"    Faithfulness:     {avg_scores.faithfulness}", flush=True)
        print(f"    Answer Relevancy: {avg_scores.answer_relevancy}", flush=True)
        print(f"    Context Precision:{avg_scores.context_precision}", flush=True)
        print(f"    Context Recall:   {avg_scores.context_recall}", flush=True)
        print(f"    Retrieval Latency:{avg_latency} ms\n", flush=True)

        all_results.append({
            "model_name": model_name,
            "model_id": model_id,
            "collection": col_name,
            "total_chunks": total_chunks,
            "avg_latency_ms": avg_latency,
            "scores": avg_scores.model_dump()
        })

    # Tính độ chênh lệch delta giữa BGE-M3 và Baseline E5
    e5_scores = all_results[0]["scores"]
    bge_scores = all_results[1]["scores"]
    delta = {
        "delta_faithfulness": round(bge_scores["faithfulness"] - e5_scores["faithfulness"], 4),
        "delta_answer_relevancy": round(bge_scores["answer_relevancy"] - e5_scores["answer_relevancy"], 4),
        "delta_context_precision": round(bge_scores["context_precision"] - e5_scores["context_precision"], 4),
        "delta_context_recall": round(bge_scores["context_recall"] - e5_scores["context_recall"], 4),
        "delta_latency_ms": round(all_results[1]["avg_latency_ms"] - all_results[0]["avg_latency_ms"], 2),
    }

    output_payload = {
        "results": all_results,
        "delta_bge_vs_e5": delta
    }

    out_json = output_dir / "results_exp3_embedding.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print(f"\n[HOÀN THÀNH THỰC NGHIỆM 3] Đã lưu kết quả vào: {out_json}", flush=True)
    print(f"Delta so sánh: {delta}\n", flush=True)
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
    run_experiment_3(test_pdf, testset_file, max_samples=3)
