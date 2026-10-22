import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# -*- coding: utf-8 -*-
"""
Test Suite cho Module Đánh giá và Thực nghiệm RAG (Phase 4 Evaluation)
Chạy: python test_evaluation_pipeline.py
"""
import sys
import os
import unittest
from pathlib import Path

# Fix Windows console unicode printing
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add repo root to sys.path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from src.core.chunking import (
    split_text_into_sentences,
    cosine_distance,
    SemanticChunker,
    create_chunker
)
from src.core.eval_schemas import EvalSample, RagasScores, EvalResultItem
from src.pipelines.evaluation import _cosine_similarity, _clean_json_response, RagasEvaluator
import numpy as np


class TestEvaluationPipeline(unittest.TestCase):

    def test_01_sentence_splitting(self):
        """Kiểm tra tách câu tiếng Việt và tiếng Anh."""
        text = "Hệ thống RAG hỗ trợ học tập cá nhân hóa. Nó sử dụng mô hình E5 để nhúng văn bản! Liệu nó có hiệu quả? Rất hiệu quả."
        sentences = split_text_into_sentences(text)
        self.assertGreaterEqual(len(sentences), 3)
        self.assertIn("Hệ thống RAG hỗ trợ học tập cá nhân hóa.", sentences)

    def test_02_cosine_distance(self):
        """Kiểm tra tính khoảng cách cosine."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        v3 = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(cosine_distance(v1, v2), 0.0, places=4)
        self.assertAlmostEqual(cosine_distance(v1, v3), 1.0, places=4)

    def test_03_breakpoint_thresholds(self):
        """Kiểm tra tính toán 3 loại ngưỡng semantic breakpoint."""
        distances = [0.1, 0.12, 0.15, 0.11, 0.85, 0.14, 0.13, 0.90]

        # Interquartile
        chunker_iqr = SemanticChunker(threshold_type="interquartile")
        thresh_iqr = chunker_iqr.calculate_breakpoint_threshold(distances)
        self.assertGreater(thresh_iqr, 0.15)

        # Standard Deviation
        chunker_std = SemanticChunker(threshold_type="standard_deviation")
        thresh_std = chunker_std.calculate_breakpoint_threshold(distances)
        self.assertGreater(thresh_std, np.mean(distances))

        # Percentile
        chunker_pct = SemanticChunker(threshold_type="percentile", threshold_amount=80.0)
        thresh_pct = chunker_pct.calculate_breakpoint_threshold(distances)
        self.assertGreater(thresh_pct, 0.14)

    def test_04_eval_schemas(self):
        """Kiểm tra tính hợp lệ của Pydantic Schemas."""
        sample = EvalSample(
            id="test_01",
            question="LoRA là gì?",
            reference_answer="LoRA là phương pháp phân rã hạng thấp."
        )
        self.assertEqual(sample.id, "test_01")

        scores = RagasScores(
            context_precision=0.9247,
            context_recall=0.9400,
            faithfulness=0.8761,
            answer_relevancy=0.8044
        )
        self.assertEqual(scores.context_precision, 0.9247)
        self.assertEqual(scores.context_recall, 0.9400)

        # Kiểm tra validation ràng buộc điểm số trong [0, 1]
        with self.assertRaises(Exception):
            RagasScores(context_precision=1.5, context_recall=0.5, faithfulness=0.5, answer_relevancy=0.5)

    def test_05_json_cleaner(self):
        """Kiểm tra bóc tách JSON an toàn từ LLM."""
        raw_markdown = "Đây là kết quả:\n```json\n[{\"verdict\": 1}]\n```\nCảm ơn bạn."
        cleaned = _clean_json_response(raw_markdown)
        self.assertEqual(cleaned, [{"verdict": 1}])

        raw_obj = "Text trước {\"key\": \"value\"} text sau"
        cleaned_obj = _clean_json_response(raw_obj)
        self.assertEqual(cleaned_obj, {"key": "value"})

    def test_06_chart_generation(self):
        """Kiểm tra module xuất biểu đồ thực nghiệm."""
        from experiments.export_charts import plot_exp1_chunking, plot_exp2_reranking
        plot_exp1_chunking()
        plot_exp2_reranking()

        out_bar = REPO_ROOT / "experiments" / "output_charts" / "chunking_barchart.png"
        out_line = REPO_ROOT / "experiments" / "output_charts" / "chunking_linetrend.png"
        out_rerank = REPO_ROOT / "experiments" / "output_charts" / "reranking_barchart.png"

        self.assertTrue(out_bar.exists(), "Tệp chunking_barchart.png phải tồn tại")
        self.assertTrue(out_line.exists(), "Tệp chunking_linetrend.png phải tồn tại")
        self.assertTrue(out_rerank.exists(), "Tệp reranking_barchart.png phải tồn tại")


if __name__ == "__main__":
    print("\n--- BẮT ĐẦU CHẠY TEST SUITE ĐÁNH GIÁ VÀ THỰC NGHIỆM ---")
    unittest.main(verbosity=2)
