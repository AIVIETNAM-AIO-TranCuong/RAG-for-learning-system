# KẾ HOẠCH KỸ THUẬT: HIỆN THỰC HÓA PHƯƠNG PHÁP ĐÁNH GIÁ HỆ THỐNG RAG
*(Dựa trên đề xuất phương pháp từ sơ đồ chiến lược và kết quả thực nghiệm trong ảnh)*

---

## 🎯 Mục tiêu & Phạm vi (Scope)

Theo đúng định hướng trong sơ đồ **Evaluation Strategy (Ảnh 2)**:
* **Trọng tâm đồ án:** Tập trung vào **Giai đoạn 1 — Tối ưu hóa tầng Truy xuất (Retriever Optimization)** đo lường qua **Context Recall** và **Context Precision**.
* **Giai đoạn 2 (Tối ưu hóa tầng Tạo sinh - Faithfulness & Answer Relevancy):** Được ghi nhận là bước đo lường gián tiếp và là hướng phát triển tương lai.
* **Mục tiêu kỹ thuật của kế hoạch này:** Xây dựng trọn bộ mã nguồn Python thực thi, chạy thực nghiệm độc lập và thu thập dữ liệu đo lường tự động từ hệ thống RAG thực tế.

```
                    ┌───────────────────────────────┐      ┌──────────────────────────────────┐
                    │     1. Optimize Retriever     │ ───▶ │     2. Optimize Generation       │
                    │      (Recall & Precision)     │      │   (Faithfulness & Answer Rel.)   │
                    └───────────────────────────────┘      └──────────────────────────────────┘
                                   ▲
                    ┌──────────────────────────────┐
                    │ This project only focuses on │
                    │          this phase          │
                    └──────────────────────────────┘
```

---

## 🏗️ Kiến trúc các Module cần hiện thực hóa

```
d:\RAG_Learning_System\
├── src/
│   ├── core/
│   │   ├── chunking.py             <-- [MỚI] Module Chunking (Recursive & Semantic 3 ngưỡng)
│   │   └── eval_schemas.py         <-- [MỚI] Pydantic schemas cho Testset & Evaluation
│   ├── pipelines/
│   │   ├── indexing.py             <-- [CẬP NHẬT] Tích hợp bộ chọn Chunking Strategy
│   │   ├── evaluation.py           <-- [MỚI] Pipeline tính RAGAS metrics (LLM-as-a-judge)
│   │   └── rag.py                  <-- [CẬP NHẬT] Hỗ trợ bật/tắt Reranker linh hoạt
│   └── interfaces/
│       └── cli.py                  <-- [CẬP NHẬT] Bổ sung command `evaluate`
├── data/
│   └── evaluation/
│       └── testset_qa_gt.json      <-- [MỚI] Bộ câu hỏi + ground truth reference answers
├── experiments/
│   ├── run_exp1_chunking.py        <-- [MỚI] Script chạy Thực nghiệm 1: 7 cấu hình Chunking
│   ├── run_exp2_reranking.py       <-- [MỚI] Script chạy Thực nghiệm 2: Baseline vs Semantic vs Reranker
│   └── export_charts.py            <-- [MỚI] Xuất biểu đồ tự động từ kết quả thực tế
└── test_evaluation_pipeline.py     <-- [MỚI] Test suite kiểm thử toàn bộ luồng đánh giá
```

---

## 📋 Chi tiết các giai đoạn thực hiện (Actionable Tasks)

### 🔹 Giai đoạn 1: Hiện thực hóa Module Semantic Chunking (`src/core/chunking.py`)
> Đề xuất từ **Ảnh 2 & Ảnh 3 (Experiment 1: Chunking Methods)**

Hiện tại hệ thống chỉ có `RecursiveCharacterTextSplitter`. Cần xây dựng module chunking độc lập và linh hoạt:

1. **Thuật toán Semantic Chunking:**
   * Tách văn bản thành các câu đơn vị dựa trên ranh giới câu (regex / nltk).
   * Tạo vector embedding cho từng câu sử dụng `Multilingual-E5-large` (từ `src.db.store`).
   * Tính khoảng cách Cosine Distance $d_i = 1 - \text{sim}(e_i, e_{i+1})$ giữa các câu liên tiếp.
   * Xác định điểm cắt phân đoạn (breakpoints) theo 3 phương pháp ngưỡng:
     * **`percentile`**: Điểm cắt là các khoảng cách vượt phân vị thứ $P$ (mặc định $90$–$95\%$).
     * **`standard_deviation` (std_dev)**: Ngưỡng $T = \mu + k \cdot \sigma$ (mặc định $k=1.0$ hoặc $1.5$).
     * **`interquartile` (IQR)**: Ngưỡng $T = Q_3 + 1.5 \cdot \text{IQR}$ (đã chứng minh hiệu quả cao nhất trên slides).
   * Gom các câu giữa các điểm cắt thành các chunk hoàn chỉnh, giữ lại metadata trang và số thứ tự.
2. **Bộ điều phối Chunking Dispatcher (`chunk_document`):**
   * Cho phép nhận tham số: `method="recursive"` (chunk_size, chunk_overlap) hoặc `method="semantic"` (breakpoint_threshold_type, buffer_size).

---

### 🔹 Giai đoạn 2: Hiện thực hóa Framework Đánh giá Ragas (`src/pipelines/evaluation.py`)
> Đề xuất từ **Ảnh 1 & Ảnh 5 (RAGAS: Retrieval-Augmented Generation Assessment)**

Cơ chế **LLM-as-a-judge** nhận 4 đầu vào để chấm điểm tự động:

1. **4 Đầu vào (Inputs):**
   * `question`: Câu hỏi từ bộ kiểm thử.
   * `generated_answer`: Câu trả lời do hệ thống RAG sinh ra.
   * `retrieved_contexts`: Danh sách các đoạn văn bản (chunks) được truy xuất từ Qdrant.
   * `reference_answer`: Đáp án chuẩn vàng (Ground Truth) do con người xây dựng.

2. **4 Chỉ số đầu ra (Output Metrics trong khoảng $[0, 1]$):**
   * **Tầng Truy xuất (Retrieval Metrics):**
     * `context_precision`: Đo lường tỷ lệ các chunk thực sự liên quan được xếp ở các thứ hạng đầu (đo năng lực của Reranker).
     * `context_recall`: Tỷ lệ các thông tin trong `reference_answer` được tìm thấy đầy đủ trong `retrieved_contexts`.
   * **Tầng Tạo sinh (Generation Metrics):**
     * `faithfulness`: Tỷ lệ các khẳng định trong `generated_answer` suy diễn trực tiếp được từ `retrieved_contexts` (chống ảo giác).
     * `answer_relevancy`: Đo độ tương đồng cosine ngữ nghĩa giữa câu hỏi gốc và câu hỏi sinh ngược từ câu trả lời.

3. **Cấu hình Judge Models:**
   * **LLM Judge:** Tích hợp `Gemini 2.5 Flash` thông qua `ChatGoogleGenerativeAI` với prompt kiểm tra định dạng JSON nghiêm ngặt và cơ chế exponential backoff.
   * **Embedding Judge:** Sử dụng `Multilingual-E5-large` để tính Answer Relevancy đồng bộ với hệ thống.

---

### 🔹 Giai đoạn 3: Xây dựng Bộ dữ liệu Đánh giá Chuẩn (`data/evaluation/testset_qa_gt.json`)

1. Xây dựng tập test set tối thiểu 10–20 mẫu câu hỏi trích từ tài liệu mẫu thực tế của dự án.
2. Cấu trúc mỗi mẫu:
   ```json
   {
     "id": "q1",
     "question": "Hệ thống RAG sử dụng những mô hình nào cho tầng truy xuất hai tầng?",
     "reference_answer": "Hệ thống sử dụng mô hình Bi-Encoder Multilingual-E5-large cho tầng truy xuất dày đặc ban đầu và mô hình Cross-Encoder BGE-Reranker-v2-m3 cho tầng tái xếp hạng chuyên sâu.",
     "target_document": "sample.pdf"
   }
   ```

---

### 🔹 Giai đoạn 4: Kịch bản Thực nghiệm 1 — So sánh Chunking Methods (`experiments/run_exp1_chunking.py`)
> Đề xuất từ **Ảnh 3 (Experimental Results: Chunking Methods)**

1. **Thực thi quét 7 cấu hình trên cùng một tài liệu:**
   * **Nhóm Recursive (4 cấu hình):**
     1. Chunk Size 1500 / Overlap 200
     2. Chunk Size 1000 / Overlap 150 *(Baseline đề xuất)*
     3. Chunk Size 800 / Overlap 100
     4. Chunk Size 500 / Overlap 50
   * **Nhóm Semantic (3 cấu hình):**
     5. Ngưỡng `interquartile`
     6. Ngưỡng `standard_deviation`
     7. Ngưỡng `percentile`
2. **Quy trình chạy:**
   * Với mỗi cấu hình: Index vào một collection độc lập trong Qdrant $\to$ Chạy RAG lấy câu trả lời và contexts $\to$ Đánh giá Ragas 4 chỉ số $\to$ Đếm số chunks sinh ra.
   * Ghi kết quả vào `experiments/results_exp1_chunking.json` và `csv`.

---

### 🔹 Giai đoạn 5: Kịch bản Thực nghiệm 2 — Ablation Study Reranker (`experiments/run_exp2_reranking.py`)
> Đề xuất từ **Ảnh 4 (Experimental Results: Reranking Cross-Encoder)**

1. **So sánh 3 chiến lược truy xuất:**
   * Chiến lược 1: **Baseline** — Recursive (1000/150), không dùng Reranker ($K=5$).
   * Chiến lược 2: **Semantic** — Semantic (Interquartile), không dùng Reranker ($K=5$).
   * Chiến lược 3: **Ours (Đề xuất)** — Recursive (1000/150) + BGE-Reranker-v2-m3 ($K_{\text{initial}}=15 \to K_{\text{rerank}}=5$).
2. **Tính mức chênh lệch ($\Delta$):**
   * Tự động tính độ tăng trưởng của Ours so với Baseline trên từng chỉ số: Context Precision (+$\Delta$), Context Recall (+$\Delta$), Answer Relevancy (+$\Delta$), Faithfulness (+$\Delta$).
   * Ghi kết quả vào `experiments/results_exp2_reranking.json` và `csv`.

---

### 🔹 Giai đoạn 6: Xuất biểu đồ trực quan tự động (`experiments/export_charts.py`)
> Tái tạo chính xác các dạng biểu đồ trong **Ảnh 3 & Ảnh 4**

* Sinh tự động 4 biểu đồ lưu vào thư mục `docs/report_latex/Figures/`:
  1. `evaluation_strategy.png` (Sơ đồ chiến lược đánh giá)
  2. `ragas_pipeline.png` (Sơ đồ Ragas pipeline)
  3. `chunking_barchart.png` (Grouped bar chart 5 cấu hình Recursive)
  4. `chunking_linetrend.png` (Line chart xu hướng thay đổi theo Chunk Size)
  5. `reranking_barchart.png` (Bar chart đối chiếu Baseline vs Semantic vs Reranker)

---

### 🔹 Giai đoạn 7: Tích hợp CLI và Viết Test Suite (`test_evaluation_pipeline.py`)

1. Bổ sung lệnh CLI Typer:
   ```bash
   python -m src.interfaces.cli evaluate --testset data/evaluation/testset_qa_gt.json --output results.json
   ```
2. Bộ test suite tự động kiểm thử tính đúng đắn của từng module:
   * Test tách Semantic Chunking với cả 3 threshold.
   * Test tính Ragas metrics với mock dữ liệu.
   * Test chạy trọn vẹn 1 câu hỏi qua toàn bộ pipeline thực nghiệm.

---

## ⏱️ Thứ tự triển khai dự kiến

| Bước | Module / Tệp tin | Nhiệm vụ chính |
|:---:|---|---|
| **1** | `src/core/chunking.py` | Cài đặt Semantic Chunking (IQR, Std, Percentile) & Recursive Splitter |
| **2** | `src/core/eval_schemas.py` | Định nghĩa schemas Pydantic cho test case, metrics và experiment results |
| **3** | `data/evaluation/testset_qa_gt.json` | Soạn thảo bộ test set câu hỏi và đáp án mẫu chuẩn |
| **4** | `src/pipelines/evaluation.py` | Hiện thực hóa pipeline chấm điểm Ragas (4 metrics) bằng LLM-as-a-judge |
| **5** | `experiments/run_exp1_chunking.py` | Kịch bản tự động chạy Thực nghiệm 1 trên 7 cấu hình |
| **6** | `experiments/run_exp2_reranking.py` | Kịch bản tự động chạy Thực nghiệm 2 (Ablation Reranker) |
| **7** | `experiments/export_charts.py` | Script xuất biểu đồ Matplotlib đồng bộ bảng màu báo cáo |
| **8** | `test_evaluation_pipeline.py` | Test suite chạy kiểm tra toàn bộ pipeline đánh giá |

---
> Nhấn **Proceed** bên dưới để bắt đầu thực hiện Bước 1 ngay lập tức.
