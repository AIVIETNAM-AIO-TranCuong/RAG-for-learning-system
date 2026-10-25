# 🧠 RAG Learning System — Intelligent Learning Assistant (NotebookLM Style)

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%2012.x-red.svg)](https://pytorch.org/)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-brightgreen.svg)](https://qdrant.tech/)
[![Google Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-orange.svg)](https://ai.google.dev/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Ragas Evaluation](https://img.shields.io/badge/Evaluation-Ragas%20Metrics-blueviolet.svg)](https://github.com/explodinggradients/ragas)
[![LaTeX Report](https://img.shields.io/badge/Thesis%20Report-LaTeX%2036%20Pages-success.svg)](docs/report_latex/report_draft.pdf)

Hệ thống **RAG (Retrieval-Augmented Generation)** chuyên sâu hỗ trợ học tập, ôn luyện và nghiên cứu thông minh từ kho tài liệu cá nhân đa định dạng. Được lấy cảm hứng từ Google NotebookLM, hệ thống cung cấp pipeline truy xuất kết hợp (**Hybrid Search Dense + Sparse BM25**), tái xếp hạng (**Cross-Encoder Reranker**), phân đoạn ngữ nghĩa tự thích ứng (**Semantic Chunking IQR**), trích dẫn nguồn minh bạch, tóm tắt tự động (**Map-Reduce**), tạo đề thi trắc nghiệm tương tác (**Interactive Quiz**) và thẻ ghi nhớ ôn tập (**Flashcards 3D**). 

Hệ thống đi kèm **Bộ thực nghiệm định lượng (Empirical Benchmark)** và **Báo cáo nghiên cứu khoa học LaTeX 36 trang** đánh giá trên tài liệu chuyên ngành phân tích dữ liệu ngân hàng 140 trang.

---

## 🌟 Các tính năng nổi bật

### 1. 📂 Nạp & Tiền xử lý đa định dạng tài liệu
- **Hỗ trợ 5 định dạng phổ biến:** Microsoft Word (`.docx`), PowerPoint (`.pptx`), Adobe PDF (`.pdf`), Markdown (`.md`) và Plain Text (`.txt`).
- **Chuẩn hóa & dọn rác văn bản:** Chuẩn hóa Unicode tiếng Việt (NFC), loại bỏ ký tự lạ, chuẩn hóa khoảng trắng với TensorFlow Text (`tf_text_utils`).
- **Bảo toàn ngữ cảnh cấu trúc:** Tự động trích xuất metadata theo từng trang (page) và từng slide kèm số thứ tự tài liệu.

### 2. 🧩 Chiến lược phân đoạn văn bản nâng cao (Chunking Strategies)
- **Recursive Character Chunking:** Phân đoạn đệ quy theo kích thước chunk và độ chồng lấp (overlap) với cấu hình cơ sở tối ưu $S=1000, O=150$.
- **Semantic Chunking:** Tách đoạn văn bản dựa trên khoảng cách ngữ nghĩa giữa các câu liên tiếp sử dụng Vector Embeddings:
  - Ngưỡng phân vị (Percentile Threshold)
  - Ngưỡng độ lệch chuẩn (Standard Deviation)
  - **Phương pháp đề xuất: Interquartile Range (IQR):** Tự động phát hiện điểm ngắt tự nhiên ($\text{Breakpoint} = Q_3 + 1.5 \times \text{IQR}$), loại bỏ hoàn toàn nhiễu từ các outlier phân mảnh mà không cần điều chỉnh siêu tham số thủ công.

### 3. 🔍 RAG Pipeline 2 tầng & Tìm kiếm kết hợp (Hybrid Search)
- **Dense Retrieval:** Sử dụng mô hình Embedding đa ngôn ngữ hàng đầu `intfloat/multilingual-e5-large` chạy tối ưu hóa trên GPU (CUDA), lưu trữ Persistent trên **Qdrant Vector Database**.
- **Sparse Retrieval (BM25 Okapi):** Truy xuất theo từ khóa chính xác, đặc biệt hiệu quả với các thuật ngữ kỹ thuật, viết tắt tài chính (CASA, NII, DAX, PostgreSQL, B6_LeakageD30).
- **Reciprocal Rank Fusion (RRF):** Kết hợp và chuẩn hóa xếp hạng giữa Dense và Sparse scores ($k=60$).
- **Tái xếp hạng (Cross-Encoder Reranking):** Ứng dụng mô hình `BAAI/bge-reranker-v2-m3` để chấm điểm tương quan ngữ nghĩa sâu câu hỏi - ngữ cảnh, chọn lọc ra Top-K tinh hoa nhất (**"Golden Fusion"**).
- **Trích dẫn nguồn chuẩn xác (Citations):** Gắn trích dẫn nguồn theo từng câu trả lời (`[S1]`, `[S2]...`) kèm thông tin trang, slide, tiêu đề và trích đoạn tham chiếu.

### 4. 📝 Tính năng học tập thông minh (NotebookLM Style)
- **Tóm tắt tài liệu tự động (Summarization):**
  - **Single-pass:** Tóm tắt nhanh đối với tài liệu ngắn.
  - **Map-Reduce:** Xử lý tài liệu dài hàng chục, hàng trăm chunks mà không bị tràn context window của LLM thông qua các template Jinja2 chuyên biệt.
- **Trắc nghiệm tương tác (MCQ Quiz):** Tự động tạo bộ câu hỏi trắc nghiệm 4 lựa chọn (A, B, C, D) theo chủ đề hoặc toàn bộ tài liệu, làm bài và chấm điểm giải thích chi tiết trực tiếp trên UI.
- **Thẻ ghi nhớ Flashcards 3D:** Ôn tập kiến thức cốt lõi với hiệu ứng lật thẻ 3D, hỗ trợ click chuột hoặc phím tắt thuận tiện: Mũi tên trái/phải (`←` / `→`) để chuyển thẻ và phím `Space` để lật.

### 5. 📊 Khung đánh giá tự động (Evaluation Framework)
- Tích hợp chuẩn đánh giá **Ragas (LLM-as-a-Judge)** tính toán 4 chỉ số chất lượng:
  - **Context Precision:** Tỷ lệ thông tin phù hợp được ưu tiên xếp ở thứ hạng đầu.
  - **Context Recall:** Độ bao phủ đầy đủ của thông tin truy xuất so với Ground Truth.
  - **Faithfulness:** Độ trung thực của câu trả lời sinh ra (không bị ảo giác/hallucination).
  - **Answer Relevancy:** Mức độ phù hợp và trực diện của câu trả lời đối với câu hỏi.
- Hỗ trợ các chỉ số truy xuất cổ điển: **Hit Rate@K** và **Mean Reciprocal Rank (MRR@K)**.

### 6. 🖥️ 3 Giao diện sử dụng toàn diện
- **Streamlit Web UI:** Giao diện trực quan, hiện đại, hỗ trợ dark mode, quản lý tài liệu, chat tương tác, làm quiz và luyện flashcards.
- **FastAPI REST Server:** Backend chuẩn RESTful, tích hợp tài liệu tương tác tự động Swagger UI / OpenAPI.
- **Typer CLI:** Công cụ dòng lệnh mạnh mẽ, hiển thị màu sắc và bảng biểu định dạng Rich.

---

## 📊 Kết quả Thực nghiệm & Benchmark (Empirical Findings)

Chuỗi 5 bài thực nghiệm định lượng được kiểm thử độc lập trên tài liệu nghiệp vụ tài chính ngân hàng thực tế 140 trang (`[Project]-Finance-Analytics-eDA.pdf`, 387 chunks, 12 bộ câu hỏi kiểm thử chuyên sâu):

### 1. So sánh tổng hợp các chiến lược cốt lõi (Exp 5 — Golden Fusion)

| Chiến lược (Strategy) | Context Precision | Context Recall | Faithfulness | Answer Relevancy | Đánh giá & Điểm đột phá |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Baseline (Recursive 1000/150)** | 0.6667 | 0.5833 | **1.0000** | 0.8691 | Cấu hình baseline ổn định nhất |
| **2. Semantic Chunking (IQR)** | 0.7889 | **1.0000** | **1.0000** | 0.9100 | Ngắt đoạn ngữ nghĩa tự thích ứng (+41.67% Recall) |
| **3. Recursive + Cross-Encoder Reranker** | 0.9556 | 0.9167 | **1.0000** | 0.9085 | Tái xếp hạng lọc nhiễu vượt trội (+28.89% Precision) |
| **4. Golden Fusion (SOTA Đề xuất)** | **0.9722** | **1.0000** | **1.0000** | **0.9350** | **Hiệu năng toàn diện cao nhất (+30.55% Prec, +41.67% Rec)** |

### 2. Điểm nổi bật qua các thực nghiệm thành phần:
- **Thực nghiệm 1 (Chunking Strategy):** Cấu hình `Recursive S:1000 / O:150` đạt đỉnh đường cong Answer Relevancy ($0.8691$) so với các kích thước $500$ ($0.7966$), $800$ ($0.8314$) và $1500$ ($0.8150$). Phương pháp `Semantic Chunking (IQR)` nâng Context Precision từ $0.6667$ lên $0.7500$ và Context Recall lên $0.6389$.
- **Thực nghiệm 2 (Cross-Encoder Reranker):** Thêm tầng Reranker `bge-reranker-v2-m3` giúp tăng Context Precision từ $0.6667$ lên $0.8889$ (+22.22%) và Context Recall từ $0.5833$ lên $0.9167$ (+33.34%).
- **Thực nghiệm 3 (Embedding Models):** `BAAI/BGE-M3` có độ trễ suy luận siêu nhanh ($69.23$ ms, giảm **42.8%** so với $120.95$ ms của `multilingual-e5-large`), trong khi `multilingual-e5-large` đạt Context Recall cao hơn ($0.9167$ so với $0.8611$).
- **Thực nghiệm 4 (Hybrid Search):** Kết hợp Dense + BM25 qua thuật toán RRF giúp truy xuất hoàn hảo các thực thể mã nguồn và công thức nghiệp vụ (như `B6_LeakageD30`, `DAX`, `CASA`), đưa Context Precision đạt $0.9722$ và Context Recall đạt $1.0000$.

> Toàn bộ biểu đồ trực quan hóa độ phân giải cao được xuất tự động tại thư mục [experiments/output_charts/](experiments/output_charts/) và [docs/report_latex/Figures/](docs/report_latex/Figures/).

---

## 📑 Báo cáo Nghiên cứu Khoa học (Thesis Report)

Dự án bao gồm tài liệu báo cáo luận văn hoàn chỉnh định dạng LaTeX chuẩn khoa học:
- **Tập tin PDF hoàn chỉnh:** [docs/report_latex/report_draft.pdf](docs/report_latex/report_draft.pdf) (36 trang, biên dịch trực tiếp từ [report_draft.tex](docs/report_latex/report_draft.tex)).
- **Cấu trúc 5 chương hoàn chỉnh:**
  - *Chương 1: Giới thiệu và Tổng quan Đề tài*
  - *Chương 2: Cơ sở Lý thuyết (RAG, Embeddings, Cross-Encoder, Ragas)*
  - *Chương 3: Thiết kế Kiến trúc Hệ thống (Two-Stage Pipeline, Map-Reduce, Hybrid Search)*
  - *Chương 4: Thực nghiệm và Đánh giá (Chuỗi 5 thực nghiệm phân tích chi tiết)*
  - *Chương 5: Kết luận và Hướng phát triển Tương lai*

---

## ⚙️ Cài đặt môi trường

### Bước 1: Tạo môi trường Conda với Python 3.11
```bash
conda create -n rag_learning python=3.11 -y
conda activate rag_learning
```

### Bước 2: Cài đặt PyTorch với hỗ trợ GPU (CUDA 12.x)
```bash
# Kiểm tra GPU trên hệ thống
nvidia-smi

# Cài đặt PyTorch tương thích CUDA 12.4
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

### Bước 3: Cài đặt các thư viện phụ thuộc
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Bước 4: Cấu hình biến môi trường (`.env`)
Tạo hoặc chỉnh sửa file `.env` tại thư mục gốc của dự án:
```env
# Embedding Model & Thiết bị phần cứng
EMBEDDING_MODEL=intfloat/multilingual-e5-large
EMBEDDING_DEVICE=cuda

# Cơ sở dữ liệu Vector Qdrant
QDRANT_PATH=./qdrant_storage
COLLECTION_NAME=documents

# Cấu hình Chunking chuẩn
CHUNK_SIZE=1000
CHUNK_OVERLAP=150

# Cấu hình Mô hình Ngôn ngữ Lớn (LLM)
LLM_BACKEND=gemini
GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
GEMINI_MODEL=gemini-flash-latest

# Cấu hình RAG & Tái xếp hạng (Reranker)
TOP_K=5
INITIAL_K=15
RERANK_K=5
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_DEVICE=cuda

# Cấu hình Tính năng Học tập (Learning Features)
SUMMARIZE_BATCH_SIZE=10
N_QUIZ_QUESTIONS=8
N_FLASHCARDS=15
```

---

## 🚀 Hướng dẫn khởi chạy ứng dụng

### 1. Khởi chạy Giao diện Web (Khuyến nghị)

Mở **2 cửa sổ Terminal** trong môi trường `rag_learning`:

**Terminal 1 — Khởi động Backend API Server (FastAPI):**
```bash
conda activate rag_learning
cd d:\RAG_Learning_System
uvicorn src.interfaces.api:app --reload --port 8000
```
- API Server: `http://127.0.0.1:8000`
- Swagger UI Docs tương tác: `http://127.0.0.1:8000/docs`

**Terminal 2 — Khởi động Frontend Web UI (Streamlit):**
```bash
conda activate rag_learning
cd d:\RAG_Learning_System
streamlit run src/interfaces/ui.py
```
- Ứng dụng sẽ tự động mở trên trình duyệt tại: `http://localhost:8501`

---

### 2. Sử dụng công cụ dòng lệnh (CLI)

Hệ thống cung cấp đầy đủ các tác vụ qua dòng lệnh `src.interfaces.cli`:

```bash
# Xem hướng dẫn các câu lệnh
python -m src.interfaces.cli --help

# Nạp tài liệu vào cơ sở dữ liệu (PDF, Word, Slide, Markdown, TXT)
python -m src.interfaces.cli index "data/pdfs/[Project]-Finance-Analytics-eDA.pdf"

# Liệt kê các tài liệu đã nạp
python -m src.interfaces.cli list-docs

# Hỏi đáp từ tài liệu kèm trích dẫn nguồn
python -m src.interfaces.cli ask "Thu nhập lãi thuần của ngân hàng được xác định như thế nào?"

# Tóm tắt tài liệu (tự động phân nhánh Single-pass / Map-Reduce)
python -m src.interfaces.cli summarize --doc-id <DOCUMENT_ID>

# Tạo câu hỏi trắc nghiệm tương tác trực tiếp trên terminal
python -m src.interfaces.cli quiz --doc-id <DOCUMENT_ID> --interactive

# Tạo bộ thẻ Flashcards ôn tập
python -m src.interfaces.cli flashcards --doc-id <DOCUMENT_ID>
```

---

## 🧪 Kiểm thử tự động (Unit Tests)

Bộ kiểm thử tự động được tổ chức trong thư mục `tests/`:

```bash
# Chạy toàn bộ test suite bằng unittest runner chuẩn
python -m unittest discover tests

# Hoặc chạy kiểm thử độc lập cho từng module cụ thể:
python -m unittest tests/test_data_preprocessing.py   # Kiểm thử tiền xử lý & chuẩn hóa Unicode
python -m unittest tests/test_multiformat_indexing.py  # Kiểm thử nạp 5 định dạng file & Vector DB
python -m unittest tests/test_rag_pipeline.py          # Kiểm thử truy xuất RAG & Cross-Encoder
python -m unittest tests/test_learning_features.py     # Kiểm thử Tóm tắt, Quiz & Flashcards
python -m unittest tests/test_interfaces.py            # Kiểm thử REST API & CLI
python -m unittest tests/test_evaluation_pipeline.py    # Kiểm thử Semantic Chunking & Ragas metrics
```

---

## 📈 Chạy Thực nghiệm & Xuất biểu đồ (Experiments)

Dự án tích hợp đầy đủ các script thực nghiệm định lượng phục vụ nghiên cứu và đánh giá:

```bash
# Thực nghiệm 1: So sánh các chiến lược Chunking (Recursive vs Semantic IQR/Percentile/Std)
python experiments/run_exp1_chunking.py

# Thực nghiệm 2: Đánh giá tác động của Cross-Encoder Reranker (Two-stage Retrieval)
python experiments/run_exp2_reranking.py

# Thực nghiệm 3: So sánh hiệu năng mô hình Embedding (Multilingual-E5 vs BGE-M3)
python experiments/run_exp3_embedding_models.py

# Thực nghiệm 4: Đánh giá Hybrid Search (Dense + Sparse BM25 RRF)
python experiments/run_exp4_hybrid_search.py

# Thực nghiệm 5: Đánh giá Golden Fusion (Hội tụ toàn diện các kỹ thuật SOTA)
python experiments/run_exp3_golden_fusion.py

# Tự động kết xuất toàn bộ biểu đồ trực quan hóa ra file PNG vào experiments/ và LaTeX/
python experiments/export_charts.py
```

---

## 📁 Cấu trúc thư mục dự án

```text
RAG_Learning_System/
├── data/                             # Dữ liệu tài liệu mẫu & tập kiểm thử Ground Truth
│   ├── clean_corpus/                 # Văn bản chuẩn hóa Markdown & PDF
│   ├── evaluation/                   # Bộ câu hỏi kiểm thử chuẩn (testset_qa_gt.json)
│   └── pdfs/                         # Tài liệu PDF chuyên ngành tài chính
├── docs/                             # Báo cáo và tài liệu dự án
│   ├── [Description]-Building-Simple-NotebookLM.pdf  # Đề bài đặc tả dự án
│   └── report_latex/                 # Toàn bộ mã nguồn LaTeX báo cáo luận văn 36 trang
│       ├── Figures/                  # Sơ đồ kiến trúc & Biểu đồ thực nghiệm định lượng
│       ├── references.bib            # Danh mục tài liệu tham khảo chuẩn BibTeX
│       ├── report_draft.tex          # Mã nguồn LaTeX bài báo cáo
│       └── report_draft.pdf          # File PDF xuất bản chính thức
├── experiments/                      # Bộ mã nguồn thực nghiệm & kết quả Benchmark
│   ├── output_charts/                # Biểu đồ kết quả trực quan hóa (.png)
│   ├── results_*.json                # Dữ liệu kết quả định lượng chi tiết (Exp 1 - Exp 5)
│   ├── run_exp1_chunking.py          # Benchmark chiến lược Chunking
│   ├── run_exp2_reranking.py         # Benchmark tác động của Reranker
│   ├── run_exp3_embedding_models.py  # Benchmark mô hình Embeddings
│   ├── run_exp3_golden_fusion.py     # Benchmark Golden Fusion đề xuất
│   ├── run_exp4_hybrid_search.py     # Benchmark Hybrid Search (BM25 + Dense)
│   ├── prepare_clean_corpus.py       # Tiền xử lý & số hóa bộ ngữ liệu chuẩn
│   ├── setup_finance_experiment.py   # Thiết lập testset chuẩn ngân hàng 140 trang
│   └── export_charts.py              # Script vẽ biểu đồ phân tích chuẩn khoa học
├── qdrant_storage/                   # Cơ sở dữ liệu Vector Qdrant lưu trữ cục bộ
├── src/                              # Mã nguồn chính của hệ thống
│   ├── core/
│   │   ├── chunking.py               # Phân đoạn văn bản (Recursive & Semantic IQR Chunking)
│   │   ├── config.py                 # Quản lý cấu hình tập trung qua Pydantic Settings
│   │   ├── eval_schemas.py           # Schema dữ liệu cho bộ chỉ số đánh giá
│   │   ├── hybrid.py                 # Hybrid Search (BM25 + Dense + RRF)
│   │   └── schemas.py                # Pydantic Models cho dữ liệu và API
│   ├── db/
│   │   └── store.py                  # Quản lý kết nối Qdrant DB và HuggingFace Embeddings
│   ├── interfaces/
│   │   ├── api.py                    # REST API Server (FastAPI)
│   │   ├── cli.py                    # Dòng lệnh tương tác CLI (Typer & Rich)
│   │   ├── styles.py                 # Custom CSS cho giao diện Streamlit
│   │   └── ui.py                     # Giao diện Web chính (Streamlit UI)
│   ├── pipelines/
│   │   ├── evaluation.py             # Pipeline đánh giá tự động (Ragas LLM-as-a-Judge)
│   │   ├── indexing.py               # Bộ nạp 5 định dạng file & Vectorization
│   │   ├── learning.py               # Pipeline Tóm tắt (Map-Reduce), Quiz & Flashcards
│   │   └── rag.py                    # Pipeline RAG (Retrieve, Rerank, Citation Q&A)
│   ├── prompts/                      # Quản lý Jinja2 Prompt Templates
│   │   ├── flashcard.j2
│   │   ├── qa.j2
│   │   ├── quiz.j2
│   │   ├── summary_map.j2
│   │   ├── summary_reduce.j2
│   │   └── summary_single.j2
│   ├── utils/
│   │   └── tf_text_utils.py          # Tiền xử lý, chuẩn hóa Unicode & dọn rác văn bản
│   └── llm.py                        # Wrapper tích hợp Google Gemini API
├── tests/                            # Bộ kiểm thử tự động (Unit Tests Suite)
│   ├── test_data_preprocessing.py    # Kiểm thử chuẩn hóa Unicode & Regex làm sạch
│   ├── test_evaluation_pipeline.py   # Kiểm thử Semantic Chunking & Ragas metrics
│   ├── test_interfaces.py            # Kiểm thử API & CLI
│   ├── test_learning_features.py     # Kiểm thử Tóm tắt, Quiz & Flashcards
│   ├── test_multiformat_indexing.py  # Kiểm thử nạp 5 định dạng tài liệu
│   └── test_rag_pipeline.py          # Kiểm thử RAG Core & Tái xếp hạng
├── .env                              # Biến môi trường & cấu hình API Key
├── .gitignore                        # Cấu hình bỏ qua file trong Git
└── requirements.txt                  # Danh mục các thư viện Python
```

---

## 📜 Giấy phép & Tác giả
- Phát triển bởi sinh viên / nhóm nghiên cứu trong khuôn khổ đồ án môn học AI VIETNAM.
- Dự án mã nguồn mở phục vụ mục đích học tập và nghiên cứu.