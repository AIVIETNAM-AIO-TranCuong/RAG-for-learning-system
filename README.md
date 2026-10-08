# 🧠 RAG Learning System — Intelligent Learning Assistant (NotebookLM Style)

Hệ thống RAG (Retrieval-Augmented Generation) hỗ trợ học tập và nghiên cứu thông minh từ tài liệu cá nhân đa định dạng. Cung cấp tính năng hỏi đáp kèm trích dẫn nguồn, tóm tắt tự động (Map-Reduce), sinh đề thi trắc nghiệm tương tác và thẻ ghi nhớ ôn tập Flashcards.

---

## 🌟 Các tính năng nổi bật

- **📂 Đa định dạng tài liệu:** Hỗ trợ nạp và phân tích **PDF (`.pdf`)**, **Microsoft Word (`.docx`)**, **PowerPoint (`.pptx`)**, **Markdown (`.md`)** và **Plain Text (`.txt`)**.
- **🔍 RAG Pipeline chuyên sâu:**
  - Tiền xử lý và chuẩn hóa Unicode với TensorFlow Text (`tf_text_utils`).
  - Embedding tiếng Việt & đa ngôn ngữ: `intfloat/multilingual-e5-large` trên GPU (CUDA).
  - Cơ sở dữ liệu Vector cục bộ: **Qdrant** lưu trữ persistent.
  - Tái xếp hạng kết quả (Reranking): **CrossEncoder** `BAAI/bge-reranker-v2-m3` tối ưu độ chính xác ngữ nghĩa.
  - Trích dẫn nguồn chuẩn xác theo từng câu trả lời: `[S1]`, `[S2]...` kèm số trang/slide và trích đoạn tham chiếu.
- **📝 Tóm tắt tài liệu thông minh (Summarization):** Tự động chuyển đổi giữa **Single-pass** (với tài liệu ngắn) và **Map-Reduce** (với tài liệu dài hàng chục chunks).
- **🎯 Trắc nghiệm tương tác (MCQ Quiz):** Tự sinh bộ câu hỏi 4 lựa chọn (A, B, C, D) theo chủ đề hoặc toàn bộ tài liệu, làm bài và chấm điểm trực tiếp trên giao diện.
- **🗂️ Thẻ ghi nhớ (Flashcards):** Ôn tập kiến thức cốt lõi, hỗ trợ click trực tiếp vào thẻ để lật mặt sau, phím tắt mũi tên trái/phải (`←` / `→`) để chuyển thẻ và phím `Space` để lật.
- **🖥️ 3 Giao diện sử dụng toàn diện:**
  - **Streamlit Web UI:** Giao diện trực quan, thân thiện, hiện đại.
  - **FastAPI REST API:** Hệ thống backend chuẩn RESTful, tích hợp tài liệu tương tác Swagger UI.
  - **Typer CLI:** Công cụ dòng lệnh mạnh mẽ với giao diện Rich màu sắc.

---

## ⚙️ Cài đặt môi trường (Conda)

### Bước 1: Tạo môi trường Conda với Python 3.11
```bash
conda create -n rag_learning python=3.11 -y
conda activate rag_learning
```

### Bước 2: Cài đặt PyTorch với hỗ trợ CUDA
```bash
# Kiểm tra phiên bản CUDA trên máy
nvidia-smi

# Cài đặt PyTorch tương thích CUDA 12.x
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
# Embedding Model & Thiết bị
EMBEDDING_MODEL=intfloat/multilingual-e5-large
EMBEDDING_DEVICE=cuda

# Qdrant Vector Storage
QDRANT_PATH=./qdrant_storage
COLLECTION_NAME=documents

# Chunking Parameters
CHUNK_SIZE=1000
CHUNK_OVERLAP=150

# LLM Backend
LLM_BACKEND=gemini
GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
GEMINI_MODEL=gemini-flash-latest

# RAG & Reranker
TOP_K=5
INITIAL_K=15
RERANK_K=5
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_DEVICE=cuda

# Learning Features
SUMMARIZE_BATCH_SIZE=10
N_QUIZ_QUESTIONS=8
N_FLASHCARDS=15
```

---

## 🚀 Hướng dẫn khởi chạy ứng dụng

### 1. Chạy Web UI (Khuyến nghị)

Mở **2 cửa sổ Terminal** trong môi trường `rag_learning`:

**Terminal 1 — Khởi động API Server Backend:**
```bash
conda activate rag_learning
cd d:\RAG_Learning_System
uvicorn src.interfaces.api:app --reload --port 8000
```
* API Server: `http://127.0.0.1:8000`
* Swagger UI Docs: `http://127.0.0.1:8000/docs`

**Terminal 2 — Khởi động Giao diện Web Streamlit:**
```bash
conda activate rag_learning
cd d:\RAG_Learning_System
streamlit run src/interfaces/ui.py
```
* Trình duyệt sẽ tự động mở tại: `http://localhost:8501`

---

### 2. Sử dụng công cụ dòng lệnh (CLI)

Hệ thống cung cấp CLI thông qua module `src.interfaces.cli`:

```bash
# Xem trợ giúp các lệnh
python -m src.interfaces.cli --help

# Nạp tài liệu mới (hỗ trợ .pdf, .docx, .pptx, .txt, .md)
python -m src.interfaces.cli index "data/pdfs/sample.pdf"
python -m src.interfaces.cli index "data/lecture_slides.pptx"

# Liệt kê tất cả tài liệu đã nạp vào Vector DB
python -m src.interfaces.cli list-docs

# Hỏi đáp từ tài liệu (RAG Q&A kèm trích dẫn)
python -m src.interfaces.cli ask "Fine-tuning LLM là gì?"

# Tóm tắt tài liệu
python -m src.interfaces.cli summarize --doc-id <DOCUMENT_ID>

# Tạo câu hỏi trắc nghiệm tương tác
python -m src.interfaces.cli quiz --doc-id <DOCUMENT_ID> --interactive

# Tạo bộ thẻ Flashcards ôn tập
python -m src.interfaces.cli flashcards --doc-id <DOCUMENT_ID>
```

---

## 📁 Cấu trúc thư mục dự án

```text
RAG_Learning_System/
├── data/                   # Thư mục lưu trữ tài liệu mẫu đầu vào
├── qdrant_storage/               # Thư mục lưu trữ cơ sở dữ liệu Vector Qdrant
├── src/
│   ├── core/
│   │   ├── config.py             # Quản lý cấu hình tập trung qua Pydantic Settings
│   │   └── schemas.py            # Khai báo cấu trúc dữ liệu (Pydantic Models)
│   ├── db/
│   │   └── store.py              # Kết nối Qdrant DB và HuggingFace Embeddings
│   ├── interfaces/
│   │   ├── api.py                # REST API Server (FastAPI)
│   │   ├── cli.py                # Dòng lệnh tương tác CLI (Typer & Rich)
│   │   ├── styles.py             # CSS giao diện tùy biến cho Streamlit
│   │   └── ui.py                 # Giao diện Web chính (Streamlit UI)
│   ├── pipelines/
│   │   ├── indexing.py           # Bộ nạp đa định dạng (PDF, DOCX, PPTX, TXT, MD) & Vectorization
│   │   ├── learning.py           # Pipeline Tóm tắt (Map-Reduce), Quiz & Flashcards
│   │   └── rag.py                # Pipeline RAG (Retrieve, CrossEncoder Rerank & Citation Q&A)
│   ├── prompts/                  # Quản lý Prompt templates qua Jinja2
│   │   ├── flashcard.j2
│   │   ├── qa.j2
│   │   ├── quiz.j2
│   │   ├── summary_map.j2
│   │   ├── summary_reduce.j2
│   │   └── summary_single.j2
│   ├── utils/
│   │   └── tf_text_utils.py      # Tiền xử lý, chuẩn hóa Unicode & dọn rác văn bản
│   └── llm.py                    # Wrapper tích hợp Google Gemini API
├── .env                          # Biến môi trường & cấu hình API Key
├── requirements.txt              # Danh sách thư viện Python
├── test_multiformat_indexing.py  # Test kiểm thử nạp đa định dạng (.pdf, .docx, .pptx, .md, .txt)
├── test_rag_pipeline.py          # Test kiểm thử RAG Core & Reranker
└── test_learning_features.py     # Test kiểm thử tính năng Tóm tắt, Quiz & Flashcards
```

---

## 🧪 Kiểm thử tự động (Unit Tests)

Chạy các kịch bản kiểm tra để đảm bảo toàn bộ hệ thống hoạt động ổn định:

```bash
# Kiểm tra nạp đa định dạng (Word, PowerPoint, PDF, Markdown, Text)
python test_multiformat_indexing.py

# Kiểm tra RAG Pipeline & Reranking
python test_rag_pipeline.py

# Kiểm tra tính năng Tóm tắt, Quiz & Flashcards
python test_learning_features.py
```