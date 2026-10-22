# Kế hoạch giai đoạn tiếp theo — Báo cáo RAG Learning System
*(Cập nhật với số liệu thực nghiệm thực tế từ slides)*

> **Trạng thái hiện tại (đã hoàn thành):**
> - ✅ Section I — Giới thiệu
> - ✅ Section II — Nghiên cứu liên quan (14 citations, 7 subsections, Table 1, 4 công thức toán học)
> - ✅ Section III — Kiến trúc hệ thống (8 Hình, 2 Bảng, 5 subsections đầy đủ)
> - ✅ Figures/: 8 file `.png` đã tích hợp
> - ✅ references.bib: 14 citation chuẩn quốc tế

---

## GIAI ĐOẠN 4 — Section IV: Thực nghiệm và Đánh giá

### 4.0 Chiến lược đánh giá và Phạm vi thực nghiệm

> **⚠️ ĐIỂM QUAN TRỌNG (từ slides):** Dự án này **chỉ tập trung vào giai đoạn 1** — Tối ưu hóa tầng Truy xuất (Retriever). Giai đoạn 2 (Tối ưu hóa tầng Tạo sinh) nằm ngoài phạm vi của đồ án.

**Sơ đồ chiến lược đánh giá (Hình 9 — cần tạo):**

```
                ┌─────────────────────────────┐      ┌──────────────────────────────────┐
                │   1. Optimize Retriever     │ ───▶ │   2. Optimize Generation         │
                │   (Recall & Precision)      │      │   (Faithfulness & Answer Rel.)   │
                └─────────────────────────────┘      └──────────────────────────────────┘
                           ▲
          ┌────────────────────────────────────┐
          │  This project only focuses on      │
          │  this phase                        │
          └────────────────────────────────────┘

  Experiment 1                    Experiment 2
  ┌──────────────┐                ┌──────────────┐
  │ Chunking     │──▶ Recursive   │  Reranking   │──▶ Cross-Encoder
  │ Methods      │──▶ Semantic    └──────────────┘
  └──────────────┘
```

**Thiết kế thực nghiệm tổng quan:**
- **Thực nghiệm 1:** So sánh 2 phương pháp chunking (Recursive vs Semantic) với nhiều cấu hình tham số
- **Thực nghiệm 2:** Ablation study thêm Cross-Encoder Reranker vào baseline tốt nhất từ Thực nghiệm 1

**Framework đánh giá — RAGAS (LLM-as-a-Judge):**

Pipeline RAGAS nhận 4 đầu vào: `Questions`, `Generated Answer`, `Retrieved Contexts`, `Reference Answer` — kết hợp Embedding Model + LLM để tính 4 chỉ số đầu ra:

| Nhóm | Chỉ số | Cơ chế tính | Phạm vi |
|------|--------|------------|---------|
| **Generation** | Faithfulness | Tỉ lệ mệnh đề suy diễn được từ Context | [0, 1] |
| **Generation** | Answer Relevancy | Cosine sim giữa câu hỏi gốc và câu hỏi sinh ngược | [0, 1] |
| **Retrieval** | Context Precision | Tỉ lệ chunk liên quan được xếp hạng ưu tiên | [0, 1] |
| **Retrieval** | Context Recall | Tỉ lệ dữ kiện trong Reference Answer có căn cứ trong Context | [0, 1] |

---

### 4.1 Thiết kế bộ câu hỏi đánh giá (Evaluation Dataset)

**Bộ test set thực tế đã được xây dựng và chạy** — từ đó có số liệu RAGAS thực bên dưới.
Mỗi mẫu đánh giá gồm: `question`, `ground_truth_answer`, `retrieved_contexts` (từ Qdrant).

---

### 4.2 Thực nghiệm 1 — So sánh phương pháp Chunking

**Giả thuyết nghiên cứu:** Chiến lược phân đoạn văn bản (Recursive vs Semantic) và các tham số kích thước chunk ảnh hưởng trực tiếp đến chất lượng truy xuất và độ trung thực tạo sinh.

#### Thiết kế thực nghiệm

**Nhóm A — Recursive Chunking (4 cấu hình):**

| Cấu hình | Chunk Size / Overlap | Số chunks | Faithfulness | Answer Rel. | Context Prec. | Context Recall |
|----------|---------------------|-----------|-------------|------------|--------------|---------------|
| Recursive S:1500/O:200 | 1500 / 200 | 246 | 0.8811 | 0.7590 | 0.8149 | **0.92** |
| **Recursive S:1000/O:150** ⭐ | **1000 / 150** | **340** | **0.8632** | **0.7292** | **0.8618** | **0.92** |
| Recursive S:800/O:100 | 800 / 100 | 401 | 0.8112 | 0.7526 | 0.8572 | 0.92 |
| Recursive S:500/O:50 | 500 / 50 | 600 | 0.7867 | 0.7509 | 0.8151 | 0.92 |

**Nhóm B — Semantic Chunking (3 cấu hình):**

| Cấu hình | Phương pháp ngưỡng | Số chunks | Faithfulness | Answer Rel. | Context Prec. | Context Recall |
|----------|--------------------|-----------|-------------|------------|--------------|---------------|
| **Semantic Interquartile** ⭐ | Interquartile | **564** | **0.8867** | **0.7959** | **0.8117** | **0.92** |
| Semantic Std_dev | Standard Deviation | 291 | 0.8693 | 0.7578 | 0.8040 | 0.90 |
| Semantic Percentile | Percentile | 465 | 0.8573 | 0.7344 | 0.8225 | 0.90 |

> **⭐ Baseline cho Thực nghiệm 2:** Recursive (1000/150) được chọn làm baseline vì đạt điểm cân bằng tốt nhất giữa Context Precision (0.8618) và số lượng chunks quản lý được (340).

**Bảng kết quả tổng hợp Thực nghiệm 1 (Table 3 trong LaTeX — SỐ LIỆU THỰC):**

| Phương pháp | Config | Chunks | Faithfulness ↑ | Answer Relev. ↑ | Context Prec. ↑ | Context Recall ↑ |
|-------------|--------|--------|---------------|----------------|----------------|-----------------|
| Recursive | 1500/200 | 246 | 0.8811 | 0.7590 | 0.8149 | 0.92 |
| Recursive | **1000/150** | 340 | 0.8632 | 0.7292 | **0.8618** | **0.92** |
| Recursive | 800/100 | 401 | 0.8112 | 0.7526 | 0.8572 | 0.92 |
| Recursive | 500/50 | 600 | 0.7867 | 0.7509 | 0.8151 | 0.92 |
| Semantic | **Interquartile** | 564 | **0.8867** | **0.7959** | 0.8117 | 0.92 |
| Semantic | Std_dev | 291 | 0.8693 | 0.7578 | 0.8040 | 0.90 |
| Semantic | Percentile | 465 | 0.8573 | 0.7344 | 0.8225 | 0.90 |

**Phân tích kết quả cần viết:**
- Recursive 1000/150 nổi bật với Context Precision cao nhất trong nhóm Recursive (0.8618)
- Context Recall = 0.92 đồng đều cho hầu hết Recursive configs → Bi-Encoder đã bao phủ tốt
- Semantic Interquartile đạt Faithfulness cao nhất (0.8867) và Answer Relevancy tốt nhất (0.7959) — gợi ý semantic chunking giữ được ngữ nghĩa tốt hơn trong từng chunk
- Chunk nhỏ (500/50 → 600 chunks) cho Faithfulness thấp nhất (0.7867) → over-fragmentation gây mất ngữ cảnh

**Hình cần tạo:**
- **Hình 10:** Bar chart "So sánh các chỉ số RAG theo cấu hình Chunk Size và Overlap" (5 cột grouped bar: S:1500/O:200, S:1200/O:200, S:1000/O:150, S:800/O:100, S:500/O:50) — **dùng lại chart từ slides**
- **Hình 11:** Line chart "Xu hướng các chỉ số khi thay đổi Chunk Size" (trục X: Chunk Size 600→1400; 4 đường: Faithfulness, Answer Relevance, Context_precision, Context_recall) — **dùng lại chart từ slides**

---

### 4.3 Thực nghiệm 2 — Tái xếp hạng bằng Cross-Encoder (Reranking)

**Giả thuyết nghiên cứu:** Tích hợp Cross-Encoder BGE-Reranker-v2-m3 vào baseline Recursive (1000/150) cải thiện đáng kể Context Precision và Context Recall so với cả baseline Recursive lẫn Semantic Interquartile.

#### Thiết kế thực nghiệm

| Cấu hình | Phương pháp | Chunks | Reranker |
|----------|------------|--------|---------|
| Baseline | Recursive (1000/150) | 340 | ❌ |
| So sánh | Semantic (Interquartile) | 564 | ❌ |
| **Đề xuất (Ours)** | **Recursive (1000/150) + Reranker** | **340** | ✅ BGE-v2-m3 |

**Bảng kết quả Thực nghiệm 2 (Table 4 trong LaTeX — SỐ LIỆU THỰC):**

| Phương pháp | Chunks | Faithfulness ↑ | Answer Relev. ↑ | Context Prec. ↑ | Context Recall ↑ |
|-------------|--------|---------------|----------------|----------------|-----------------|
| Recursive (1000/150) | 340 | 0.8632 | 0.7292 | 0.8618 | 0.9200 |
| Semantic (Interquartile) | 564 | **0.8867** | **0.7959** | 0.8117 | 0.9200 |
| **Recursive + Reranker** | **340** | 0.8761 | 0.8044 | **0.9247** | **0.9400** |
| **Δ (Recursive vs Ours)** | — | **+0.0129** | **+0.0752** | **+0.0629** | **+0.0200** |

**Phân tích kết quả cần viết:**
- Reranker cải thiện **Context Precision** mạnh nhất: 0.8618 → **0.9247** (+6.29pp) — chứng minh Cross-Encoder lọc nhiễu hiệu quả, đưa chunks liên quan nhất lên đầu
- **Context Recall** tăng từ 0.92 → **0.94** (+2pp) — mặc dù số chunks nạp vào LLM ít hơn (K=5), độ bao phủ dữ kiện vẫn cao hơn
- **Answer Relevancy** tăng đáng kể 0.7292 → **0.8044** (+7.52pp) — phản ánh việc ngữ cảnh chính xác hơn giúp LLM trả lời trúng trọng tâm hơn
- **Faithfulness** tăng nhẹ 0.8632 → 0.8761 (+1.29pp) — tầng Retrieval tốt hơn gián tiếp giảm thiểu ảo giác
- Semantic Interquartile cho Faithfulness cao hơn Recursive (0.8867 vs 0.8632) nhưng kém hơn về Context Precision (0.8117 vs 0.9247 khi có Reranker)

**Hình cần tạo:**
- **Hình 12:** Grouped bar chart "So sánh các chiến lược: Baseline vs Semantic vs Reranker" (4 nhóm: Faithfulness, Answer Relevance, Context Precision, Context Recall; 3 bars mỗi nhóm) — **dùng lại chart từ slides**, đạt giá trị 0.86/0.89/0.88 → 0.73/0.80/0.80 → 0.86/0.81/0.92 → 0.92/0.92/0.94

---

### 4.4 Tóm tắt và Thảo luận (Discussion)

**Bảng tổng hợp so sánh toàn bộ (Table 5 — Comprehensive Summary):**

| Phương pháp | Chunks | Faithfulness ↑ | Answer Relev. ↑ | Context Prec. ↑ | Context Recall ↑ |
|-------------|--------|---------------|----------------|----------------|-----------------|
| Recursive 1500/200 | 246 | 0.8811 | 0.7590 | 0.8149 | 0.92 |
| Recursive 1000/150 (Baseline) | 340 | 0.8632 | 0.7292 | 0.8618 | 0.92 |
| Recursive 800/100 | 401 | 0.8112 | 0.7526 | 0.8572 | 0.92 |
| Recursive 500/50 | 600 | 0.7867 | 0.7509 | 0.8151 | 0.92 |
| Semantic Interquartile | 564 | **0.8867** | **0.7959** | 0.8117 | 0.92 |
| Semantic Std_dev | 291 | 0.8693 | 0.7578 | 0.8040 | 0.90 |
| Semantic Percentile | 465 | 0.8573 | 0.7344 | 0.8225 | 0.90 |
| **Recursive 1000/150 + Reranker** ⭐ | **340** | 0.8761 | 0.8044 | **0.9247** | **0.9400** |

**Nội dung thảo luận cần viết:**
1. **Tại sao Recursive 1000/150 thắng về Precision:** Kích thước chunk cân bằng — đủ ngắn để Bi-Encoder embed ngữ nghĩa tập trung, đủ dài để chứa ngữ cảnh đầy đủ một ý tưởng
2. **Tại sao Semantic Chunking cho Faithfulness cao hơn:** Chunks được cắt theo ranh giới ngữ nghĩa tự nhiên → mỗi chunk mang một ý trọn vẹn → LLM ít bị prompt mâu thuẫn hơn
3. **Tại sao Reranker tăng mạnh Context Precision:** Cross-Encoder xem xét tương tác token-level giữa query và từng chunk → phân biệt được các chunks "gần đúng" và "đúng thực sự" tốt hơn Bi-Encoder
4. **Trade-off:** Reranker tăng latency thêm ~200-300ms so với One-Stage → chấp nhận được trong giáo dục (ưu tiên độ chính xác)
5. **Phạm vi nghiên cứu:** Dự án tập trung tối ưu tầng Retriever (Recall & Precision) → Giai đoạn 2 (Faithfulness & Generation) là hướng phát triển tương lai

---

## GIAI ĐOẠN 4b — Tạo các hình ảnh thực nghiệm (Figures 9–12)

| Hình | Tên file | Nội dung | Nguồn |
|------|----------|----------|-------|
| **Hình 9** | `ragas_pipeline.png` | Sơ đồ RAGAS pipeline (4 inputs → Framework → 4 Scores) | Tạo mới theo slides |
| **Hình 10** | `chunking_barchart.png` | Bar chart grouped 5 Recursive configs × 4 metrics | Tái tạo từ slides |
| **Hình 11** | `chunking_linetrend.png` | Line chart xu hướng 4 metrics theo Chunk Size | Tái tạo từ slides |
| **Hình 12** | `reranking_barchart.png` | Bar chart 3 configs × 4 metrics (Baseline vs Semantic vs Reranker) | Tái tạo từ slides |

> [!TIP]
> Tất cả 4 hình có thể được tạo bằng Python `matplotlib`/`seaborn` với style nhất quán (màu xanh `#344F1E`, font serif) rồi xuất `.png` → nhúng vào LaTeX.

---

## GIAI ĐOẠN 5 — Section V: Kết luận và Hướng phát triển

### 5.1 Kết luận (Conclusion)

**Cấu trúc đoạn kết:**
1. **Tóm tắt đóng góp kỹ thuật:** 4 trụ cột — Two-Stage Retrieval, Verifiable Citations, Adaptive Map-Reduce, Active Recall Toolkit
2. **Kết quả thực nghiệm nổi bật:**
   - Recursive Chunking 1000/150 + BGE Reranker đạt **Context Precision = 0.9247** (+6.29pp vs baseline)
   - **Context Recall = 0.9400** — độ bao phủ dữ kiện cao nhất
   - **Answer Relevancy = 0.8044** (+7.52pp vs baseline)
3. **Ý nghĩa học thuật:** Chứng minh thực nghiệm rằng Two-Stage Retrieval là lựa chọn tối ưu Pareto cho RAG giáo dục

### 5.2 Hạn chế (Limitations)

1. **Phạm vi thực nghiệm hẹp:** Chỉ tối ưu tầng Retriever; tầng Generation (Faithfulness, Answer Relevancy) chưa được tối ưu hóa có hệ thống
2. **Đơn phương thức (Unimodal):** Chưa xử lý được hình ảnh, sơ đồ, bảng biểu nhúng trong PDF
3. **Phụ thuộc API:** GeminiBackend yêu cầu kết nối internet và quota API
4. **Bộ test nhỏ:** Bộ đánh giá có giới hạn; chưa đại diện cho đa dạng miền tri thức

### 5.3 Hướng phát triển tương lai (Future Work)

1. **Tối ưu hóa tầng Generation (Giai đoạn 2):** Nghiên cứu các kỹ thuật prompt engineering nâng cao, fine-tuning LLM để cải thiện Faithfulness và Answer Relevancy
2. **Multimodal RAG:** Ứng dụng ColPali, GPT-4V để hiểu và truy xuất từ hình ảnh và bảng biểu trong tài liệu
3. **Graph RAG:** Xây dựng Knowledge Graph kết hợp vector search cho câu hỏi Multi-hop
4. **Adaptive Learning:** Theo dõi lịch sử quiz, tự động điều chỉnh độ khó dựa trên hiệu suất người học
5. **Production Deployment:** Docker + Qdrant Cloud + Prometheus/Grafana monitoring

---

## GIAI ĐOẠN 6 — Kiểm tra và Biên dịch LaTeX

### 6.1 Danh mục kiểm tra (Checklist)

- [ ] Tất cả `\cite{...}` có entry trong `references.bib`
- [ ] Tất cả `\ref{...}` và `\label{...}` khớp nhau
- [ ] Xóa toàn bộ dòng `% TODO` còn sót
- [ ] `\thehinh` đếm đúng từ Hình 1 đến **Hình 12**
- [ ] Bảng Table 1 (Section II), Table 2 (Section III), Table 3–5 (Section IV) có `\label` nhất quán
- [ ] Các file PNG: `ragas_pipeline.png`, `chunking_barchart.png`, `chunking_linetrend.png`, `reranking_barchart.png` đặt đúng trong `Figures/`
- [ ] Cài package bị thiếu: `ragged2e` (sửa lỗi compile hiện tại)

### 6.2 Quy trình biên dịch

```bash
pdflatex -interaction=nonstopmode report_draft.tex
bibtex report_draft
pdflatex -interaction=nonstopmode report_draft.tex
pdflatex -interaction=nonstopmode report_draft.tex
```

---

## Tóm tắt thứ tự thực hiện (Đã cập nhật với số liệu thực)

| Bước | Nhiệm vụ | Kết quả bàn giao |
|------|----------|-----------------|
| **4.0** | Viết intro Section IV: Chiến lược đánh giá 2 giai đoạn + giải thích RAGAS | LaTeX text |
| **4b-H9** | Tạo `ragas_pipeline.png` — sơ đồ RAGAS | File PNG |
| **4.2** | Viết Thực nghiệm 1 (Chunking Methods) + **Table 3** với số liệu thực | LaTeX text + table |
| **4b-H10** | Tạo `chunking_barchart.png` — bar chart 5 configs | File PNG |
| **4b-H11** | Tạo `chunking_linetrend.png` — line chart xu hướng chunk size | File PNG |
| **4.3** | Viết Thực nghiệm 2 (Reranking) + **Table 4** với số liệu thực | LaTeX text + table |
| **4b-H12** | Tạo `reranking_barchart.png` — bar chart 3 chiến lược | File PNG |
| **4.4** | Viết thảo luận tổng hợp + **Table 5** toàn diện | LaTeX text + table |
| **5.1–5.3** | Viết Section V — Kết luận, Hạn chế, Hướng phát triển | LaTeX text |
| **6.1–6.2** | Kiểm tra cross-refs + biên dịch PDF hoàn chỉnh | File PDF |

---

> [!IMPORTANT]
> **Số liệu đã được xác nhận từ slides thực nghiệm thực tế.** Không cần ghi chú `(estimated)`. Các con số trong Table 3–5 là kết quả RAGAS đo được trên hệ thống thực.

> [!NOTE]
> **Phạm vi Section IV được thu hẹp đúng thực tế:** Chỉ trình bày tối ưu tầng Retriever (2 experiments). Không có Thực nghiệm 3 về Generation Quality (đó là hướng nghiên cứu tương lai). Section IV bao gồm: Eval Strategy → RAGAS Framework → Exp 1 Chunking → Exp 2 Reranking → Tổng hợp & Thảo luận.

> [!TIP]
> Hãy dùng lệnh `/goal` nếu muốn tôi tự động thực hiện toàn bộ kế hoạch này từ đầu đến cuối.
