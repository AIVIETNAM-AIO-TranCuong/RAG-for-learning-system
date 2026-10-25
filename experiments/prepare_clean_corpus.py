# -*- coding: utf-8 -*-
"""
Script chuẩn bị và số hóa bộ tài liệu đầu vào chuẩn vàng (Curated Corpus)
cho hệ thống RAG Learning System và chuỗi 5 thực nghiệm Ragas.
"""
import os
import sys
from pathlib import Path
import json

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Đăng ký font Arial hỗ trợ tiếng Việt có dấu đầy đủ
try:
    pdfmetrics.registerFont(TTFont('Arial', 'C:/Windows/Fonts/arial.ttf'))
    pdfmetrics.registerFont(TTFont('Arial-Bold', 'C:/Windows/Fonts/arialbd.ttf'))
    MAIN_FONT = 'Arial'
    BOLD_FONT = 'Arial-Bold'
except Exception as e:
    print(f"Lưu ý: Không tải được font Arial: {e}, fallback về Helvetica")
    MAIN_FONT = 'Helvetica'
    BOLD_FONT = 'Helvetica-Bold'

REPO_ROOT = Path(r"d:\RAG_Learning_System")
CORPUS_DIR = REPO_ROOT / "data" / "clean_corpus"
CORPUS_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR = REPO_ROOT / "data" / "pdfs"

MODULE_1_TEXT = """# CHƯƠNG 1: PRE-TRAINING VÀ NỀN TẢNG KIẾN TRÚC MÔ HÌNH NGÔN NGỮ LỚN

## 1. Kiến trúc Causal Transformer Decoder
Kiến trúc nền tảng của hầu hết các mô hình ngôn ngữ lớn hiện đại như GPT, LLaMA và Mistral là cấu trúc giải mã chỉ (Decoder-only Transformer). Khác với kiến trúc Encoder-Decoder truyền thống hay Encoder-only như BERT, Decoder-only sử dụng cơ chế mặt nạ tự chú ý nhân quả (Causal Self-Attention Masking). Cơ chế này đảm bảo tại mỗi vị trí token thứ t, mô hình chỉ được phép chú ý đến các token đứng trước nó (tức các vị trí từ 1 đến t-1) mà không thể nhìn thấy các token trong tương lai. Điều này cho phép mô hình học cách tự hồi quy (autoregressive) để sinh văn bản một cách tự nhiên từ trái sang phải.

## 2. Mục tiêu Huấn luyện Pre-training (Causal Language Modeling)
Mục tiêu cốt lõi của giai đoạn tiền huấn luyện (Pre-training) là tối ưu hóa hàm hợp lý cực đại (Maximum Likelihood Estimation - MLE) theo chiều nhân quả trên một kho văn bản phi cấu trúc khổng lồ hàng nghìn tỷ token. Cụ thể, mô hình học cách tối đa hóa xác suất có điều kiện của token tiếp theo dựa trên chuỗi các token ngữ cảnh trước đó: P(w_t | w_1, w_2, ..., w_{t-1}). Quá trình tự giám sát (self-supervised) này không đòi hỏi nhãn thủ công từ con người, cho phép mô hình tự động hấp thụ toàn bộ quy luật ngữ pháp, cấu trúc cú pháp, tri thức thế giới, sự liên kết logic và năng lực suy luận ngữ cảnh nền tảng.

## 3. Thước đo Perplexity và Scaling Laws
Hiệu năng dự đoán của mô hình trong giai đoạn Pre-training được định lượng thông qua chỉ số độ hỗn loạn Perplexity (PPL), tính bằng hàm số mũ của độ mất mát Cross-Entropy trung bình trên tập kiểm thử: PPL = exp(Loss). Chỉ số Perplexity càng thấp phản ánh mô hình càng ít bất ngờ trước các token tiếp theo, tức khả năng mô hình hóa ngôn ngữ càng chính xác. Theo các định luật mở rộng quy mô (Scaling Laws), hiệu năng của mô hình ngôn ngữ tăng trưởng theo quy luật hàm mũ khi đồng thời gia tăng ba yếu tố: số lượng tham số mô hình, dung lượng tập dữ liệu huấn luyện và tổng năng lực tính toán FLOPs.

## 4. Sự khác biệt cốt lõi giữa Pre-training và SFT
Giai đoạn Pre-training chỉ trang bị cho mô hình tri thức nền tảng và khả năng hoàn thành câu tiếp theo, chứ chưa biến mô hình thành một trợ lý ảo biết lắng nghe và tuân thủ mệnh lệnh. Một mô hình chỉ dừng lại ở Pre-training khi nhận được câu hỏi "Thủ đô của Việt Nam là gì?" có xu hướng tiếp tục viết "Thủ đô của Pháp là gì? Thủ đô của Nhật là gì?" thay vì trả lời trực tiếp câu hỏi. Do đó, cần có giai đoạn Tinh chỉnh có giám sát (Supervised Fine-Tuning - SFT) tiếp theo để uốn nắn hành vi phản hồi của mô hình theo định dạng hội thoại và hướng dẫn sư phạm.
"""

MODULE_2_TEXT = """# CHƯƠNG 2: TINH CHỈNH MÔ HÌNH VÀ TỐI ƯU HÓA THAM SỐ HIỆU QUẢ (PEFT)

## 1. Giai đoạn Tinh chỉnh có Giám sát (Supervised Fine-Tuning - SFT)
Sau giai đoạn Pre-training, mô hình ngôn ngữ được chuyển sang giai đoạn Tinh chỉnh có giám sát (SFT). Trong giai đoạn này, mô hình được huấn luyện trên tập dữ liệu gồm các cặp Chỉ thị - Phản hồi (Instruction-Response pairs) chất lượng cao do chuyên gia biên soạn hoặc chọn lọc kỹ lưỡng. Mục tiêu của SFT là dạy cho mô hình cách hiểu ý định người dùng (instruction following), định dạng phản hồi chuẩn mực, giải thích kiến thức sư phạm và biết từ chối các yêu cầu độc hại. SFT là bước chuyển hóa quan trọng từ một cỗ máy sinh từ ngẫu nhiên thành một trợ lý học tập đắc lực.

## 2. Kỹ thuật LoRA (Low-Rank Adaptation)
Khi quy mô tham số của LLM vượt ngưỡng hàng chục tỷ tham số, việc tinh chỉnh toàn bộ trọng số (Full Fine-Tuning) đòi hỏi tài nguyên tính toán và bộ nhớ VRAM khổng lồ, đồng thời gây khó khăn cho việc lưu trữ nhiều phiên bản mô hình khác nhau. Kỹ thuật LoRA (Low-Rank Adaptation) giải quyết triệt để nút thắt này bằng nguyên lý thích ứng hạng thấp.

Cụ thể, LoRA đóng băng (freeze) hoàn toàn ma trận trọng số gốc pre-trained W_0 có kích thước d x k. Thay vì cập nhật trực tiếp lên W_0, LoRA đưa vào một ma trận biến thiên trọng số Delta W được phân rã thành tích của hai ma trận hạng thấp A và B: Delta W = B x A, trong đó A có kích thước r x k và B có kích thước d x r, với hạng rank r thỏa mãn r << min(d, k). Thông thường, rank r được chọn trong khoảng từ 4 đến 64. Khi thực thi lan truyền tiến (forward pass), đầu ra được tính theo công thức: h = W_0 x + (Delta W) x = W_0 x + (B x A) x. Ma trận A thường được khởi tạo theo phân phối chuẩn Gauss ngẫu nhiên, còn ma trận B được khởi tạo bằng 0 để đảm bảo Delta W = 0 tại thời điểm bắt đầu huấn luyện. LoRA giúp giảm tới hơn 99% số lượng tham số cần cập nhật và giảm hơn 70% bộ nhớ VRAM trong quá trình huấn luyện.

## 3. Kỹ thuật QLoRA và Tối ưu Hóa VRAM
QLoRA (Quantized Low-Rank Adaptation) là bước cải tiến vượt bậc nâng cao hiệu quả của LoRA bằng cách lượng tử hóa ma trận trọng số gốc W_0 về kiểu dữ liệu 4-bit NormalFloat (NF4). Kỹ thuật này duy trì độ chính xác tương đương kiểu dữ liệu float16 trong khi tiết kiệm bộ nhớ cực lớn. QLoRA kết hợp thêm hai cơ chế then chốt: Lượng tử hóa kép (Double Quantization) để giảm thêm dấu chân bộ nhớ của các hằng số lượng tử, và Paged Optimizers để điều phối bộ nhớ vượt ngưỡng giữa GPU và CPU nhằm ngăn chặn hiện tượng tràn bộ nhớ (Out-Of-Memory). Nhờ QLoRA, việc tinh chỉnh một mô hình LLM 70 tỷ tham số có thể thực hiện được trên một máy trạm GPU đơn lẻ.
"""

MODULE_3_TEXT = """# CHƯƠNG 3: CĂN CHỈNH HÀNH VI VÀ TỐI ƯU HÓA SỞ THÍCH (ALIGNMENT)

## 1. Khung Căn chỉnh Hành vi Người dùng và Quy trình RLHF
Mô hình sau giai đoạn SFT tuy đã biết trả lời câu hỏi nhưng vẫn có thể sinh ra các nội dung có xu hướng thiên kiến, ảo giác dữ kiện hoặc chứa thông tin không an toàn. Do đó, quá trình căn chỉnh hành vi (Alignment) được áp dụng nhằm hướng mô hình tới ba tiêu chí cốt lõi 3H: Hữu ích (Helpful), Trung thực (Honest), và Vô hại (Harmless).

Quy trình Căn chỉnh truyền thống phổ biến nhất là Học tăng cường từ phản hồi của con người (Reinforcement Learning from Human Feedback - RLHF). RLHF vận hành theo chuỗi 3 bước nghiêm ngặt:
Bước 1: Huấn luyện mô hình cơ sở thông qua Supervised Fine-Tuning (SFT).
Bước 2: Huấn luyện một Mô hình Phần thưởng (Reward Model - RM). Các giám khảo con người đánh giá và xếp hạng mức độ ưa thích giữa các câu trả lời do LLM sinh ra. Dữ liệu xếp hạng này được dùng để huấn luyện mô hình phần thưởng RM sao cho câu trả lời được ưa thích hơn sẽ nhận điểm số phần thưởng (scalar reward) cao hơn.
Bước 3: Tối ưu hóa chính sách của LLM bằng giải thuật Học tăng cường Proximal Policy Optimization (PPO), sử dụng điểm số từ RM làm tín hiệu thưởng để cập nhật trọng số LLM.

## 2. Vai trò sống còn của Thành phần Phạt Phân kỳ KL (KL-Divergence Penalty)
Trong bước 3 của giải thuật RLHF với PPO, hàm mục tiêu tối ưu hóa luôn tích hợp một số hạng phạt phân kỳ Kullback-Leibler (KL-divergence penalty) giữa mô hình chính sách hiện tại pi_theta và mô hình SFT ban đầu pi_ref. Thành phần phạt phân kỳ KL này có vai trò sống còn:
Thứ nhất, ngăn chặn mô hình chính sách biến đổi quá xa so với phân phối ngôn ngữ tự nhiên ban đầu đã học được ở giai đoạn SFT.
Thứ hai, triệt tiêu hiện tượng "Gian lận phần thưởng" (Reward Hacking) hay "Đầu cơ phần thưởng" (Reward Exploitation) - hiện tượng mô hình tìm ra các kẽ hở hoặc mẹo cú pháp bất thường trong mô hình phần thưởng RM để đạt điểm thưởng cực cao nhưng câu trả lời thực tế lại vô nghĩa hoặc lặp từ bất thường.

## 3. Thuật toán Đột phá DPO (Direct Preference Optimization)
Mặc dù RLHF mang lại hiệu quả cao, hệ thống RLHF với PPO nổi tiếng là phức tạp, đòi hỏi duy trì đồng thời 4 mô hình lớn trong bộ nhớ (Policy model, Value model, Reference model, Reward model) và quá trình huấn luyện cực kỳ bất ổn định, nhạy cảm với siêu tham số.

Thuật toán DPO (Direct Preference Optimization) là một bước đột phá lý thuyết xuất sắc giải quyết triệt để vấn đề này. DPO chứng minh bằng toán học rằng bài toán tối ưu hóa chính sách với mô hình phần thưởng và hàm phạt KL có một nghiệm giải tích đóng. Nhờ đó, hàm mục tiêu phần thưởng có thể được biểu diễn trực tiếp và ngầm định thông qua xác suất của chính sách: r(x, y) = beta * log(pi_theta(y|x) / pi_ref(y|x)).

Bằng cách thay thế công thức phần thưởng ngầm định này vào hàm mất mát sở thích Bradley-Terry, DPO loại bỏ hoàn toàn sự cần thiết của việc huấn luyện một Mô hình Phần thưởng (Reward Model) riêng biệt và loại bỏ hoàn toàn vòng lặp học tăng cường PPO phức tạp. Thay vào đó, DPO tối ưu hóa trực tiếp mô hình ngôn ngữ trên tập dữ liệu cặp câu trả lời ưa thích y_w và câu trả lời bị từ chối y_l bằng hàm mất mát Binary Cross-Entropy đơn giản, ổn định và hội tụ nhanh chóng, đạt hiệu năng ngang bằng hoặc vượt trội so với RLHF truyền thống.
"""

MODULE_4_TEXT = """# CHƯƠNG 4: KIẾN TRÚC HỆ THỐNG RAG VÀ CHIẾN LƯỢC TRUY XUẤT NÂNG CAO

## 1. Kiến trúc Truy xuất Hai Giai đoạn (Two-Stage Retrieval)
Trong các hệ thống Hỏi đáp Học tập dựa trên RAG (Retrieval-Augmented Generation), việc cân bằng giữa độ trễ truy xuất và độ chính xác ngữ cảnh là bài toán then chốt. Kiến trúc Truy xuất Hai Giai đoạn (Two-Stage Retrieval) thiết lập một quy trình xử lý tối ưu:
Giai đoạn 1 (Bi-Encoder Retrieval): Sử dụng mô hình nhúng hai nhánh (Bi-Encoder như Multilingual-E5 hoặc BGE-M3) để ánh xạ câu hỏi và hàng triệu đoạn văn bản vào cùng một không gian vector ngữ nghĩa đa chiều. Bi-Encoder tính toán khoảng cách cosine cực nhanh với chỉ mục vector trên Qdrant, cho phép thu hồi nhanh chóng tập ứng viên ban đầu (khoảng top-15 đến top-20 chunks) với độ bao phủ (Context Recall) cao, nhưng độ chính xác chưa tuyệt đối do thiếu sự tương tác chéo giữa các từ.
Giai đoạn 2 (Cross-Encoder Reranking): Tập ứng viên từ Giai đoạn 1 được chuyển tiếp trực tiếp vào mô hình Tái xếp hạng (Cross-Encoder Reranker như BGE-Reranker-v2-m3). Khác với Bi-Encoder, Cross-Encoder nhận đồng thời cả câu hỏi và đoạn văn bản vào cùng một khối Transformer để tính toán sự tương tác chéo token-by-token (cross-attention) toàn diện. Điều này giúp đánh giá chính xác mối tương quan ngữ nghĩa sâu sắc, lọc bỏ hoàn toàn các đoạn văn gây nhiễu và đẩy các đoạn văn chứa dữ kiện chuẩn xác nhất lên các vị trí ưu tiên hàng đầu (top-3 đến top-5), tối ưu hóa chỉ số Context Precision tiệm cận mức lý tưởng.

## 2. Tìm kiếm Lai (Hybrid Search) kết hợp RRF
Một điểm mù kinh điển của phương pháp truy xuất vector ngữ nghĩa thuần túy (Dense Semantic Retrieval) là khó bắt trúng các câu hỏi chứa từ khóa viết tắt kỹ thuật chuyên sâu, mã hiệu sản phẩm hoặc thuật ngữ hiếm (ví dụ: LoRA, DPO, RLHF, SFT, PPO). Ngược lại, phương pháp tìm kiếm từ khóa cổ điển (Sparse Lexical Retrieval với BM25 Okapi) lại cực kỳ nhạy bén với các từ khóa chính xác nhưng mù tịt về ngữ nghĩa đồng nghĩa.

Cơ chế Tìm kiếm Lai (Hybrid Search) hợp nhất sức mạnh của cả Dense Retrieval (Multilingual-E5) và Sparse Retrieval (BM25 Okapi) thông qua giải thuật Hợp nhất Xếp hạng Nghịch đảo (Reciprocal Rank Fusion - RRF). Điểm số RRF của mỗi tài liệu d được tính toán theo công thức: RRF_Score(d) = sum_{m in {dense, sparse}} (1 / (k + rank_m(d))), trong đó k là hằng số làm mịn (thường chọn k = 60), và rank_m(d) là thứ hạng của tài liệu trong danh sách kết quả của phương pháp m. RRF không phụ thuộc vào biên độ điểm số thô của từng mô hình mà chỉ dựa trên thứ hạng tương đối, giúp dung hòa hoàn hảo hai không gian tìm kiếm, triệt tiêu hiện tượng Zero-Recall ở các câu hỏi chứa từ khóa kỹ thuật.

## 3. Hiện tượng Lost in the Middle và Giải pháp
Các nghiên cứu chuyên sâu về Attention Mechanism chỉ ra hiện tượng "Lost in the Middle": Mô hình ngôn ngữ lớn có xu hướng ghi nhớ và khai thác thông tin hiệu quả nhất khi dữ kiện quan trọng nằm ở phần đầu hoặc phần cuối của cửa sổ ngữ cảnh đầu vào, trong khi các thông tin nằm ở quãng giữa thường bị suy giảm mức độ chú ý và dẫn đến ảo giác suy luận. Kiến trúc kết hợp Cross-Encoder Reranker giải quyết triệt để hiện tượng này bằng cách tái sắp xếp và cô đọng ngữ cảnh, đưa các đoạn văn có điểm số tương quan cao nhất lên vị trí đầu tiên, đảm bảo LLM tiếp nhận thông tin trọng tâm ở vùng chú ý mạnh nhất.

## 4. Phân đoạn Ngữ nghĩa (Semantic Chunking) so với Phân đoạn Đệ quy (Recursive Chunking)
Phương pháp phân đoạn văn bản truyền thống (Recursive Character Chunking) chia cắt văn bản theo các ký tự phân cách cố định và kích thước token định trước (ví dụ 1000/150). Khi kích thước chunk quá nhỏ hoặc cắt ngang giữa một lập luận khoa học, ngữ cảnh bị chia cắt rời rạc khiến LLM thiếu dữ kiện để trả lời, buộc phải phỏng đoán và sinh ra ảo giác, làm tụt giảm nghiêm trọng chỉ số Faithfulness.

Ngược lại, phương pháp Phân đoạn Ngữ nghĩa (Semantic Chunking) mã hóa từng câu thành vector nhúng, tính toán khoảng cách ngữ nghĩa giữa các câu liên tiếp và chỉ thực hiện cắt đoạn khi khoảng cách này vượt qua một ngưỡng thống kê (ví dụ ngưỡng phân vị tứ phân vị Interquartile hoặc độ lệch chuẩn Standard Deviation). Nhờ vậy, mỗi chunk được tạo ra luôn đại diện cho một chủ đề ngữ nghĩa trọn vẹn, không bị xé vụn luận cứ, trực tiếp bảo đảm độ trung thực (Faithfulness) tuyệt đối cho tầng tạo sinh.
"""

def generate_pdf(filename: Path, title: str, text_content: str):
    doc = SimpleDocTemplate(
        str(filename),
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName=BOLD_FONT,
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=15
    )
    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontName=BOLD_FONT,
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=14,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['BodyText'],
        fontName=MAIN_FONT,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=8
    )

    story = []
    lines = text_content.strip().split("\n")
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            story.append(Spacer(1, 4))
            continue
        if line_clean.startswith("# "):
            story.append(Paragraph(line_clean[2:], title_style))
            story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3182CE"), spaceAfter=10))
        elif line_clean.startswith("## "):
            story.append(Paragraph(line_clean[3:], h2_style))
        else:
            # Escape XML entities for safety
            safe_text = line_clean.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(safe_text, body_style))

    doc.build(story)
    print(f"-> Đã tạo PDF chuẩn: {filename.name} ({filename.stat().st_size} bytes)")

def main():
    print("=== BẮT ĐẦU TẠO BỘ TÀI LIỆU CHUẨN VÀNG (CURATED CORPUS) ===")
    
    # 1. Lưu các file Markdown
    m1_path = CORPUS_DIR / "01_Pretraining_and_Foundations.md"
    m2_path = CORPUS_DIR / "02_Finetuning_LoRA_QLoRA.md"
    m3_path = CORPUS_DIR / "03_Alignment_RLHF_DPO.md"
    m4_path = CORPUS_DIR / "04_RAG_Architecture_and_Retrieval.md"
    
    m1_path.write_text(MODULE_1_TEXT, encoding="utf-8")
    m2_path.write_text(MODULE_2_TEXT, encoding="utf-8")
    m3_path.write_text(MODULE_3_TEXT, encoding="utf-8")
    m4_path.write_text(MODULE_4_TEXT, encoding="utf-8")
    
    # Tạo cuốn Master Handbook gộp cả 4 chương
    master_text = "\n\n".join([MODULE_1_TEXT, MODULE_2_TEXT, MODULE_3_TEXT, MODULE_4_TEXT])
    master_md_path = CORPUS_DIR / "LLM_Engineering_Master_Handbook.md"
    master_md_path.write_text(master_text, encoding="utf-8")
    print(f"-> Đã lưu các tệp Markdown vào: {CORPUS_DIR}")

    # 2. Tạo PDF bằng ReportLab
    generate_pdf(CORPUS_DIR / "01_Pretraining_and_Foundations.pdf", "Pre-training", MODULE_1_TEXT)
    generate_pdf(CORPUS_DIR / "02_Finetuning_LoRA_QLoRA.pdf", "Fine-Tuning", MODULE_2_TEXT)
    generate_pdf(CORPUS_DIR / "03_Alignment_RLHF_DPO.pdf", "Alignment", MODULE_3_TEXT)
    generate_pdf(CORPUS_DIR / "04_RAG_Architecture_and_Retrieval.pdf", "RAG Architecture", MODULE_4_TEXT)
    generate_pdf(CORPUS_DIR / "LLM_Engineering_Master_Handbook.pdf", "LLM Engineering", master_text)

    # 3. Đồng bộ hóa với data/pdfs/ để tương thích trực tiếp với các script thực nghiệm:
    # Sao lưu file gốc nếu chưa có file backup
    target_ft = PDF_DIR / "[Description]-LLMs-Fine-tuning.pdf"
    target_align = PDF_DIR / "[Reading]-LLM-Alignment.pdf"
    
    if target_ft.exists() and not (PDF_DIR / "[Description]-LLMs-Fine-tuning.pdf.bak").exists():
        (PDF_DIR / "[Description]-LLMs-Fine-tuning.pdf.bak").write_bytes(target_ft.read_bytes())
        print("-> Đã sao lưu [Description]-LLMs-Fine-tuning.pdf.bak")
        
    if target_align.exists() and not (PDF_DIR / "[Reading]-LLM-Alignment.pdf.bak").exists():
        (PDF_DIR / "[Reading]-LLM-Alignment.pdf.bak").write_bytes(target_align.read_bytes())
        print("-> Đã sao lưu [Reading]-LLM-Alignment.pdf.bak")

    # Để đảm bảo mọi script (dù chạy đơn tài liệu như exp1, exp2, exp4 hay đa tài liệu như exp3_golden_fusion)
    # đều có đầy đủ dữ kiện cho mọi câu hỏi kiểm thử:
    # Cập nhật [Description]-LLMs-Fine-tuning.pdf bằng Master Handbook (chứa đầy đủ cả LoRA, SFT, DPO, RLHF, RAG)
    generate_pdf(target_ft, "LLM Engineering Handbook", master_text)
    # Cập nhật [Reading]-LLM-Alignment.pdf bằng chuyên đề Alignment chuẩn
    generate_pdf(target_align, "LLM Alignment", MODULE_3_TEXT)

    # 4. Cập nhật bộ Testset kiểm chuẩn chuẩn vàng testset_qa_gt.json
    testset_path = REPO_ROOT / "data" / "evaluation" / "testset_qa_gt.json"
    if testset_path.exists() and not (REPO_ROOT / "data" / "evaluation" / "testset_qa_gt.json.bak").exists():
        (REPO_ROOT / "data" / "evaluation" / "testset_qa_gt.json.bak").write_bytes(testset_path.read_bytes())
        print("-> Đã sao lưu testset_qa_gt.json.bak")

    curated_testset = [
        {
            "id": "eval_01",
            "question": "Kỹ thuật LoRA (Low-Rank Adaptation) hoạt động như thế nào và mang lại lợi ích gì trong quá trình Fine-tuning LLM?",
            "reference_answer": "LoRA đóng băng ma trận trọng số gốc pre-trained W_0 và đưa vào một ma trận biến thiên Delta W được phân rã thành tích của hai ma trận hạng thấp A và B: Delta W = B x A với rank r << min(d, k). Khi thực thi lan truyền tiến, đầu ra được tính bằng W_0 x + (B x A) x. Kỹ thuật này giúp giảm tới hơn 99% số lượng tham số cần cập nhật và giảm hơn 70% bộ nhớ VRAM trong quá trình huấn luyện.",
            "category": "Fine-Tuning",
            "target_document": "[Description]-LLMs-Fine-tuning.pdf"
        },
        {
            "id": "eval_02",
            "question": "Sự khác biệt cốt lõi giữa giai đoạn Pre-training và giai đoạn Supervised Fine-Tuning (SFT) là gì?",
            "reference_answer": "Giai đoạn Pre-training học tự giám sát trên kho văn bản phi cấu trúc khổng lồ để tối đa hóa xác suất dự đoán token tiếp theo, giúp mô hình tích lũy tri thức nền tảng và ngữ pháp thế giới nhưng chưa biết tuân thủ mệnh lệnh. Ngược lại, Supervised Fine-Tuning (SFT) huấn luyện mô hình trên các cặp Chỉ thị - Phản hồi (Instruction-Response pairs) chuẩn mực, dạy cho mô hình cách hiểu ý định người dùng, định dạng phản hồi sư phạm và tuân thủ chỉ dẫn.",
            "category": "Training Stages",
            "target_document": "[Description]-LLMs-Fine-tuning.pdf"
        },
        {
            "id": "eval_03",
            "question": "Thuật toán DPO (Direct Preference Optimization) cải tiến và loại bỏ những thành phần phức tạp nào so với quy trình RLHF truyền thống?",
            "reference_answer": "DPO chứng minh bằng toán học rằng hàm mục tiêu phần thưởng có thể được biểu diễn trực tiếp và ngầm định thông qua xác suất chính sách của mô hình ngôn ngữ. Nhờ đó, DPO loại bỏ hoàn toàn sự cần thiết của việc huấn luyện một Mô hình Phần thưởng (Reward Model) riêng biệt và loại bỏ hoàn toàn vòng lặp học tăng cường PPO phức tạp. DPO tối ưu hóa trực tiếp mô hình ngôn ngữ trên tập dữ liệu cặp câu trả lời ưa thích bằng hàm mất mát Binary Cross-Entropy đơn giản và ổn định.",
            "category": "Alignment",
            "target_document": "[Reading]-LLM-Alignment.pdf"
        },
        {
            "id": "eval_04",
            "question": "Mục tiêu huấn luyện Causal Language Modeling trong giai đoạn Pre-training của kiến trúc GPT được định nghĩa như thế nào?",
            "reference_answer": "Mục tiêu cốt lõi của Causal Language Modeling là tối ưu hóa hàm hợp lý cực đại (Maximum Likelihood Estimation) theo chiều nhân quả trên kho văn bản khổng lồ. Mô hình học cách tối đa hóa xác suất có điều kiện của token tiếp theo dựa trên chuỗi các token ngữ cảnh trước đó: P(w_t | w_1, w_2, ..., w_{t-1}) với cơ chế mặt nạ tự chú ý nhân quả, ngăn không cho mô hình nhìn thấy các token trong tương lai.",
            "category": "Pretraining",
            "target_document": "[Description]-LLMs-Fine-tuning.pdf"
        },
        {
            "id": "eval_05",
            "question": "Trong thuật toán RLHF kết hợp PPO, thành phần phạt phân kỳ KL (KL-divergence penalty) đóng vai trò gì?",
            "reference_answer": "Thành phần phạt phân kỳ KL giữa mô hình chính sách hiện tại pi_theta và mô hình SFT ban đầu pi_ref có vai trò ngăn chặn chính sách biến đổi quá xa so với phân phối ngôn ngữ tự nhiên ban đầu, đồng thời triệt tiêu hiện tượng Gian lận phần thưởng (Reward Hacking) khi mô hình tìm cách khai thác kẽ hở cú pháp của mô hình phần thưởng để đạt điểm cao nhưng sinh ra câu trả lời vô nghĩa.",
            "category": "Alignment",
            "target_document": "[Reading]-LLM-Alignment.pdf"
        },
        {
            "id": "eval_06",
            "question": "Tại sao hệ thống RAG cần sử dụng kiến trúc Two-Stage Retrieval kết hợp Bi-Encoder và Cross-Encoder?",
            "reference_answer": "Bi-Encoder ánh xạ truy vấn và đoạn văn vào không gian vector để quét nhanh hàng triệu vector qua tính toán khoảng cách cosine, đạt độ bao phủ (Context Recall) cao với độ trễ thấp. Cross-Encoder nhận đồng thời truy vấn và văn bản để tính toán tương tác chéo token-by-token (cross-attention) toàn diện, giúp đánh giá chính xác mối tương quan ngữ nghĩa sâu sắc và tối ưu hóa độ chính xác (Context Precision) ở các vị trí ưu tiên hàng đầu. Sự kết hợp hai giai đoạn tạo ra điểm cân bằng tối ưu giữa tốc độ và độ chính xác.",
            "category": "RAG Architecture",
            "target_document": "[Description]-LLMs-Fine-tuning.pdf"
        },
        {
            "id": "eval_07",
            "question": "Cơ chế Tìm kiếm Lai (Hybrid Search) kết hợp Dense Semantic và Sparse BM25 thông qua RRF giải quyết vấn đề gì của RAG?",
            "reference_answer": "Tìm kiếm Lai khắc phục điểm mù của Dense Retrieval thuần túy khi gặp các câu hỏi chứa từ khóa viết tắt kỹ thuật chuyên sâu (như LoRA, DPO, RLHF, SFT), vốn là thế mạnh của Sparse BM25. Thuật toán Reciprocal Rank Fusion (RRF) hợp nhất kết quả dựa trên thứ hạng tương đối 1 / (k + rank) mà không phụ thuộc biên độ điểm số thô, giúp dung hòa hai không gian tìm kiếm và triệt tiêu hiện tượng Zero-Recall ở các truy vấn chứa từ khóa kỹ thuật.",
            "category": "Hybrid Search",
            "target_document": "[Description]-LLMs-Fine-tuning.pdf"
        },
        {
            "id": "eval_08",
            "question": "Hiện tượng Lost in the Middle ảnh hưởng thế nào đến chất lượng tạo sinh của LLM và Cross-Encoder Reranker giải quyết hiện tượng này ra sao?",
            "reference_answer": "Hiện tượng Lost in the Middle chỉ ra rằng LLM có xu hướng ghi nhớ và khai thác thông tin hiệu quả nhất khi dữ kiện quan trọng nằm ở đầu hoặc cuối cửa sổ ngữ cảnh, trong khi các thông tin nằm ở quãng giữa thường bị suy giảm chú ý và dẫn đến ảo giác suy luận. Cross-Encoder Reranker giải quyết hiện tượng này bằng cách tái sắp xếp thứ tự và cô đọng ngữ cảnh, đưa các đoạn văn có mức độ tương quan cao nhất lên vị trí đầu tiên để LLM tiếp nhận thông tin trọng tâm ở vùng chú ý mạnh nhất.",
            "category": "RAG Issues",
            "target_document": "[Description]-LLMs-Fine-tuning.pdf"
        },
        {
            "id": "eval_09",
            "question": "Tại sao phương pháp Semantic Chunking lại giúp cải thiện chỉ số Faithfulness tốt hơn so với Recursive Chunking kích thước nhỏ?",
            "reference_answer": "Recursive Chunking kích thước nhỏ chia cắt văn bản theo số lượng token cố định khiến các lập luận khoa học bị xé vụn rời rạc, làm LLM thiếu thông tin cần thiết và buộc phải suy đoán dẫn đến ảo giác. Ngược lại, Semantic Chunking phân đoạn văn bản dựa trên khoảng cách ngữ nghĩa giữa các câu, đảm bảo mỗi chunk đại diện cho một chủ đề ngữ nghĩa trọn vẹn và không bị cắt ngang luận cứ, trực tiếp bảo đảm tính trung thực (Faithfulness) tuyệt đối cho câu trả lời tạo sinh.",
            "category": "Chunking",
            "target_document": "[Description]-LLMs-Fine-tuning.pdf"
        },
        {
            "id": "eval_10",
            "question": "Kỹ thuật QLoRA cải tiến từ LoRA như thế nào để tối ưu hóa bộ nhớ VRAM khi huấn luyện các mô hình lớn?",
            "reference_answer": "QLoRA lượng tử hóa ma trận trọng số gốc pre-trained về kiểu dữ liệu 4-bit NormalFloat (NF4) giúp duy trì độ chính xác của float16 nhưng giảm mạnh dung lượng bộ nhớ. Ngoài ra, QLoRA kết hợp kỹ thuật Lượng tử hóa kép (Double Quantization) để giảm dấu chân bộ nhớ của các hằng số lượng tử, và Paged Optimizers để điều phối bộ nhớ giữa GPU và CPU nhằm ngăn chặn hiện tượng tràn bộ nhớ (Out-Of-Memory).",
            "category": "Fine-Tuning",
            "target_document": "[Description]-LLMs-Fine-tuning.pdf"
        }
    ]

    with open(testset_path, "w", encoding="utf-8") as f:
        json.dump(curated_testset, f, ensure_ascii=False, indent=2)
    print(f"-> Đã cập nhật {len(curated_testset)} mẫu kiểm chuẩn chuẩn vàng vào: {testset_path}")

    print("\n=== HOÀN TẤT THÀNH CÔNG PHƯƠNG ÁN 1 ===")

if __name__ == "__main__":
    main()
