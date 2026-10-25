# -*- coding: utf-8 -*-
"""
Thiết lập bộ testset chuẩn vàng và cấu hình thực nghiệm
cho tài liệu 140 trang: [Project]-Finance-Analytics-eDA.pdf
"""
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(r"d:\RAG_Learning_System")
TARGET_PDF = REPO_ROOT / "data" / "pdfs" / "[Project]-Finance-Analytics-eDA.pdf"
TESTSET_PATH = REPO_ROOT / "data" / "evaluation" / "testset_qa_gt.json"

FINANCE_TESTSET = [
    {
        "id": "eval_fin_01",
        "question": "Mục tiêu tổng quát của dự án phân tích tiền gửi tại AIV Bank là gì và được tổ chức theo bốn nhóm mục tiêu nào?",
        "reference_answer": "Mục tiêu tổng quát của dự án là đánh giá diễn biến huy động, nhận diện các yếu tố ảnh hưởng đến chi phí và khả năng duy trì nguồn vốn, từ đó đề xuất các hướng quản trị có cơ sở dẫn chứng từ dữ liệu trực quan. Bốn nhóm mục tiêu gồm: (1) Đánh giá quy mô và cơ cấu danh mục; (2) Phân tích khả năng duy trì tiền gửi khi đến hạn; (3) Đánh giá chi phí huy động và đặc điểm kỳ hạn; (4) Quản trị dòng tiền thanh khoản và rủi ro sau đáo hạn.",
        "category": "Project Objective",
        "target_document": "[Project]-Finance-Analytics-eDA.pdf"
    },
    {
        "id": "eval_fin_02",
        "question": "Sự khác biệt trong cách ghi nhận tiền gửi và khoản cho vay giữa khách hàng và ngân hàng thương mại là gì?",
        "reference_answer": "Đối với khách hàng, tiền gửi là tài sản vì họ có quyền nhận lại tiền theo thỏa thuận; còn đối với ngân hàng, tiền gửi là khoản nợ phải trả đi kèm nghĩa vụ hoàn trả gốc và lãi. Ngược lại, khoản cho vay là tài sản của ngân hàng vì ngân hàng có quyền thu hồi gốc và nhận lãi, trong khi đối với khách hàng vay thì đó là nghĩa vụ nợ phải trả.",
        "category": "Banking Foundation",
        "target_document": "[Project]-Finance-Analytics-eDA.pdf"
    },
    {
        "id": "eval_fin_03",
        "question": "Thu nhập lãi thuần của ngân hàng được xác định như thế nào và tại sao tiền gửi không kỳ hạn giúp duy trì nguồn vốn chi phí thấp?",
        "reference_answer": "Thu nhập lãi thuần được xác định bằng chênh lệch giữa thu nhập lãi và chi phí lãi (ví dụ thu nhập lãi 100 tỷ trừ chi phí lãi 60 tỷ bằng 40 tỷ thu nhập lãi thuần). Tiền gửi không kỳ hạn thường có lãi suất thấp hơn tiền gửi có kỳ hạn, do đó khi khách hàng duy trì số dư tài khoản thanh toán và nhận lương qua ngân hàng sẽ giúp ngân hàng duy trì nguồn vốn với chi phí lãi thấp hơn.",
        "category": "Cost of Funds",
        "target_document": "[Project]-Finance-Analytics-eDA.pdf"
    },
    {
        "id": "eval_fin_04",
        "question": "Khi kết nối Power BI với PostgreSQL, nếu gặp thông báo nhắc SSL hoặc certificate trust thì nguyên nhân và cách sửa là gì?",
        "reference_answer": "Nguyên nhân là do mục Encrypt connection vẫn còn được đánh dấu trong khi máy chủ PostgreSQL sử dụng chứng chỉ tự ký. Cách sửa là bỏ đánh dấu Encrypt connection; nếu hộp thoại không hiện lại thì vào File -> Options and settings -> Data source settings, chọn nguồn dữ liệu và bấm Edit Permissions để điều chỉnh.",
        "category": "Data Integration",
        "target_document": "[Project]-Finance-Analytics-eDA.pdf"
    },
    {
        "id": "eval_fin_05",
        "question": "Khi kết nối Power BI với PostgreSQL gặp lỗi could not connect hoặc timeout thì nguyên nhân xuất phát từ đâu?",
        "reference_answer": "Nguyên nhân do máy chủ cơ sở dữ liệu chưa được mở hoặc do mạng đang sử dụng chặn cổng kết nối 5432 của PostgreSQL. Cần kiểm tra trạng thái hoạt động của máy chủ và kiểm tra xem tường lửa mạng có chặn cổng 5432 hay không.",
        "category": "Data Integration",
        "target_document": "[Project]-Finance-Analytics-eDA.pdf"
    },
    {
        "id": "eval_fin_06",
        "question": "Trong quy trình dựng báo cáo Power BI, hộp thoại kết nối PostgreSQL cần chọn chế độ Data Connectivity mode là gì và hai tham số cần tạo là gì?",
        "reference_answer": "Hộp thoại kết nối PostgreSQL cần chọn Data Connectivity mode là Import. Hai tham số cần tạo trong ngăn Queries là PgServer và PgDatabase với kiểu dữ liệu Text.",
        "category": "Power BI Config",
        "target_document": "[Project]-Finance-Analytics-eDA.pdf"
    },
    {
        "id": "eval_fin_07",
        "question": "Mười lăm bảng nạp dữ liệu bằng truy vấn SQL từ PostgreSQL vào Power BI được phân thành các nhóm số dòng như thế nào?",
        "reference_answer": "15 bảng nạp từ SQL được chia làm 4 nhóm: Nhóm thứ nhất gồm 9 bảng có đúng 49 dòng (mỗi bảng trả về một dòng cho mỗi tháng); Nhóm thứ hai gồm 3 bảng có số dòng là bội của 49 (A1 có 98 dòng, A3 có 441 dòng, B3 có 98 dòng); Nhóm thứ ba gồm A4_CoCauPhanKhuc có 353 dòng; Nhóm còn lại gồm B5 có 31 dòng và B4 có 41 dòng.",
        "category": "Data Modeling",
        "target_document": "[Project]-Finance-Analytics-eDA.pdf"
    },
    {
        "id": "eval_fin_08",
        "question": "Tại sao trong mô hình báo cáo Power BI cần thiết lập Sort by column cho tám cột và dựng các bảng bằng DAX?",
        "reference_answer": "Cần thiết lập Sort by column cho 8 cột để đảm bảo các trường dữ liệu hiển thị theo đúng thứ tự logic thời gian và phân loại thay vì thứ tự bảng chữ cái abc mặc định. Các bảng dựng bằng DAX (như DimMonth) giúp hoàn thiện mô hình dữ liệu sao (Star Schema) và phục vụ tính toán các measure phân tích tài chính.",
        "category": "DAX & Modeling",
        "target_document": "[Project]-Finance-Analytics-eDA.pdf"
    },
    {
        "id": "eval_fin_09",
        "question": "Bảng B6_LeakageD30 trong dự án phân tích tiền gửi tại AIV Bank phản ánh điều gì?",
        "reference_answer": "Bảng B6_LeakageD30 phản ánh dòng tiền rò rỉ (leakage) và biến động số dư tiền gửi của khách hàng sau khi sổ tiết kiệm đáo hạn trong khung thời gian 30 ngày (D30), phục vụ đánh giá mức độ giữ chân nguồn vốn và rủi ro thanh khoản.",
        "category": "Liquidity Analytics",
        "target_document": "[Project]-Finance-Analytics-eDA.pdf"
    },
    {
        "id": "eval_fin_10",
        "question": "Tại sao sự tập trung nguồn vốn vào một số ít khách hàng lớn hoặc nhiều khoản tiền gửi cùng đáo hạn ngắn hạn lại gây rủi ro cho ngân hàng?",
        "reference_answer": "Nếu nguồn vốn tập trung vào một số khách hàng lớn thì khi các khách hàng này rút tiền ngân hàng sẽ chịu tác động thanh khoản rất lớn. Tương tự, nếu nhiều khoản tiền gửi cùng đến hạn trong thời gian ngắn sẽ làm tăng nhu cầu huy động thay thế khẩn cấp và gia tăng rủi ro thanh khoản, buộc ngân hàng phải theo dõi chặt chẽ lịch đáo hạn và tỷ lệ tái tục.",
        "category": "Risk Management",
        "target_document": "[Project]-Finance-Analytics-eDA.pdf"
    }
]

def main():
    print("=== THIẾT LẬP BỘ ĐÁNH GIÁ CHO TÀI LIỆU FINANCE ANALYTICS 140 TRANG ===")
    assert TARGET_PDF.exists(), f"Không tìm thấy tệp {TARGET_PDF}"
    
    with open(TESTSET_PATH, "w", encoding="utf-8") as f:
        json.dump(FINANCE_TESTSET, f, ensure_ascii=False, indent=2)
    print(f"-> Đã ghi 10 mẫu kiểm chuẩn mới vào: {TESTSET_PATH}")
    print("=== HOÀN TẤT THIẾT LẬP ===")

if __name__ == "__main__":
    main()
