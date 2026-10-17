# UI demo độc lập

Demo HTML/CSS/JavaScript cho API hiện có. Không cần npm, không thay đổi Streamlit, pipeline hay dữ liệu Qdrant.

Chạy từ thư mục gốc dự án:

```powershell
python -m http.server 5173 --bind 127.0.0.1 --directory demo-ui
```

- Mở <http://127.0.0.1:5173/> để kết nối trực tiếp backend, cùng địa chỉ mặc định như Streamlit: `http://127.0.0.1:8000`. Đổi địa chỉ bằng nút trạng thái kết nối trên đầu trang.
- UI luôn dùng API thật, không có chế độ dữ liệu mẫu hay tự thay câu trả lời khi API lỗi. Liên kết cũ có `?sample=1` cũng mở API thật.
- Backend dùng cách chạy hiện có: `python -m uvicorn src.interfaces.api:app --port 8000` (cần môi trường đã cài đủ requirements của backend).

Demo nối các endpoint `/health`, `/documents`, `/upload`, `/ask`, `/summarize`, `/quiz`, `/flashcards`. Hỏi đáp nhận một tài liệu hoặc toàn bộ thư viện; ba công cụ học tập nhận một tài liệu theo hợp đồng API hiện tại. Upload nhiều file theo thứ tự, không ép lập chỉ mục lại tài liệu trùng. API chưa báo tiến độ trung gian nên UI hiển thị chung “Đang tải lên và lập chỉ mục”, không tạo phần trăm giả.

Để sinh nội dung, backend cần API key LLM và các thư viện/model đã được cấu hình như khi dùng Streamlit. Khi thiếu cấu hình hoặc backend trả lỗi, UI hiển thị lỗi thật và giữ câu hỏi để thử lại. UI không nhận hay lưu API key LLM; khóa tiếp tục nằm trong cấu hình backend.

Tin nhắn và kết quả chỉ tồn tại trong phiên trang, tách theo tài liệu và kết nối. Reload sẽ xóa phiên; địa chỉ API được nhớ trong localStorage. Backend hiện xử lý từng câu hỏi độc lập, demo không giả lập khả năng nhớ lịch sử của API. POST không tự retry. Không có chức năng xóa tài liệu trong demo.

Enter gửi câu hỏi, Shift+Enter xuống dòng. Ở bộ thẻ, ←/→ chuyển và Space lật khi không nhập liệu; nút lật cũng dùng được bằng bàn phím. Bấm trích dẫn để đọc đoạn nguồn trong panel bên phải.

Các tệp: `index.html` (khung), `styles.css` (token và giao diện), `app.mjs` (tương tác), `api.mjs` (API client), `sample.mjs` (fixture chỉ dành cho kiểm thử, không được UI import). Font Inter có fallback hệ thống khi không có mạng. Không có script CDN.

Kiểm tra adapter và dữ liệu mẫu bằng Node.js 20+ (chỉ phục vụ kiểm thử, không cần để chạy UI):

```powershell
node --test demo-ui/api.test.mjs
```

## Bản full stack trên Vercel

Bản public tại <https://rag-learning-system-ui.vercel.app/> dùng UI này cùng backend
FastAPI cloud dưới `/api`, Gemini API và Qdrant Cloud. Project build từ **gốc repository**,
preset **FastAPI**, production branch **`quan/ui`**. Xem [hướng dẫn cloud](../CLOUD_DEPLOYMENT.md).
UI giữ nguyên bố cục và bốn công cụ. Upload tối đa 4.000.000 byte/file; API key chỉ nằm ở backend.

## Publish riêng UI trên Vercel (tùy chọn)

Có thể tiếp tục publish riêng thư mục này khi muốn kết nối một backend HTTPS khác.

Chọn thư mục gốc dự án Vercel là `demo-ui`, preset **Other**, build command `node build.mjs`, output directory `public`. Cấu hình đã có trong `vercel.json`. CLI có thể triển khai bằng `vercel --prod --yes` khi đứng trong `demo-ui` và đã đăng nhập Vercel.

Build chỉ xuất 4 tệp giao diện chạy thật; fixture, test, backend và dữ liệu không nằm trong bundle. Thư mục build `demo-ui/public/` và metadata `.vercel/` không được commit.

Bản public cần backend HTTPS riêng. Nếu chưa có, UI vẫn xuất bản được và hiển thị nút **Kết nối backend**; không tự gọi localhost của người truy cập. Khi có backend, nhập URL trong UI hoặc đặt biến build `PUBLIC_API_URL=https://...` trên Vercel rồi redeploy để áp dụng mặc định cho mọi người. Backend phải cho phép origin Vercel qua CORS. API key LLM vẫn cấu hình ở backend, không đặt vào `PUBLIC_API_URL` hay mã frontend.

Chạy UI trên localhost vẫn tự nối `http://127.0.0.1:8000` như trước.
