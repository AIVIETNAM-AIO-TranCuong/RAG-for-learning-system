"""
Streamlit Web UI -- Phase 4
Giao dien web hien dai, tuong tac voi RAG Learning System thong qua FastAPI REST API.

Chay ung dung:
  streamlit run src/interfaces/ui.py
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import httpx
import streamlit as st
import streamlit.components.v1 as components

# Dam bao import duoc src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.interfaces.styles import (
    inject_custom_styles,
    render_header,
    render_status_pill,
    render_flashcard_card,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG Learning System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_styles()


# ============================================================
# API CLIENT
# ============================================================

class ApiClient:
    """HTTP Client de giao tiep voi FastAPI server."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base_url, timeout=120.0)

    def check_health(self) -> bool:
        try:
            resp = self.client.get("/health", timeout=10.0)
            return resp.status_code == 200
        except Exception:
            return False

    def _handle(self, resp: httpx.Response) -> Dict[str, Any]:
        try:
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            try:
                detail = resp.json().get("detail", str(e))
                raise RuntimeError(detail) from None
            except Exception:
                raise RuntimeError(str(e)) from None

    def list_documents(self) -> List[Dict[str, Any]]:
        resp = self.client.get("/documents")
        return self._handle(resp)

    def upload_document(self, file_bytes: bytes, filename: str, force: bool = False) -> Dict[str, Any]:
        files = {"file": (filename, file_bytes)}
        data = {"force_reindex": str(force).lower()}
        resp = self.client.post("/upload", files=files, data=data, timeout=300.0)
        return self._handle(resp)

    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        resp = self.client.delete(f"/documents/{doc_id}")
        return self._handle(resp)

    def ask(self, question: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
        payload = {"question": question, "document_id": doc_id}
        resp = self.client.post("/ask", json=payload)
        return self._handle(resp)

    def summarize(self, doc_id: str) -> Dict[str, Any]:
        payload = {"document_id": doc_id}
        resp = self.client.post("/summarize", json=payload)
        return self._handle(resp)

    def create_quiz(self, doc_id: str, topic: Optional[str] = None, n_questions: int = 8) -> Dict[str, Any]:
        payload = {"document_id": doc_id, "topic": topic, "n_questions": n_questions}
        resp = self.client.post("/quiz", json=payload)
        return self._handle(resp)

    def create_flashcards(self, doc_id: str, topic: Optional[str] = None, n_cards: int = 15) -> Dict[str, Any]:
        payload = {"document_id": doc_id, "topic": topic, "n_cards": n_cards}
        resp = self.client.post("/flashcards", json=payload)
        return self._handle(resp)


class LocalPipelineClient:
    """Client chạy trực tiếp pipelines trong cùng tiến trình (dành cho Streamlit Cloud hoặc Standalone)."""

    def check_health(self) -> bool:
        return True

    def list_documents(self) -> List[Dict[str, Any]]:
        from src.db.store import get_store
        return get_store().list_documents()

    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        from src.db.store import get_store
        get_store().delete_document(doc_id)
        return {"status": "deleted", "document_id": doc_id}

    def upload_document(self, file_bytes: bytes, filename: str, force: bool = True) -> Dict[str, Any]:
        import tempfile
        from src.pipelines.indexing import index_document
        from src.db.store import get_store

        temp_dir = tempfile.mkdtemp(prefix="rag_upload_")
        temp_path = Path(temp_dir) / filename
        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        doc_id = index_document(temp_path, force_reindex=force)
        docs = get_store().list_documents()
        chunks = 0
        for d in docs:
            if d.get("document_id") == doc_id:
                chunks = d.get("total_chunks", 0)
                break
        return {"document_id": doc_id, "filename": filename, "total_chunks": chunks}

    def ask(self, question: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
        from src.pipelines.rag import answer
        res = answer(query=question, document_id=doc_id)
        return res.model_dump()

    def summarize(self, doc_id: str) -> Dict[str, Any]:
        from src.pipelines.learning import summarize
        res = summarize(document_id=doc_id)
        return res.model_dump()

    def create_quiz(self, doc_id: str, topic: Optional[str] = None, n_questions: int = 8) -> Dict[str, Any]:
        from src.pipelines.learning import generate_quiz
        res = generate_quiz(document_id=doc_id, topic=topic, n_questions=n_questions)
        return res.model_dump()

    def create_flashcards(self, doc_id: str, topic: Optional[str] = None, n_cards: int = 15) -> Dict[str, Any]:
        from src.pipelines.learning import generate_flashcards
        res = generate_flashcards(document_id=doc_id, topic=topic, n_cards=n_cards)
        return res.model_dump()


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "api_url" not in st.session_state:
    st.session_state.api_url = "http://127.0.0.1:8000"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

if "flashcards_data" not in st.session_state:
    st.session_state.flashcards_data = []

if "fc_index" not in st.session_state:
    st.session_state.fc_index = 0

if "fc_flipped" not in st.session_state:
    st.session_state.fc_flipped = False

if "summary_data" not in st.session_state:
    st.session_state.summary_data = None


# Kiểm tra chế độ kết nối: API Server hoặc Standalone (cho Streamlit Cloud)
remote_api = ApiClient(base_url=st.session_state.api_url)
is_server_online = remote_api.check_health()
is_local_host = "127.0.0.1" in st.session_state.api_url or "localhost" in st.session_state.api_url

if is_server_online:
    api = remote_api
    is_online = True
    connection_mode = "FastAPI Backend"
elif is_local_host:
    api = remote_api
    is_online = False
    connection_mode = "Server Offline"
else:
    api = LocalPipelineClient()
    is_online = True
    connection_mode = "Cloud Standalone"


# ============================================================
# SIDEBAR: DOCUMENT & SERVER MANAGEMENT
# ============================================================

with st.sidebar:
    st.title("⚙️ Thiết lập Hệ thống")
    col_pill, col_ref = st.columns([2, 1])
    with col_pill:
        st.markdown(render_status_pill(is_online), unsafe_allow_html=True)
    with col_ref:
        if st.button("🔄 Thử lại", help="Kiểm tra lại kết nối"):
            st.rerun()

    st.caption(f"Chế độ: **{connection_mode}**")

    # Cau hinh URL API (tuy chon neu muon tro ve backend rieng)
    with st.expander("🌐 Kết nối API"):
        new_url = st.text_input("FastAPI Server URL", value=st.session_state.api_url)
        if new_url != st.session_state.api_url:
            st.session_state.api_url = new_url
            st.rerun()
        if is_server_online:
            st.success("🟢 Đang kết nối tới server này")
        else:
            st.info("ℹ️ Đang chạy chế độ Trực tiếp nội bộ (Standalone/Cloud).")

    st.divider()

    # Section: Upload tai lieu
    st.subheader("📤 Nạp tài liệu mới")
    uploaded_file = st.file_uploader(
        "Chọn file tài liệu:",
        type=["pdf", "docx", "pptx", "txt", "md"],
    )

    if uploaded_file and st.button("🚀 Nạp vào hệ thống", use_container_width=True):
        if not is_online:
            st.error("Server API đang offline, không thể nạp tài liệu.")
        else:
            with st.spinner("Đang nạp tài liệu"):
                try:
                    res = api.upload_document(
                        file_bytes=uploaded_file.getvalue(),
                        filename=uploaded_file.name,
                        force=True,
                    )
                    st.success(f"Nạp thành công! ID: `{res['document_id'][:12]}...` ({res['total_chunks']} chunks)")
                    st.rerun()
                except Exception as e:
                    st.error(f"Lỗi nạp tài liệu: {str(e)}")

    st.divider()

    # Section: Chon tai lieu lam viec
    st.subheader("📚 Tài liệu đã nạp")
    documents = []
    if is_online:
        try:
            documents = api.list_documents()
        except Exception:
            documents = []

    doc_options = {"all": "🌐 Tất cả tài liệu"}
    doc_map = {}
    for d in documents:
        doc_id = d.get("document_id", "")
        fn = d.get("filename", "unknown")
        ext = Path(fn).suffix.lower()
        if ext == ".pdf":
            icon = "📕"
        elif ext == ".docx":
            icon = "📘"
        elif ext == ".pptx":
            icon = "📙"
        elif ext in {".txt", ".md"}:
            icon = "📝"
        else:
            icon = "📄"
        label = f"{icon} {fn}"
        doc_options[doc_id] = label
        doc_map[doc_id] = d

    selected_doc_id = st.selectbox(
        "Tài liệu đang chọn:",
        options=list(doc_options.keys()),
        format_func=lambda x: doc_options.get(x, x),
    )

    # Thao tac xoa tai lieu
    if selected_doc_id != "all":
        info = doc_map.get(selected_doc_id, {})
        st.caption(f"**ID:** `{selected_doc_id}`")
        st.caption(f"**Tổng số chunks:** {info.get('total_chunks', 0)}")
        st.caption(f"**Số trang:** {info.get('max_page', 0)}")

        if st.button("🗑️ Xóa tài liệu này", type="secondary", use_container_width=True):
            try:
                api.delete_document(selected_doc_id)
                st.success("Đã xóa tài liệu khỏi Qdrant.")
                st.rerun()
            except Exception as e:
                st.error(f"Lỗi xóa: {e}")


# ============================================================
# MAIN CONTENT AREA
# ============================================================

render_header(
    title="🧠 RAG Learning System",
    subtitle="Trợ lý học tập thông minh dựa trên tài liệu: Hỏi đáp trích dẫn, Tóm tắt, Tạo Quiz & Flashcards",
)

tab_qa, tab_summary, tab_quiz, tab_flashcards = st.tabs([
    "💬 Hỏi đáp (Q&A)",
    "📝 Tóm tắt (Summary)",
    "🎯 Trắc nghiệm (Quiz)",
    "🗂️ Thẻ ghi nhớ (Flashcards)",
])


# ============================================================
# TAB 1: HOI DAP (Q&A)
# ============================================================

with tab_qa:
    st.subheader("💬 Hỏi đáp tri thức từ tài liệu")
    target_text = doc_options.get(selected_doc_id, "")
    st.caption(f"Ngữ cảnh hiện tại: **{target_text}**")

    # Form dat cau hoi
    with st.form("qa_form", clear_on_submit=False):
        user_question = st.text_input(
            "Nhập câu hỏi của bạn:",
            placeholder="Ví dụ: Tóm tắt các đặc điểm nổi bật của dự án...",
        )
        submit_qa = st.form_submit_button("🔍 Gửi câu hỏi", use_container_width=True)

    if submit_qa:
        if not user_question.strip():
            st.warning("Vui lòng nhập nội dung câu hỏi.")
        elif not is_online:
            st.error("Server API đang offline. Vui lòng bật FastAPI server.")
        else:
            with st.spinner("Đang tìm kiếm tài liệu, rerank và tổng hợp câu trả lời..."):
                try:
                    doc_filter = None if selected_doc_id == "all" else selected_doc_id
                    ans_data = api.ask(question=user_question.strip(), doc_id=doc_filter)
                    st.session_state.chat_history.insert(0, {
                        "question": user_question.strip(),
                        "answer": ans_data.get("answer", ""),
                        "citations": ans_data.get("citations", []),
                        "doc_label": target_text,
                    })
                except Exception as e:
                    st.error(f"Lỗi khi trả lời: {str(e)}")

    # Hien thi lich su hoi dap
    if st.session_state.chat_history:
        for idx, item in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"### ❓ {item['question']}")
                st.caption(f"Ngữ cảnh: {item['doc_label']}")
                st.markdown(item["answer"])

                # Hien thi Accordion chi tiet citations
                citations = item.get("citations", [])
                if citations:
                    with st.expander(f"📖 Nguồn trích dẫn ({len(citations)} trích đoạn)"):
                        for c in citations:
                            marker = c.get("marker", "S")
                            chunk_data = c.get("chunk", {})
                            meta = chunk_data.get("metadata", {})
                            fn = meta.get("filename", "unknown")
                            page = meta.get("page", 0)
                            content = chunk_data.get("content", "")
                            score = chunk_data.get("score", 0.0)

                            st.markdown(
                                f"<span class='citation-badge'>[{marker}]</span> "
                                f"**{fn}** (Trang {page})",
                                unsafe_allow_html=True,
                            )
                            st.info(content)
                st.divider()
    else:
        st.info("Chưa có câu hỏi nào. Hãy nhập câu hỏi đầu tiên ở trên!")


# ============================================================
# TAB 2: TOM TAT (SUMMARY)
# ============================================================

with tab_summary:
    st.subheader("📝 Tóm tắt nội dung tài liệu")

    if selected_doc_id == "all":
        st.warning("Vui lòng chọn một tài liệu cụ thể ở cột bên trái để thực hiện tóm tắt.")
    else:
        st.caption(f"Đang chọn: **{doc_options.get(selected_doc_id, '')}**")

        col_btn, _ = st.columns([1, 2])
        with col_btn:
            generate_sum_btn = st.button("📑 Tạo bản tóm tắt", use_container_width=True)

        if generate_sum_btn:
            if not is_online:
                st.error("Server API đang offline.")
            else:
                with st.spinner("Đang phân tích và tóm tắt"):
                    try:
                        summary_res = api.summarize(selected_doc_id)
                        st.session_state.summary_data = summary_res
                    except Exception as e:
                        st.error(f"Lỗi tóm tắt: {str(e)}")

        # Hien thi ket qua tom tat
        if st.session_state.summary_data:
            s = st.session_state.summary_data
            st.markdown("### 📄 Nội dung bản tóm tắt:")
            st.markdown(s.get("content", ""))


# ============================================================
# TAB 3: TRAC NGHIEM (QUIZ)
# ============================================================

with tab_quiz:
    st.subheader("🎯 Tạo câu hỏi trắc nghiệm & Kiểm tra kiến thức")

    if selected_doc_id == "all":
        st.warning("Vui lòng chọn một tài liệu cụ thể ở cột bên trái để tạo đề trắc nghiệm.")
    else:
        col_q1, col_q2, col_q3 = st.columns([2, 1, 1])
        with col_q1:
            quiz_topic = st.text_input("Chủ đề tập trung (tùy chọn)", placeholder="Để trống nếu muốn tạo từ toàn bộ")
        with col_q2:
            quiz_num = st.slider("Số lượng câu hỏi", min_value=1, max_value=15, value=5)
        with col_q3:
            st.write("")
            st.write("")
            btn_create_quiz = st.button("🎲 Tạo đề thi", use_container_width=True)

        if btn_create_quiz:
            if not is_online:
                st.error("Server API đang offline.")
            else:
                with st.spinner(f"Đang sinh {quiz_num} câu hỏi trắc nghiệm..."):
                    try:
                        topic_param = quiz_topic.strip() if quiz_topic.strip() else None
                        q_res = api.create_quiz(
                            doc_id=selected_doc_id,
                            topic=topic_param,
                            n_questions=quiz_num,
                        )
                        st.session_state.quiz_data = q_res.get("items", [])
                        st.session_state.quiz_submitted = False
                        st.session_state.user_answers = {}
                        st.rerun()
                    except Exception as e:
                        st.error(f"Lỗi tạo quiz: {str(e)}")

        # Giao dien lam bai thi
        if st.session_state.quiz_data:
            st.divider()
            st.markdown("### 📝 Bài thi trắc nghiệm tương tác")

            if "user_answers" not in st.session_state:
                st.session_state.user_answers = {}

            with st.form("quiz_submission_form"):
                for idx, item in enumerate(st.session_state.quiz_data):
                    q_text = item.get("question", "")
                    options = item.get("options", [])
                    st.markdown(f"**Câu {idx + 1}: {q_text}**")

                    selected = st.radio(
                        f"Lựa chọn cho câu {idx + 1}:",
                        options=options,
                        key=f"quiz_q_{idx}",
                        label_visibility="collapsed",
                    )
                    # Trich xuat ky tu A/B/C/D
                    st.session_state.user_answers[idx] = selected[:1].upper() if selected else ""
                    st.write("")

                submit_quiz_btn = st.form_submit_button("🏁 Nộp bài & Chấm điểm", use_container_width=True)

            if submit_quiz_btn:
                st.session_state.quiz_submitted = True

            # Hien thi ket qua cham diem
            if st.session_state.quiz_submitted:
                correct_count = 0
                total_q = len(st.session_state.quiz_data)

                for idx, item in enumerate(st.session_state.quiz_data):
                    user_ans = st.session_state.user_answers.get(idx, "")
                    real_ans = item.get("answer", "")
                    if user_ans == real_ans:
                        correct_count += 1

                pct = (correct_count / total_q) * 100
                st.divider()
                st.markdown(f"## 🏆 Kết quả: **{correct_count} / {total_q} câu đúng ({pct:.1f}%)**")

                if pct >= 80:
                    st.balloons()
                    st.success("🌟 Xuất sắc! Bạn đã nắm rất vững kiến thức tài liệu.")
                elif pct >= 50:
                    st.info("👍 Khá tốt! Hãy ôn lại các câu trả lời sai nhé.")
                else:
                    st.warning("⚠️ Cần cố gắng ôn tập thêm tài liệu.")

                # Review chi tiet
                st.markdown("### 🔍 Đáp án chi tiết & Giải thích:")
                for idx, item in enumerate(st.session_state.quiz_data):
                    user_ans = st.session_state.user_answers.get(idx, "")
                    real_ans = item.get("answer", "")
                    is_correct = (user_ans == real_ans)

                    color_box = "quiz-correct" if is_correct else "quiz-wrong"
                    icon = "✅" if is_correct else "❌"

                    st.markdown(f"""
                    <div class="quiz-box {color_box}">
                        <b>{icon} Câu {idx + 1}: {item.get('question')}</b><br>
                        - Bạn chọn: <b>{user_ans}</b><br>
                        - Đáp án đúng: <b>{real_ans}</b><br>
                        <small><em>Giải thích: {item.get('explanation', '')}</em></small>
                    </div>
                    """, unsafe_allow_html=True)


# ============================================================
# TAB 4: THE GHI NHO (FLASHCARDS)
# ============================================================

with tab_flashcards:
    st.subheader("🗂️ Ôn tập với Thẻ ghi nhớ (Flashcards)")

    if selected_doc_id == "all":
        st.warning("Vui lòng chọn một tài liệu cụ thể ở cột bên trái để tạo Flashcards.")
    else:
        col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
        with col_f1:
            fc_topic = st.text_input("Chủ đề thẻ (tùy chọn)", placeholder="Ví dụ: Định nghĩa, Mô hình kinh doanh...", key="fc_topic_input")
        with col_f2:
            fc_num = st.slider("Số lượng thẻ", min_value=1, max_value=20, value=8, key="fc_slider")
        with col_f3:
            st.write("")
            st.write("")
            btn_create_fc = st.button("🎴 Tạo Flashcards", use_container_width=True)

        if btn_create_fc:
            if not is_online:
                st.error("Server API đang offline.")
            else:
                with st.spinner(f"Đang sinh {fc_num} thẻ ghi nhớ..."):
                    try:
                        t_param = fc_topic.strip() if fc_topic.strip() else None
                        fc_res = api.create_flashcards(
                            doc_id=selected_doc_id,
                            topic=t_param,
                            n_cards=fc_num,
                        )
                        st.session_state.flashcards_data = fc_res.get("cards", [])
                        st.session_state.fc_index = 0
                        st.session_state.fc_flipped = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"Lỗi tạo flashcards: {str(e)}")

        # Hien thi the tuong tac Flip-card
        cards = st.session_state.flashcards_data
        if cards:
            total_cards = len(cards)
            curr_idx = st.session_state.fc_index
            curr_card = cards[curr_idx]

            # Render the gradient to ro (nhap chuot truc tiep vao the de lat)
            render_flashcard_card(
                front=curr_card.get("front", ""),
                back=curr_card.get("back", ""),
                is_flipped=st.session_state.fc_flipped,
                index=curr_idx + 1,
                total=total_cards,
            )

            # Lang nghe su kien: Nhap chuot truc tiep vao the de lat + Phim tat ban phim
            components.html(
                """
                <script>
                (function() {
                    const parentDoc = window.parent.document;

                    function findButton(text) {
                        const btns = parentDoc.querySelectorAll('button');
                        for (let b of btns) {
                            if (b.innerText && b.innerText.includes(text) && !b.disabled) {
                                return b;
                            }
                        }
                        return null;
                    }

                    // 1. Lang nghe click truc tiep tren the Flashcard
                    if (window.parent.__fcClickHandler) {
                        parentDoc.removeEventListener('click', window.parent.__fcClickHandler);
                    }
                    window.parent.__fcClickHandler = function(e) {
                        const card = e.target.closest('.flashcard-container');
                        if (card) {
                            const btn = findButton('Lật');
                            if (btn) {
                                btn.click();
                            }
                        }
                    };
                    parentDoc.addEventListener('click', window.parent.__fcClickHandler);

                    // 2. Lang nghe phim tat ban phim (Mui ten trai, phai, space, enter)
                    if (window.parent.__fcKeyHandler) {
                        parentDoc.removeEventListener('keydown', window.parent.__fcKeyHandler);
                    }
                    window.parent.__fcKeyHandler = function(e) {
                        const activeTag = parentDoc.activeElement ? parentDoc.activeElement.tagName.toLowerCase() : '';
                        if (activeTag === 'input' || activeTag === 'textarea') {
                            return;
                        }

                        if (e.key === 'ArrowLeft') {
                            const btn = findButton('Thẻ trước');
                            if (btn) {
                                e.preventDefault();
                                btn.click();
                            }
                        } else if (e.key === 'ArrowRight') {
                            const btn = findButton('Thẻ tiếp');
                            if (btn) {
                                e.preventDefault();
                                btn.click();
                            }
                        } else if (e.key === ' ' || e.key === 'Enter' || e.key === 'ArrowUp' || e.key === 'ArrowDown') {
                            const btn = findButton('Lật');
                            if (btn) {
                                e.preventDefault();
                                btn.click();
                            }
                        }
                    };
                    parentDoc.addEventListener('keydown', window.parent.__fcKeyHandler);
                })();
                </script>
                """,
                height=0,
                width=0,
            )

            # Controls dieu huong
            col_c1, col_c2, col_c3, col_c4 = st.columns([1, 2, 1, 1])

            with col_c1:
                if st.button("⬅️ Thẻ trước", disabled=(curr_idx == 0), use_container_width=True):
                    st.session_state.fc_index -= 1
                    st.session_state.fc_flipped = False
                    st.rerun()

            with col_c2:
                flip_label = "🔄 Lật mặt trước" if st.session_state.fc_flipped else "🔄 Lật mặt sau (Xem giải thích)"
                if st.button(flip_label, use_container_width=True, type="primary"):
                    st.session_state.fc_flipped = not st.session_state.fc_flipped
                    st.rerun()

            with col_c3:
                if st.button("Thẻ tiếp ➡️", disabled=(curr_idx >= total_cards - 1), use_container_width=True):
                    st.session_state.fc_index += 1
                    st.session_state.fc_flipped = False
                    st.rerun()

            with col_c4:
                import random
                if st.button("🔀 Xáo trộn", use_container_width=True):
                    random.shuffle(st.session_state.flashcards_data)
                    st.session_state.fc_index = 0
                    st.session_state.fc_flipped = False
                    st.rerun()

            # Danh sach toan bo the dang bang
            with st.expander(f"📋 Xem toàn bộ danh sách {total_cards} thẻ"):
                for i, c in enumerate(cards, start=1):
                    st.markdown(f"**{i}. {c.get('front')}**")
                    st.caption(f"👉 {c.get('back')}")
                    st.write("")
