"""
Styles & UI Components for Streamlit Web UI -- Phase 4
Cung cap Custom CSS va HTML renderers hien dai, tao trai nghiem tuong tu NotebookLM.
"""

import streamlit as st


CUSTOM_CSS = """
<style>
/* ============================================================
   GLOBAL STYLES & TYPOGRAPHY
   ============================================================ */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* App Header styling */
.main-header {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: white;
    padding: 1.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.25);
}
.main-header h1 {
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
    color: white !important;
}
.main-header p {
    font-size: 1rem;
    opacity: 0.9;
    margin: 0.5rem 0 0 0;
}

/* ============================================================
   CARD CONTAINERS & METRICS
   ============================================================ */
.custom-card {
    background: var(--background-color, #ffffff);
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 14px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.custom-card:hover {
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
}

/* Citation inline badge */
.citation-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white !important;
    font-weight: 600;
    font-size: 0.75rem;
    padding: 2px 7px;
    border-radius: 6px;
    margin: 0 3px;
    vertical-align: middle;
    box-shadow: 0 2px 5px rgba(99, 102, 241, 0.3);
}

/* Status indicator badge */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 20px;
}
.status-online {
    background-color: rgba(34, 197, 94, 0.15);
    color: #16a34a;
}
.status-offline {
    background-color: rgba(239, 68, 68, 0.15);
    color: #dc2626;
}
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}
.dot-online { background-color: #16a34a; }
.dot-offline { background-color: #dc2626; }

/* ============================================================
   QUIZ COMPONENT STYLES
   ============================================================ */
.quiz-box {
    border-left: 4px solid #6366f1;
    background: rgba(99, 102, 241, 0.03);
    padding: 1rem 1.25rem;
    border-radius: 0 12px 12px 0;
    margin-bottom: 1.25rem;
}
.quiz-correct {
    border-left-color: #22c55e !important;
    background: rgba(34, 197, 94, 0.08) !important;
}
.quiz-wrong {
    border-left-color: #ef4444 !important;
    background: rgba(239, 68, 68, 0.08) !important;
}

/* ============================================================
   FLASHCARD COMPONENT STYLES
   ============================================================ */
.flashcard-container {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    color: white;
    min-height: 250px;
    border-radius: 20px;
    padding: 3rem 2rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    box-shadow: 0 12px 28px -6px rgba(49, 46, 129, 0.4);
    margin: 1.5rem 0 1rem 0;
    border: 1px solid rgba(255, 255, 255, 0.12);
    cursor: pointer;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    user-select: none;
}
.flashcard-container:hover {
    transform: translateY(-4px);
    box-shadow: 0 18px 36px -4px rgba(79, 70, 229, 0.5);
    border-color: rgba(165, 180, 252, 0.4);
}
.flashcard-container:active {
    transform: translateY(-1px) scale(0.995);
}
.flashcard-tag {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #a5b4fc;
    font-weight: 700;
    margin-bottom: 0.75rem;
}
.flashcard-text {
    font-size: 1.35rem;
    font-weight: 600;
    line-height: 1.6;
    margin: 1rem 0;
}
.flashcard-back-text {
    font-size: 1.15rem;
    font-weight: 400;
    line-height: 1.6;
    color: #e0e7ff;
    margin: 1rem 0;
}
.flashcard-hint {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 1rem;
    padding-top: 0.6rem;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    width: 100%;
}

/* Streamlit Tabs Enhancements */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    padding: 10px 20px;
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background-color: rgba(99, 102, 241, 0.12) !important;
    color: #4f46e5 !important;
}
</style>
"""


def inject_custom_styles():
    """Tiem doan CSS tuy chinh vao giao dien Streamlit."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header(title: str, subtitle: str):
    """In tieu de header banner dep mat."""
    html = f"""
    <div class="main-header">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_status_pill(is_online: bool) -> str:
    """Tra ve doan HTML hien thi trang thai online/offline."""
    if is_online:
        return """
        <div class="status-pill status-online">
            <span class="status-dot dot-online"></span>
            <span>API Online</span>
        </div>
        """
    else:
        return """
        <div class="status-pill status-offline">
            <span class="status-dot dot-offline"></span>
            <span>API Offline</span>
        </div>
        """


def render_flashcard_card(front: str, back: str, is_flipped: bool, index: int, total: int):
    """Render the flashcard voi giao dien the gradient hien dai, nhap chuot de lat."""
    tag = f"MẶT SAU — ĐÁP ÁN (THẺ {index}/{total})" if is_flipped else f"MẶT TRƯỚC — KHÁI NIỆM (THẺ {index}/{total})"
    content = back if is_flipped else front
    text_class = "flashcard-back-text" if is_flipped else "flashcard-text"
    flip_hint = "🔄 Nhấp vào thẻ để lật lại mặt trước" if is_flipped else "🔄 Nhấp vào thẻ để xem đáp án"

    html = f"""
    <div id="flashcard-active-card" class="flashcard-container">
        <div class="flashcard-tag">{tag}</div>
        <div class="{text_class}">{content}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
