import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Đồng bộ hóa st.secrets sang os.environ khi chạy trên Streamlit Cloud
try:
    import streamlit as _st
    if hasattr(_st, "secrets"):
        for _k, _v in _st.secrets.items():
            if isinstance(_v, str):
                os.environ.setdefault(_k, _v)
                os.environ.setdefault(_k.upper(), _v)
except Exception:
    pass


class Settings(BaseSettings):
    # Embedding
    embedding_model: str = Field(default="intfloat/multilingual-e5-large")
    embedding_device: str = Field(default="cuda")
    hf_token: str = Field(default="")              # HuggingFace token cho gated models

    # Qdrant Database
    qdrant_path: str = Field(default="./qdrant_storage")
    collection_name: str = Field(default="documents")

    # Chunking
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=150)

    # LLM Backend (gemini | hf_local | vllm)
    llm_backend: str = Field(default="gemini")
    gemini_api_key: str = Field(default="")
    gemini_model: str = Field(default="gemini-2.5-flash-lite")
    hf_model_id: str = Field(default="")           # cho hf_local backend
    vllm_base_url: str = Field(default="")          # cho vllm backend

    # RAG Retrieval
    top_k: int = Field(default=5)                   # chunks trả về cuối cùng
    initial_k: int = Field(default=15)              # lấy ra trước khi rerank
    rerank_k: int = Field(default=5)                # giữ lại sau rerank
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3")
    reranker_device: str = Field(default="cuda")

    # Learning Features (Phase 3)
    summarize_batch_size: int = Field(default=10)   # chunks mỗi batch khi Map-Reduce
    n_quiz_questions: int = Field(default=8)         # số câu hỏi quiz
    n_flashcards: int = Field(default=15)            # số flashcards

    # Đọc cấu hình từ file .env
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
