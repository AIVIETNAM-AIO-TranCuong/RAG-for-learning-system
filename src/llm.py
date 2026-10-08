from abc import ABC, abstractmethod
from typing import Optional

from src.core.config import settings


class LLMBackend(ABC):
    """Abstract interface cho tat ca LLM backends."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Goi LLM va tra ve text response."""
        ...


class GeminiBackend(LLMBackend):
    """Backend dung Google Gemini API. Yeu cau: GEMINI_API_KEY trong .env"""

    def __init__(self):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Can cai dat: pip install google-generativeai langchain-google-genai")

        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY chua duoc cau hinh. Them vao .env: GEMINI_API_KEY=your_key_here")

        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model)
        print(f"Gemini backend initialized: {settings.gemini_model}")

    def generate(self, prompt: str) -> str:
        import time
        max_retries = 4
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    if attempt < max_retries - 1:
                        sleep_s = 3 * (attempt + 1)
                        print(f"  [Gemini Rate Limit] Doi {sleep_s}s de retry...")
                        time.sleep(sleep_s)
                        continue
                raise


class HFLocalBackend(LLMBackend):
    """Backend dung HuggingFace Transformers local. Yeu cau: HF_MODEL_ID trong .env"""

    def __init__(self):
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise ImportError("Can cai dat: pip install transformers accelerate")

        if not settings.hf_model_id:
            raise ValueError("HF_MODEL_ID chua duoc cau hinh. Them vao .env: HF_MODEL_ID=ten_model")

        print(f"Loading HuggingFace model: {settings.hf_model_id}...")
        self.pipe = hf_pipeline(
            "text-generation",
            model=settings.hf_model_id,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"HF local backend initialized: {settings.hf_model_id}")

    def generate(self, prompt: str) -> str:
        outputs = self.pipe(prompt, max_new_tokens=1024, do_sample=False, return_full_text=False)
        return outputs[0]["generated_text"].strip()


class vLLMBackend(LLMBackend):
    """Backend goi vLLM server qua OpenAI-compatible REST API. Yeu cau: VLLM_BASE_URL trong .env"""

    def __init__(self):
        try:
            import httpx
        except ImportError:
            raise ImportError("Can cai dat: pip install httpx")

        if not settings.vllm_base_url:
            raise ValueError("VLLM_BASE_URL chua duoc cau hinh. Them vao .env: VLLM_BASE_URL=http://localhost:8000")

        self.base_url = settings.vllm_base_url.rstrip("/")
        self.http_client = httpx.Client(timeout=120.0)
        print(f"vLLM backend initialized: {self.base_url}")

    def generate(self, prompt: str) -> str:
        response = self.http_client.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": prompt, "max_tokens": 1024, "temperature": 0.1},
        )
        response.raise_for_status()
        return response.json()["choices"][0]["text"].strip()

    def __del__(self):
        if hasattr(self, "http_client"):
            self.http_client.close()


_llm_instance: Optional[LLMBackend] = None


def get_llm() -> LLMBackend:
    """Khoi tao va tra ve LLM backend instance (Singleton)."""
    global _llm_instance

    if _llm_instance is None:
        backend = settings.llm_backend.lower()
        if backend == "gemini":
            _llm_instance = GeminiBackend()
        elif backend == "hf_local":
            _llm_instance = HFLocalBackend()
        elif backend == "vllm":
            _llm_instance = vLLMBackend()
        else:
            raise ValueError(f"LLM backend khong hop le: '{backend}'. Cac gia tri hop le: 'gemini', 'hf_local', 'vllm'")

    return _llm_instance


def reset_llm() -> None:
    """Reset singleton instance (dung cho testing hoac doi backend runtime)."""
    global _llm_instance
    _llm_instance = None