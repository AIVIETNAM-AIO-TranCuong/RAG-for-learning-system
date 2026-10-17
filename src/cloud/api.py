import asyncio
import hashlib
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.core.schemas import FlashcardSet, QuizSet, RagAnswer, Summary
from .ai import GeminiAI
from .config import CloudSettings, MAX_UPLOAD_BYTES, OPERATION_SECONDS
from .documents import document_digest, extract_chunks, safe_filename
from .errors import provider_error
from .service import CloudService
from .store import CloudStore


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=12000)
    document_id: str | None = None


class DocumentRequest(BaseModel):
    document_id: str = Field(min_length=1)


class QuizRequest(DocumentRequest):
    topic: str | None = Field(default=None, max_length=2000)
    n_questions: int = Field(default=8, ge=1, le=20)


class FlashcardsRequest(DocumentRequest):
    topic: str | None = Field(default=None, max_length=2000)
    n_cards: int = Field(default=15, ge=1, le=30)


class CdnStaticFiles(StaticFiles):
    async def check_config(self):
        # CDN promotion can remove the local asset directory from the runtime bundle.
        # Unmatched URLs must still return 404 instead of a missing-directory error.
        if self.directory is not None and Path(self.directory).is_dir():
            await super().check_config()


async def execute(operation):
    try:
        async with asyncio.timeout(OPERATION_SECONDS):
            return await operation()
    except Exception as error:
        raise provider_error(error) from None


def create_app(settings: CloudSettings | None = None, service=None, *, ui_directory: Path | None = None) -> FastAPI:
    settings = settings or CloudSettings()

    @asynccontextmanager
    async def lifespan(app):
        yield
        runtime = app.state.service
        if runtime and service is None:
            await runtime.ai.close()
            await runtime.store.close()

    app = FastAPI(title="RAG Learning Cloud API", version="1.0.0", lifespan=lifespan,
                  docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)
    app.state.service = service
    app.add_middleware(CORSMiddleware, allow_origins=[
        "https://rag-learning-system-ui.vercel.app", "http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    def runtime():
        if app.state.service is None:
            missing = settings.missing()
            if missing:
                raise HTTPException(503, "Backend chưa được cấu hình: " + ", ".join(missing))
            if not settings.qdrant_url.startswith("https://"):
                raise HTTPException(503, "QDRANT_URL phải dùng HTTPS.")
            app.state.service = CloudService(CloudStore(settings), GeminiAI(settings))
        return app.state.service

    router = APIRouter(prefix="/api")

    @router.get("/health")
    async def health():
        try:
            async with asyncio.timeout(8):
                await runtime().store.ping()
            return dict(status="healthy", llm_configured=True, qdrant_connected=True,
                        llm_backend="gemini", gemini_model=settings.model, max_upload_bytes=MAX_UPLOAD_BYTES)
        except Exception as error:
            detail = error.detail if isinstance(error, HTTPException) and error.status_code == 503 else "Không kết nối được Qdrant Cloud."
            return JSONResponse(status_code=503, content=dict(status="unavailable", detail=detail,
                llm_configured=bool(settings.gemini_api_key), qdrant_connected=False))

    @router.get("/documents")
    async def documents():
        return await execute(lambda: runtime().store.documents())

    @router.post("/ask", response_model=RagAnswer)
    async def ask(request: AskRequest):
        if not request.question.strip():
            raise HTTPException(422, "Vui lòng nhập câu hỏi.")
        return await execute(lambda: runtime().ask(request.question.strip(), request.document_id))

    @router.post("/summarize", response_model=Summary)
    async def summarize(request: DocumentRequest):
        return await execute(lambda: runtime().summarize(request.document_id))

    @router.post("/quiz", response_model=QuizSet)
    async def quiz(request: QuizRequest):
        return await execute(lambda: runtime().learning("quiz", request.document_id, request.topic, request.n_questions))

    @router.post("/flashcards", response_model=FlashcardSet)
    async def flashcards(request: FlashcardsRequest):
        return await execute(lambda: runtime().learning("flashcard", request.document_id, request.topic, request.n_cards))

    @router.post("/upload")
    async def upload(file: UploadFile = File(...), force_reindex: bool = Form(False)):
        try:
            if force_reindex:
                raise HTTPException(422, "Demo công khai không cho phép ghi đè tài liệu.")
            filename = safe_filename(file.filename or "")

            async def process():
                # All file I/O uses ephemeral storage. Original bytes never persist in the deployment.
                with tempfile.TemporaryDirectory(prefix="rag-cloud-") as directory:
                    path = Path(directory) / filename
                    total, digest = 0, hashlib.sha256()
                    with path.open("wb") as target:
                        while data := await file.read(64 * 1024):
                            total += len(data)
                            if total > MAX_UPLOAD_BYTES:
                                raise HTTPException(413, "File vượt giới hạn 4.000.000 byte (4 MB).")
                            digest.update(data)
                            target.write(data)
                    if total == 0:
                        raise HTTPException(422, "File trống, vui lòng chọn tài liệu có nội dung.")
                    doc_id = document_digest(filename, digest.hexdigest())
                    chunks = await asyncio.to_thread(extract_chunks, path, filename, doc_id)
                    count = await runtime().index(chunks)
                    return dict(document_id=doc_id, filename=filename, total_chunks=count,
                                message="Nạp tài liệu thành công.")

            return await execute(process)
        finally:
            await file.close()

    app.include_router(router)
    if ui_directory is not None:
        # Declare the mount explicitly so Vercel can promote the built files to its CDN.
        # API routes precede the mount; html=True also serves index.html at '/'.
        app.mount("/", CdnStaticFiles(directory=str(ui_directory), html=True, check_dir=False), name="ui")
    return app


app = create_app(ui_directory=Path(__file__).resolve().parents[2] / "public")
