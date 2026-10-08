import hashlib
from pathlib import Path
from uuid import uuid4
from typing import List, Union

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http.models import PointStruct

from src.core.config import settings
from src.core.schemas import ChunkMetadata
from src.db.store import get_store
from src.utils.tf_text_utils import preprocess_batch

# Danh sách các định dạng tài liệu được hỗ trợ
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".md"}


def _make_document_id(file_path: Path) -> str:
    """Tạo ID duy nhất cho tài liệu dựa trên tên file và kích thước."""
    file_size = file_path.stat().st_size
    raw = f"{file_path.name}_{file_size}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _load_pdf(path: Path) -> List[Document]:
    """Đọc tệp PDF theo từng trang."""
    try:
        loader = PyPDFLoader(str(path))
        return loader.load()
    except Exception as e:
        raise ValueError(f"Lỗi khi đọc file PDF '{path.name}': {str(e)}")


def _load_docx(path: Path) -> List[Document]:
    """Đọc tệp Word (.docx) và gom đoạn văn/bảng thành các trang tài liệu logic."""
    try:
        import docx
    except ImportError:
        raise ImportError("Thiếu thư viện 'python-docx'. Cài đặt bằng: pip install python-docx")

    try:
        doc = docx.Document(str(path))
    except Exception as e:
        raise ValueError(f"Lỗi khi mở file Word '{path.name}': {str(e)}")

    pages: List[Document] = []
    current_page_text: List[str] = []
    current_words = 0
    page_idx = 1

    # Đọc các đoạn văn (paragraphs)
    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        current_page_text.append(text)
        current_words += len(text.split())
        # Cứ mỗi ~350-400 từ gom thành 1 trang logic để trích dẫn trang rõ ràng
        if current_words >= 350:
            pages.append(Document(
                page_content="\n\n".join(current_page_text),
                metadata={"page": page_idx, "source": str(path.resolve())}
            ))
            current_page_text = []
            current_words = 0
            page_idx += 1

    # Đọc các bảng biểu (tables)
    for table in doc.tables:
        table_rows = []
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                table_rows.append(row_text)
        if table_rows:
            current_page_text.append("\n".join(table_rows))
            current_words += sum(len(r.split()) for r in table_rows)
            if current_words >= 350:
                pages.append(Document(
                    page_content="\n\n".join(current_page_text),
                    metadata={"page": page_idx, "source": str(path.resolve())}
                ))
                current_page_text = []
                current_words = 0
                page_idx += 1

    if current_page_text:
        pages.append(Document(
            page_content="\n\n".join(current_page_text),
            metadata={"page": page_idx, "source": str(path.resolve())}
        ))

    return pages


def _load_pptx(path: Path) -> List[Document]:
    """Đọc tệp PowerPoint (.pptx), mỗi slide được tính là 1 trang (page = slide number)."""
    try:
        import pptx
    except ImportError:
        raise ImportError("Thiếu thư viện 'python-pptx'. Cài đặt bằng: pip install python-pptx")

    try:
        prs = pptx.Presentation(str(path))
    except Exception as e:
        raise ValueError(f"Lỗi khi mở file PowerPoint '{path.name}': {str(e)}")

    pages: List[Document] = []
    for slide_idx, slide in enumerate(prs.slides, start=1):
        slide_texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    t = paragraph.text.strip()
                    if t:
                        slide_texts.append(t)
            elif shape.has_table:
                for row in shape.table.rows:
                    row_t = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_t:
                        slide_texts.append(row_t)
        if slide_texts:
            pages.append(Document(
                page_content="\n".join(slide_texts),
                metadata={"page": slide_idx, "source": str(path.resolve())}
            ))

    return pages


def _load_text(path: Path) -> List[Document]:
    """Đọc tệp văn bản thuần (.txt) hoặc Markdown (.md)."""
    encodings = ("utf-8", "utf-8-sig", "latin-1", "cp1252")
    content = ""
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

    if not content.strip():
        return []

    # Gom theo đoạn thành các trang logic ~400 từ
    paragraphs = content.split("\n\n")
    pages: List[Document] = []
    current_page_text: List[str] = []
    current_words = 0
    page_idx = 1

    for p in paragraphs:
        p_clean = p.strip()
        if not p_clean:
            continue
        current_page_text.append(p_clean)
        current_words += len(p_clean.split())
        if current_words >= 400:
            pages.append(Document(
                page_content="\n\n".join(current_page_text),
                metadata={"page": page_idx, "source": str(path.resolve())}
            ))
            current_page_text = []
            current_words = 0
            page_idx += 1

    if current_page_text:
        pages.append(Document(
            page_content="\n\n".join(current_page_text),
            metadata={"page": page_idx, "source": str(path.resolve())}
        ))

    return pages


def load_file_to_documents(path: Path) -> List[Document]:
    """
    Dispatcher đọc nội dung tệp theo định dạng (PDF, DOCX, PPTX, TXT, MD).
    """
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _load_pdf(path)
    elif ext == ".docx":
        return _load_docx(path)
    elif ext == ".pptx":
        return _load_pptx(path)
    elif ext in {".txt", ".md"}:
        return _load_text(path)
    else:
        raise ValueError(
            f"Định dạng tệp '{ext}' không được hỗ trợ. "
            f"Các định dạng hỗ trợ: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def index_document(file_path: Union[str, Path], force_reindex: bool = False) -> str:
    """
    Đọc tài liệu đa định dạng (PDF, DOCX, PPTX, TXT, MD), tiền xử lý, chunking, embedding và lưu vào Qdrant.
    
    Args:
        file_path: Đường dẫn đến tệp tài liệu
        force_reindex: Nếu True, xóa dữ liệu cũ và index lại nếu tài liệu đã tồn tại
        
    Returns:
        document_id của tài liệu đã được index
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Định dạng tệp '{ext}' không được hỗ trợ. "
            f"Các định dạng hỗ trợ: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    document_id = _make_document_id(path)
    store = get_store()

    # Kiểm tra xem tài liệu đã từng được index chưa
    existing_records = store.scroll_by_document(document_id)
    if existing_records:
        if not force_reindex:
            print(f"Tài liệu {path.name} (ID: {document_id}) đã tồn tại. Bỏ qua index.")
            return document_id
        else:
            print(f"Tài liệu {path.name} (ID: {document_id}) đã tồn tại. Đang xóa dữ liệu cũ và index lại...")
            store.delete_document(document_id)

    # 1. LOAD TÀI LIỆU THEO ĐỊNH DẠNG
    print(f"Đang đọc tài liệu: {path.name} ({ext})...")
    pages = load_file_to_documents(path)

    if not pages:
        raise ValueError(f"Tệp {path.name} không có nội dung chữ để trích xuất.")

    # Lấy mảng văn bản thô
    raw_texts = [page.page_content for page in pages]

    # 2. PREPROCESS TEXT
    print("Tiền xử lý văn bản...")
    cleaned_texts = preprocess_batch(raw_texts)

    for i, page in enumerate(pages):
        page.page_content = cleaned_texts[i]

    # 3. CHUNKING TEXT
    print(f"Chunking văn bản (chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(pages)

    if not chunks:
        print("Cảnh báo: Không tạo được chunk nào từ tệp này.")
        return document_id

    # 4. EMBEDDING & CHUẨN BỊ VECTORS
    print(f"Đang tạo embedding và chuẩn bị lưu {len(chunks)} chunks vào Qdrant...")
    points = []
    chunk_texts = [chunk.page_content for chunk in chunks]
    embeddings_list = store.embeddings.embed_documents(chunk_texts)

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings_list)):
        chunk_id = f"{document_id}_{i}"
        
        # Schema metadata chuẩn
        metadata = ChunkMetadata(
            document_id=document_id,
            filename=path.name,
            source=str(path.resolve()),
            page=chunk.metadata.get("page", 1),
            chunk_id=chunk_id,
            section=None
        )

        payload = metadata.model_dump()
        payload["content"] = chunk.page_content

        point = PointStruct(
            id=str(uuid4()),
            vector=embedding,
            payload=payload
        )
        points.append(point)

    # 5. UPSERT VÀO QDRANT
    print(f"Đang lưu {len(points)} chunks vào Qdrant...")
    store.upsert_chunks(points)
    print(f"Đã lập chỉ mục thành công tài liệu '{path.name}' với {len(chunks)} chunks.")

    return document_id


# Alias tương thích ngược với code cũ
index_pdf = index_document
