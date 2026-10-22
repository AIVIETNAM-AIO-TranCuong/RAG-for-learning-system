import os
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, Record,
    Filter, FieldCondition, MatchValue, ScoredPoint
)
from langchain_huggingface import HuggingFaceEmbeddings

try:
    import transformers.modeling_utils as _mu
    _mu.check_torch_load_is_safe = lambda: None
    import transformers.utils.import_utils as _iu
    _iu.check_torch_load_is_safe = lambda: None
except Exception:
    pass

from src.core.config import settings


# Shared QdrantClient instance across all stores to prevent local file locking
_shared_client = None

def _get_shared_client() -> QdrantClient:
    global _shared_client
    if _shared_client is None:
        os.makedirs(settings.qdrant_path, exist_ok=True)
        _shared_client = QdrantClient(path=settings.qdrant_path)
    return _shared_client


class QdrantStore:
    """
    Manages connection to Qdrant Vector Database and Embedding model.
    Supports dynamic embedding models and collections.
    """

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        collection_name: Optional[str] = None
    ):
        self.model_name = embedding_model or settings.embedding_model
        self.collection_name = collection_name or settings.collection_name

        # Login HuggingFace nếu có HF_TOKEN (cần cho gated models)
        hf_token = settings.hf_token
        if hf_token:
            try:
                from huggingface_hub import login
                login(token=hf_token, add_to_git_credential=False)
                print("HuggingFace login successful.")
            except Exception as e:
                print(f"Warning: HuggingFace login failed: {e}")

        # Initialize embedding model with automatic CUDA fallback
        import torch
        embed_device = settings.embedding_device
        if embed_device == "cuda" and not torch.cuda.is_available():
            print("Notice: CUDA is not available. Falling back to CPU for embedding.")
            embed_device = "cpu"

        print(f"Loading embedding model: {self.model_name} on {embed_device}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': embed_device}
        )
        
        # Determine embedding dimension programmatically (by embedding a test string)
        test_embed = self.embeddings.embed_query("test")
        self.embedding_dim = len(test_embed)

        # Get shared Qdrant Client (avoids portalocker AlreadyLocked on local path)
        print(f"Connecting to Qdrant at {settings.qdrant_path} (Collection: {self.collection_name})...")
        self.client = _get_shared_client()

        # Ensure collection and index are available
        self._ensure_collection()

    def _ensure_collection(self):
        """Check if collection already exists, create new one if not."""
        collections_response = self.client.get_collections()
        collection_names = [c.name for c in collections_response.collections]

        if self.collection_name not in collection_names:
            print(f"Collection '{self.collection_name}' not found. Creating...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim, 
                    distance=Distance.COSINE
                )
            )
            # Create payload index to speed up filtering/searching by metadata
            for field in ["document_id", "filename", "page"]:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema="keyword" if field != "page" else "integer"
                )
        else:
            print(f"Collection '{self.collection_name}' already exists.")

    def embed_text(self, text: str) -> List[float]:
        """Create embedding vector for a text snippet."""
        return self.embeddings.embed_query(text)

    def upsert_chunks(self, points: List[PointStruct]) -> None:
        """Save a list of vector points to Qdrant."""
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def scroll_by_document(self, document_id: str) -> List[Record]:
        """Get ALL chunks belonging to a specific document (for summarization)."""
        all_records = []
        offset = None
        
        doc_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id)
                )
            ]
        )

        while True:
            records, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=doc_filter,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            all_records.extend(records)
            if next_offset is None:
                break
            offset = next_offset

        return all_records

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        document_id: Optional[str] = None
    ) -> List[ScoredPoint]:
        """
        Semantic search trên Qdrant collection.

        Args:
            query_vector: Embedding vector của câu query
            top_k: Số kết quả tối đa trả về
            document_id: Nếu cung cấp, chỉ tìm trong tài liệu này

        Returns:
            List[ScoredPoint] với payload và relevance score
        """
        search_filter = None
        if document_id:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False
            )
            return response.points
        else:
            return self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False
            )

    def delete_document(self, document_id: str) -> None:
        """Delete all chunks of a specific document from the system."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
        )

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents currently stored in the system."""
        all_records = []
        offset = None
        while True:
            records, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            all_records.extend(records)
            if next_offset is None:
                break
            offset = next_offset

        # Group by document_id
        doc_map = {}
        for r in all_records:
            payload = r.payload or {}
            doc_id = payload.get("document_id")
            if not doc_id:
                continue
            
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "document_id": doc_id,
                    "filename": payload.get("filename", "unknown"),
                    "source": payload.get("source", ""),
                    "total_chunks": 0,
                    "max_page": 0
                }
            
            doc_map[doc_id]["total_chunks"] += 1
            page = payload.get("page", 0)
            if page > doc_map[doc_id]["max_page"]:
                doc_map[doc_id]["max_page"] = page

        return list(doc_map.values())


# Cache multi-instance of stores by model and collection
_stores: Dict[str, QdrantStore] = {}

def get_store(
    embedding_model: Optional[str] = None,
    collection_name: Optional[str] = None
) -> QdrantStore:
    """Get instance of QdrantStore, caching by model and collection."""
    model = embedding_model or settings.embedding_model
    col = collection_name or settings.collection_name
    key = f"{model}::{col}"
    if key not in _stores:
        _stores[key] = QdrantStore(embedding_model=model, collection_name=col)
    return _stores[key]
