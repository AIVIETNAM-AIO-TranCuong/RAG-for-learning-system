import os
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, Record,
    Filter, FieldCondition, MatchValue, ScoredPoint
)
from langchain_huggingface import HuggingFaceEmbeddings

from src.core.config import settings


class QdrantStore:
    """
    Manages connection to Qdrant Vector Database and Embedding model.
    """

    def __init__(self):
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

        print(f"Loading embedding model: {settings.embedding_model} on {embed_device}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'device': embed_device}
        )
        
        # Determine embedding dimension programmatically (by embedding a test string)
        test_embed = self.embeddings.embed_query("test")
        self.embedding_dim = len(test_embed)

        # Initialize Qdrant Client
        os.makedirs(settings.qdrant_path, exist_ok=True)
        print(f"Connecting to Qdrant at {settings.qdrant_path}...")
        self.client = QdrantClient(path=settings.qdrant_path)

        # Ensure collection and index are available
        self._ensure_collection()

    def _ensure_collection(self):
        """Check if collection already exists, create new one if not."""
        collections_response = self.client.get_collections()
        collection_names = [c.name for c in collections_response.collections]

        if settings.collection_name not in collection_names:
            print(f"Collection '{settings.collection_name}' not found. Creating...")
            self.client.create_collection(
                collection_name=settings.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim, 
                    distance=Distance.COSINE
                )
            )
            # Create payload index to speed up filtering/searching by metadata
            for field in ["document_id", "filename", "page"]:
                self.client.create_payload_index(
                    collection_name=settings.collection_name,
                    field_name=field,
                    field_schema="keyword" if field != "page" else "integer"
                )
        else:
            print(f"Collection '{settings.collection_name}' already exists.")

    def embed_text(self, text: str) -> List[float]:
        """Create embedding vector for a text snippet."""
        return self.embeddings.embed_query(text)

    def upsert_chunks(self, points: List[PointStruct]) -> None:
        """Save a list of vector points to Qdrant."""
        self.client.upsert(
            collection_name=settings.collection_name,
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
                collection_name=settings.collection_name,
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
                collection_name=settings.collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False
            )
            return response.points
        else:
            return self.client.search(
                collection_name=settings.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False
            )

    def delete_document(self, document_id: str) -> None:
        """Delete all chunks of a specific document from the system."""
        self.client.delete(
            collection_name=settings.collection_name,
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
                collection_name=settings.collection_name,
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


# Global variable to store the single instance (Singleton)
_store_instance = None

def get_store() -> QdrantStore:
    """Get instance of QdrantStore, initialize if it doesn't exist."""
    global _store_instance
    if _store_instance is None:
        _store_instance = QdrantStore()
    return _store_instance
