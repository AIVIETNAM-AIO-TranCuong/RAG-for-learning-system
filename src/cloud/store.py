import asyncio
from uuid import NAMESPACE_URL, uuid5

from fastapi import HTTPException
from qdrant_client import AsyncQdrantClient, models

from .config import CloudSettings, EMBEDDING_DIMENSIONS


def ready_filter(document_id: str | None = None):
    must = [models.FieldCondition(key="ready", match=models.MatchValue(value=True))]
    if document_id:
        must.append(models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)))
    return models.Filter(must=must)


def point_id(chunk: dict) -> str:
    return str(uuid5(NAMESPACE_URL, f"rag-cloud:{chunk['document_id']}:{chunk['chunk_id']}"))


class CloudStore:
    def __init__(self, settings: CloudSettings):
        self.settings = settings
        self.client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key,
                                        timeout=30, prefer_grpc=False, check_compatibility=False)
        self._initialized = False
        self._lock = asyncio.Lock()

    async def close(self):
        await self.client.close()

    async def ensure(self):
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return
            name = self.settings.collection
            if not await self.client.collection_exists(name):
                try:
                    await self.client.create_collection(name, vectors_config=models.VectorParams(
                        size=EMBEDDING_DIMENSIONS, distance=models.Distance.COSINE))
                except Exception as error:
                    if getattr(error, "status_code", None) not in (409,):
                        raise
            info = await self.client.get_collection(name)
            vector = info.config.params.vectors
            if not isinstance(vector, models.VectorParams) or vector.size != EMBEDDING_DIMENSIONS or vector.distance != models.Distance.COSINE:
                raise HTTPException(503, "Collection cloud có cấu hình vector không tương thích. Hãy dùng collection mới.")
            first, _ = await self.client.scroll(name, limit=1, with_payload=True, with_vectors=False)
            if first and (first[0].payload or {}).get("embedding_model") != self.settings.embedding_model:
                raise HTTPException(503, "Model embedding khác model của collection. Cần migration sang collection mới.")
            for field, schema in (("document_id", models.PayloadSchemaType.KEYWORD), ("ready", models.PayloadSchemaType.BOOL)):
                await self.client.create_payload_index(name, field_name=field, field_schema=schema, wait=True)
            self._initialized = True

    async def ping(self):
        await self.ensure()
        await self.client.get_collection(self.settings.collection)

    async def chunks(self, document_id: str | None = None):
        await self.ensure()
        records, offset = [], None
        while True:
            page, offset = await self.client.scroll(
                self.settings.collection, scroll_filter=ready_filter(document_id),
                limit=128, offset=offset, with_payload=True, with_vectors=False)
            records.extend(dict(record.payload or {}) for record in page)
            if offset is None:
                break
        return sorted(records, key=lambda p: (p.get("document_id", ""), p.get("page", 0), p.get("position", 0), p.get("chunk_id", "")))

    async def documents(self):
        documents = {}
        for chunk in await self.chunks():
            doc_id = chunk["document_id"]
            document = documents.setdefault(doc_id, dict(document_id=doc_id, filename=chunk["filename"],
                source=chunk["filename"], total_chunks=0, max_page=0))
            document["total_chunks"] += 1
            document["max_page"] = max(document["max_page"], chunk.get("page", 0))
        return sorted(documents.values(), key=lambda d: d["filename"].casefold())

    async def search(self, vector: list[float], document_id: str | None = None):
        await self.ensure()
        result = await self.client.query_points(self.settings.collection, query=vector,
            query_filter=ready_filter(document_id), limit=15, with_payload=True, with_vectors=False)
        return [dict(point.payload or {}, score=point.score) for point in result.points]

    async def save(self, chunks: list[dict], vectors: list[list[float]]):
        if not chunks or len(chunks) != len(vectors):
            raise ValueError("One vector is required for each chunk")
        if len({chunk["document_id"] for chunk in chunks}) != 1:
            raise ValueError("Save one document at a time")
        await self.ensure()
        points = [models.PointStruct(id=point_id(chunk), vector=vector, payload=dict(
            chunk, source=chunk["filename"], ready=False, embedding_model=self.settings.embedding_model))
            for chunk, vector in zip(chunks, vectors, strict=True)]
        # Pending points are invisible to every read/search. Stable IDs allow safe retries.
        for offset in range(0, len(points), 64):
            await self.client.upsert(self.settings.collection, points=points[offset:offset + 64], wait=True)
        await self.client.set_payload(self.settings.collection, payload={"ready": True},
                                      points=[point.id for point in points], wait=True)
