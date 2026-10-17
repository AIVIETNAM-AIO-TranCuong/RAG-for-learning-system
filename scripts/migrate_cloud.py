"""Read a consistent copy of local Qdrant and re-embed it into Qdrant Cloud.

Export only: uv run python -m scripts.migrate_cloud --export-only
Migrate:     uv run python -m scripts.migrate_cloud
"""
import argparse
import asyncio
import json
import re
import shutil
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from qdrant_client import QdrantClient

from src.cloud.ai import GeminiAI
from src.cloud.config import CloudSettings
from src.cloud.service import CloudService
from src.cloud.store import CloudStore
from src.cloud.errors import provider_error

ROOT = Path(__file__).resolve().parents[1]


class MigrationAI(GeminiAI):
    """Only the offline migrator retries rate-limited, idempotent embedding calls."""
    async def embed(self, text, *, title=None):
        for attempt in range(6):
            try:
                return await super().embed(text, title=title)
            except Exception as error:
                if str(getattr(error, "code", None)) != "429" or attempt == 5:
                    raise
                delay = 60
                for detail in getattr(error, "details", None) or []:
                    if isinstance(detail, dict) and "retryDelay" in detail:
                        match = re.match(r"([\d.]+)s", str(detail["retryDelay"]))
                        if match:
                            delay = max(1, min(120, float(match[1]) + 1))
                print(f"Embedding provider rate limit; retrying in {delay:.0f}s", flush=True)
                await asyncio.sleep(delay)


def export_local(source: Path, collection="documents") -> list[dict]:
    """SQLite backup allows a consistent read while the original client holds its lock."""
    source = source.resolve()
    with tempfile.TemporaryDirectory(prefix="rag-migration-") as temporary:
        snapshot = Path(temporary)
        shutil.copyfile(source / "meta.json", snapshot / "meta.json")
        target = snapshot / "collection" / collection
        target.mkdir(parents=True)
        with closing(sqlite3.connect((source / "collection" / collection / "storage.sqlite").as_uri() + "?mode=ro", uri=True)) as original:
            with closing(sqlite3.connect(target / "storage.sqlite")) as backup:
                original.backup(backup)
        client = QdrantClient(path=str(snapshot))
        try:
            chunks, offset = [], None
            while True:
                records, offset = client.scroll(collection, limit=128, offset=offset,
                                                with_payload=True, with_vectors=False)
                for record in records:
                    payload = dict(record.payload or {})
                    payload["source"] = payload["filename"]
                    suffix = re.search(r"_(\d+)$", payload["chunk_id"])
                    payload["position"] = int(suffix[1]) if suffix else 0
                    chunks.append(payload)
                if offset is None:
                    break
            return sorted(chunks, key=lambda p: (p["document_id"], p["page"], p["position"]))
        finally:
            client.close()


async def migrate(chunks, settings):
    missing = settings.missing()
    if missing:
        raise SystemExit("Configure .env.cloud first: " + ", ".join(missing))
    store, ai = CloudStore(settings), MigrationAI(settings)
    service = CloudService(store, ai)
    try:
        for doc_id in sorted({chunk["document_id"] for chunk in chunks}):
            expected = [chunk for chunk in chunks if chunk["document_id"] == doc_id]
            existing = await store.chunks(doc_id)
            fields = ("document_id", "chunk_id", "filename", "page", "content")
            signature = lambda records: {tuple(record.get(field) for field in fields) for record in records}
            if existing and signature(existing) != signature(expected):
                raise SystemExit(f"Cloud document {doc_id} differs from snapshot; refusing to overwrite")
            count = await service.index(expected)
            actual = await store.chunks(doc_id)
            if signature(actual) != signature(expected):
                raise SystemExit(f"Verification failed for {doc_id}")
            print(f"Verified {doc_id}: {count} chunks", flush=True)
        print(f"Migration verified: {len({c['document_id'] for c in chunks})} documents / {len(chunks)} chunks")
    finally:
        await ai.close()
        await store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "qdrant_storage")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "cloud-migration.json")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    chunks = export_local(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Snapshot exported: {len({c['document_id'] for c in chunks})} documents / {len(chunks)} chunks")
    if not args.export_only:
        asyncio.run(migrate(chunks, CloudSettings()))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(provider_error(error).detail) from None
