# Cloud deployment (Vercel + Gemini + Qdrant Cloud)

The public demo shares one source library. Its UI and four study tools retain the
existing contracts. Local Streamlit, CLI, pipelines, and Qdrant storage are unchanged.

## Configure

Use Python 3.12, Node 24, and uv. Keep a separate cloud environment:

```powershell
$env:UV_PROJECT_ENVIRONMENT = '.venv-cloud'
uv sync --frozen
Copy-Item .env.cloud.example .env.cloud # only when .env.cloud does not exist
```

Fill `GEMINI_API_KEY`, `QDRANT_URL` (HTTPS), and `QDRANT_API_KEY` in the ignored
`.env.cloud`. Use a Qdrant Free Cluster and the Gemini free tier. Do not enable paid
billing automatically. Local model settings in `.env` do not override cloud models.

Cloud model overrides: `CLOUD_GEMINI_MODEL`, `CLOUD_EMBEDDING_MODEL`,
`CLOUD_COLLECTION`. Defaults are Gemini 3.5 Flash-Lite, Gemini Embedding 2 (768
dimensions), and `documents_gemini_embedding2_768`. Changing embedding models
requires a new collection and a fresh migration, even when dimensions match.

```powershell
uv run python -m pytest
uv run python -m scripts.migrate_cloud --export-only
uv run python -m scripts.migrate_cloud
uv run uvicorn src.cloud.api:app --port 8001
```

Migration copies SQLite with the backup API, then opens only the temporary copy
with Qdrant. It preserves IDs, text and pages, discards old vectors, and removes
machine-specific source paths. The ignored export is `data/cloud-migration.json`.
Reruns verify existing documents and resume without duplicates. Only migration
retries provider-limited embedding calls; it can take several minutes on free tiers.

## Vercel

Use the repository root, FastAPI preset, production branch `quan/ui`, and the
tracked root `vercel.json`. The lockfile includes only cloud runtime dependencies.
Original `requirements.txt` remains for local use and is not the cloud install source.
The build command emits exactly four UI runtime files into root `public/` for CDN
delivery. The browser resolves `/api` against its current origin, including previews.

Add the three secrets to **both Preview and Production** Vercel environments. They
must never be `PUBLIC_*` variables. Use the existing `rag-learning-system-ui` project.
Deploy a preview, run smoke checks, then deploy/promote production. Keep the previous
UI deployment available for rollback. GitHub pushes to `quan/ui` trigger production
only after the repository is connected with that production branch.

## Behavior and limits

- API: `/api/health`, `/api/documents`, `/api/upload`, `/api/ask`, `/api/summarize`,
  `/api/quiz`, `/api/flashcards`; interactive docs at `/api/docs`.
- Uploads: PDF, DOCX, PPTX, TXT and MD; maximum exactly 4,000,000 bytes each.
  Scanned PDFs require OCR beforehand. Original file bytes are temporary only.
- Every chunk has a separate embedding. Pending indexes are hidden until commit.
  Public endpoints do not delete or overwrite documents.
- Retrieval filters selected sources before choosing 15 candidates and reranking
  to five. Reranking failure uses the original top five. Citation snapshots include
  content, file and page. Chat remains independent turns as in the current demo.
- Summary maps batches of ten, with at most two map calls at once. Learning context
  is bounded at 48,000 characters and sampled across the ordered document.
- Requests have a 240-second deadline under the Vercel 300-second limit. The client
  does not automatically retry generation POSTs.
- There is no application quota or login. Provider limits still apply and return
  readable errors: 404 source, 413 file size, 422 file/input, 429 provider quota,
  502 upstream/invalid AI output, 503 configuration/readiness, 504 timeout.
- API credentials, local data, environments and test fixtures are excluded from deployment.

## Verification

```powershell
uv run python -m pytest
node --test demo-ui/api.test.mjs
node demo-ui/build.mjs --cloud
```

Production checks must use real Gemini and Qdrant: migrated library, upload, source
filtering, question/citation, summary, quiz, cards, and document persistence after
reload. A healthy route verifies configuration and Qdrant connectivity; the release
smoke tests separately verify actual Gemini generation and embeddings.
