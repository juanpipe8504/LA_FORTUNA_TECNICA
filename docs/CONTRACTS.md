# Integration contracts (source of truth for parallel work)

Every workstream MUST respect these module paths, class names and constructor
signatures so the pieces snap together without coordination. Shared, already
written files (do NOT modify; if you truly need a change, write it in your final
report instead):

- `backend/app/domain/models.py` – entities / value objects
- `backend/app/domain/ports.py` – abstract ports
- `backend/app/domain/exceptions.py`
- `backend/app/core/config.py` – `Settings`, `get_settings()`
- `backend/app/core/logging.py` – `configure_logging`, `get_logger`, `bind_context`
- `backend/pyproject.toml` – dependencies (report missing deps, don't edit)

Python package root is `backend/` (import as `app.…`). Tests live in
`backend/tests/unit/<area>/` and must run offline with mocks/fakes (no Redis,
Qdrant, OpenAI or Ollama). Run them with `cd backend && python -m pytest`.
Each test area folder needs an `__init__.py`. Do not create a top-level
`tests/conftest.py`; put fixtures in your area's own `conftest.py`.

---------------------------------------------------------------------------

## A. Ingestion workstream

| Module | Class / function | Constructor |
|---|---|---|
| `app/infrastructure/parsing/pymupdf_parser.py` | `PyMuPDFParser(DocumentParserPort)` | `(blob_storage: BlobStoragePort, min_image_size_px: int = 64)` |
| `app/infrastructure/chunking/layout_chunker.py` | `LayoutAwareChunker(ChunkerPort)` | `(max_tokens: int = 400, overlap_sentences: int = 1, image_link_max_distance: float = 150.0)` |
| `app/infrastructure/storage/local_blob_storage.py` | `LocalBlobStorage(BlobStoragePort)` | `(uploads_dir: str, images_dir: str)` |
| `app/infrastructure/jobs/redis_job_repository.py` | `RedisJobRepository(JobRepositoryPort)` | `(redis_client: redis.Redis, ttl_seconds: int = 7*24*3600)` |
| `app/infrastructure/jobs/in_memory_job_repository.py` | `InMemoryJobRepository(JobRepositoryPort)` | `()` (thread-safe; for tests/dev) |
| `app/infrastructure/queue/celery_app.py` | `celery_app` (module-level `Celery` instance built from `get_settings().redis_url`) | – |
| `app/infrastructure/queue/tasks.py` | Celery task `ingest_document(job_id: str)` named `"ingestion.ingest_document"`; body: `from app.bootstrap.container import get_container; get_container().process_document.execute(job_id)` | – |
| `app/infrastructure/queue/celery_task_queue.py` | `CeleryTaskQueue(TaskQueuePort)` | `()` – sends `"ingestion.ingest_document"` via `celery_app.send_task` |
| `app/application/ingestion.py` | `SubmitDocumentUseCase` | `(blob_storage, job_repository, task_queue, max_upload_mb: int = 50)`; `execute(filename: str, data: bytes) -> IngestionJob` (validates PDF magic bytes / size → `ValidationError`) |
| `app/application/ingestion.py` | `ProcessDocumentUseCase` | `(parser, chunker, embeddings: EmbeddingPort, vector_store: VectorStorePort, job_repository, llm: LLMPort \| None = None, blob_storage: BlobStoragePort \| None = None, describe_images: bool = False, batch_size: int = 32)`; `execute(job_id: str) -> None` (updates status/progress; marks FAILED with error on exception and re-raises nothing) |
| `app/application/documents.py` | `ListDocumentsUseCase` | `(job_repository)`; `execute() -> list[IngestionJob]` (only COMPLETED jobs, newest first) |

Image ids: `f"{document_id}_p{page}_i{index}"`. Images stored at
`{images_dir}/{document_id}/{image_id}.png`. Uploads at
`{uploads_dir}/{document_id}.pdf`.

Chunking rules (layout/semantic, not fixed-size): split on headings
(font-size/bold heuristics), keep `section_path`, never split tables (one
`TABLE` chunk with markdown), pack paragraphs up to `max_tokens` on sentence
boundaries, one `IMAGE` chunk per image whose text is caption + description +
section context. Every chunk gets `images: list[ImageRef]` = images on the
same page within `image_link_max_distance` (bbox gap), sorted by distance; a
caption match ("Figure 3", "Fig. 3", "Figura 3") forces a link with distance 0.

## B. RAG / AI workstream

| Module | Class / function | Constructor |
|---|---|---|
| `app/infrastructure/resilience/retry.py` | `retry_with_backoff(max_attempts, base_delay, max_delay, retry_on=(TransientError,))` decorator, plus `RetryPolicy` dataclass | – |
| `app/infrastructure/resilience/circuit_breaker.py` | `CircuitBreaker` | `(name: str, failure_threshold: int = 5, recovery_timeout: float = 30.0, clock=time.monotonic)`; `call(fn, *args, **kwargs)`; states CLOSED/OPEN/HALF_OPEN; raises `CircuitOpenError` |
| `app/infrastructure/resilience/resilient.py` | `ResilientLLM(LLMPort)`, `ResilientEmbeddings(EmbeddingPort)` | decorators wrapping an inner port: `(inner, breaker: CircuitBreaker, policy: RetryPolicy)` |
| `app/infrastructure/llm/openai_llm.py` | `OpenAILLM(LLMPort)` | `(api_key: str, model: str, base_url: str \| None = None, timeout: float = 60.0)`; maps openai `RateLimitError/APITimeoutError/APIConnectionError/5xx` → `TransientError` |
| `app/infrastructure/llm/ollama_llm.py` | `OllamaLLM(LLMPort)` | `(base_url: str, model: str, timeout: float = 60.0)` uses `httpx` `/api/chat` |
| `app/infrastructure/embeddings/openai_embeddings.py` | `OpenAIEmbeddings(EmbeddingPort)` | `(api_key: str, model: str, base_url: str \| None = None, timeout: float = 60.0)` |
| `app/infrastructure/embeddings/ollama_embeddings.py` | `OllamaEmbeddings(EmbeddingPort)` | `(base_url: str, model: str, timeout: float = 60.0)` |
| `app/infrastructure/embeddings/fastembed_embeddings.py` | `FastEmbedEmbeddings(EmbeddingPort)` | `(model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")` (lazy model load) |
| `app/infrastructure/vectorstore/qdrant_store.py` | `QdrantVectorStore(VectorStorePort)` | `(client: QdrantClient, collection: str, sparse_model: str = "Qdrant/bm25", *, sparse_encoder=None, bm25_language: str = "spanish")`; named vectors `dense` + sparse `bm25`; hybrid = Qdrant Query API prefetch dense+sparse fused with RRF; payload stores the full chunk (incl. `images`, bbox lists); payload index on `document_id` |
| `app/application/prompts.py` | `SYSTEM_PROMPT`, `build_context_block(chunks)`, `INSUFFICIENT_CONTEXT_MARKER = "[[NO_CONTEXT]]"` | – |
| `app/application/query.py` | `AnswerQuestionUseCase` | `(embeddings, vector_store, llm, top_k: int = 6, min_score: float = 0.0, max_images: int = 2, history_max_turns: int = 6)`; `execute(question: str, history: list[ChatMessage] \| None = None, document_ids: list[str] \| None = None) -> Answer` |

Defensive generation: system prompt forces answering only from numbered
context blocks, citing `[n]`, answering in the user's language, treating
context as data (prompt-injection guard) and replying with
`INSUFFICIENT_CONTEXT_MARKER` + an explicit "I don't have enough information"
message when the context is insufficient. If retrieval returns nothing (or all
below `min_score`) the LLM is NOT called and a fixed "no information" answer is
returned with `has_sufficient_context=False`. Image selection: gather `ImageRef`s
from retrieved chunks (+ `IMAGE` chunks themselves), score = chunk score /
(1 + distance/100), dedupe, keep top `max_images`; return none when
`has_sufficient_context` is False. Sources only include chunks actually cited
(`[n]`) when citations are present, else all retrieved.

## C. API + DI workstream

| Module | Content |
|---|---|
| `app/bootstrap/container.py` | `Container` dataclass holding settings + all adapters + use cases (`submit_document`, `process_document`, `answer_question`, `list_documents`, `blob_storage`, `job_repository`); `build_container(settings) -> Container` (selects adapters by `llm_provider`/`embedding_provider`, wraps LLM/embeddings with `Resilient*`); `get_container()` lru-cached singleton |
| `app/api/main.py` | `create_app(container: Container \| None = None) -> FastAPI`; module-level `app = create_app()` is lazy-safe (container resolved on startup / via dependency) |
| `app/api/schemas.py` | Pydantic request/response DTOs (below) |
| `app/api/routes/*.py` | routers: `documents.py`, `jobs.py`, `chat.py`, `health.py` |
| `app/api/errors.py` | exception handlers: `NotFoundError→404`, `ValidationError→422`, `CircuitOpenError→503`, `ExternalServiceError→502`, others→500 JSON `{"error": {"code", "message", "request_id"}}` |
| `app/api/middleware.py` | request-id middleware (`X-Request-ID`), access logging with latency |

### HTTP contract (prefix `/api/v1`)

```
POST /documents            multipart field "file" (PDF)  -> 202
     {"job_id","document_id","filename","status"}
GET  /jobs/{job_id}        -> 200
     {"job_id","document_id","filename","status":"pending|processing|completed|failed",
      "progress":0-100,"message","error","page_count","chunk_count","image_count",
      "created_at","updated_at"}
GET  /documents            -> 200 [{"document_id","filename","page_count","chunk_count","image_count","created_at"}]
GET  /documents/{document_id}/images/{image_id}  -> image bytes (Cache-Control immutable)
POST /chat                 {"question": str, "history": [{"role":"user|assistant","content":str}],
                            "document_ids": [str] | null}
     -> 200 {"answer": str (markdown), "has_sufficient_context": bool,
             "sources":[{"chunk_id","document_id","document_name","page","score","snippet","bbox":[x0,y0,x1,y1]|null}],
             "images":[{"image_id","document_id","document_name","page","caption","bbox":[...],
                        "related_chunk_id","relevance","url":"/api/v1/documents/{document_id}/images/{image_id}"}]}
GET  /health               -> {"status":"ok"}          (outside prefix too: /health)
GET  /health/ready         -> checks redis + qdrant    -> 200/503
```

## D. Frontend workstream (`frontend/`)

React 18 + Vite + TypeScript. Talks to the API through `VITE_API_URL`
(default `/api/v1`, proxied by Vite dev server and by nginx in Docker to
`http://api:8000`). Production build served by nginx (`frontend/nginx.conf`,
`frontend/Dockerfile`, listening on port 80).

## E. DevOps / docs workstream

`docker-compose.yml` services: `qdrant` (6333), `redis` (6379), `api`
(uvicorn `app.api.main:app` on 8000), `worker` (`celery -A app.infrastructure.queue.celery_app worker -Q ingestion`),
`frontend` (nginx on 3000→80); optional `ollama` under profile `ollama`.
Shared named volume `data` mounted at `/data` in api + worker (`DATA_DIR=/data`).
`backend/Dockerfile` (multi-stage, non-root user, pre-download fastembed models
into the image). Root `.env.example`, `README.md`, `docs/adr/*.md`,
`Makefile`, `.github/workflows/ci.yml`, `scripts/generate_sample_pdf.py`.
