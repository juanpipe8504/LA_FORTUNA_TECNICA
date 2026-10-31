# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Stage 1: builder – compile wheels for the package and its dependencies
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies first (better layer caching), then the package itself.
COPY pyproject.toml ./
COPY app ./app
RUN pip install --upgrade pip setuptools wheel \
    && pip install .

# Pre-download the fastembed models (dense + BM25 sparse) so the runtime image
# works offline and the first request/ingestion is not slowed down.
ENV FASTEMBED_CACHE_PATH=/opt/models/fastembed \
    HF_HOME=/opt/models/huggingface
RUN python -c "from fastembed import TextEmbedding, SparseTextEmbedding; \
TextEmbedding('BAAI/bge-small-en-v1.5'); \
SparseTextEmbedding('Qdrant/bm25'); \
print('fastembed models cached')"

# ---------------------------------------------------------------------------
# Stage 2: runtime – slim image, non-root user
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    FASTEMBED_CACHE_PATH=/opt/models/fastembed \
    HF_HOME=/opt/models/huggingface \
    DATA_DIR=/data

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app \
    && mkdir -p /data /app \
    && chown -R app:app /data /app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=app:app /opt/models /opt/models
COPY --chown=app:app app ./app

USER app

EXPOSE 8000
VOLUME ["/data"]

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)" || exit 1

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
