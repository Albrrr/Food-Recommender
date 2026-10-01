# syntax=docker/dockerfile:1

# Foodify API.
#
# The embedding and reranker models are baked into the image rather than pulled
# at runtime: it makes the image large (~3GB) but container boot is a local
# load rather than a 1.3GB download, which is what matters for cold starts.
# To trade the other way, delete the warm-models step and mount a cache volume.

FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models \
    PYTHONPATH=/app/backend

WORKDIR /app

# CPU-only torch. The default index serves CUDA builds, which add several GB
# of libraries this image can never use.
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.14.0

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Pre-download models so startup does no network I/O.
COPY backend/config.py /app/backend/config.py
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
from fastembed.rerank.cross_encoder import TextCrossEncoder
SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
PY

COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app:app", "--app-dir", "/app/backend", "--host", "0.0.0.0", "--port", "8000"]
