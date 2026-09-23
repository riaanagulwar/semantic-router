# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Carrying the venv over whole keeps console scripts (uvicorn) on PATH.
COPY --from=builder /opt/venv /opt/venv

# Set as ENV (not just ARG) so the model baked in below is by definition
# the same one app.config.settings loads at runtime — the two can't drift.
ARG EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    EMBEDDING_MODEL_NAME=${EMBEDDING_MODEL_NAME}

COPY . .

# Bake the embedding model into the image at build time so a fresh
# container never hits the network for it on startup (it's otherwise a
# ~80MB download on every cold start).
RUN python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['EMBEDDING_MODEL_NAME'])" \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
