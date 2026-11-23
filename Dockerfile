# syntax=docker/dockerfile:1
# ShelfScan: the API and the demo page in one CPU-only image. The two models (~245 MB)
# are mounted read-only at run time (docker-compose.yml), not baked into a layer.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.17 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1
WORKDIR /app

# Dependencies first (a cached layer), exactly as locked: CPU-only torch, no dev or
# eval groups. uv's download cache stays in a BuildKit cache mount, not in the image.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY app/ app/
COPY services/ services/
COPY schemas/ schemas/
COPY demo/ demo/
COPY run.py ./

# Non-root; only the output folder is writable (a recursive chown of /app would copy
# the whole virtual environment into a new layer).
RUN useradd --uid 1000 --create-home shelfscan \
 && mkdir -p data/outputs \
 && chown shelfscan:shelfscan data/outputs
USER shelfscan

ENV PATH=/app/.venv/bin:$PATH \
    SHELFSCAN_MODELS_DIR=/models \
    HF_HUB_OFFLINE=1
EXPOSE 8001
# Liveness only: /ready is 503 for the few seconds the models take to load.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health')" || exit 1
# 0.0.0.0 inside the container; docker-compose publishes it on 127.0.0.1 only.
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "8001"]
