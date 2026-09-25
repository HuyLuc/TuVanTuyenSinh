# syntax=docker/dockerfile:1

# ---- Bước build: cài thư viện bằng uv ----
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app

# Cài thư viện trước (tách layer để đổi code không phải cài lại)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# ---- Bước chạy ----
FROM python:3.12-slim
# Hugging Face Spaces chạy container với UID 1000
RUN useradd -m -u 1000 app \
    && mkdir -p /home/app/.cache/huggingface /app/data \
    && chown -R app:app /home/app/.cache /app
WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
# Dữ liệu dựng sẵn trên máy (Qdrant local, SQLite) được đóng gói vào image để bản deploy chỉ đọc.
COPY --chown=app:app data/ /app/data/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/data \
    HF_HOME=/home/app/.cache/huggingface

USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["uvicorn", "tuyensinh.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
