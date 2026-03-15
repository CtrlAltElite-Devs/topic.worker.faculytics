FROM runpod/pytorch:2.2.0-py3.11-cuda12.1.1-devel-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml .python-version ./
# Lock file may not exist yet during initial build
COPY uv.loc[k] ./

RUN uv sync --frozen --no-dev --no-install-project || uv sync --no-dev --no-install-project

# Bake LaBSE into image (~1.8 GB) to avoid cold-start download
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/LaBSE')"

COPY src/ src/

CMD ["uv", "run", "python", "-m", "src.handler"]
