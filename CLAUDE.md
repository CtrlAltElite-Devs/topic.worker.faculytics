# CLAUDE.md

## Project Overview

BERTopic topic modeling worker for the Faculytics analysis pipeline. Deployed on RunPod serverless. Receives pre-cleaned text + pre-computed LaBSE embeddings, runs BERTopic clustering, and returns topics, assignments, and quality metrics.

## Main Caller

The NestJS API at `api.faculytics/` dispatches topic modeling jobs via BullMQ. The `TopicModelProcessor` (extending `RunPodBatchProcessor`) POSTs to this worker's RunPod endpoint.

**Contract schemas in the API repo:**
- `src/modules/analysis/dto/topic-model-worker.dto.ts` — request/response schemas (Zod)
- `src/modules/analysis/processors/topic-model.processor.ts` — persistence logic

## Common Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Lint
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Local smoke test (calls handler() directly, bypasses RunPod)
uv run python test_local.py

# Docker
docker build -t topic-worker .
docker run --gpus all topic-worker
```

## Local Testing

There are two ways to test during development:

### Smoke test (real BERTopic, no RunPod)

`test_local.py` calls `handler()` directly with 30 synthetic items and clustered embeddings. It uses reduced params (`min_topic_size=5`, `nr_topics=3`) so it works with a small dataset.

```bash
uv sync
uv run python test_local.py
```

First run downloads LaBSE (~1.8 GB). Subsequent runs use the cached model. Works on CPU (slow) or GPU (fast). Prints topics, assignments, and quality metrics.

### Mock worker (no BERTopic, tests API integration)

The API repo's mock worker (`api.faculytics/mock-worker/`) has a `/topic-model` endpoint that returns fake results. Use this to test the full API pipeline flow without running the real worker:

```bash
cd ../api.faculytics
docker compose up          # starts Redis + mock worker
# Set TOPIC_MODEL_WORKER_URL=http://localhost:3001/topic-model in .env
npm run start:dev          # then trigger a pipeline via the API
```

## Architecture

- **Runtime**: RunPod serverless handler (`src/handler.py`)
- **Pipeline**: BERTopic with UMAP + HDBSCAN + KeyBERTInspired
- **Model**: LaBSE (baked into Docker image, used for KeyBERT keyword extraction)
- **Error strategy**: Domain errors return `status: "failed"` in the response (RunPod wraps in `output`, BullMQ won't retry). Infrastructure errors raise exceptions (RunPod returns error → BullMQ retries).

### File Structure

```
src/
├── config.py       # LABSE_MODEL, DEVICE, WORKER_VERSION, DEFAULT_PARAMS (RUN 012)
├── models.py       # Pydantic request/response schemas (match Zod DTOs)
├── topic_model.py  # run_bertopic(), extract_topic_info(), get_assignments()
├── evaluate.py     # compute_metrics() — NPMI, diversity, outlier ratio, silhouette, embedding coherence
└── handler.py      # RunPod handler entry point, model loading
```

### Key Design Decisions

- **No preprocessing**: Text arrives pre-cleaned from the API (`cleanedComment`). Embeddings were also generated on cleaned text.
- **RUN 012 defaults**: `min_topic_size=15, nr_topics=20, umap_n_neighbors=20, umap_n_components=10` — proven optimal from experimentation.
- **LaBSE loaded at container start**: Global singleton, needed for KeyBERTInspired keyword extraction on GPU.

## Configuration

No environment variables needed — configuration is in `src/config.py`. RunPod provides GPU and networking.
