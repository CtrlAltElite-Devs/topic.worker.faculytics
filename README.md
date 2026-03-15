# Topic Modeling Worker — Faculytics

BERTopic topic modeling worker for the Faculytics analysis pipeline. Deployed on [RunPod serverless](https://docs.runpod.io/serverless/overview). Receives pre-cleaned text with pre-computed [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) embeddings, runs BERTopic clustering, and returns topics, assignments, and quality metrics.

## API Contract

The worker runs as a RunPod serverless handler. The API wraps the payload in `{ "input": ... }` and receives the result in `{ "output": ... }`.

### Request (inside `input`)

```json
{
  "jobId": "uuid",
  "version": "1.0",
  "type": "topic-model",
  "items": [
    {
      "submissionId": "uuid",
      "text": "Maayo kaayo siya mo explain sa among lessons",
      "embedding": [0.01, 0.02, "... (768 floats)"]
    }
  ],
  "params": {
    "min_topic_size": 15,
    "nr_topics": 20,
    "umap_n_neighbors": 20,
    "umap_n_components": 10
  },
  "metadata": { "pipelineId": "uuid", "runId": "uuid" },
  "publishedAt": "2026-03-16T00:00:00.000Z"
}
```

`params` is optional — defaults to RUN 012 values when omitted.

### Success response (inside `output`)

```json
{
  "version": "1.0.0",
  "status": "completed",
  "topics": [
    {
      "topicIndex": 0,
      "rawLabel": "0_teaching_clarity",
      "keywords": ["explain", "clear", "lessons", "understand", "method"],
      "docCount": 42
    }
  ],
  "assignments": [
    { "submissionId": "uuid", "topicIndex": 0, "probability": 0.87 }
  ],
  "metrics": {
    "npmi_coherence": 0.15,
    "topic_diversity": 0.78,
    "outlier_ratio": 0.12,
    "silhouette_score": 0.35,
    "embedding_coherence": 0.55
  },
  "outlierCount": 18,
  "completedAt": "2026-03-16T00:05:00.000Z"
}
```

### Error response (inside `output` — domain errors, no BullMQ retry)

```json
{
  "version": "1.0.0",
  "status": "failed",
  "error": "Received 8 items, need at least 15 (min_topic_size) for topic modeling",
  "completedAt": "2026-03-16T00:00:01.000Z"
}
```

Domain errors (too few items, zero-vector embeddings, 0 topics produced) return `status: "failed"` so BullMQ does not retry. Infrastructure errors (CUDA OOM, model corruption) raise exceptions — RunPod returns an error status and BullMQ retries with exponential backoff.

## Quick Start

### Local smoke test

Calls `handler()` directly with synthetic data — no RunPod needed.

```bash
uv sync
uv run python test_local.py
```

First run downloads LaBSE (~1.8 GB). Works on CPU (slow) or GPU (fast). Prints topics, assignments, and quality metrics.

### Unit tests

```bash
uv run pytest
```

### Lint & format

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

### Docker

```bash
docker build -t topic-worker .
docker run --gpus all topic-worker
```

The Dockerfile bakes LaBSE into the image to avoid cold-start downloads on RunPod.

### Mock worker (API integration testing)

The API repo's mock worker returns fake topic results without running BERTopic:

```bash
cd ../api.faculytics
docker compose up
# Set TOPIC_MODEL_WORKER_URL=http://localhost:3001/topic-model in .env
npm run start:dev
```

## Default Parameters (RUN 012)

These defaults were tuned during experimentation in `topic-modeling.faculytics/`:

| Parameter | Default | Description |
|---|---|---|
| `min_topic_size` | `15` | Minimum documents per cluster (HDBSCAN `min_cluster_size`) |
| `nr_topics` | `20` | Target number of topics (BERTopic topic reduction) |
| `umap_n_neighbors` | `20` | UMAP locality parameter |
| `umap_n_components` | `10` | UMAP output dimensions |

Override via the `params` field in the request payload.

## Architecture

```
src/
├── config.py       # LABSE_MODEL, DEVICE, WORKER_VERSION, DEFAULT_PARAMS
├── models.py       # Pydantic request/response schemas (match API Zod DTOs)
├── topic_model.py  # run_bertopic(), extract_topic_info(), get_assignments()
├── evaluate.py     # compute_metrics() — NPMI, diversity, outlier ratio, silhouette, embedding coherence
└── handler.py      # RunPod handler entry point, LaBSE loading at startup
```

- **LaBSE** is loaded once at container start as a global singleton — used by [KeyBERTInspired](https://maartengr.github.io/BERTopic/getting_started/representation/representation.html#keybertinspired) for language-agnostic keyword extraction
- **No text preprocessing** in the worker — text arrives pre-cleaned from the API (`cleanedComment`). Embeddings were also generated on cleaned text, so everything is aligned
- **Multilingual stop words** — English, Cebuano, and Tagalog function words are excluded from c-TF-IDF to improve topic discrimination
- **Contract compliance** — Pydantic schemas match the Zod schemas in `api.faculytics/src/modules/analysis/dto/topic-model-worker.dto.ts`

## Quality Metrics

The worker computes and returns five quality metrics:

| Metric | Target | Description |
|---|---|---|
| `npmi_coherence` | > 0.1 | Keyword co-occurrence coherence (NPMI) |
| `topic_diversity` | > 0.7 | Ratio of unique keywords across topics |
| `outlier_ratio` | < 0.2 | Fraction of documents not assigned to any topic |
| `silhouette_score` | higher is better | Cluster separation quality (-1 to 1) |
| `embedding_coherence` | > 0.5 | Pairwise cosine similarity of keyword embeddings per topic |
