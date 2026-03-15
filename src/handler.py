"""
RunPod serverless handler for topic modeling.
"""

import logging
from datetime import datetime, timezone

import numpy as np
import runpod
from sentence_transformers import SentenceTransformer

from .config import DEFAULT_PARAMS, DEVICE, LABSE_MODEL, WORKER_VERSION
from .evaluate import compute_metrics
from .models import TopicModelRequest, TopicModelResponse
from .topic_model import extract_topic_info, get_assignments, run_bertopic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load LaBSE once at container start (used by KeyBERTInspired for keyword extraction)
logger.info(f"Loading {LABSE_MODEL} on {DEVICE}...")
embed_model = SentenceTransformer(LABSE_MODEL, device=DEVICE)
logger.info("Model loaded.")


def _fail(error: str) -> dict:
    """Build a domain-error response (no BullMQ retry)."""
    return TopicModelResponse(
        version=WORKER_VERSION,
        status="failed",
        error=error,
        completedAt=datetime.now(timezone.utc).isoformat(),
    ).model_dump()


def handler(event: dict) -> dict:
    """RunPod handler entry point."""
    try:
        input_data = event.get("input", event)

        # Parse + validate
        request = TopicModelRequest.model_validate(input_data)

        # Merge params with defaults
        params = {**DEFAULT_PARAMS}
        if request.params:
            provided = request.params.model_dump(exclude_none=True)
            params.update(provided)

        min_topic_size = params["min_topic_size"]

        # Validate minimum item count
        if len(request.items) < min_topic_size:
            return _fail(
                f"Received {len(request.items)} items, need at least {min_topic_size} "
                f"(min_topic_size) for topic modeling"
            )

        # Extract data
        texts = [item.text for item in request.items]
        submission_ids = [item.submissionId for item in request.items]
        embeddings = np.array([item.embedding for item in request.items], dtype=np.float32)

        # Validate embeddings
        if embeddings.shape[1] != 768:
            return _fail(f"Expected 768-dim embeddings, got {embeddings.shape[1]}")

        # Check for zero vectors
        norms = np.linalg.norm(embeddings, axis=1)
        zero_mask = norms < 1e-8
        if zero_mask.all():
            return _fail("All embeddings are zero vectors")
        if zero_mask.any():
            logger.warning(f"Filtering {zero_mask.sum()} zero-vector embeddings")
            valid_mask = ~zero_mask
            texts = [t for t, v in zip(texts, valid_mask) if v]
            submission_ids = [s for s, v in zip(submission_ids, valid_mask) if v]
            embeddings = embeddings[valid_mask]

            if len(texts) < min_topic_size:
                return _fail(
                    f"After filtering zero vectors, only {len(texts)} items remain "
                    f"(need {min_topic_size})"
                )

        # Run BERTopic
        model = run_bertopic(embeddings, texts, params, embed_model)

        # Extract results
        topics_info = extract_topic_info(model)

        if len(topics_info) == 0:
            return _fail("BERTopic produced 0 topics")

        assignments = get_assignments(model, texts, submission_ids, embeddings)

        outlier_count = sum(1 for t in model.topics_ if t == -1)

        # Compute metrics
        metrics = compute_metrics(
            model, model.topics_, texts, embeddings, embed_model=embed_model
        )

        response = TopicModelResponse(
            version=WORKER_VERSION,
            status="completed",
            topics=topics_info,
            assignments=assignments,
            metrics=metrics,
            outlierCount=outlier_count,
            completedAt=datetime.now(timezone.utc).isoformat(),
        )

        return response.model_dump()

    except Exception as e:
        logger.exception("Unexpected error in handler")
        # Infrastructure error — let RunPod return error status → BullMQ will retry
        raise


runpod.serverless.start({"handler": handler})
