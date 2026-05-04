"""
RunPod serverless handler for topic modeling.
"""

import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import runpod
from sentence_transformers import SentenceTransformer

from .config import DEFAULT_PARAMS, DEVICE, LABSE_MODEL, WORKER_VERSION
from .evaluate import compute_metrics
from .models import TopicModelRequest, TopicModelResponse
from .topic_model import (
    TIER1_DEFAULTS,
    assign_multi_topic,
    audit_aspect_coverage,
    build_assignments_from_multi,
    build_bertopic_guided,
    extract_topic_info,
    extract_topic_info_multi,
    get_assignments,
    map_topics_to_aspects,
    run_bertopic,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load LaBSE once at container start (used by KeyBERTInspired for keyword extraction)
logger.info(f"Loading {LABSE_MODEL} on {DEVICE}...")
embed_model = SentenceTransformer(LABSE_MODEL, device=DEVICE)
logger.info("Model loaded.")


def _fail(error: str) -> dict:
    """Build a domain-error response (no BullMQ retry)."""
    # exclude_none=True keeps the failure payload to only populated keys.
    # Note: 1.0.0 _fail used model_dump() (no exclude_none), so 1.1.0 omits
    # explicit nulls that 1.0.0 included (topics, assignments, metrics,
    # outlierCount). Optional-typed schemas (Zod, Pydantic) accept both;
    # consumers using `'foo' in obj` membership checks will see a change.
    return TopicModelResponse(
        version=WORKER_VERSION,
        status="failed",
        error=error,
        completedAt=datetime.now(UTC).isoformat(),
    ).model_dump(exclude_none=True)


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

        # Auto-scale params for small datasets so HDBSCAN can find clusters
        n_items = len(request.items)
        if n_items < min_topic_size * 4:
            scaled_min = max(3, n_items // 10)
            if scaled_min < min_topic_size:
                logger.info(
                    f"Small dataset ({n_items} items): scaling min_topic_size "
                    f"{min_topic_size} → {scaled_min}"
                )
                params["min_topic_size"] = scaled_min
                min_topic_size = scaled_min

        if n_items < 100:
            # Scale UMAP neighbors — too many neighbors washes out local structure
            scaled_neighbors = max(5, n_items // 4)
            if scaled_neighbors < params["umap_n_neighbors"]:
                logger.info(
                    f"Small dataset ({n_items} items): scaling umap_n_neighbors "
                    f"{params['umap_n_neighbors']} → {scaled_neighbors}"
                )
                params["umap_n_neighbors"] = scaled_neighbors

            # Reduce UMAP dimensions — high-dim output is too sparse for HDBSCAN
            max_components = 5
            if params["umap_n_components"] > max_components:
                logger.info(
                    f"Small dataset ({n_items} items): scaling umap_n_components "
                    f"{params['umap_n_components']} → {max_components}"
                )
                params["umap_n_components"] = max_components

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

        # Run BERTopic — branch on tier1_guided
        tier1_guided = bool(params.get("tier1_guided", TIER1_DEFAULTS["tier1_guided"]))

        aspect_mapping_payload: dict[str, dict[str, Any]] | None = None

        if tier1_guided:
            # Pre-fit aspect coverage audit (informational, non-gating)
            coverage = audit_aspect_coverage(texts)
            logger.info(f"aspect coverage: {coverage}")
            weak = [a for a, c in coverage.items() if c < 50]
            if weak:
                logger.warning(f"aspects with <50 hits — review seeds: {weak}")

            model = build_bertopic_guided(embeddings, texts, params, embed_model)

            # Compute aspect mapping once and reuse for both topics_info and the
            # response payload. (Earlier draft computed it twice — saved here.)
            aspect_mapping = map_topics_to_aspects(
                model,
                embed_model,
                match_threshold=params.get("match_threshold", TIER1_DEFAULTS["match_threshold"]),
                representation="Main",
            )

            topics_info = extract_topic_info_multi(model, aspect_mapping=aspect_mapping)
            if len(topics_info) == 0:
                return _fail("BERTopic produced 0 topics")

            multi = assign_multi_topic(
                model,
                texts,
                primary_threshold=params.get(
                    "primary_threshold", TIER1_DEFAULTS["primary_threshold"]
                ),
                secondary_threshold=params.get(
                    "secondary_threshold", TIER1_DEFAULTS["secondary_threshold"]
                ),
                secondary_gap_max=params.get(
                    "secondary_gap_max", TIER1_DEFAULTS["secondary_gap_max"]
                ),
            )
            assignments = build_assignments_from_multi(multi, submission_ids)

            # Stringify topic-id keys for JSON object semantics in the response.
            aspect_mapping_payload = {
                str(tid): {"aspect": v["aspect"], "similarity": v["similarity"]}
                for tid, v in aspect_mapping.items()
            }
        else:
            # Legacy path — preserved verbatim for atomic rollback
            model = run_bertopic(embeddings, texts, params, embed_model)
            topics_info = extract_topic_info(model)
            if len(topics_info) == 0:
                return _fail("BERTopic produced 0 topics")
            assignments = get_assignments(model, texts, submission_ids, embeddings)

        outlier_count = sum(1 for t in model.topics_ if t == -1)

        # Compute metrics — unchanged
        metrics = compute_metrics(model, model.topics_, texts, embeddings, embed_model=embed_model)

        response = TopicModelResponse(
            version=WORKER_VERSION,
            status="completed",
            topics=topics_info,
            assignments=assignments,
            metrics=metrics,
            outlierCount=outlier_count,
            aspectMapping=aspect_mapping_payload,  # None on legacy path
            completedAt=datetime.now(UTC).isoformat(),
        )

        # exclude_none=True keeps the legacy-path payload byte-identical to 1.0.0
        # (modulo the bumped `version` string).
        return response.model_dump(exclude_none=True)

    except Exception:
        logger.exception("Unexpected error in handler")
        # Infrastructure error — let RunPod return error status → BullMQ will retry
        raise


runpod.serverless.start({"handler": handler})
