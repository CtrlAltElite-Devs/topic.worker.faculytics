"""
Pydantic request/response schemas.
Matches the Zod schemas in api.faculytics/src/modules/analysis/dto/topic-model-worker.dto.ts
"""

from pydantic import BaseModel, ConfigDict

# --- Request ---


class RequestItem(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    submissionId: str
    text: str
    embedding: list[float]


class RequestParams(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Existing (1.0.0)
    min_topic_size: int | None = None
    nr_topics: int | None = None
    umap_n_neighbors: int | None = None
    umap_n_components: int | None = None
    hdbscan_min_samples: int | None = None
    # NEW (Tier 1, 1.1.0)
    tier1_guided: bool | None = None
    match_threshold: float | None = None
    primary_threshold: float | None = None
    secondary_threshold: float | None = None
    secondary_gap_max: float | None = None
    mmr_diversity: float | None = None


class TopicModelRequest(BaseModel):
    """Tolerates extra envelope fields (jobId, version, type, metadata, publishedAt)."""

    model_config = ConfigDict(extra="ignore")

    items: list[RequestItem]
    params: RequestParams | None = None


# --- Response ---


class TopicItem(BaseModel):
    # Existing (1.0.0)
    topicIndex: int
    rawLabel: str
    keywords: list[str]
    docCount: int
    # NEW (Tier 1, 1.1.0) — optional for legacy-path compat
    aspectLabel: str | None = None
    aspectSimilarity: float | None = None
    isEmergent: bool | None = None
    keywordsMmr: list[str] | None = None


class AssignmentItem(BaseModel):
    # Existing (1.0.0)
    submissionId: str
    topicIndex: int
    probability: float
    # NEW (Tier 1, 1.1.0) — optional for legacy-path compat
    secondaryTopicIndex: int | None = None
    secondaryProbability: float | None = None
    isMultiTopic: bool | None = None


class AspectMappingEntry(BaseModel):
    aspect: str
    similarity: float


class MetricsResult(BaseModel):
    npmi_coherence: float | None = None
    topic_diversity: float | None = None
    outlier_ratio: float | None = None
    silhouette_score: float | None = None
    embedding_coherence: float | None = None


class TopicModelResponse(BaseModel):
    version: str
    status: str  # "completed" | "failed"
    topics: list[TopicItem] | None = None
    assignments: list[AssignmentItem] | None = None
    metrics: MetricsResult | None = None
    outlierCount: int | None = None
    error: str | None = None
    completedAt: str
    # NEW (Tier 1, 1.1.0)
    aspectMapping: dict[str, AspectMappingEntry] | None = None
