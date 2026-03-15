"""
Topic model evaluation metrics.
Adapted from topic-modeling.faculytics/src/evaluate.py
"""

import logging
from typing import Any

import numpy as np
from bertopic import BERTopic
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


def compute_npmi_coherence(model: BERTopic, texts: list[str], top_n: int = 10) -> float:
    """NPMI coherence — measures keyword co-occurrence. Target: > 0.1"""
    tokenized = [text.lower().split() for text in texts]
    dictionary = Dictionary(tokenized)

    topic_words = []
    for topic_id in model.get_topic_info()["Topic"]:
        if topic_id == -1:
            continue
        words = model.get_topic(topic_id)
        if words:
            topic_words.append([word for word, _ in words[:top_n]])

    if not topic_words:
        return 0.0

    try:
        coherence_model = CoherenceModel(
            topics=topic_words,
            texts=tokenized,
            dictionary=dictionary,
            coherence="c_npmi",
            window_size=10,
        )
        return coherence_model.get_coherence()
    except Exception as e:
        logger.warning(f"Coherence calculation failed: {e}")
        return 0.0


def compute_topic_diversity(model: BERTopic, top_n: int = 10) -> float:
    """Ratio of unique words across all topics. Target: > 0.7"""
    all_words = []
    for topic_id in model.get_topic_info()["Topic"]:
        if topic_id == -1:
            continue
        words = model.get_topic(topic_id)
        if words:
            all_words.extend([word for word, _ in words[:top_n]])

    if not all_words:
        return 0.0
    return len(set(all_words)) / len(all_words)


def compute_outlier_ratio(model: BERTopic) -> float:
    """Ratio of documents assigned to outlier topic (-1). Target: < 0.2"""
    topic_info = model.get_topic_info()
    outlier_row = topic_info[topic_info["Topic"] == -1]
    if outlier_row.empty:
        return 0.0
    return outlier_row["Count"].values[0] / topic_info["Count"].sum()


def compute_silhouette(topics: list[int], embeddings: np.ndarray) -> float:
    """Silhouette score for topic clusters. Range: -1 to 1, higher is better."""
    valid_mask = np.array(topics) != -1
    valid_topics = np.array(topics)[valid_mask]
    valid_embeddings = embeddings[valid_mask]

    if len(set(valid_topics)) < 2 or len(valid_topics) < 10:
        return 0.0

    try:
        return silhouette_score(valid_embeddings, valid_topics, metric="cosine")
    except Exception as e:
        logger.warning(f"Silhouette calculation failed: {e}")
        return 0.0


def compute_embedding_coherence(model: BERTopic, embed_model=None) -> float:
    """
    Embedding-based coherence: average pairwise cosine similarity
    between top keyword embeddings per topic. Target: > 0.5
    """
    if embed_model is None:
        return 0.0

    scores = []
    for topic_id in model.get_topic_info()["Topic"]:
        if topic_id == -1:
            continue
        words = model.get_topic(topic_id)
        if not words or len(words) < 2:
            continue
        keywords = [w for w, _ in words[:10]]
        try:
            vecs = embed_model.encode(keywords, show_progress_bar=False, normalize_embeddings=True)
            n = len(vecs)
            sim_sum = 0.0
            count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    sim_sum += float(np.dot(vecs[i], vecs[j]))
                    count += 1
            if count > 0:
                scores.append(sim_sum / count)
        except Exception:
            continue

    return round(float(np.mean(scores)), 4) if scores else 0.0


def compute_metrics(
    model: BERTopic,
    topics: list[int],
    texts: list[str],
    embeddings: np.ndarray,
    embed_model=None,
) -> dict[str, Any]:
    """Compute all evaluation metrics for a topic model."""
    logger.info("Computing evaluation metrics...")

    return {
        "embedding_coherence": compute_embedding_coherence(model, embed_model=embed_model),
        "npmi_coherence": round(compute_npmi_coherence(model, texts), 4),
        "topic_diversity": round(compute_topic_diversity(model), 4),
        "outlier_ratio": round(compute_outlier_ratio(model), 4),
        "silhouette_score": round(compute_silhouette(topics, embeddings), 4),
    }
