"""
BERTopic wrapper for topic modeling with pre-computed embeddings.
Adapted from topic-modeling.faculytics/src/topic_model.py
"""

import logging
from typing import Any

import numpy as np
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from .config import DEFAULT_PARAMS

logger = logging.getLogger(__name__)

# Stop words: English + high-frequency Cebuano/Tagalog function words
# These are grammatical tokens that dominate c-TF-IDF without adding topic signal
MULTILINGUAL_STOP_WORDS = list({
    # Role/title words
    "propesor", "estudyante", "guro", "magaaral", "maestra", "maestro",
    "students", "student", "maam", "sir", "professor", "teacher", "instructor",
    "faculty", "atty", "miss",
    # English stop words (core subset)
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "that", "this",
    "these", "those", "it", "its", "i", "me", "my", "we", "our", "you",
    "your", "he", "she", "they", "his", "her", "their", "as", "not",
    "also", "so", "if", "than", "then", "when", "what", "how", "all",
    "very", "just", "more", "about", "up", "out", "no", "him",
    # Cebuano function words + possessives + filler
    "ang", "nga", "sa", "ni", "si", "ug", "og", "kay", "ba", "na",
    "man", "lang", "ra", "jud", "gyud", "ko", "mo", "mi", "siya",
    "niya", "nako", "imo", "iya", "ato", "nato", "namo", "kamo",
    "sila", "nila", "kang", "kanang", "kanila", "dili", "wala",
    "naa", "adto", "diri", "didto", "pero",
    # Cebuano possessives
    "iyang", "kanyang", "among", "atong", "ilang", "inyong",
    # High-frequency Cebuano filler
    "kaayo", "mao", "bawat", "aralin", "klase", "gawain",
    # Generic action verbs
    "nagbibigay", "naguse", "nagtatakda", "naggagamit", "ginagamit",
    "nagbibigay ng", "ginagamit ng",
    # Generic English filler
    "really", "much", "am", "way", "feel", "truly",
    # Generic time/context words
    "every", "during", "specific", "even",
    # Tagalog function words
    "ng", "mga", "ay", "nang", "rin", "din", "po", "ho", "yung",
    "kasi", "naman", "pa", "pag", "kung", "dahil", "para",
    "hindi", "at", "o", "namin", "natin",
    "nila", "ito", "iyon", "yon", "dito", "doon",
})


def run_bertopic(
    embeddings: np.ndarray,
    texts: list[str],
    params: dict[str, Any],
    embed_model: SentenceTransformer,
) -> BERTopic:
    """
    Run BERTopic with pre-computed embeddings.

    Args:
        embeddings: Pre-computed LaBSE embeddings (n_samples, 768).
        texts: Cleaned text documents.
        params: Hyperparameters (merged with DEFAULT_PARAMS).
        embed_model: SentenceTransformer for KeyBERTInspired keyword extraction.

    Returns:
        Fitted BERTopic model.
    """
    p = {**DEFAULT_PARAMS, **params}
    min_topic_size = p["min_topic_size"]
    nr_topics = p.get("nr_topics")
    umap_n_neighbors = p["umap_n_neighbors"]
    umap_n_components = p["umap_n_components"]

    logger.info(
        f"BERTopic params: min_topic_size={min_topic_size}, nr_topics={nr_topics}, "
        f"umap_n_neighbors={umap_n_neighbors}, umap_n_components={umap_n_components}"
    )

    umap_model = UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=umap_n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    min_samples = p.get("hdbscan_min_samples", 5)
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words=MULTILINGUAL_STOP_WORDS,
        min_df=2,
    )

    representation_model = KeyBERTInspired()

    topic_model = BERTopic(
        embedding_model=embed_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        nr_topics=nr_topics,
        verbose=True,
    )

    np.random.seed(42)

    logger.info(f"Fitting BERTopic on {len(texts)} documents...")
    topic_model.fit_transform(texts, embeddings=embeddings)

    n_topics = len(set(topic_model.topics_)) - (1 if -1 in topic_model.topics_ else 0)
    n_outliers = sum(1 for t in topic_model.topics_ if t == -1)
    logger.info(f"Found {n_topics} topics, {n_outliers} outliers ({n_outliers / len(texts) * 100:.1f}%)")

    return topic_model


def extract_topic_info(model: BERTopic) -> list[dict[str, Any]]:
    """Extract topic information from a fitted BERTopic model."""
    topic_info = model.get_topic_info()
    results = []

    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]
        if topic_id == -1:
            continue

        topic_words = model.get_topic(topic_id)
        keywords = [word for word, _ in topic_words[:10]] if topic_words else []

        results.append({
            "topicIndex": topic_id,
            "rawLabel": row.get("Name", f"Topic_{topic_id}"),
            "keywords": keywords,
            "docCount": row["Count"],
        })

    return results


def get_assignments(
    model: BERTopic,
    texts: list[str],
    submission_ids: list[str],
    embeddings: np.ndarray,
) -> list[dict[str, Any]]:
    """Build per-document topic assignments with probabilities."""
    topics = model.topics_
    probs = model.probabilities_

    assignments = []
    for i, (topic_id, submission_id) in enumerate(zip(topics, submission_ids)):
        if topic_id == -1:
            continue

        prob = float(probs[i]) if probs is not None and i < len(probs) else 0.0
        # For reduced topics, probs is a matrix — take the max for the assigned topic
        if probs is not None and probs.ndim == 2:
            prob = float(probs[i].max()) if i < len(probs) else 0.0

        assignments.append({
            "submissionId": submission_id,
            "topicIndex": int(topic_id),
            "probability": round(prob, 4),
        })

    return assignments
