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
MULTILINGUAL_STOP_WORDS = list(
    {
        # Role/title words
        "propesor",
        "estudyante",
        "guro",
        "magaaral",
        "maestra",
        "maestro",
        "students",
        "student",
        "maam",
        "sir",
        "professor",
        "teacher",
        "instructor",
        "faculty",
        "atty",
        "miss",
        # English stop words (core subset)
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "they",
        "his",
        "her",
        "their",
        "as",
        "not",
        "also",
        "so",
        "if",
        "than",
        "then",
        "when",
        "what",
        "how",
        "all",
        "very",
        "just",
        "more",
        "about",
        "up",
        "out",
        "no",
        "him",
        # Cebuano function words + possessives + filler
        "ang",
        "nga",
        "sa",
        "ni",
        "si",
        "ug",
        "og",
        "kay",
        "ba",
        "na",
        "man",
        "lang",
        "ra",
        "jud",
        "gyud",
        "ko",
        "mo",
        "mi",
        "siya",
        "niya",
        "nako",
        "imo",
        "iya",
        "ato",
        "nato",
        "namo",
        "kamo",
        "sila",
        "nila",
        "kang",
        "kanang",
        "kanila",
        "dili",
        "wala",
        "naa",
        "adto",
        "diri",
        "didto",
        "pero",
        # Cebuano possessives
        "iyang",
        "kanyang",
        "among",
        "atong",
        "ilang",
        "inyong",
        # High-frequency Cebuano filler
        "kaayo",
        "mao",
        "bawat",
        "aralin",
        "klase",
        "gawain",
        # Generic action verbs
        "nagbibigay",
        "naguse",
        "nagtatakda",
        "naggagamit",
        "ginagamit",
        "nagbibigay ng",
        "ginagamit ng",
        # Generic English filler
        "really",
        "much",
        "am",
        "way",
        "feel",
        "truly",
        # Generic time/context words
        "every",
        "during",
        "specific",
        "even",
        # Tagalog function words
        "ng",
        "mga",
        "ay",
        "nang",
        "rin",
        "din",
        "po",
        "ho",
        "yung",
        "kasi",
        "naman",
        "pa",
        "pag",
        "kung",
        "dahil",
        "para",
        "hindi",
        "at",
        "o",
        "namin",
        "natin",
        "nila",
        "ito",
        "iyon",
        "yon",
        "dito",
        "doon",
    }
)


# ─── Tier 1 aspect taxonomy ─────────────────────────────────────────
# Hardcoded module constants for v1. Per-pipeline configurable taxonomy
# is Tier 2.5 (out of scope here). Order in ASPECT_NAMES MUST match
# SEED_TOPIC_LIST — BERTopic uses positional alignment.
#
# Source of truth: experiments/topic-modeling-notebook/lib.py:238-282
# (the exact ACADEMIC_ASPECTS dict that produced run_nb_015_real_filtered).
ACADEMIC_ASPECTS: dict[str, list[str]] = {
    "teaching_clarity": [
        "malinaw",
        "explain",
        "explanation",
        "understand",
        "masabtan",
        "paliwanag",
        "klaro",
        "lecture",
        "discussion",
        "delivery",
    ],
    "assessment_grading": [
        "grade",
        "grading",
        "exam",
        "quiz",
        "score",
        "marks",
        "rubric",
        "criteria",
        "fair",
        "grado",
        "pagsulit",
    ],
    "workload_pacing": [
        "workload",
        "deadline",
        "many",
        "daghan",
        "marami",
        "submission",
        "requirement",
        "buhat",
        "pace",
        "fast",
        "slow",
        "rushed",
    ],
    "teaching_methodology": [
        "teaching",
        "pagtuturo",
        "pagtudlo",
        "method",
        "style",
        "paraan",
        "approach",
        "strategy",
        "technique",
        "engaging",
        "boring",
    ],
    "instructor_attitude": [
        "approachable",
        "kind",
        "buotan",
        "mabait",
        "intimidating",
        "strict",
        "patient",
        "respectful",
        "professional",
        "respeto",
        "rude",
    ],
    "responsiveness_support": [
        "questions",
        "responsive",
        "answer",
        "tubag",
        "sagot",
        "support",
        "help",
        "tabang",
        "tulong",
        "available",
        "reply",
    ],
    "punctuality_attendance": [
        "late",
        "absent",
        "regular",
        "kanunay",
        "lagi",
        "palagi",
        "tardy",
        "minutes",
        "on time",
        "schedule",
        "miss",
    ],
    "lms_digital_tools": [
        "portal",
        "lms",
        "online",
        "moodle",
        "post",
        "upload",
        "digital",
        "platform",
        "system",
        "module",
        "asynchronous",
    ],
    "real_world_relevance": [
        "real world",
        "practical",
        "application",
        "totoong",
        "kinabuhi",
        "buhay",
        "example",
        "halimbawa",
        "relevant",
        "useful",
    ],
    "language_communication": [
        "english",
        "cebuano",
        "bisaya",
        "tagalog",
        "speak",
        "istorya",
        "pronunciation",
        "fluent",
        "language",
        "communication",
    ],
}

ASPECT_NAMES: list[str] = list(ACADEMIC_ASPECTS.keys())
SEED_TOPIC_LIST: list[list[str]] = list(ACADEMIC_ASPECTS.values())

# Tier 1 default thresholds (validated in run_nb_015_real_filtered).
# NOTE: tier1_guided defaults to False in 1.1.0 — the API caller must
# opt in with `params.tier1_guided: true`. Will flip to True in 1.2.0
# once the sibling API spec (Zod + persistence) is verified in production.
TIER1_DEFAULTS = {
    "tier1_guided": False,
    "match_threshold": 0.65,
    "primary_threshold": 0.20,
    "secondary_threshold": 0.15,
    "secondary_gap_max": 0.20,
    "mmr_diversity": 0.4,
    "hdbscan_min_samples": 5,
}


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
        min_df=1,
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
    outlier_pct = n_outliers / len(texts) * 100
    logger.info(f"Found {n_topics} topics, {n_outliers} outliers ({outlier_pct:.1f}%)")

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

        results.append(
            {
                "topicIndex": topic_id,
                "rawLabel": row.get("Name", f"Topic_{topic_id}"),
                "keywords": keywords,
                "docCount": row["Count"],
            }
        )

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

        assignments.append(
            {
                "submissionId": submission_id,
                "topicIndex": int(topic_id),
                "probability": round(prob, 4),
            }
        )

    return assignments


def build_bertopic_guided(
    embeddings: np.ndarray,
    texts: list[str],
    params: dict[str, Any],
    embed_model: SentenceTransformer,
) -> BERTopic:
    """
    Tier 1 guided BERTopic build.

    Differences from run_bertopic:
      * seed_topic_list = SEED_TOPIC_LIST (biases topic merging toward ASPECT_NAMES)
      * calculate_probabilities=True (required for approximate_distribution → multi-topic)
      * representation_model = {"Main": KeyBERTInspired(), "MMR": MaximalMarginalRelevance(...)}
        (gives every topic two keyword lists; "MMR" is exposed as keywordsMmr)

    All other wiring (UMAP, HDBSCAN, CountVectorizer) matches run_bertopic
    except CountVectorizer.min_df=2 (vs run_bertopic's min_df=1) — this
    matches lib.py and therefore matches what nb_015 produced.
    """
    from bertopic.representation import MaximalMarginalRelevance

    p = {**DEFAULT_PARAMS, **params}
    min_topic_size = p["min_topic_size"]
    nr_topics = p.get("nr_topics")
    umap_n_neighbors = p["umap_n_neighbors"]
    umap_n_components = p["umap_n_components"]
    hdbscan_min_samples = p.get("hdbscan_min_samples", TIER1_DEFAULTS["hdbscan_min_samples"])
    mmr_diversity = p.get("mmr_diversity", TIER1_DEFAULTS["mmr_diversity"])

    logger.info(
        f"[guided] BERTopic params: min_topic_size={min_topic_size}, "
        f"nr_topics={nr_topics}, umap_n_neighbors={umap_n_neighbors}, "
        f"umap_n_components={umap_n_components}, "
        f"hdbscan_min_samples={hdbscan_min_samples}, "
        f"mmr_diversity={mmr_diversity}, n_seeds={len(SEED_TOPIC_LIST)}"
    )

    umap_model = UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=umap_n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=hdbscan_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words=MULTILINGUAL_STOP_WORDS,
        min_df=2,  # NOTE: differs from run_bertopic (min_df=1); matches lib.py + nb_015
    )
    representation_model = {
        "Main": KeyBERTInspired(),
        "MMR": MaximalMarginalRelevance(diversity=mmr_diversity),
    }

    topic_model = BERTopic(
        embedding_model=embed_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        seed_topic_list=SEED_TOPIC_LIST,
        calculate_probabilities=True,
        nr_topics=nr_topics,
        verbose=True,
    )

    np.random.seed(42)  # match run_bertopic determinism
    logger.info(f"Fitting guided BERTopic on {len(texts)} documents...")
    topic_model.fit_transform(texts, embeddings=embeddings)

    n_topics = len(set(topic_model.topics_)) - (1 if -1 in topic_model.topics_ else 0)
    n_outliers = sum(1 for t in topic_model.topics_ if t == -1)
    outlier_pct = n_outliers / len(texts) * 100
    logger.info(f"[guided] Found {n_topics} topics, {n_outliers} outliers ({outlier_pct:.1f}%)")

    return topic_model


def audit_aspect_coverage(
    texts: list[str],
    aspects: dict[str, list[str]] | None = None,
) -> dict[str, int]:
    """
    Pre-fit sanity check: how many documents contain at least one seed word
    per aspect? Aspects with <50 hits are likely too narrow for the data —
    informational only, does NOT gate the fit.

    Returns: {aspect_name: doc_count}. Word-boundary match on lowercased text
    (so "late" does not match "related"). Multi-word seeds like "real world"
    work because re.escape handles the embedded space and \\b anchors at
    word transitions on both sides. Adversarial review F5.
    """
    import re

    if aspects is None:
        aspects = ACADEMIC_ASPECTS

    lowered = [t.lower() for t in texts]
    out: dict[str, int] = {}
    for name, seeds in aspects.items():
        # Compile one alternation per aspect so each text only scans once.
        pattern = re.compile(r"\b(?:" + "|".join(re.escape(s.lower()) for s in seeds) + r")\b")
        out[name] = sum(1 for t in lowered if pattern.search(t))
    return out


def map_topics_to_aspects(
    model: BERTopic,
    embed_model: SentenceTransformer,
    match_threshold: float = TIER1_DEFAULTS["match_threshold"],
    representation: str = "Main",
    top_n_keywords: int = 10,
) -> dict[int, dict[str, Any]]:
    """
    Map each discovered topic to the closest academic aspect by comparing
    topic-keyword centroids to aspect-seed centroids in LaBSE space.

    A topic is labeled "emergent" when best similarity < match_threshold.
    Outlier topic -1 always maps to {"aspect": "outlier", "similarity": 0.0}.

    Returns:
        {topic_id: {"aspect": str, "similarity": float}}
    """
    aspect_names = list(ACADEMIC_ASPECTS.keys())
    aspect_centroids = []
    for name in aspect_names:
        seed_embs = embed_model.encode(
            ACADEMIC_ASPECTS[name],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        aspect_centroids.append(np.asarray(seed_embs).mean(axis=0))
    aspect_matrix = np.vstack(aspect_centroids)
    # Re-L2-normalize after averaging (averaged unit vectors are not unit length).
    aspect_matrix = aspect_matrix / (np.linalg.norm(aspect_matrix, axis=1, keepdims=True) + 1e-12)

    out: dict[int, dict[str, Any]] = {}
    for topic_id in model.get_topic_info()["Topic"]:
        topic_id_int = int(topic_id)
        if topic_id_int == -1:
            out[-1] = {"aspect": "outlier", "similarity": 0.0}
            continue

        # Pull keywords from the requested representation when present (multi-rep
        # models built by build_bertopic_guided have topic_aspects_["Main"]).
        words: list[tuple[str, float]] = []
        if (
            representation
            and isinstance(model.topic_aspects_, dict)
            and representation in model.topic_aspects_
        ):
            words = model.topic_aspects_[representation].get(topic_id_int, [])
        if not words:
            words = model.get_topic(topic_id_int) or []

        if not words:
            out[topic_id_int] = {"aspect": "emergent", "similarity": 0.0}
            continue

        keywords = [w for w, _ in words[:top_n_keywords]]
        kw_embs = embed_model.encode(keywords, show_progress_bar=False, normalize_embeddings=True)
        topic_centroid = np.asarray(kw_embs).mean(axis=0)
        topic_centroid = topic_centroid / (np.linalg.norm(topic_centroid) + 1e-12)

        sims = aspect_matrix @ topic_centroid
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim < match_threshold:
            out[topic_id_int] = {"aspect": "emergent", "similarity": round(best_sim, 4)}
        else:
            out[topic_id_int] = {
                "aspect": aspect_names[best_idx],
                "similarity": round(best_sim, 4),
            }
    return out


def assign_multi_topic(
    model: BERTopic,
    texts: list[str],
    primary_threshold: float = TIER1_DEFAULTS["primary_threshold"],
    secondary_threshold: float = TIER1_DEFAULTS["secondary_threshold"],
    secondary_gap_max: float = TIER1_DEFAULTS["secondary_gap_max"],
    window: int = 4,
    stride: int = 1,
) -> list[dict[str, Any]]:
    """
    Per-document soft topic assignment via BERTopic.approximate_distribution.

    A document gets a secondary topic only when:
      - secondary_score >= secondary_threshold
      - (primary_score - secondary_score) < secondary_gap_max

    Requires the model to have been built with calculate_probabilities=True
    (build_bertopic_guided sets this).

    Returns: list of dicts with keys
        primary_topic, secondary_topic, primary_confidence,
        secondary_confidence, is_multi_topic
    Order matches input `texts` order. NO submissionId — that's added by
    build_assignments_from_multi.
    """
    assert getattr(model, "calculate_probabilities", False), (
        "assign_multi_topic requires a BERTopic model built with "
        "calculate_probabilities=True (use build_bertopic_guided)"
    )

    distributions, _ = model.approximate_distribution(texts, window=window, stride=stride)
    distributions = np.asarray(distributions)

    topic_ids = [t for t in model.get_topic_info()["Topic"].tolist() if t != -1]
    topic_ids_arr = np.array(topic_ids)

    # Defensive clip: if approximate_distribution emits more columns than there are
    # non-outlier topics, indexing topic_ids_arr by argsort over all columns would
    # raise IndexError. Clip to the topic-id count and actually do the "clipping"
    # the warning log promises. (Adversarial review F1.)
    if distributions.shape[1] != len(topic_ids):
        logger.warning(
            f"distribution shape {distributions.shape} does not match "
            f"non-outlier topic count {len(topic_ids)}; clipping to {len(topic_ids)} columns"
        )
        if distributions.ndim == 2 and distributions.shape[1] > len(topic_ids):
            distributions = distributions[:, : len(topic_ids)]

    results: list[dict[str, Any]] = []
    for dist in distributions:
        if dist.size == 0 or dist.sum() == 0.0:
            results.append(
                {
                    "primary_topic": -1,
                    "secondary_topic": None,
                    "primary_confidence": 0.0,
                    "secondary_confidence": None,
                    "is_multi_topic": False,
                }
            )
            continue

        order = np.argsort(dist)[::-1]
        top1_idx = order[0]
        top1_p = float(dist[top1_idx])
        primary = int(topic_ids_arr[top1_idx]) if top1_p >= primary_threshold else -1

        secondary: int | None = None
        top2_p_val: float | None = None
        if dist.size >= 2 and primary != -1:
            top2_idx = order[1]
            top2_p = float(dist[top2_idx])
            if top2_p >= secondary_threshold and (top1_p - top2_p) < secondary_gap_max:
                secondary = int(topic_ids_arr[top2_idx])
                top2_p_val = top2_p

        results.append(
            {
                "primary_topic": primary,
                "secondary_topic": secondary,
                "primary_confidence": round(top1_p, 4),
                "secondary_confidence": round(top2_p_val, 4) if top2_p_val is not None else None,
                "is_multi_topic": secondary is not None,
            }
        )
    return results


def extract_topic_info_multi(
    model: BERTopic,
    aspect_mapping: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Topic info extraction enriched with aspect labels and MMR keywords.

    Output shape mirrors extract_topic_info() (topicIndex/rawLabel/keywords/docCount)
    plus optional aspectLabel/aspectSimilarity/isEmergent/keywordsMmr fields.

    Reads model.topic_aspects_["Main"] for primary keywords (with fallback
    to model.get_topic) and model.topic_aspects_["MMR"] for MMR keywords
    when present. When aspect_mapping is None, omits the three aspect fields.
    """
    topic_info = model.get_topic_info()
    has_mmr = isinstance(model.topic_aspects_, dict) and "MMR" in model.topic_aspects_

    results: list[dict[str, Any]] = []
    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:
            continue

        main_words = model.get_topic(topic_id) or []
        keywords_main = [w for w, _ in main_words[:10]]
        keywords_mmr: list[str] = []
        if has_mmr:
            mmr_words = model.topic_aspects_["MMR"].get(topic_id, [])
            keywords_mmr = [w for w, _ in mmr_words[:10]]

        entry: dict[str, Any] = {
            "topicIndex": topic_id,
            "rawLabel": row.get("Name", f"Topic_{topic_id}"),
            "keywords": keywords_main,
            "docCount": int(row["Count"]),
        }
        if has_mmr:
            entry["keywordsMmr"] = keywords_mmr
        if aspect_mapping is not None and topic_id in aspect_mapping:
            asp = aspect_mapping[topic_id]
            entry["aspectLabel"] = asp["aspect"]
            entry["aspectSimilarity"] = round(float(asp["similarity"]), 4)
            entry["isEmergent"] = asp["aspect"] == "emergent"
        results.append(entry)
    return results


def build_assignments_from_multi(
    multi: list[dict[str, Any]],
    submission_ids: list[str],
) -> list[dict[str, Any]]:
    """
    Convert assign_multi_topic output into worker AssignmentItem dicts.

    Mirrors get_assignments behavior:
      - Skips documents whose primary_topic == -1 (outliers).
      - Preserves order from the input texts/submission_ids alignment.
      - Rounds probabilities to 4 decimal places.

    Returns: list of dicts with keys
        submissionId, topicIndex, probability, secondaryTopicIndex,
        secondaryProbability, isMultiTopic
    """
    assert len(multi) == len(submission_ids), (
        f"multi length {len(multi)} != submission_ids length {len(submission_ids)}"
    )

    assignments: list[dict[str, Any]] = []
    for entry, sub_id in zip(multi, submission_ids):
        primary = entry["primary_topic"]
        if primary == -1:
            continue

        item: dict[str, Any] = {
            "submissionId": sub_id,
            "topicIndex": int(primary),
            "probability": round(float(entry["primary_confidence"]), 4),
            "isMultiTopic": bool(entry["is_multi_topic"]),
        }
        if entry["secondary_topic"] is not None:
            item["secondaryTopicIndex"] = int(entry["secondary_topic"])
        if entry["secondary_confidence"] is not None:
            item["secondaryProbability"] = round(float(entry["secondary_confidence"]), 4)
        assignments.append(item)

    return assignments
