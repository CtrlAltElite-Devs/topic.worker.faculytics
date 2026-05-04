"""Unit tests for Tier 1 guided BERTopic constants and builder.

The builder tests use unittest.mock to patch BERTopic.fit_transform so the
test exercises the real BERTopic constructor (seed_topic_list, calculate_probabilities,
representation_model wiring) without needing LaBSE to be loaded or GPU available.
"""

from unittest.mock import patch

import numpy as np
import pytest

from src.topic_model import (
    ACADEMIC_ASPECTS,
    ASPECT_NAMES,
    SEED_TOPIC_LIST,
    TIER1_DEFAULTS,
    audit_aspect_coverage,
    build_bertopic_guided,
)


def _fake_fit_transform(self, documents, embeddings=None, **kwargs):
    """Stand-in for BERTopic.fit_transform.

    The real method has the *side effect* of setting ``self.topics_`` on the
    instance. Our builder logs ``topic_model.topics_`` after fit, so the patch
    must reproduce that side effect (otherwise it stays None and AttributeError
    fires inside the post-fit logging block). autospec=True ensures the bound
    instance is passed as the first arg.
    """
    n = len(documents)
    self.topics_ = [0] * n
    return (np.zeros(n, dtype=int), None)


class TestTier1Constants:
    def test_aspect_names_count(self):
        assert len(ASPECT_NAMES) == 10

    def test_seed_list_position_aligned(self):
        # Positional alignment is required by BERTopic seed_topic_list semantics.
        assert len(SEED_TOPIC_LIST) == len(ASPECT_NAMES)
        for seeds in SEED_TOPIC_LIST:
            assert isinstance(seeds, list)
            assert len(seeds) >= 10

    def test_aspect_keys_match_lib(self):
        # Mirrors lib.py:238 — explicit set keeps the seed order honest.
        expected = {
            "teaching_clarity",
            "assessment_grading",
            "workload_pacing",
            "teaching_methodology",
            "instructor_attitude",
            "responsiveness_support",
            "punctuality_attendance",
            "lms_digital_tools",
            "real_world_relevance",
            "language_communication",
        }
        assert set(ACADEMIC_ASPECTS.keys()) == expected

    def test_tier1_defaults(self):
        # tier1_guided defaults to False in 1.1.0 — API caller opts in.
        assert TIER1_DEFAULTS["tier1_guided"] is False
        assert TIER1_DEFAULTS["match_threshold"] == 0.65
        assert TIER1_DEFAULTS["primary_threshold"] == 0.20
        assert TIER1_DEFAULTS["secondary_threshold"] == 0.15
        assert TIER1_DEFAULTS["secondary_gap_max"] == 0.20
        assert TIER1_DEFAULTS["mmr_diversity"] == 0.4
        assert TIER1_DEFAULTS["hdbscan_min_samples"] == 5


class _FakeEmbedModel:
    """Minimal stand-in for SentenceTransformer — never called during the
    patched fit, but the type signature must accept it."""

    def encode(self, *args, **kwargs):
        raise AssertionError("encode should not be called when fit is patched")


class TestBuildBertopicGuidedShape:
    """Verify build_bertopic_guided wires the BERTopic instance correctly.

    Patches BERTopic.fit_transform so we never actually fit (no LaBSE, no UMAP
    fit, no HDBSCAN fit) — we only assert on the constructor-time configuration
    that distinguishes guided builds from the legacy run_bertopic.
    """

    def test_guided_model_sets_seed_list_and_probabilities(self):
        pytest.importorskip("bertopic")
        # 30 random embeddings — never actually consumed because fit is patched.
        embeddings = np.random.RandomState(42).randn(30, 768).astype(np.float32)
        texts = [f"doc {i}" for i in range(30)]
        params = {
            "min_topic_size": 5,
            "nr_topics": 3,
            "umap_n_neighbors": 5,
            "umap_n_components": 3,
            "hdbscan_min_samples": 2,
            "mmr_diversity": 0.4,
        }

        with patch(
            "bertopic.BERTopic.fit_transform", autospec=True, side_effect=_fake_fit_transform
        ):
            model = build_bertopic_guided(embeddings, texts, params, _FakeEmbedModel())

        # Seeded topics are the production-distinguishing config.
        assert model.seed_topic_list == SEED_TOPIC_LIST
        # Required for approximate_distribution → multi-topic assignment.
        assert model.calculate_probabilities is True
        # Sibling representation: Main (KeyBERTInspired) + MMR (MaximalMarginalRelevance).
        assert isinstance(model.representation_model, dict)
        assert set(model.representation_model.keys()) == {"Main", "MMR"}

    def test_mmr_diversity_threads_through(self):
        pytest.importorskip("bertopic")
        from bertopic.representation import MaximalMarginalRelevance

        embeddings = np.random.RandomState(0).randn(30, 768).astype(np.float32)
        texts = [f"doc {i}" for i in range(30)]
        params = {
            "min_topic_size": 5,
            "nr_topics": 3,
            "umap_n_neighbors": 5,
            "umap_n_components": 3,
            "mmr_diversity": 0.7,  # non-default, verify it lands on the MMR rep
        }

        with patch(
            "bertopic.BERTopic.fit_transform", autospec=True, side_effect=_fake_fit_transform
        ):
            model = build_bertopic_guided(embeddings, texts, params, _FakeEmbedModel())

        mmr = model.representation_model["MMR"]
        assert isinstance(mmr, MaximalMarginalRelevance)
        assert mmr.diversity == 0.7


class TestAuditAspectCoverage:
    """Regression suite for adversarial F5 — substring -> word-boundary fix."""

    def test_word_boundary_avoids_late_in_related(self):
        # Pre-fix behavior: "late" was matched inside "related" → false positive
        # for punctuality_attendance. Post-fix: word-boundary anchors.
        out = audit_aspect_coverage(
            ["this is a related concept"],
            aspects={"punctuality_attendance": ["late"]},
        )
        assert out["punctuality_attendance"] == 0

    def test_exact_word_match_counts(self):
        out = audit_aspect_coverage(
            ["the professor was late again"],
            aspects={"punctuality_attendance": ["late"]},
        )
        assert out["punctuality_attendance"] == 1

    def test_multi_word_seed_matches_phrase(self):
        # "real world" is one of the canonical seeds — it must still match.
        out = audit_aspect_coverage(
            ["this applies to the real world setting"],
            aspects={"real_world_relevance": ["real world"]},
        )
        assert out["real_world_relevance"] == 1

    def test_default_aspects_are_used_when_none(self):
        # Regression-protect the `aspects=None` branch — must use ACADEMIC_ASPECTS.
        out = audit_aspect_coverage(["the rubric was fair this term"])
        assert set(out.keys()) == set(ACADEMIC_ASPECTS.keys())
        # "rubric" + "fair" both appear in assessment_grading seeds → 1 hit.
        assert out["assessment_grading"] == 1
