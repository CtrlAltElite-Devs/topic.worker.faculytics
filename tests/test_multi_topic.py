"""Unit tests for assign_multi_topic threshold logic and build_assignments_from_multi."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.topic_model import assign_multi_topic, build_assignments_from_multi


def _make_fake_model(topic_ids: list[int], distributions: np.ndarray):
    info = pd.DataFrame({"Topic": topic_ids, "Count": [10] * len(topic_ids)})
    return SimpleNamespace(
        get_topic_info=lambda: info,
        approximate_distribution=lambda texts, window=4, stride=1: (distributions, None),
        calculate_probabilities=True,
    )


class TestAssignMultiTopic:
    def test_asserts_calculate_probabilities(self):
        model = SimpleNamespace(calculate_probabilities=False)
        with pytest.raises(AssertionError, match="calculate_probabilities=True"):
            assign_multi_topic(model, ["doc"])

    def test_below_primary_threshold_is_outlier(self):
        # top1 prob = 0.10 < primary_threshold 0.20 → primary becomes -1
        model = _make_fake_model([0, 1, 2], np.array([[0.10, 0.05, 0.03]]))
        out = assign_multi_topic(model, ["doc"], primary_threshold=0.20)
        assert out[0]["primary_topic"] == -1
        assert out[0]["is_multi_topic"] is False

    def test_clear_primary_no_secondary(self):
        # top1=0.50, top2=0.05 → gap 0.45 >= 0.20, no secondary
        model = _make_fake_model([0, 1, 2], np.array([[0.50, 0.05, 0.02]]))
        out = assign_multi_topic(
            model,
            ["doc"],
            primary_threshold=0.20,
            secondary_threshold=0.15,
            secondary_gap_max=0.20,
        )
        assert out[0]["primary_topic"] == 0
        assert out[0]["secondary_topic"] is None
        assert out[0]["is_multi_topic"] is False

    def test_secondary_gap_within_window(self):
        # top1=0.30, top2=0.20 → gap 0.10 < 0.20, top2>=0.15 → secondary set
        model = _make_fake_model([0, 1, 2], np.array([[0.30, 0.20, 0.05]]))
        out = assign_multi_topic(
            model,
            ["doc"],
            primary_threshold=0.20,
            secondary_threshold=0.15,
            secondary_gap_max=0.20,
        )
        assert out[0]["primary_topic"] == 0
        assert out[0]["secondary_topic"] == 1
        assert out[0]["is_multi_topic"] is True
        assert out[0]["secondary_confidence"] == 0.20

    def test_secondary_below_secondary_threshold(self):
        # top1=0.30, top2=0.10 → top2 < 0.15, no secondary
        model = _make_fake_model([0, 1, 2], np.array([[0.30, 0.10, 0.05]]))
        out = assign_multi_topic(
            model,
            ["doc"],
            primary_threshold=0.20,
            secondary_threshold=0.15,
            secondary_gap_max=0.20,
        )
        assert out[0]["secondary_topic"] is None
        assert out[0]["is_multi_topic"] is False

    def test_zero_distribution_returns_outlier(self):
        model = _make_fake_model([0, 1], np.array([[0.0, 0.0]]))
        out = assign_multi_topic(model, ["doc"])
        assert out[0]["primary_topic"] == -1
        assert out[0]["primary_confidence"] == 0.0
        assert out[0]["is_multi_topic"] is False

    def test_distribution_columns_exceed_topics_clip_safely(self):
        # Regression for adversarial F1: when approximate_distribution returns
        # more columns than there are non-outlier topics, the warning log
        # used to claim "clipping" without actually clipping — argsort across
        # the wider matrix would pick an out-of-bounds index, raising
        # IndexError. With the clip in place, the function must select only
        # among real topics and not crash.
        model = _make_fake_model([0, 1], np.array([[0.05, 0.40, 0.50]]))
        out = assign_multi_topic(
            model,
            ["doc"],
            primary_threshold=0.10,
            secondary_threshold=0.10,
            secondary_gap_max=0.20,
        )
        # Only topic_ids [0, 1] exist — the 0.50 column is dropped.
        assert out[0]["primary_topic"] in {0, 1}
        # Top1 is now column index 1 (value 0.40), top2 is column index 0 (value 0.05).
        assert out[0]["primary_confidence"] == 0.40


class TestBuildAssignmentsFromMulti:
    def test_filters_outliers(self):
        multi = [
            {
                "primary_topic": -1,
                "secondary_topic": None,
                "primary_confidence": 0.0,
                "secondary_confidence": None,
                "is_multi_topic": False,
            },
            {
                "primary_topic": 3,
                "secondary_topic": None,
                "primary_confidence": 0.42,
                "secondary_confidence": None,
                "is_multi_topic": False,
            },
        ]
        out = build_assignments_from_multi(multi, ["s1", "s2"])
        assert len(out) == 1
        assert out[0]["submissionId"] == "s2"
        assert out[0]["topicIndex"] == 3

    def test_includes_secondary_when_present(self):
        multi = [
            {
                "primary_topic": 1,
                "secondary_topic": 4,
                "primary_confidence": 0.30,
                "secondary_confidence": 0.20,
                "is_multi_topic": True,
            },
        ]
        out = build_assignments_from_multi(multi, ["s1"])
        assert out[0]["secondaryTopicIndex"] == 4
        assert out[0]["secondaryProbability"] == 0.20
        assert out[0]["isMultiTopic"] is True

    def test_omits_secondary_when_absent(self):
        multi = [
            {
                "primary_topic": 1,
                "secondary_topic": None,
                "primary_confidence": 0.55,
                "secondary_confidence": None,
                "is_multi_topic": False,
            },
        ]
        out = build_assignments_from_multi(multi, ["s1"])
        assert "secondaryTopicIndex" not in out[0]
        assert "secondaryProbability" not in out[0]
        assert out[0]["isMultiTopic"] is False

    def test_length_mismatch_raises(self):
        with pytest.raises(AssertionError):
            build_assignments_from_multi(
                [
                    {
                        "primary_topic": 0,
                        "secondary_topic": None,
                        "primary_confidence": 0.5,
                        "secondary_confidence": None,
                        "is_multi_topic": False,
                    }
                ],
                ["s1", "s2"],
            )

    def test_probability_rounded_to_4dp(self):
        multi = [
            {
                "primary_topic": 0,
                "secondary_topic": None,
                "primary_confidence": 0.123456789,
                "secondary_confidence": None,
                "is_multi_topic": False,
            },
        ]
        out = build_assignments_from_multi(multi, ["s1"])
        assert out[0]["probability"] == 0.1235
