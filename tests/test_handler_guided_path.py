"""Unit tests for the request/response schema surface of the Tier 1 guided path."""

from src.models import (
    AspectMappingEntry,
    AssignmentItem,
    RequestParams,
    TopicItem,
    TopicModelResponse,
)


class TestRequestParamsExtension:
    def test_accepts_tier1_params(self):
        p = RequestParams.model_validate(
            {
                "min_topic_size": 15,
                "tier1_guided": True,
                "match_threshold": 0.65,
                "primary_threshold": 0.20,
                "secondary_threshold": 0.15,
                "secondary_gap_max": 0.20,
                "mmr_diversity": 0.4,
                "hdbscan_min_samples": 5,
            }
        )
        assert p.tier1_guided is True
        assert p.match_threshold == 0.65
        assert p.mmr_diversity == 0.4

    def test_legacy_request_still_works(self):
        # 1.0.0-shaped request must still validate cleanly.
        p = RequestParams.model_validate(
            {
                "min_topic_size": 15,
                "nr_topics": 20,
                "umap_n_neighbors": 20,
                "umap_n_components": 10,
            }
        )
        assert p.tier1_guided is None
        assert p.match_threshold is None


class TestTopicItemExtension:
    def test_legacy_topic_validates(self):
        t = TopicItem.model_validate(
            {
                "topicIndex": 0,
                "rawLabel": "0_foo_bar",
                "keywords": ["foo", "bar"],
                "docCount": 12,
            }
        )
        assert t.aspectLabel is None
        assert t.keywordsMmr is None

    def test_guided_topic_with_all_new_fields(self):
        t = TopicItem.model_validate(
            {
                "topicIndex": 0,
                "rawLabel": "0_foo_bar",
                "keywords": ["foo", "bar"],
                "docCount": 12,
                "aspectLabel": "teaching_methodology",
                "aspectSimilarity": 0.83,
                "isEmergent": False,
                "keywordsMmr": ["alpha", "beta"],
            }
        )
        assert t.aspectLabel == "teaching_methodology"
        assert t.aspectSimilarity == 0.83
        assert t.isEmergent is False
        assert t.keywordsMmr == ["alpha", "beta"]


class TestAssignmentItemExtension:
    def test_single_topic_assignment(self):
        a = AssignmentItem.model_validate(
            {"submissionId": "s1", "topicIndex": 0, "probability": 0.42},
        )
        assert a.secondaryTopicIndex is None
        assert a.isMultiTopic is None

    def test_multi_topic_assignment(self):
        a = AssignmentItem.model_validate(
            {
                "submissionId": "s1",
                "topicIndex": 0,
                "probability": 0.30,
                "secondaryTopicIndex": 4,
                "secondaryProbability": 0.20,
                "isMultiTopic": True,
            }
        )
        assert a.secondaryTopicIndex == 4
        assert a.secondaryProbability == 0.20
        assert a.isMultiTopic is True


class TestAspectMapping:
    def test_aspect_mapping_entry(self):
        e = AspectMappingEntry.model_validate({"aspect": "emergent", "similarity": 0.55})
        assert e.aspect == "emergent"
        assert e.similarity == 0.55

    def test_response_with_aspect_mapping(self):
        r = TopicModelResponse.model_validate(
            {
                "version": "1.1.0",
                "status": "completed",
                "completedAt": "2026-05-04T00:00:00Z",
                "aspectMapping": {
                    "-1": {"aspect": "outlier", "similarity": 0.0},
                    "0": {"aspect": "teaching_methodology", "similarity": 0.83},
                },
            }
        )
        assert r.aspectMapping["0"].aspect == "teaching_methodology"
        assert r.aspectMapping["-1"].aspect == "outlier"


class TestExcludeNoneSemantics:
    def test_legacy_response_dump_omits_new_fields(self):
        r = TopicModelResponse(
            version="1.1.0",
            status="completed",
            topics=[TopicItem(topicIndex=0, rawLabel="t", keywords=["x"], docCount=5)],
            assignments=[AssignmentItem(submissionId="s1", topicIndex=0, probability=0.5)],
            outlierCount=0,
            completedAt="2026-05-04T00:00:00Z",
        )
        dumped = r.model_dump(exclude_none=True)
        # New top-level field absent
        assert "aspectMapping" not in dumped
        # New per-topic fields absent
        assert "aspectLabel" not in dumped["topics"][0]
        assert "keywordsMmr" not in dumped["topics"][0]
        # New per-assignment fields absent
        assert "secondaryTopicIndex" not in dumped["assignments"][0]
        assert "isMultiTopic" not in dumped["assignments"][0]
