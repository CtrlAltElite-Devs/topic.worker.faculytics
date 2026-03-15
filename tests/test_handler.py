"""Tests for the RunPod handler (input validation and domain error paths)."""

import pytest

# Import only models and config — avoid importing handler directly since
# it loads the model at module level. Test the logic functions instead.
from src.models import TopicModelRequest, RequestItem, RequestParams
from src.config import DEFAULT_PARAMS


class TestTopicModelRequest:
    """Test Pydantic request schema parsing."""

    def test_valid_request(self):
        data = {
            "items": [
                {"submissionId": "s1", "text": "hello world test", "embedding": [0.1] * 768},
            ],
            "params": {"min_topic_size": 10},
        }
        request = TopicModelRequest.model_validate(data)
        assert len(request.items) == 1
        assert request.params.min_topic_size == 10

    def test_extra_envelope_fields_ignored(self):
        """RunPod handler receives the full API envelope with extra fields."""
        data = {
            "jobId": "some-uuid",
            "version": "1.0",
            "type": "topic-model",
            "items": [
                {"submissionId": "s1", "text": "hello world test", "embedding": [0.1] * 768},
            ],
            "metadata": {"pipelineId": "p1", "runId": "r1"},
            "publishedAt": "2026-03-16T00:00:00.000Z",
        }
        request = TopicModelRequest.model_validate(data)
        assert len(request.items) == 1

    def test_default_params_when_none(self):
        data = {
            "items": [
                {"submissionId": "s1", "text": "hello world test", "embedding": [0.1] * 768},
            ],
        }
        request = TopicModelRequest.model_validate(data)
        assert request.params is None

    def test_request_item_extra_fields_ignored(self):
        item = RequestItem.model_validate({
            "submissionId": "s1",
            "text": "hello",
            "embedding": [0.1] * 768,
            "extraField": "ignored",
        })
        assert item.submissionId == "s1"


class TestDefaultParams:
    """Verify RUN 012 defaults are set correctly."""

    def test_run_012_defaults(self):
        assert DEFAULT_PARAMS["min_topic_size"] == 15
        assert DEFAULT_PARAMS["nr_topics"] == 20
        assert DEFAULT_PARAMS["umap_n_neighbors"] == 20
        assert DEFAULT_PARAMS["umap_n_components"] == 10
