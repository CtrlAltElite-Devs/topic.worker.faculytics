"""Tests for topic_model module (unit tests that don't require GPU)."""

from src.topic_model import MULTILINGUAL_STOP_WORDS


class TestStopWords:
    """Verify multilingual stop words list."""

    def test_contains_english_stop_words(self):
        for word in ["the", "a", "and", "is", "are", "was"]:
            assert word in MULTILINGUAL_STOP_WORDS

    def test_contains_cebuano_stop_words(self):
        for word in ["ang", "nga", "sa", "ni", "ug", "og", "kay"]:
            assert word in MULTILINGUAL_STOP_WORDS

    def test_contains_tagalog_stop_words(self):
        for word in ["ng", "mga", "ay", "nang", "po", "hindi"]:
            assert word in MULTILINGUAL_STOP_WORDS

    def test_contains_role_words(self):
        for word in ["propesor", "estudyante", "teacher", "professor", "student"]:
            assert word in MULTILINGUAL_STOP_WORDS

    def test_no_duplicates_in_semantics(self):
        # The list is built from a set literal, so no duplicates
        assert len(MULTILINGUAL_STOP_WORDS) == len(set(MULTILINGUAL_STOP_WORDS))
