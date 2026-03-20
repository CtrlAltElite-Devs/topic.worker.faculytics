"""Tests for topic_model module (unit tests that don't require GPU)."""

from sklearn.feature_extraction.text import CountVectorizer

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


class TestCountVectorizerConfig:
    """Verify CountVectorizer config doesn't crash on small topic clusters."""

    def _make_vectorizer(self):
        return CountVectorizer(
            ngram_range=(1, 2),
            stop_words=MULTILINGUAL_STOP_WORDS,
            min_df=1,
        )

    def test_single_class_document(self):
        """With only 1 topic, BERTopic produces 1 class document. min_df=1 must not crash."""
        vectorizer = self._make_vectorizer()
        docs = ["teacher very helpful clear explanation"]
        matrix = vectorizer.fit_transform(docs)
        assert matrix.shape[0] == 1
        assert len(vectorizer.get_feature_names_out()) > 0

    def test_two_class_documents(self):
        """With 2 topics, BERTopic produces 2 class documents."""
        vectorizer = self._make_vectorizer()
        docs = [
            "teacher very helpful clear explanation",
            "strict grading unfair exam difficult",
        ]
        matrix = vectorizer.fit_transform(docs)
        assert matrix.shape[0] == 2
        assert len(vectorizer.get_feature_names_out()) > 0
