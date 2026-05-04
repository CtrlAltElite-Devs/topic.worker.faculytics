"""Unit tests for map_topics_to_aspects (no real LaBSE / no real BERTopic)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.topic_model import map_topics_to_aspects


class _FakeEmbedModel:
    """Returns one-hot-ish vectors keyed by the first character of each input."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def encode(self, items, show_progress_bar=False, normalize_embeddings=True):
        out = np.zeros((len(items), self.dim), dtype=np.float32)
        for i, s in enumerate(items):
            idx = (ord(s[0].lower()) - ord("a")) % self.dim if s else 0
            out[i, idx] = 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
            out = out / norms
        return out


def _make_fake_model(
    topic_ids: list[int],
    topic_keywords: dict[int, list[str]],
    use_topic_aspects: bool = True,
):
    """
    Real BERTopic instances initialize self.topic_aspects_ as {} (empty dict),
    not None. Guided models populate it with {"Main": {...}, "MMR": {...}}.

    use_topic_aspects=True  → simulates a guided model (production hot path).
                              map_topics_to_aspects pulls keywords from
                              topic_aspects_["Main"][tid].
    use_topic_aspects=False → simulates a legacy model (fallback path).
                              map_topics_to_aspects falls back to model.get_topic(tid).
    """
    info = pd.DataFrame({"Topic": topic_ids, "Count": [10] * len(topic_ids)})
    if use_topic_aspects:
        topic_aspects = {
            "Main": {tid: [(w, 1.0) for w in kws] for tid, kws in topic_keywords.items()},
            "MMR": {tid: [(w, 1.0) for w in kws[:5]] for tid, kws in topic_keywords.items()},
        }
    else:
        topic_aspects = {}  # mirrors real BERTopic init (empty dict, not None)
    return SimpleNamespace(
        get_topic_info=lambda: info,
        get_topic=lambda tid: [(w, 1.0) for w in topic_keywords.get(tid, [])],
        topic_aspects_=topic_aspects,
    )


class TestMapTopicsToAspects:
    def test_outlier_maps_to_outlier(self):
        model = _make_fake_model([-1], {-1: []})
        out = map_topics_to_aspects(model, _FakeEmbedModel(), match_threshold=0.5)
        assert out[-1] == {"aspect": "outlier", "similarity": 0.0}

    def test_below_threshold_is_emergent_via_main_representation(self):
        # Production hot path: keywords come from topic_aspects_["Main"].
        # All keywords start with 'z' — never overlaps with seed letters.
        model = _make_fake_model([0], {0: ["zzz", "zzy", "zzx"]}, use_topic_aspects=True)
        out = map_topics_to_aspects(
            model, _FakeEmbedModel(), match_threshold=0.99, representation="Main"
        )
        assert out[0]["aspect"] == "emergent"
        assert 0.0 <= out[0]["similarity"] <= 1.0

    def test_keyword_alignment_via_main_representation(self):
        # Production hot path with match_threshold=0.0 → always pick the best aspect.
        model = _make_fake_model([0], {0: ["malinaw", "explain", "klaro"]}, use_topic_aspects=True)
        out = map_topics_to_aspects(
            model, _FakeEmbedModel(), match_threshold=0.0, representation="Main"
        )
        assert out[0]["aspect"] != "emergent"
        assert out[0]["aspect"] != "outlier"

    def test_falls_back_to_get_topic_when_no_topic_aspects(self):
        # Legacy/fallback path: topic_aspects_ is empty dict, function should
        # fall through to model.get_topic(topic_id).
        model = _make_fake_model([0], {0: ["malinaw", "explain", "klaro"]}, use_topic_aspects=False)
        out = map_topics_to_aspects(
            model, _FakeEmbedModel(), match_threshold=0.0, representation="Main"
        )
        # Same outcome as the hot path because get_topic returns the same data —
        # but execution went through the fallback branch (verified by topic_aspects_={}).
        assert out[0]["aspect"] != "emergent"

    def test_similarity_rounded_to_4dp(self):
        model = _make_fake_model([0], {0: ["malinaw"]}, use_topic_aspects=True)
        out = map_topics_to_aspects(model, _FakeEmbedModel(), match_threshold=0.0)
        sim = out[0]["similarity"]
        # Round-trip check — value should be representable in 4dp
        assert round(sim, 4) == sim
