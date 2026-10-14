import pytest
from src.semantic_ranking import rank_evidence, rank_and_combine_evidence, rerank_with_cross_encoder


class DummyCrossEncoder:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs, show_progress_bar=False):
        return self.scores[:len(pairs)]


def test_rank_evidence_strings():
    claim = "Albert Einstein developed the theory of relativity"
    evidence = [
        "Relativity is a physical theory of gravitation developed by Albert Einstein.",
        "Mona Lisa is a painting by Leonardo da Vinci.",
        "The moon is the earth's natural satellite."
    ]
    ranked = rank_evidence(claim, evidence, top_k=2)
    assert len(ranked) == 2
    # L'evidenza su Einstein deve essere al primo posto
    assert "Einstein" in ranked[0]["text"]
    assert ranked[0]["semantic_score"] > ranked[1]["semantic_score"]


def test_rank_evidence_dicts():
    claim = "Albert Einstein developed the theory of relativity"
    evidence = [
        {"linearized": "Albert Einstein developed theory of relativity"},
        {"text": "Leonardo da Vinci painted the Mona Lisa"},
        {"snippet": "The moon is a satellite"},
    ]
    ranked = rank_evidence(claim, evidence, top_k=3)
    assert len(ranked) == 3
    # Primo elemento: Einstein
    assert "linearized" in ranked[0]
    assert "theory of relativity" in ranked[0]["linearized"]
    assert "semantic_score" in ranked[0]
    assert "semantic_score" in ranked[1]
    assert "semantic_score" in ranked[2]
    assert ranked[0]["semantic_score"] > ranked[1]["semantic_score"]


def test_rank_evidence_top_k():
    claim = "Test claim"
    evidence = ["A", "B", "C", "D", "E"]
    ranked = rank_evidence(claim, evidence, top_k=2)
    assert len(ranked) == 2
    
    ranked_all = rank_evidence(claim, evidence, top_k=10)
    assert len(ranked_all) == 5


def test_rank_and_combine_evidence():
    claim = "Albert Einstein developed the theory of relativity"
    text_evidence = [{"text": "Einstein was born in Germany."}]
    kg_evidence = [{"triple": ["Albert Einstein", "developed", "relativity"], "linearized": "Albert Einstein developed relativity"}]
    web_evidence = [{"title": "Theory of Relativity", "snippet": "General relativity is a theory of gravitation."}]
    
    ranked, premise = rank_and_combine_evidence(
        claim,
        text_evidence=text_evidence,
        kg_evidence=kg_evidence,
        web_evidence=web_evidence,
        top_k=2
    )
    
    assert len(ranked) == 2
    assert len(premise) > 0
    # La premessa deve contenere i tag opportuni
    assert "Knowledge Graph:" in premise or "Wikipedia (Theory of Relativity):" in premise or "Textual Source:" in premise


def test_rank_and_combine_evidence_deduplicates_before_rerank(monkeypatch):
    seen = {}

    def fake_rank(claim_text, evidence_items, top_k=5):
        return evidence_items

    def fake_rerank(claim_text, evidence_items, top_k=None):
        seen["count"] = len(evidence_items)
        return evidence_items

    monkeypatch.setattr("src.semantic_ranking.rank_evidence", fake_rank)
    monkeypatch.setattr("src.semantic_ranking.rerank_with_cross_encoder", fake_rerank)
    monkeypatch.setattr("src.semantic_ranking.get_pipeline", lambda name, default=None: True)

    ranked, premise = rank_and_combine_evidence(
        "Ada wrote notes",
        text_evidence=[{"text": "Ada wrote notes."}],
        kg_evidence=[{"linearized": "Ada wrote notes."}],
        top_k=5,
    )

    assert seen["count"] == 1
    assert len(ranked) == 1
    assert premise


def test_rerank_cross_encoder_single_weak_candidate_is_not_forced_to_one(monkeypatch):
    monkeypatch.setattr("src.semantic_ranking.get_cross_encoder", lambda: DummyCrossEncoder([-10.0]))
    monkeypatch.setattr("src.semantic_ranking.get_threshold", lambda name, default=0.0: 0.5)

    ranked = rerank_with_cross_encoder("claim", [{"text": "unrelated"}], top_k=1)

    assert ranked == []


def test_rerank_cross_encoder_uses_absolute_scores(monkeypatch):
    monkeypatch.setattr("src.semantic_ranking.get_cross_encoder", lambda: DummyCrossEncoder([-10.0, 10.0]))
    monkeypatch.setattr("src.semantic_ranking.get_threshold", lambda name, default=0.0: 0.5)

    ranked = rerank_with_cross_encoder(
        "claim",
        [{"text": "weak"}, {"text": "strong"}],
        top_k=2,
    )

    assert [item["text"] for item in ranked] == ["strong"]
    assert ranked[0]["cross_encoder_score"] > 0.99
