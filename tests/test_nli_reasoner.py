def test_normalize_claim_for_nli_uses_evidence_vocabulary():
    from src.nli_reasoner import normalize_claim_for_nli

    premise = "Francesco Totti is an Italian former professional footballer."

    assert normalize_claim_for_nli("Francesco Totti is a footbal player", premise) == "Francesco Totti has been a football player"
    assert normalize_claim_for_nli("Francesco Totti is a footbal player") == "Francesco Totti is a footbal player"


def test_normalize_claim_for_nli_preserves_definite_current_claims():
    from src.nli_reasoner import normalize_claim_for_nli

    premise = "Jane Doe is a former president of Exampleland."

    assert normalize_claim_for_nli("Jane Doe is the president of Exampleland", premise) == "Jane Doe is the president of Exampleland"


def test_structured_nli_returns_decision_evidence(monkeypatch):
    from src import nli_reasoner

    monkeypatch.setattr(nli_reasoner, "predict_nli_batch", lambda claim, premises: [
        {
            "verdict": "SUPPORTED",
            "nli_label": "entailment",
            "probs": {"SUPPORTED": 0.9, "REFUTED": 0.05, "NOT ENOUGH INFO": 0.05},
            "confidence": 0.9,
        },
        {
            "verdict": "NOT ENOUGH INFO",
            "nli_label": "neutral",
            "probs": {"SUPPORTED": 0.1, "REFUTED": 0.1, "NOT ENOUGH INFO": 0.8},
            "confidence": 0.8,
        },
    ])

    result = nli_reasoner.structured_nli_pipeline("Ada wrote notes", [
        {
            "linearized": "Ada Lovelace wrote notes.",
            "source_sentence": "Ada Lovelace wrote notes on the Analytical Engine.",
            "source_type": "kg",
            "source": "Ada Lovelace",
            "cross_encoder_score": 0.8,
        },
        {
            "linearized": "Charles Babbage designed the Analytical Engine.",
            "source_type": "kg",
            "source": "Analytical Engine",
            "cross_encoder_score": 0.7,
        },
    ])

    assert result["verdict"] == "SUPPORTED"
    assert result["decision_evidence"][0]["text"] == "Ada Lovelace wrote notes on the Analytical Engine."
    assert result["decision_evidence"][0]["source"] == "Ada Lovelace"


def test_predict_nli_batch_uses_one_model_call(monkeypatch):
    import torch
    from src import nli_reasoner

    calls = {"tokenizer": 0, "model": 0}

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        def __call__(self, premises, claims, **kwargs):
            calls["tokenizer"] += 1
            assert premises == ["Ada wrote notes.", "Babbage designed engines."]
            assert len(claims) == 2
            return Batch()

    class Config:
        id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}

    class Model:
        config = Config()

        def __call__(self, **kwargs):
            calls["model"] += 1
            return type("Output", (), {"logits": torch.tensor([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]])})()

    monkeypatch.setattr(nli_reasoner, "_load_nli", lambda: (Tokenizer(), Model()))

    results = nli_reasoner.predict_nli_batch("Ada wrote notes", [
        "Ada wrote notes.",
        "",
        "Babbage designed engines.",
    ])

    assert calls == {"tokenizer": 1, "model": 1}
    assert [r["verdict"] for r in results] == ["SUPPORTED", "NOT ENOUGH INFO", "REFUTED"]
