import json
from pathlib import Path
from src.eval import evaluate_nli, predict_base_claims


def test_predict_base_claims(tmp_path):
    mock_data = [
        {
            "claim_id": "1",
            "claim_text": "Albert Einstein developed the theory of relativity",
            "text_evidence": [{"text": "Albert Einstein developed relativity."}],
            "kg_evidence": [],
            "web_evidence": [],
            "nli_prediction": None
        }
    ]
    json_file = tmp_path / "base_claims.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(mock_data, f, ensure_ascii=False, indent=2)
        
    # Esegue le predizioni
    predict_base_claims(json_file)
    
    # Legge il risultato
    with open(json_file, "r", encoding="utf-8") as f:
        updated = json.load(f)
        
    assert updated[0]["nli_prediction"] is not None
    assert updated[0]["nli_prediction"] in ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO"]


def test_evaluate_nli_with_file(tmp_path):
    # Crea un mock di dati recuperati
    mock_data = [
        {
            "claim_id": "18079",  # ID plausibile
            "claim_text": "Mock claim",
            "text_evidence": [{"text": "Mock evidence text"}],
            "kg_evidence": [],
            "web_evidence": []
        }
    ]
    json_file = tmp_path / "retrieved_evidence.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(mock_data, f, ensure_ascii=False, indent=2)
        
    # Esegue evaluate_nli con il percorso del file
    # Impostiamo num_samples basso per non scaricare troppi dati o rallentare
    metrics, results = evaluate_nli(num_samples=2, evidence_key=str(json_file), split="train")
    
    assert "accuracy_pct" in metrics
    assert len(results) == 2
