"""
Reusable NLI evaluation for the fact-checking pipeline.

Evaluates how well a BERT-family model classifies HoVer claims given evidence,
computing accuracy, confusion matrix, and per-class metrics.

Usage:
    from src.eval import evaluate_nli, print_metrics
    metrics, results = evaluate_nli(model_name, num_samples=500)
    print_metrics(metrics)
"""
import json
from pathlib import Path
import threading
from functools import lru_cache

# Suppress Hugging Face auto-conversion thread exception
def _ignore_auto_conversion_error(args):
    if args.thread and args.thread.name == "Thread-auto_conversion":
        return
    if threading.__excepthook__:
        threading.__excepthook__(args)

threading.excepthook = _ignore_auto_conversion_error

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HOVER_DATASET = "Dzeniks/hover"
DEFAULT_MODEL = "microsoft/deberta-base-mnli"
EVAL_DIR = Path("data/eval")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_nli2hover(model):
    """Build NLI-class-id → HoVer-label mapping from model's id2label config.
    
    Returns mapping where neutral labels map to None (skipped in evaluation),
    since HoVer is a binary classification task (Supports/Refutes).
    """
    id2label = model.config.id2label
    mapping = {}
    for idx, label in id2label.items():
        label_lower = label.lower()
        if "entailment" in label_lower:
            mapping[idx] = 0  # Supports
        elif "contradiction" in label_lower:
            mapping[idx] = 1  # Refutes
        else:
            mapping[idx] = None  # neutral → skipped (not in HoVer label space)
    return mapping


@lru_cache(maxsize=None)
def load_model(model_name=DEFAULT_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def predict(tokenizer, model, premise, hypothesis, max_length=512):
    inputs = tokenizer(
        premise, hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    ).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits.argmax(dim=-1).item()


def evaluate_nli(model_name=DEFAULT_MODEL, num_samples=200,
                 evidence_key="evidence", split="test"):
    """Evaluate NLI accuracy using dataset evidence.

    Args:
        model_name: HuggingFace model name.
        num_samples: Number of samples to evaluate (None = all).
        evidence_key: 'evidence' for gold evidence, or path to BM25 JSON.
        split: Dataset split ('test' or 'train').

    Returns:
        (metrics_dict, results_list)
    """
    tokenizer, model = load_model(model_name)
    nli2hover = build_nli2hover(model)
    dataset = load_dataset(HOVER_DATASET, split=split)
    dataset = dataset.shuffle(seed=42)
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    retrieved_data = {}
    if evidence_key != "evidence":
        path = Path(evidence_key)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw_list = json.load(f)
                retrieved_data = {str(item["claim_id"]): item for item in raw_list}

    correct = 0
    total = 0
    skipped = 0
    confusion = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    results = []

    for row in dataset:
        claim = row["claim"]
        gold_label = row["label"]
        claim_id = str(row["id"])

        if evidence_key == "evidence":
            premise = row["evidence"]
        else:
            item = retrieved_data.get(claim_id)
            if item:
                from src.semantic_ranking import rank_and_combine_evidence
                _, premise = rank_and_combine_evidence(
                    claim,
                    text_evidence=item.get("text_evidence"),
                    kg_evidence=item.get("kg_evidence"),
                    web_evidence=item.get("web_evidence"),
                    top_k=5
                )
            else:
                premise = ""

        nli_id = predict(tokenizer, model, premise, claim)
        predicted = nli2hover.get(nli_id)

        if predicted is None:
            skipped += 1
            results.append({
                "id": row["id"],
                "claim": claim,
                "nli_label": model.config.id2label[nli_id],
                "predicted": None,
                "gold_label": gold_label,
                "correct": None,
                "skipped": True,
            })
            continue

        is_correct = predicted == gold_label
        if is_correct:
            correct += 1
        total += 1

        if gold_label == 0 and predicted == 0:
            confusion["TP"] += 1
        elif gold_label == 0 and predicted == 1:
            confusion["FN"] += 1
        elif gold_label == 1 and predicted == 1:
            confusion["TN"] += 1
        else:
            confusion["FP"] += 1

        results.append({
            "id": row["id"],
            "claim": claim,
            "nli_label": model.config.id2label[nli_id],
            "predicted": predicted,
            "gold_label": gold_label,
            "correct": is_correct,
            "skipped": False,
        })

    accuracy = correct / total * 100 if total else 0
    precision = confusion["TP"] / (confusion["TP"] + confusion["FP"]) * 100 if (confusion["TP"] + confusion["FP"]) else 0
    recall = confusion["TP"] / (confusion["TP"] + confusion["FN"]) * 100 if (confusion["TP"] + confusion["FN"]) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    metrics = {
        "model": model_name,
        "num_samples": total,
        "skipped_neutral": skipped,
        "accuracy_pct": round(accuracy, 2),
        "precision_pct": round(precision, 2),
        "recall_pct": round(recall, 2),
        "f1_score": round(f1, 2),
        "correct": correct,
        "total": total,
        "confusion": confusion,
    }
    return metrics, results


def save_results(metrics, results, name="nli_eval"):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2, ensure_ascii=False)
    return path


def print_metrics(metrics):
    print(f"Model: {metrics['model']}")
    print(f"Samples: {metrics['num_samples']} (skipped neutral: {metrics.get('skipped_neutral', 0)})")
    print(f"Accuracy: {metrics['accuracy_pct']}% ({metrics['correct']}/{metrics['total']})")
    print(f"Precision: {metrics['precision_pct']}%")
    print(f"Recall: {metrics['recall_pct']}%")
    print(f"F1: {metrics['f1_score']}")
    print(f"Confusion: {metrics['confusion']}")


def predict_base_claims(json_path, output_path=None, model_name=DEFAULT_MODEL):
    """Loads claims from base_claims.json, computes NLI prediction for each claim
    using the ranked evidence, updates the 'nli_prediction' field, and saves
    the updated JSON.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"File {json_path} not found.")
        
    with open(path, "r", encoding="utf-8") as f:
        claims_list = json.load(f)
        
    tokenizer, model = load_model(model_name)
    from src.semantic_ranking import rank_and_combine_evidence
    
    id2label = model.config.id2label
    
    print(f"Running NLI prediction on {len(claims_list)} claims using {model_name}...")
    for item in claims_list:
        claim_text = item["claim_text"]
        
        _, premise = rank_and_combine_evidence(
            claim_text,
            text_evidence=item.get("text_evidence"),
            kg_evidence=item.get("kg_evidence"),
            web_evidence=item.get("web_evidence"),
            top_k=5
        )
        
        if not premise:
            item["nli_prediction"] = "NOT ENOUGH INFO"
            continue
            
        nli_id = predict(tokenizer, model, premise, claim_text)
        nli_label = id2label[nli_id].lower()
        
        if "entail" in nli_label:
            item["nli_prediction"] = "SUPPORTED"
        elif "contradict" in nli_label:
            item["nli_prediction"] = "REFUTED"
        else:
            item["nli_prediction"] = "NOT ENOUGH INFO"
            
    out_path = Path(output_path) if output_path else path
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(claims_list, f, ensure_ascii=False, indent=2)
        
    print(f"NLI predictions successfully saved to {out_path}")
    return claims_list


def evaluate_with_llm(claims_list, llm_judge_fn=None):
    """Evaluate claims using LLM-as-judge instead of NLI.

    Args:
        claims_list: List of dicts with claim_text, text_evidence, kg_evidence, web_evidence
        llm_judge_fn: Function(claim_text, evidence_texts) -> verdict dict

    Returns:
        List with 'llm_prediction' and 'llm_reasoning' added.
    """
    if llm_judge_fn is None:
        from src.llm_judge import evaluate as llm_judge_fn

    from src.semantic_ranking import rank_and_combine_evidence

    print(f"Running LLM evaluation on {len(claims_list)} claims...")
    for item in claims_list:
        claim_text = item["claim_text"]
        evidence_texts = []
        for key in ("text_evidence", "kg_evidence", "web_evidence"):
            for ev in item.get(key, []):
                if isinstance(ev, dict):
                    text = ev.get("text") or ev.get("linearized") or ev.get("content") or ""
                    evidence_texts.append(text)

        item["llm_prediction"] = "NOT ENOUGH INFO"
        item["llm_reasoning"] = ""

        if not evidence_texts:
            continue

        result = llm_judge_fn(claim_text, evidence_texts, item.get("gold_label", -1))
        item["llm_prediction"] = result.get("verdict", "NOT ENOUGH INFO")
        item["llm_reasoning"] = result.get("reasoning", "")

    return claims_list
