import torch
import torch.nn.functional as F
import difflib
import re
from typing import List, Dict, Any, Optional

from src.config import get_model, get_threshold, get_pipeline
from src.eval import DEVICE, load_model


_nli_cache = [None, None]


def normalize_claim_for_nli(claim_text: str, premise: str = "") -> str:
    if not premise:
        return claim_text

    vocab = set()
    for word in re.findall(r"[A-Za-z]{5,}", premise):
        lower = word.lower()
        vocab.add(lower)
        if len(lower) > 6 and lower.endswith("er"):
            vocab.add(lower[:-2])
    if not vocab:
        return claim_text

    def replace(match):
        token = match.group(0)
        lower = token.lower()
        if lower in vocab or len(lower) < 5:
            return token
        matches = difflib.get_close_matches(lower, vocab, n=1, cutoff=0.78)
        if not matches:
            return token
        candidate = matches[0]
        if token[0].isupper():
            return candidate.capitalize()
        return candidate

    normalized = re.sub(r"[A-Za-z]+", replace, claim_text)
    if re.search(r"\b(former|retired|previously)\b", premise, flags=re.IGNORECASE):
        normalized = re.sub(r"\bis\s+(a|an)\s+", r"has been \1 ", normalized, flags=re.IGNORECASE)
    return normalized


def _load_nli():
    """Load NLI model and tokenizer."""
    if _nli_cache[0] is None:
        model_name = get_model("nli", "tasksource/deberta-base-long-nli")
        tokenizer, model = load_model(model_name)
        _nli_cache[:] = [tokenizer, model]
    return _nli_cache[0], _nli_cache[1]


def predict_nli(claim_text: str, premise: str, tokenizer=None, model=None) -> Dict[str, Any]:
    """Classify a claim against one premise and return verdict probabilities."""
    if tokenizer is None or model is None:
        tokenizer, model = _load_nli()
    claim_text = normalize_claim_for_nli(claim_text, premise)
    
    inputs = tokenizer(premise, claim_text, return_tensors="pt",
                       truncation=True, max_length=1024, padding=True).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze().tolist()
    nli_id = logits.argmax(dim=-1).item()
    nli_label = model.config.id2label[nli_id]
    id2label = model.config.id2label
    
    verdict_probs = {"SUPPORTED": 0.0, "REFUTED": 0.0, "NOT ENOUGH INFO": 0.0}
    for idx, prob in enumerate(probs):
        label = id2label[idx].lower()
        if "entail" in label:
            verdict_probs["SUPPORTED"] += prob
        elif "contradict" in label:
            verdict_probs["REFUTED"] += prob
        else:
            verdict_probs["NOT ENOUGH INFO"] += prob
    
    label_lower = nli_label.lower()
    if "entail" in label_lower:
        verdict = "SUPPORTED"
    elif "contradict" in label_lower:
        verdict = "REFUTED"
    else:
        verdict = "NOT ENOUGH INFO"
    
    confidence = round(verdict_probs.get(verdict, 0.0), 4)
    
    return {
        "verdict": verdict,
        "nli_label": nli_label,
        "probs": verdict_probs,
        "confidence": confidence,
    }


def _format_nli_result(model, nli_id, probs) -> Dict[str, Any]:
    nli_label = model.config.id2label[nli_id]
    verdict_probs = {"SUPPORTED": 0.0, "REFUTED": 0.0, "NOT ENOUGH INFO": 0.0}
    for idx, prob in enumerate(probs):
        label = model.config.id2label[idx].lower()
        if "entail" in label:
            verdict_probs["SUPPORTED"] += prob
        elif "contradict" in label:
            verdict_probs["REFUTED"] += prob
        else:
            verdict_probs["NOT ENOUGH INFO"] += prob

    label_lower = nli_label.lower()
    if "entail" in label_lower:
        verdict = "SUPPORTED"
    elif "contradict" in label_lower:
        verdict = "REFUTED"
    else:
        verdict = "NOT ENOUGH INFO"

    return {
        "verdict": verdict,
        "nli_label": nli_label,
        "probs": verdict_probs,
        "confidence": round(verdict_probs.get(verdict, 0.0), 4),
    }


def predict_nli_batch(claim_text: str, premises: List[str]) -> List[Dict[str, Any]]:
    """Run NLI for multiple premises against the same claim."""
    tokenizer, model = _load_nli()
    results = [None] * len(premises)
    non_empty = []
    for i, premise in enumerate(premises):
        if not premise.strip():
            results[i] = {
                "verdict": "NOT ENOUGH INFO",
                "nli_label": "neutral",
                "probs": {"SUPPORTED": 0.0, "REFUTED": 0.0, "NOT ENOUGH INFO": 1.0},
                "confidence": 1.0,
            }
            continue
        non_empty.append((i, premise, normalize_claim_for_nli(claim_text, premise)))

    if non_empty:
        inputs = tokenizer(
            [premise for _, premise, _ in non_empty],
            [claim for _, _, claim in non_empty],
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True,
        ).to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs_batch = F.softmax(logits, dim=-1).tolist()
            nli_ids = logits.argmax(dim=-1).tolist()

        for (i, _, _), nli_id, probs in zip(non_empty, nli_ids, probs_batch):
            results[i] = _format_nli_result(model, nli_id, probs)

    return results


def aggregate_verdicts(
    evidence_results: List[Dict[str, Any]], 
    evidence_weights: List[float],
    claim_text: str = ""
) -> Dict[str, Any]:
    """Combine NLI results, weighting each vote by relevance and confidence."""
    if not evidence_results:
        return {
            "verdict": "NOT ENOUGH INFO",
            "confidence": 1.0,
            "probs": {"SUPPORTED": 0.0, "REFUTED": 0.0, "NOT ENOUGH INFO": 1.0},
            "reasoning": "No evidence provided",
            "evidence_votes": [],
        }
    
    for i, result in enumerate(evidence_results):
        if result["verdict"] == "REFUTED" and result["confidence"] > 0.7:
            return {
                "verdict": "REFUTED",
                "confidence": result["confidence"],
                "probs": result["probs"],
                "reasoning": f"Strong contradiction from evidence {i}: {result.get('premise', '')[:100]}",
                "evidence_votes": evidence_results,
            }
    
    nli_conf_threshold = get_threshold("nli_confidence", 0.5)
    
    supported_weight = 0.0
    refuted_weight = 0.0
    nei_weight = 0.0
    evidence_votes = []
    
    for i, (result, weight) in enumerate(zip(evidence_results, evidence_weights)):
        if result["confidence"] < nli_conf_threshold:
            nei_weight += weight * 0.5
            continue
        
        nli_conf = result["confidence"]
        effective_weight = weight * nli_conf
        
        evidence_votes.append({
            "index": i,
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "weight": weight,
            "effective_weight": effective_weight,
            "nli_label": result["nli_label"],
        })
        
        if result["verdict"] == "SUPPORTED":
            supported_weight += effective_weight
        elif result["verdict"] == "REFUTED":
            refuted_weight += effective_weight
        else:
            # Neutral evidence has less weight than a clear verdict.
            nei_weight += effective_weight * 0.3
    
    total_weight = supported_weight + refuted_weight + nei_weight
    
    if total_weight == 0:
        return {
            "verdict": "NOT ENOUGH INFO",
            "confidence": 1.0,
            "probs": {"SUPPORTED": 0.0, "REFUTED": 0.0, "NOT ENOUGH INFO": 1.0},
            "reasoning": "No confident evidence",
            "evidence_votes": evidence_votes,
        }
    
    probs = {
        "SUPPORTED": round(supported_weight / total_weight, 4),
        "REFUTED": round(refuted_weight / total_weight, 4),
        "NOT ENOUGH INFO": round(nei_weight / total_weight, 4),
    }
    
    if probs["REFUTED"] > probs["SUPPORTED"] and probs["REFUTED"] > probs["NOT ENOUGH INFO"]:
        verdict = "REFUTED"
        confidence = probs["REFUTED"]
    elif probs["SUPPORTED"] > probs["REFUTED"] and probs["SUPPORTED"] > probs["NOT ENOUGH INFO"]:
        verdict = "SUPPORTED"
        confidence = probs["SUPPORTED"]
    else:
        verdict = "NOT ENOUGH INFO"
        confidence = probs["NOT ENOUGH INFO"]
    
    confidence = round(confidence, 4)
    
    supporting = [v for v in evidence_votes if v["verdict"] == "SUPPORTED"]
    refuting = [v for v in evidence_votes if v["verdict"] == "REFUTED"]
    
    reasoning_parts = []
    if supporting:
        reasoning_parts.append(f"{len(supporting)} evidence support (weight: {sum(v['effective_weight'] for v in supporting):.2f})")
    if refuting:
        reasoning_parts.append(f"{len(refuting)} evidence refute (weight: {sum(v['effective_weight'] for v in refuting):.2f})")
    
    reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Insufficient evidence"
    
    return {
        "verdict": verdict,
        "confidence": confidence,
        "probs": probs,
        "reasoning": reasoning,
        "evidence_votes": evidence_votes,
    }


def structured_nli_pipeline(
    claim_text: str,
    ranked_evidence: List[Dict[str, Any]],
    tokenizer=None,
    model=None
) -> Dict[str, Any]:
    """Classify each evidence item and combine the resulting votes."""
    if not ranked_evidence:
        return {
            "verdict": "NOT ENOUGH INFO",
            "confidence": 1.0,
            "probs": {"SUPPORTED": 0.0, "REFUTED": 0.0, "NOT ENOUGH INFO": 1.0},
            "reasoning": "No evidence provided",
        }
    
    premises = []
    weights = []
    for item in ranked_evidence:
        text = (
            item.get("linearized") 
            or item.get("text") 
            or item.get("content") 
            or item.get("snippet") 
            or ""
        )
        premises.append(text)
        weights.append(item.get("cross_encoder_score", item.get("semantic_score", 0.5)))
    
    evidence_results = predict_nli_batch(claim_text, premises)
    
    final = aggregate_verdicts(evidence_results, weights, claim_text)
    final["per_evidence"] = [
        {**er, "premise": p[:200]} 
        for er, p in zip(evidence_results, premises)
    ]
    enriched = []
    for i, (result, item, premise, weight) in enumerate(zip(evidence_results, ranked_evidence, premises, weights)):
        evidence_text = item.get("source_sentence") or premise
        enriched.append({
            "index": i,
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "weight": weight,
            "effective_weight": round(weight * result["confidence"], 4),
            "nli_label": result["nli_label"],
            "source_type": item.get("source_type", "evidence"),
            "source": item.get("source") or item.get("source_wiki") or item.get("title") or "",
            "text": evidence_text[:500],
        })

    deciding = [e for e in enriched if e["verdict"] == final["verdict"]]
    if not deciding:
        deciding = enriched
    deciding.sort(key=lambda e: e["effective_weight"], reverse=True)
    final["decision_evidence"] = deciding[:3]
    
    return final
