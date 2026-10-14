import re
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from src.config import get_model
from src.eval import DEVICE, load_model as load_nli_model

_nli_cache: List[Optional[Any]] = [None, None]


def _load_roberta_nli():
    if _nli_cache[0] is None:
        model_name = get_model("nli", "roberta-large-mnli")
        tokenizer, model = load_nli_model(model_name)
        _nli_cache[:] = [tokenizer, model]
    return _nli_cache[0], _nli_cache[1]


def _get_triple(item: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    triple = item.get("triple")
    if isinstance(triple, (list, tuple)) and len(triple) == 3:
        return str(triple[0]), str(triple[1]), str(triple[2])
    parts = item.get("linearized", "").split(None, 2)
    if len(parts) == 3:
        subj, pred, rest = parts
        obj = rest.rstrip(".")
        return subj, pred, obj
    return None


def build_multihop_premise(
    kg_evidence: List[Dict[str, Any]],
    text_evidence: List[Dict[str, Any]],
) -> str:
    """Build an entity-centered premise that connects related evidence.

    Groups triples by subject entity so RoBERTa can see all attributes
    of the same entity in one passage — enabling multi-hop detection.
    """
    entity_attrs = defaultdict(list)
    for item in kg_evidence:
        t = _get_triple(item)
        if t:
            subj, pred, obj = t
            entity_attrs[subj].append(f"{pred} {obj}")

    paragraphs = []
    for entity, attrs in entity_attrs.items():
        if attrs:
            sentences = []
            for a in attrs:
                if not sentences:
                    sentences.append(f"{entity} {a}")
                else:
                    sentences.append(a)
            paragraphs.append("; ".join(sentences) + ".")

    for item in text_evidence or []:
        text = item.get("text", "")
        if text:
            paragraphs.append(text)

    return " ".join(paragraphs)


def _map_nli_label(model, label: str) -> str:
    label_lower = label.lower()
    if "entail" in label_lower:
        return "SUPPORTED"
    elif "contradict" in label_lower:
        return "REFUTED"
    return "NOT ENOUGH INFO"


def predict_multihop_nli(
    claim_text: str,
    kg_evidence: List[Dict[str, Any]],
    text_evidence: List[Dict[str, Any]],
    premise: Optional[str] = None,
) -> Dict[str, Any]:
    """Enhanced NLI with graph-grounded multi-hop premise.

    Builds entity-connected premise from KG triples + wiki text,
    then runs RoBERTa NLI against the claim.

    Returns dict with: verdict, confidence, probs, premise, evidence_counts.
    """
    tokenizer, model = _load_roberta_nli()

    if premise is None:
        premise = build_multihop_premise(kg_evidence, text_evidence)
    if not premise.strip():
        return {
            "verdict": "NOT ENOUGH INFO",
            "confidence": 1.0,
            "probs": {"SUPPORTED": 0.0, "REFUTED": 0.0, "NOT ENOUGH INFO": 1.0},
            "premise": "",
            "evidence_counts": {"kg": len(kg_evidence), "text": len(text_evidence)},
        }

    inputs = tokenizer(
        premise,
        claim_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze().tolist()
        nli_id = logits.argmax(dim=-1).item()

    nli_label = model.config.id2label[nli_id]

    verdict_probs = {"SUPPORTED": 0.0, "REFUTED": 0.0, "NOT ENOUGH INFO": 0.0}
    prob_list = probs if isinstance(probs, list) else [probs]
    for idx, prob in enumerate(prob_list):
        label = _map_nli_label(model, model.config.id2label[idx])
        verdict_probs[label] += float(prob)

    verdict = _map_nli_label(model, nli_label)
    confidence = round(verdict_probs[verdict], 4)

    return {
        "verdict": verdict,
        "confidence": confidence,
        "probs": verdict_probs,
        "nli_label": nli_label,
        "premise": premise,
        "evidence_counts": {"kg": len(kg_evidence), "text": len(text_evidence)},
    }


def predict_nli_simple(claim_text: str, premise: str) -> Dict[str, Any]:
    """Simple RoBERTa NLI for a single claim-premise pair.

    Used for faithfulness checks and other non-multihop contexts.
    """
    tokenizer, model = _load_roberta_nli()

    if not premise.strip():
        return {
            "verdict": "NOT ENOUGH INFO",
            "confidence": 1.0,
            "probs": {"SUPPORTED": 0.0, "REFUTED": 0.0, "NOT ENOUGH INFO": 1.0},
        }

    inputs = tokenizer(
        premise,
        claim_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze().tolist()
        nli_id = logits.argmax(dim=-1).item()

    nli_label = model.config.id2label[nli_id]

    verdict_probs = {"SUPPORTED": 0.0, "REFUTED": 0.0, "NOT ENOUGH INFO": 0.0}
    prob_list = probs if isinstance(probs, list) else [probs]
    for idx, prob in enumerate(prob_list):
        label = _map_nli_label(model, model.config.id2label[idx])
        verdict_probs[label] += float(prob)

    verdict = _map_nli_label(model, nli_label)

    return {
        "verdict": verdict,
        "confidence": round(verdict_probs[verdict], 4),
        "probs": verdict_probs,
        "nli_label": nli_label,
    }


def faithfulness_check(
    claim_text: str,
    reasoning_text: str,
    evidence_texts: List[str],
) -> Dict[str, Any]:
    """Verify LLM reasoning is faithful to evidence using RoBERTa NLI.

    Parses reasoning into atomic claims, checks each against evidence.
    Returns dict with faithful flag, unfaithful_claims, and per-claim scores.
    """
    tokenizer, model = _load_roberta_nli()

    atomic_claims = [s.strip() for s in re.split(r"[.!]\s*", reasoning_text) if s.strip()]
    if not atomic_claims:
        atomic_claims = [reasoning_text]

    evidence_str = " ".join(evidence_texts) if evidence_texts else ""

    results = []
    unfaithful = 0
    for claim in atomic_claims:
        inputs = tokenizer(
            evidence_str,
            claim,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(DEVICE)

        with torch.no_grad():
            logits = model(**inputs).logits
            nli_id = logits.argmax(dim=-1).item()

        label = _map_nli_label(model, model.config.id2label[nli_id])
        is_faithful = label != "REFUTED"
        if not is_faithful:
            unfaithful += 1
        results.append({"claim": claim, "verdict": label, "faithful": is_faithful})

    total = len(results)
    return {
        "faithful": unfaithful == 0,
        "faithfulness_score": round((total - unfaithful) / total, 4) if total else 1.0,
        "unfaithful_count": unfaithful,
        "total_claims": total,
        "claims": results,
    }
