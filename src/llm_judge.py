"""
LLM-as-judge for the fact-checking pipeline.
Uses Mistral (via Ollama) to produce a verdict + reasoning from the same evidence.
"""

import re
from typing import Any, Dict, List, Optional

from src.wiki_llm_extractor import _ollama_model, _ollama_available, _ollama_generate

_MAX_EVIDENCE_CHARS = 8000

_JUDGE_PROMPT = """You are a precise fact-checker. Determine if the claim is SUPPORTED,
REFUTED, or NOT ENOUGH INFO based on the evidence provided.

First, identify the key entities and claims in the statement being checked.
Then, examine the evidence for each part.
Finally, make your decision.

Claim: {claim}

Evidence:
{evidence}

Reply with EXACTLY one line containing ONLY the verdict (SUPPORTED, REFUTED, or
NOT ENOUGH INFO), then a blank line, then your step-by-step reasoning."""


def _format_evidence(ranked_evidence: List[Dict[str, Any]]) -> str:
    parts = []
    char_count = 0
    for item in ranked_evidence:
        text = (
            item.get("linearized")
            or item.get("text")
            or item.get("content")
            or item.get("snippet")
            or ""
        )
        source = item.get("source") or item.get("source_wiki") or item.get("title") or ""
        if source:
            block = f"[{source}] {text}"
        else:
            block = text
        if char_count + len(block) > _MAX_EVIDENCE_CHARS:
            remaining = _MAX_EVIDENCE_CHARS - char_count
            if remaining > 60:
                parts.append(block[:remaining])
            break
        parts.append(block)
        char_count += len(block) + 1
    return "\n".join(parts)


def _parse_verdict(raw: str) -> Dict[str, Any]:
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    if not lines:
        return {"verdict": "NOT ENOUGH INFO", "reasoning": raw[:200]}

    first = lines[0].upper().rstrip(".,!?:;")
    if "SUPPORTED" in first:
        verdict = "SUPPORTED"
    elif "REFUTED" in first or "CONTRADICT" in first:
        verdict = "REFUTED"
    elif "NOT ENOUGH INFO" in first or "NEI" in first or "INSUFFICIENT" in first:
        verdict = "NOT ENOUGH INFO"
    else:
        verdict = "NOT ENOUGH INFO"

    reasoning = "\n".join(lines[1:]).strip() if len(lines) > 1 else raw[:200]
    return {"verdict": verdict, "reasoning": reasoning}


def judge(claim_text: str, ranked_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run LLM-as-judge: feed claim + evidence to Mistral, return verdict + reasoning.

    Returns dict with keys: verdict, reasoning, raw_response.
    Falls back to {"verdict": None} if Ollama is down.
    """
    model = _ollama_model()
    if not _ollama_available(model):
        return {"verdict": None, "reasoning": "LLM unavailable", "raw_response": ""}

    evidence_text = _format_evidence(ranked_evidence)
    prompt = _JUDGE_PROMPT.format(claim=claim_text, evidence=evidence_text)

    try:
        raw = _ollama_generate(model, prompt, max_new_tokens=512, temperature=0.0)
    except Exception as e:
        return {"verdict": None, "reasoning": f"LLM error: {e}", "raw_response": ""}

    result = _parse_verdict(raw)
    result["raw_response"] = raw
    return result


def evaluate(claim_text: str, evidence_texts: List[str], gold_label: int) -> Dict[str, Any]:
    """Run LLM judge and compare to gold label for eval metrics.

    Args:
        claim_text: The claim to evaluate.
        evidence_texts: List of evidence text strings.
        gold_label: 0 for SUPPORTED, 1 for REFUTED.

    Returns dict with: correct, verdict, gold_verdict, reasoning.
    """
    evidence_items = [{"text": t, "source": ""} for t in evidence_texts]
    result = judge(claim_text, evidence_items)
    verdict = result.get("verdict")
    if verdict is None:
        return {"correct": False, "verdict": None, "reasoning": "LLM unavailable"}
    gold_verdict = "SUPPORTED" if gold_label == 0 else "REFUTED"
    return {
        "correct": verdict == gold_verdict,
        "verdict": verdict,
        "gold_verdict": gold_verdict,
        "reasoning": result.get("reasoning", ""),
    }
