import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.config import get_model, get_pipeline as get_config_value

_CONTROL_RE = re.compile(r"<\|.*?\|>|```(?:\w+)?|```")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SECTION_RE = r"##\s*{name}\s*\n(.*?)(?=\n##\s+|\Z)"
_MAX_SOURCE_CHARS = 1200
_BAD_ENTITY_RE = re.compile(
    r"^(?:a|an|and|as|at|by|for|from|in|into|of|on|or|the|to|until|with)$",
    re.IGNORECASE,
)

_OLLAMA_MODEL_CACHE = None


def _ollama_base() -> str:
    return get_model("ollama_base_url", "http://localhost:11434")


def _ollama_model() -> str:
    global _OLLAMA_MODEL_CACHE
    if _OLLAMA_MODEL_CACHE is None:
        _OLLAMA_MODEL_CACHE = get_model(
            "ollama_llm", get_model("llm", "nemotron-3-super:cloud"))
    return _OLLAMA_MODEL_CACHE


def _ollama_available(model: str) -> bool:
    """Start Ollama if needed and pull a missing model."""
    base = _ollama_base()
    try:
        r = requests.get(f"{base}/api/tags", timeout=3)
        if r.status_code != 200:
            raise requests.RequestException("bad status")
    except requests.RequestException:
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        except Exception:
            pass
        import time
        time.sleep(3)
        try:
            r = requests.get(f"{base}/api/tags", timeout=3)
            if r.status_code != 200:
                return False
        except requests.RequestException:
            return False

    models = [m.get("name") for m in r.json().get("models", [])]
    if model in models:
        return True
    try:
        pr = requests.post(f"{base}/api/pull", json={"model": model, "stream": False},
                          timeout=300)
        return pr.status_code == 200
    except requests.RequestException:
        return False


def _ollama_generate(model: str, prompt: str, max_new_tokens: Optional[int],
                    temperature: float) -> str:
    base = _ollama_base()
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "num_predict": int(max_new_tokens) if max_new_tokens else 512,
        },
    }
    r = requests.post(f"{base}/api/generate", json=payload, timeout=180)
    r.raise_for_status()
    return r.json().get("response", "")


def _clean_text(text: str) -> str:
    return _THINK_RE.sub("", _CONTROL_RE.sub("", str(text or ""))).strip()


def _section(text: str, name: str) -> Optional[str]:
    match = re.search(_SECTION_RE.format(name=re.escape(name)), _clean_text(text), re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def _run_llm_prompt(prompt: str, max_new_tokens: Optional[int], temperature: float) -> str:
    model = _ollama_model()
    if not _ollama_available(model):
        raise RuntimeError("ollama-unavailable: Ollama server or model not available")
    return _ollama_generate(model, prompt, max_new_tokens, temperature)


def _clean_relation_field(text: str) -> str:
    text = _clean_text(text).strip("()[]{} -*\t")
    text = re.sub(r"^(?:subject|predicate|object)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("[[", "").replace("]]", "")
    return re.sub(r"\s+", " ", text).strip(" .;:-")


def _parse_relation_line(line: str) -> Optional[Tuple[str, str, str]]:
    line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", _clean_text(line)).strip()
    if "|" not in line or "Source Text:" in line:
        return None
    if line.startswith("(") and line.endswith(")"):
        line = line[1:-1]

    parts = [_clean_relation_field(part) for part in line.split("|", 2)]
    if len(parts) != 3 or not all(parts):
        return None
    if _BAD_ENTITY_RE.match(parts[0]) or _BAD_ENTITY_RE.match(parts[2]):
        return None
    return parts[0], parts[1], parts[2]


def _parse_llm_relations(wiki_content: str) -> List[Dict[str, Any]]:
    relations_text = _section(wiki_content, "Relations") or _clean_text(wiki_content)
    triples = []
    seen = set()

    for line in relations_text.splitlines():
        parsed = _parse_relation_line(line)
        if not parsed:
            continue
        key = tuple(part.lower() for part in parsed)
        if key in seen:
            continue
        seen.add(key)
        subject, predicate, obj = parsed
        triples.append({
            "triple": [subject, predicate, obj],
            "linearized": f"{subject} {predicate} {obj}",
            "source_sentence": "",
            "method": "llm",
            "confidence": 0.8,
            "negated": False,
            "entity_types": {},
        })

    return triples


def _parse_llm_entities(wiki_content: str) -> List[str]:
    text = _section(wiki_content, "Entities") or _clean_text(wiki_content)
    entities = re.findall(r"\[\[([^\]]+)\]\]", text)
    if not entities and _section(wiki_content, "Entities") is not None:
        entities = [re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line) for line in text.splitlines()]
    return [entity for entity in dict.fromkeys(_clean_relation_field(e) for e in entities) if entity]


def _parse_llm_summary(wiki_content: str) -> str:
    summary = _section(wiki_content, "Summary")
    if not summary:
        return ""
    summary = re.sub(r"\[\[([^\]]+)\]\]", r"\1", _clean_text(summary))
    return re.sub(r"\s+", " ", summary).strip()


def _relation_prompt(source_text: str, title: str) -> str:
    return f"""Extract factual knowledge graph relations from the source.
Output ONLY this format, one fact per line:
- ([[Subject]] | predicate | [[Object]])

Rules:
- Use only facts explicitly supported by the source.
- Keep entity names short and canonical.
- Subject and object must be real entities, dates, places, works, or organizations.
- Do not use dangling words like "from", "to", "in", or "of" as subject/object.
- Return 5-10 relations.

Source Title: {title}
Source Text:
{source_text[:_MAX_SOURCE_CHARS]}
"""


def _wiki_prompt(source_text: str, title: str) -> str:
    return f"""Create a concise markdown wiki page from the source.
Output exactly these sections: # {title}, ## Summary, ## Entities, ## Relations.
Relations must be lines like: - ([[Subject]] | predicate | [[Object]])
Subject and object must be real entities, dates, places, works, or organizations; never dangling words like "from" or "to".

Source Title: {title}
Source Text:
{source_text[:_MAX_SOURCE_CHARS]}
"""


def _extract_llm_wiki_content(source_text: str, title: str, max_new_tokens: Optional[int] = None, temperature: float = 0.0) -> str:
    return _run_llm_prompt(_wiki_prompt(source_text, title), max_new_tokens, temperature)


def _extract_llm_relations_content(source_text: str, title: str, max_new_tokens: Optional[int] = None, temperature: float = 0.0) -> str:
    return _run_llm_prompt(_relation_prompt(source_text, title), max_new_tokens, temperature)


def extract_with_llm(source_text: str, title: str) -> Dict[str, Any]:
    max_tokens = get_config_value("llm_max_new_tokens")
    temperature = get_config_value("llm_temperature", 0.0)

    wiki_content = ""
    relation_tokens = None if max_tokens is None else max(128, min(max_tokens, 256))
    relations_content = _extract_llm_relations_content(source_text, title, relation_tokens, temperature)
    triples = _parse_llm_relations(relations_content)
    if not triples:
        wiki_content = _extract_llm_wiki_content(source_text, title, max_tokens, temperature)
        triples = _parse_llm_relations(wiki_content)

    summary = _parse_llm_summary(wiki_content) or source_text[:500]
    entities = _parse_llm_entities(wiki_content or relations_content)
    if not entities:
        entities = list(dict.fromkeys(x for t in triples for x in (t["triple"][0], t["triple"][2])))

    return {
        "wiki_content": _build_wiki_content(title, summary, entities, triples),
        "triples": triples,
        "entities": entities,
        "summary": summary,
    }


def extract_hybrid(source_text: str, title: str, nlp) -> Dict[str, Any]:
    from src.wiki_compiler import _extract_entities_from_text, _extract_triples_from_text, _strip_html

    plain_text = _strip_html(source_text) if "<" in source_text[:100] else source_text
    regex_triples = _extract_triples_from_text(plain_text, nlp, title)
    regex_entities = _extract_entities_from_text(plain_text, nlp)
    summary = plain_text[:1000]

    try:
        llm_result = extract_with_llm(source_text, title)
    except Exception as exc:
        print(f"      LLM extraction failed: {exc}, using regex only")
        llm_result = {"summary": "", "entities": [], "triples": []}

    if len(llm_result["summary"]) > 20:
        summary = llm_result["summary"]

    text_lower = plain_text.lower()
    triples, entities = list(regex_triples), list(regex_entities)
    seen_triples = {(t["triple"][0].lower(), t["triple"][1].lower(), t["triple"][2].lower()) for t in triples}
    seen_entities = {entity.lower() for entity in entities}
    valid_triples = valid_entities = 0

    for entity in llm_result["entities"]:
        key = entity.lower()
        if key not in seen_entities and key in text_lower:
            seen_entities.add(key)
            entities.append(entity)
            valid_entities += 1

    for triple in llm_result["triples"]:
        subject, predicate, obj = triple["triple"]
        key = (subject.lower(), predicate.lower(), obj.lower())
        if key in seen_triples or subject.lower() not in text_lower or obj.lower() not in text_lower:
            continue
        seen_triples.add(key)
        triples.append(triple)
        valid_triples += 1

    print(f"      Hybrid extraction: Added {valid_triples} valid LLM triples, {valid_entities} valid LLM entities to regex baseline")
    return {
        "wiki_content": _build_wiki_content(title, summary, entities, triples),
        "triples": triples,
        "entities": entities,
        "summary": summary,
    }


def _build_wiki_content(title: str, summary: str, entities: List[str], triples: List[Dict[str, Any]]) -> str:
    lines = [f"# {title}", "", "## Summary", summary, "", "## Entities"]
    lines.extend(f"* [[{entity}]]" for entity in entities[:50])
    lines.extend(["", "## Relations"])
    for triple in triples[:50]:
        subject, predicate, obj = triple["triple"]
        lines.append(f"- ([[{subject}]] | {predicate} | [[{obj}]])")
    return "\n".join(lines)
