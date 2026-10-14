import json
import re
from pathlib import Path

import spacy
from src.config import get_model, get_retrieval, get_threshold

ENTITY_DIR = Path("data/entities")

RELEVANT_LABELS = {
    "PERSON", "ORG", "GPE", "LOC", "DATE",
    "EVENT", "FAC", "LAW", "NORP", "PRODUCT",
    "WORK_OF_ART", "LANGUAGE",
}

HIGH_VALUE_LABELS = {
    "PERSON", "ORG", "GPE", "LOC", "EVENT", "PRODUCT", "WORK_OF_ART", "FAC",
}

_YEAR_RE = re.compile(r"^\d{4}$")
_DATE_FOCUS_RE = re.compile(r"\b(when|date|year|month|day)\b", re.I)
_LEADING_ARTICLE_RE = re.compile(r"^(a|an|the)\s+", re.I)


def ensure_spacy_model():
    model_name = get_model("spacy", "en_core_web_sm")
    try:
        spacy.load(model_name)
    except OSError:
        from spacy.cli import download
        download(model_name)

_nlp_cache = None


def load_nlp():
    global _nlp_cache
    if _nlp_cache is None:
        ensure_spacy_model()
        _nlp_cache = spacy.load(get_model("spacy", "en_core_web_sm"))
    return _nlp_cache


def extract_entities(claim_text, nlp=None):
    if nlp is None:
        nlp = load_nlp()
    doc = nlp(claim_text)
    seen = set()
    entities = []
    for ent in doc.ents:
        if ent.label_ in RELEVANT_LABELS and ent.text.lower() not in seen:
            seen.add(ent.text.lower())
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })
    return entities


def rank_entities_for_retrieval(entities, claim_text="", *, max_primary=None,
                                allow_date_entities=None, min_entity_chars=None):
    max_primary = max_primary if max_primary is not None else get_retrieval("max_primary_entities", 5)
    allow_date_entities = (
        allow_date_entities
        if allow_date_entities is not None
        else get_retrieval("allow_date_entities", False)
    )
    min_entity_chars = min_entity_chars if min_entity_chars is not None else get_retrieval("min_entity_chars", 3)
    date_focused = bool(allow_date_entities or _DATE_FOCUS_RE.search(claim_text or ""))

    items = []
    seen = set()
    for i, ent in enumerate(entities):
        text = (ent.get("text") if isinstance(ent, dict) else str(ent)).strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        items.append({**ent, "text": text, "_i": i, "_key": key, "role": "primary", "discard_reason": None})

    for ent in items:
        text = ent["text"]
        label = ent.get("label", "")
        lower = ent["_key"]
        if len(text) < min_entity_chars:
            ent["role"], ent["discard_reason"] = "discarded", "too_short"
        elif (label == "DATE" or _YEAR_RE.match(text)) and not date_focused:
            ent["role"], ent["discard_reason"] = "discarded", "date"
        elif "'s " in lower and len(text.split()) > 2:
            ent["role"], ent["discard_reason"] = "discarded", "possessive_chunk"

    for ent in items:
        if ent["role"] == "discarded":
            continue
        lower = ent["_key"]
        for other in items:
            if other is ent or other["role"] == "discarded":
                continue
            other_lower = other["_key"]
            if lower in other_lower and lower != other_lower:
                if _LEADING_ARTICLE_RE.match(other_lower):
                    other["role"], other["discard_reason"] = "context", "article_chunk"
                else:
                    ent["role"], ent["discard_reason"] = "discarded", "covered_by_longer_entity"
                break

    primaries = [e for e in items if e["role"] == "primary"]
    primaries.sort(key=lambda e: (
        0 if e.get("label") in HIGH_VALUE_LABELS else 1,
        -len(e["text"]),
        e["_i"],
    ))
    for ent in primaries[max_primary:]:
        ent["role"], ent["discard_reason"] = "context", "primary_limit"

    items.sort(key=lambda e: (
        {"primary": 0, "context": 1, "discarded": 2}[e["role"]],
        0 if e.get("label") in HIGH_VALUE_LABELS else 1,
        -len(e["text"]),
        e["_i"],
    ))
    return [{k: v for k, v in e.items() if not k.startswith("_")} for e in items]


def is_generic_chunk(chunk):
    text = chunk.text.strip()
    has_propn = any(t.pos_ == "PROPN" for t in chunk)
    if chunk[0].pos_ == "DET" and not has_propn:
        return True
    if len(text) < 4 and not any(t.pos_ == "PROPN" for t in chunk) and text[0].islower():
        return True
    return False


def extract_from_noun_chunks(doc, entities, seen, sbert, prototypes, diagnostics=None):
    from sentence_transformers import util
    base_threshold = get_threshold("entity_prototype", 0.45)
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip()
        chunk_lower = chunk_text.lower()
        if chunk_lower in seen or len(chunk_text) <= 2:
            if diagnostics is not None:
                diagnostics.append({
                    "text": chunk_text,
                    "start": chunk.start_char,
                    "end": chunk.end_char,
                    "scores": {},
                    "best_type": "already extracted" if chunk_lower in seen else "too short",
                    "best_score": 0.0,
                    "threshold": None,
                    "is_entity_candidate": False,
                })
            continue
        if is_generic_chunk(chunk):
            if diagnostics is not None:
                diagnostics.append({
                    "text": chunk_text,
                    "start": chunk.start_char,
                    "end": chunk.end_char,
                    "scores": {},
                    "best_type": "generic chunk",
                    "best_score": 0.0,
                    "threshold": None,
                    "is_entity_candidate": False,
                })
            continue

        has_propn = any(t.pos_ == "PROPN" for t in chunk)
        is_single_upper = len(chunk_text.split()) == 1 and chunk_text[0].isupper()

        chunk_emb = sbert.encode(chunk_text, convert_to_tensor=True)
        scores = {}
        best_type = None
        best_score = 0.0
        for etype, proto_emb in prototypes.items():
            score = float(util.cos_sim(chunk_emb, proto_emb).item())
            scores[etype] = score
            if score > best_score:
                best_score = score
                best_type = etype

        if has_propn or is_single_upper:
            threshold = max(0.30, base_threshold - 0.15)
        elif len(chunk_text) >= 4:
            threshold = max(0.25, base_threshold - 0.20)
        else:
            threshold = base_threshold

        if best_score >= threshold:
            entities.append({"text": chunk_text, "label": best_type, "start": chunk.start_char, "end": chunk.end_char})
            seen.add(chunk_lower)
        elif has_propn:
            entities.append({"text": chunk_text, "label": "MISC", "start": chunk.start_char, "end": chunk.end_char})
            seen.add(chunk_lower)

        if diagnostics is not None:
            diagnostics.append({
                "text": chunk_text,
                "start": chunk.start_char,
                "end": chunk.end_char,
                "scores": scores,
                "best_type": best_type,
                "best_score": best_score,
                "threshold": threshold,
                "is_entity_candidate": best_score >= threshold,
            })


_CONNECTIVES = {"of", "in", "at", "and", "the", "for", "del", "della",
                 "von", "van", "der", "den", "de", "da", "el", "la",
                 "le", "dos", "do", "das", "di", "du", "des", "y"}
_SKIP_CAPS = {"the", "this", "these", "those", "it", "its", "they",
              "them", "their", "we", "us", "our", "you", "your",
              "he", "him", "his", "she", "her", "hers"}


def extract_uppercase_words(doc, seen):
    new_entities = []
    i = 0
    while i < len(doc):
        t = doc[i]
        w = t.text
        if not w or not w[0].isupper():
            i += 1
            continue
        if w.lower() in _SKIP_CAPS:
            i += 1
            continue
        if t.is_sent_start and t.pos_ not in ("PROPN", "NOUN", "ADJ"):
            i += 1
            continue
        if len(w) == 1 and w.isalpha():
            if i + 1 >= len(doc) or not doc[i + 1].text[0].isupper():
                i += 1
                continue
        if len(w) < 3 and not (len(w) == 1 and w.isalpha()):
            i += 1
            continue

        parts = [w]
        j = i + 1
        while j < len(doc):
            t_next = doc[j]
            w_next = t_next.text
            t_is_named = w_next[0].isupper() and (len(w_next) > 2 or (len(w_next) == 1 and w_next.isalpha()))
            if t_is_named:
                parts.append(w_next)
            elif w_next.lower() in _CONNECTIVES and j + 1 < len(doc) and doc[j + 1].text[0].isupper():
                parts.append(w_next)
            else:
                break
            j += 1

        compound = " ".join(parts)
        start_char = doc[i].idx
        end_char = doc[j - 1].idx + len(doc[j - 1].text)
        if compound.lower() not in seen:
            new_entities.append({"text": compound, "label": "MISC", "start": start_char, "end": end_char})
            seen.add(compound.lower())
        i = j

    return new_entities


def extract_all(claims, batch_size=64):
    nlp = load_nlp()
    results = []
    for claim in claims:
        entities = extract_entities(claim["claim_text"], nlp=nlp)
        results.append({
            "claim_id": claim["claim_id"],
            "claim_text": claim["claim_text"],
            "entities": entities,
        })
    return results


def save_entities(data, split="train"):
    ENTITY_DIR.mkdir(parents=True, exist_ok=True)
    path = ENTITY_DIR / f"{split}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def load_entities(split="train"):
    path = ENTITY_DIR / f"{split}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
