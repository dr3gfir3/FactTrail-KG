import json
from pathlib import Path

import pytest

from src.entity_extractor import (
    RELEVANT_LABELS,
    extract_entities,
    extract_all,
    rank_entities_for_retrieval,
    save_entities,
    load_entities,
)


def test_relevant_labels_includes_common_types():
    assert "PERSON" in RELEVANT_LABELS
    assert "ORG" in RELEVANT_LABELS
    assert "GPE" in RELEVANT_LABELS
    assert "LOC" in RELEVANT_LABELS


def test_extract_entities_person():
    entities = extract_entities("Peder Severin Krøyer was a Danish painter.")
    texts = [e["text"] for e in entities]
    assert "Peder Severin Krøyer" in texts


def test_extract_entities_deduplicates():
    entities = extract_entities("Peder Severin Krøyer and Peder Severin Krøyer both liked painting.")
    texts = [e["text"] for e in entities]
    assert texts.count("Peder Severin Krøyer") == 1


def test_extract_entities_output_structure():
    entities = extract_entities("Michelle Obama visited London in 2020.")
    for e in entities:
        assert "text" in e
        assert "label" in e
        assert "start" in e
        assert "end" in e


def test_extract_all_structure():
    claims = [
        {"claim_id": "1", "claim_text": "Barack Obama was president."},
        {"claim_id": "2", "claim_text": "The Eiffel Tower is in Paris."},
    ]
    results = extract_all(claims)
    assert len(results) == 2
    assert results[0]["claim_id"] == "1"
    assert "entities" in results[0]
    assert isinstance(results[0]["entities"], list)


def test_save_and_load_entities(tmp_path):
    from src.entity_extractor import save_entities, load_entities
    
    entities = [{"text": "Einstein", "label": "PERSON", "start": 0, "end": 8}]
    file_path = tmp_path / "entities.json"
    
    save_entities(entities, file_path)
    loaded = load_entities(file_path)
    
    assert len(loaded) == 1
    assert loaded[0]["text"] == "Einstein"
    assert loaded[0]["label"] == "PERSON"


def test_extract_uppercase_words_spans():
    from src.entity_extractor import extract_uppercase_words
    from src.entity_extractor import load_nlp
    
    nlp = load_nlp()
    text = "The quick brown Fox jumped over the Lazy Dog."
    doc = nlp(text)
    seen = set(["the", "quick", "brown", "jumped", "over"])
    
    entities = extract_uppercase_words(doc, seen)
    
    assert len(entities) == 2
    # Fox
    assert entities[0]["text"] == "Fox"
    assert entities[0]["start"] == 16
    assert entities[0]["end"] == 19
    # Lazy Dog
    assert entities[1]["text"] == "Lazy Dog"
    assert entities[1]["start"] == 36
    assert entities[1]["end"] == 44


def test_extract_uppercase_words_skips_sentence_starters():
    from src.entity_extractor import extract_uppercase_words
    from src.entity_extractor import load_nlp
    
    nlp = load_nlp()
    text = "According to science, evolution is a fact. However, some people disagree."
    doc = nlp(text)
    seen = set(["to", "science,", "evolution", "is", "a", "fact.", "some", "people", "disagree."])
    
    # "According" and "However" should be skipped because they are sent_start and ADP/ADV
    entities = extract_uppercase_words(doc, seen)
    
    assert len(entities) == 0


def test_rank_entities_for_retrieval_jwst_noise():
    claim = "The James Webb Space Telescope launched on an Ariane 5 rocket from French Guiana in 2022."
    ranked = rank_entities_for_retrieval([
        {"text": "James Webb Space Telescope", "label": "PRODUCT", "start": 4, "end": 30},
        {"text": "Telescope", "label": "MISC", "start": 22, "end": 30},
        {"text": "Ariane 5", "label": "PRODUCT", "start": 46, "end": 54},
        {"text": "an Ariane 5 rocket", "label": "PRODUCT", "start": 43, "end": 61},
        {"text": "French Guiana", "label": "GPE", "start": 67, "end": 80},
        {"text": "French", "label": "NORP", "start": 67, "end": 73},
        {"text": "Guiana", "label": "GPE", "start": 74, "end": 80},
        {"text": "2022", "label": "DATE", "start": 84, "end": 88},
        {"text": "Guiana's first full-color images", "label": "MISC", "start": 74, "end": 107},
    ], claim)

    roles = {e["text"]: e["role"] for e in ranked}

    assert roles["James Webb Space Telescope"] == "primary"
    assert roles["Ariane 5"] == "primary"
    assert roles["French Guiana"] == "primary"
    assert roles["French"] == "discarded"
    assert roles["Guiana"] == "discarded"
    assert roles["Telescope"] == "discarded"
    assert roles["2022"] == "discarded"
    assert roles["Guiana's first full-color images"] == "discarded"
    assert roles["an Ariane 5 rocket"] == "context"


def test_rank_entities_prefers_longer_overlaps():
    ranked = rank_entities_for_retrieval([
        {"text": "French", "label": "NORP", "start": 0, "end": 6},
        {"text": "Guiana", "label": "GPE", "start": 7, "end": 13},
        {"text": "French Guiana", "label": "GPE", "start": 0, "end": 13},
        {"text": "Telescope", "label": "MISC", "start": 30, "end": 39},
        {"text": "James Webb Space Telescope", "label": "PRODUCT", "start": 14, "end": 39},
    ], "French Guiana hosts a launch site for the James Webb Space Telescope.")

    roles = {e["text"]: e["role"] for e in ranked}

    assert roles["French Guiana"] == "primary"
    assert roles["James Webb Space Telescope"] == "primary"
    assert roles["French"] == "discarded"
    assert roles["Guiana"] == "discarded"
    assert roles["Telescope"] == "discarded"
