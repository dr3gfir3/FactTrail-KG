"""Tests for knowledge_graph module — focusing on add_triples index correctness."""
import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch


@pytest.fixture(autouse=True)
def tmp_kg(tmp_path, monkeypatch):
    """Use a temporary KG file for each test."""
    kg_file = tmp_path / "knowledge_graph.json"
    emb_file = tmp_path / "kg" / "triple_embeddings.npy"
    emb_idx_file = tmp_path / "kg" / "triple_embeddings_index.json"
    monkeypatch.setattr("src.knowledge_graph.KG_FILE", kg_file)
    monkeypatch.setattr("src.knowledge_graph.KG_EMBEDDINGS_FILE", emb_file)
    monkeypatch.setattr("src.knowledge_graph.KG_EMBEDDINGS_INDEX_FILE", emb_idx_file)
    from src.knowledge_graph import _invalidate_cache
    _invalidate_cache()
    return kg_file


def test_add_triples_basic():
    from src.knowledge_graph import add_triples, search_by_entity, _load

    triples = [
        {"triple": ["Einstein", "born in", "Germany"], "linearized": "Einstein born in Germany"},
        {"triple": ["Einstein", "won", "Nobel Prize"], "linearized": "Einstein won Nobel Prize"},
    ]
    added = add_triples("Test Page", triples)
    assert len(added) == 2

    results = search_by_entity("einstein")
    assert len(results) == 2
    texts = {r["linearized"] for r in results}
    assert "Einstein born in Germany" in texts
    assert "Einstein won Nobel Prize" in texts


def test_add_triples_keeps_metadata():
    from src.knowledge_graph import add_triples, search_by_entity

    add_triples("Test Page", [{
        "triple": ["Ada Lovelace", "wrote", "Notes"],
        "linearized": "Ada Lovelace wrote Notes",
        "source_sentence": "Ada Lovelace wrote notes on the Analytical Engine.",
        "method": "regex",
        "confidence": 0.7,
        "negated": False,
        "entity_types": {"Ada Lovelace": "PERSON", "Notes": "WORK_OF_ART"},
    }])

    result = search_by_entity("ada lovelace")[0]
    assert result["source_sentence"] == "Ada Lovelace wrote notes on the Analytical Engine."
    assert result["method"] == "regex"
    assert result["confidence"] == 0.7
    assert result["negated"] is False
    assert result["entity_types"]["Ada Lovelace"] == "PERSON"


def test_search_hybrid_applies_predicate_weights():
    from src.knowledge_graph import add_triples, search_hybrid

    add_triples("Page", [
        {"triple": ["X", "born in", "Y"], "linearized": "X born in Y"},
        {"triple": ["X", "founded", "Z"], "linearized": "X founded Z"},
    ])

    results = search_hybrid("X")
    assert results[0]["triple"] == ["X", "founded", "Z"]
    assert results[0]["predicate_weight"] > results[1]["predicate_weight"]


def test_add_triples_deduplicates():
    from src.knowledge_graph import add_triples, _load

    triples = [
        {"triple": ["A", "rel", "B"], "linearized": "A rel B"},
        {"triple": ["A", "rel", "B"], "linearized": "A rel B"},  # duplicate
        {"triple": ["C", "rel", "D"], "linearized": "C rel D"},
    ]
    added = add_triples("Test Page", triples)
    assert len(added) == 2  # only 2 unique

    data = _load()
    assert len(data["triples"]) == 2


def test_entity_search_normalizes_wiki_markup_and_spaces():
    from src.knowledge_graph import add_triples, search_by_entity

    add_triples("Page", [{
        "triple": ["[[Ada   Lovelace]]", "wrote", "Notes"],
        "linearized": "Ada Lovelace wrote Notes",
    }])

    assert search_by_entity("ada lovelace")[0]["linearized"] == "Ada Lovelace wrote Notes"
    assert search_by_entity("[[Ada Lovelace]]")[0]["linearized"] == "Ada Lovelace wrote Notes"


def test_add_triples_deduplicates_normalized_markup():
    from src.knowledge_graph import add_triples, _load

    add_triples("Page", [
        {"triple": ["[[Ada Lovelace]]", "wrote", "Notes"], "linearized": "Ada wrote Notes"},
        {"triple": ["Ada   Lovelace", "wrote", "[[Notes]]"], "linearized": "Ada wrote Notes"},
    ])

    assert len(_load()["triples"]) == 1


def test_add_triples_index_correct_with_duplicates():
    """B2 regression test: entity_index must point to correct triple positions
    even when some triples are skipped as duplicates."""
    from src.knowledge_graph import add_triples, search_by_entity, _load

    # First batch
    batch1 = [
        {"triple": ["A", "rel1", "B"], "linearized": "A rel1 B"},
        {"triple": ["C", "rel2", "D"], "linearized": "C rel2 D"},
    ]
    add_triples("Page1", batch1)

    # Second batch: first triple is a duplicate of batch1[0], second is new
    batch2 = [
        {"triple": ["A", "rel1", "B"], "linearized": "A rel1 B"},  # duplicate
        {"triple": ["E", "rel3", "F"], "linearized": "E rel3 F"},  # new, index should be 2
    ]
    add_triples("Page2", batch2)

    data = _load()
    assert len(data["triples"]) == 3  # 2 from batch1 + 1 new from batch2

    # The critical test: search_by_entity("e") should return the correct triple
    results = search_by_entity("e")
    assert len(results) == 1
    assert results[0]["linearized"] == "E rel3 F"

    results_f = search_by_entity("f")
    assert len(results_f) == 1
    assert results_f[0]["linearized"] == "E rel3 F"


def test_clear_and_get_stats():
    from src.knowledge_graph import add_triples, clear, get_stats

    triples = [{"triple": ["X", "r", "Y"], "linearized": "X r Y"}]
    add_triples("Page", triples)
    stats = get_stats()
    assert stats["total_triples"] == 1
    assert stats["pages"] == 1

    clear()
    stats = get_stats()
    assert stats["total_triples"] == 0


def test_add_triples_deletes_stale_embeddings(tmp_kg):
    from src.knowledge_graph import add_triples, _save_embeddings
    import src.knowledge_graph as kg

    _save_embeddings(np.zeros((1, 2)), {"0": "old"})
    assert kg.KG_EMBEDDINGS_FILE.exists()
    assert kg.KG_EMBEDDINGS_INDEX_FILE.exists()

    add_triples("Page", [{"triple": ["A", "r", "B"], "linearized": "A r B"}])

    assert not kg.KG_EMBEDDINGS_FILE.exists()
    assert not kg.KG_EMBEDDINGS_INDEX_FILE.exists()


def test_clear_deletes_embeddings(tmp_kg):
    from src.knowledge_graph import clear, _save_embeddings
    import src.knowledge_graph as kg

    _save_embeddings(np.zeros((1, 2)), {"0": "old"})
    clear()

    assert not kg.KG_EMBEDDINGS_FILE.exists()
    assert not kg.KG_EMBEDDINGS_INDEX_FILE.exists()


def test_search_semantic_ignores_mismatched_embeddings(tmp_kg):
    from src.knowledge_graph import _save_embeddings, search_semantic
    import src.knowledge_graph as kg

    _save_embeddings(np.zeros((1, 2)), {"0": "old"})

    assert search_semantic("query") == []
    assert not kg.KG_EMBEDDINGS_FILE.exists()
    assert not kg.KG_EMBEDDINGS_INDEX_FILE.exists()
