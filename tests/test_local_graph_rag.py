import numpy as np
from pathlib import Path
import types
import sys


def test_bm25_hits_prefers_lexical_match():
    from src.local_graph_rag import _bm25_hits

    triples = [
        {"linearized": "Ada Lovelace wrote notes"},
        {"linearized": "Paris is in France"},
    ]

    scores = _bm25_hits("Ada notes", triples, 2)

    assert list(scores)[0] == 0
    assert scores[0] == 1.0


def test_retrieve_falls_back_to_bm25_without_qdrant(monkeypatch):
    import src.knowledge_graph as kg
    from src.knowledge_graph import add_triples
    from src.local_graph_rag import retrieve

    kg_file = Path("data/test_local_graph_rag_kg.json")
    try:
        kg_file.unlink()
    except FileNotFoundError:
        pass
    monkeypatch.setattr(kg, "KG_FILE", kg_file)
    kg._invalidate_cache()
    monkeypatch.setattr("src.local_graph_rag._get_client", lambda: None)

    class Model:
        def encode(self, text, convert_to_numpy=False, **_):
            rows = 1 if isinstance(text, str) else len(text)
            return np.ones((rows, 3)) if rows != 1 else np.ones(3)

    monkeypatch.setattr("src.semantic_ranking.get_sbert_model", lambda: Model())
    add_triples("Page", [{"triple": ["Ada Lovelace", "wrote", "Notes"], "linearized": "Ada Lovelace wrote Notes"}])

    results = retrieve([{"canonical": "Ada Lovelace", "aliases": []}], 5)

    assert results[0]["retrieval_method"] == "local_graph_rag"
    assert results[0]["bm25_score"] > 0

    kg_file.unlink()
    kg._invalidate_cache()


def test_ensure_index_rebuilds_when_triples_change_same_count(monkeypatch):
    import src.local_graph_rag as rag

    class Client:
        def __init__(self):
            self.upserts = 0

        def upsert(self, **_):
            self.upserts += 1

    class Model:
        def encode(self, text, convert_to_numpy=False, **_):
            return np.ones((len(text), 3))

    client = Client()
    meta_file = Path("data/test_runtime/local_graph_rag_meta.json")
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        meta_file.unlink()
    except FileNotFoundError:
        pass
    monkeypatch.setattr(rag, "INDEX_META_FILE", meta_file)
    monkeypatch.setattr(rag, "_get_client", lambda: client)
    monkeypatch.setattr(rag, "_collection_size", lambda _: 1)
    monkeypatch.setattr(rag, "_recreate_collection", lambda *_: None)
    monkeypatch.setitem(sys.modules, "qdrant_client", types.SimpleNamespace(
        models=types.SimpleNamespace(PointStruct=lambda **kwargs: kwargs)
    ))

    rag._ensure_index([{"linearized": "Ada wrote notes"}], Model())
    rag._ensure_index([{"linearized": "Ada wrote books"}], Model())

    assert client.upserts == 2
    meta_file.unlink()
