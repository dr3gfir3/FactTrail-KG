import hashlib
import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

QDRANT_DIR = Path("data/qdrant")
COLLECTION = "wiki_kg_triples"
INDEX_META_FILE = QDRANT_DIR / f"{COLLECTION}.json"

# Per-process caches keyed by the triples hash, so we don't re-validate the
# Qdrant index or rebuild BM25 on every retrieve() call.
_index_cache = {"hash": None, "valid": False}
_bm25_cache = {"hash": None, "bm25": None, "tokenized": None}


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


_client_singleton = None


def _get_client():
    global _client_singleton
    if _client_singleton is not None:
        return _client_singleton
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        return None
    QDRANT_DIR.mkdir(parents=True, exist_ok=True)
    # Creating a local QdrantClient is expensive (~20s); build it once per process.
    _client_singleton = QdrantClient(path=str(QDRANT_DIR))
    return _client_singleton


def _recreate_collection(client, size):
    from qdrant_client import models

    try:
        client.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
        )
    except AttributeError:
        client.delete_collection(COLLECTION)
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
        )


def _collection_size(client):
    try:
        return client.count(COLLECTION, exact=True).count
    except Exception:
        return -1


def _triples_hash(triples):
    payload = [t.get("linearized", "") for t in triples]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()


def _index_current(triples, client):
    if _collection_size(client) != len(triples) or not INDEX_META_FILE.exists():
        return False
    try:
        meta = json.loads(INDEX_META_FILE.read_text(encoding="utf-8"))
    except Exception:
        return False
    return meta.get("triples_hash") == _triples_hash(triples)


def _ensure_index(triples, model):
    client = _get_client()
    if not client or not triples:
        return client
    h = _triples_hash(triples)
    # Skip the expensive count()+file read if we already validated this KG version.
    if _index_cache.get("hash") == h and _index_cache.get("valid"):
        return client
    if _index_current(triples, client):
        _index_cache["hash"] = h
        _index_cache["valid"] = True
        return client

    texts = [t["linearized"] for t in triples]
    vectors = model.encode(texts, convert_to_numpy=True)
    _recreate_collection(client, int(vectors.shape[1]))

    from qdrant_client import models

    points = [
        models.PointStruct(id=i, vector=vectors[i].tolist(), payload={"idx": i, "text": texts[i]})
        for i in range(len(triples))
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    INDEX_META_FILE.write_text(json.dumps({"triples_hash": _triples_hash(triples)}), encoding="utf-8")
    _index_cache["hash"] = h
    _index_cache["valid"] = True
    return client


def _get_bm25(triples):
    """Build (and memoize) the BM25 index for the current KG version."""
    h = _triples_hash(triples)
    if _bm25_cache.get("hash") == h and _bm25_cache.get("bm25") is not None:
        return _bm25_cache["bm25"], _bm25_cache["tokenized"]
    tokenized = [_tokenize(t["linearized"]) for t in triples]
    bm25 = BM25Okapi(tokenized)
    _bm25_cache["hash"] = h
    _bm25_cache["bm25"] = bm25
    _bm25_cache["tokenized"] = tokenized
    return bm25, tokenized


def _vector_hits(client, query_vector, top_k):
    if not client:
        return {}
    try:
        points = client.query_points(collection_name=COLLECTION, query=query_vector, limit=top_k).points
    except AttributeError:
        points = client.search(collection_name=COLLECTION, query_vector=query_vector, limit=top_k)
    return {int(p.payload["idx"]): float(p.score) for p in points}


def _bm25_hits(query, triples, top_k):
    bm25, tokenized = _get_bm25(triples)
    if not any(tokenized):
        return {}
    scores = bm25.get_scores(_tokenize(query))
    if not len(scores):
        return {}
    if float(np.max(scores)) <= 0:
        q = set(_tokenize(query))
        scores = np.array([len(q & set(tokens)) for tokens in tokenized], dtype=float)
        if float(np.max(scores)) <= 0:
            return {}
    order = np.argsort(scores)[::-1][:top_k]
    max_score = float(scores[order[0]]) or 1.0
    return {int(i): float(scores[i]) / max_score for i in order if scores[i] > 0}


def retrieve(entities, max_per_entity=50):
    from src.knowledge_graph import _load, _normalize_entity, search_indices_for_term
    from src.semantic_ranking import get_sbert_model

    data = _load()
    triples = data.get("triples", [])
    if not triples:
        return []

    names = []
    exact = set()
    for entity in entities:
        terms = [str(entity)]
        if isinstance(entity, dict):
            canonical = entity.get("canonical", entity.get("text", ""))
            terms = [canonical] + entity.get("aliases", [])
        for term in terms:
            if not term:
                continue
            names.append(str(term))
            exact.update(search_indices_for_term(term, max_per_entity))

    query = " ".join(dict.fromkeys(names))
    if not query:
        return []

    model = get_sbert_model()
    client = _ensure_index(triples, model)
    qvec = model.encode(query, convert_to_numpy=True).tolist()
    limit = max(max_per_entity, 20)
    vector = _vector_hits(client, qvec, limit)
    lexical = _bm25_hits(query, triples, limit)

    scored = {}
    for idx in set(vector) | set(lexical) | exact:
        score = (0.45 * vector.get(idx, 0.0)) + (0.45 * lexical.get(idx, 0.0)) + (0.10 if idx in exact else 0.0)
        if score <= 0:
            continue
        item = triples[idx].copy()
        item.update({
            "qdrant_score": round(vector.get(idx, 0.0), 4),
            "bm25_score": round(lexical.get(idx, 0.0), 4),
            "graph_score": 1.0 if idx in exact else 0.0,
            "combined_score": round(score, 4),
            "retrieval_method": "local_graph_rag",
        })
        scored[idx] = item

    return sorted(scored.values(), key=lambda x: x["combined_score"], reverse=True)[:max_per_entity]
