import json, re
from pathlib import Path
import numpy as np

KG_FILE = Path("data/knowledge_graph.json")
KG_EMBEDDINGS_FILE = Path("data/kg/triple_embeddings.npy")
KG_EMBEDDINGS_INDEX_FILE = Path("data/kg/triple_embeddings_index.json")

_kg_cache = None


def _normalize_entity(value):
    text = str(value or "").replace("[[", "").replace("]]", "")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _normalize_page_key(title):
    """Case/space-normalized key for page_index to avoid duplicate-case JSON keys."""
    return re.sub(r"\s+", " ", str(title or "")).strip().lower()


def _triple_key(triple):
    return (
        _normalize_entity(triple[0]),
        re.sub(r"\s+", " ", str(triple[1] or "")).strip().lower(),
        _normalize_entity(triple[2]),
    )


def _invalidate_cache():
    """Clear the in-memory KG cache (useful for testing)."""
    global _kg_cache
    _kg_cache = None


def _load():
    global _kg_cache
    if _kg_cache is not None:
        return _kg_cache

    if KG_FILE.exists():
        with open(KG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"triples": [], "page_index": {}}
    
    if "entity_index" not in data:
        data["entity_index"] = {}
        for i, t in enumerate(data["triples"]):
            for entity in [t["triple"][0], t["triple"][2]]:
                entity_lower = _normalize_entity(entity)
                if entity_lower not in data["entity_index"]:
                    data["entity_index"][entity_lower] = []
                data["entity_index"][entity_lower].append(i)
        _save(data)
    _kg_cache = data
    return data


def _save(data):
    global _kg_cache
    KG_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Normalize page_index keys to avoid duplicate-case JSON keys (which break reload)
    pi = data.get("page_index")
    if isinstance(pi, dict):
        norm_pi = {}
        for k, v in pi.items():
            nk = _normalize_page_key(k)
            norm_pi[nk] = int(norm_pi.get(nk, 0)) + int(v if isinstance(v, int) else 0)
        data["page_index"] = norm_pi
    with open(KG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    _kg_cache = data


def repair_kg():
    """Recompute page_index (case-folded) and entity_index from triples.

    Safe to call on a corrupt/duplicate-key KG file; rewrites a clean file.
    """
    data = _load()
    data["page_index"] = {}
    data["entity_index"] = {}
    for i, t in enumerate(data["triples"]):
        src = t.get("source") or t.get("source_wiki") or "unknown"
        key = _normalize_page_key(src)
        data["page_index"][key] = int(data["page_index"].get(key, 0)) + 1
        for entity in [t["triple"][0], t["triple"][2]]:
            el = _normalize_entity(entity)
            data["entity_index"].setdefault(el, [])
            if i not in data["entity_index"][el]:
                data["entity_index"][el].append(i)
    _save(data)
    return len(data["triples"])


def _delete_embeddings():
    for path in (KG_EMBEDDINGS_FILE, KG_EMBEDDINGS_INDEX_FILE):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _save_embeddings(embeddings: np.ndarray, index_map: dict):
    """Save triple embeddings to disk."""
    KG_EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.save(KG_EMBEDDINGS_FILE, embeddings)
    with open(KG_EMBEDDINGS_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index_map, f, ensure_ascii=False, indent=2)


def _load_embeddings():
    """Load triple embeddings from disk."""
    if KG_EMBEDDINGS_FILE.exists() and KG_EMBEDDINGS_INDEX_FILE.exists():
        embeddings = np.load(KG_EMBEDDINGS_FILE)
        with open(KG_EMBEDDINGS_INDEX_FILE, "r", encoding="utf-8") as f:
            index_map = json.load(f)
        return embeddings, index_map
    return None, None


def build_triple_embeddings(model=None):
    """Build and save embeddings for all triples in the KG."""
    data = _load()
    triples = data["triples"]
    
    if not triples:
        return None, None
    
    if model is None:
        from src.semantic_ranking import get_sbert_model
        model = get_sbert_model()
    
    texts = [t["linearized"] for t in triples]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    index_map = {str(i): t["linearized"] for i, t in enumerate(triples)}
    
    _save_embeddings(embeddings, index_map)
    return embeddings, index_map


def search_semantic(query: str, model=None, top_k=50, min_score=0.3):
    """Semantic search over triple embeddings using cosine similarity."""
    from src.config import get_threshold
    from sentence_transformers import util
    
    embeddings, index_map = _load_embeddings()
    if embeddings is None:
        return []
    data = _load()
    if len(embeddings) != len(data["triples"]):
        _delete_embeddings()
        return []
    
    if model is None:
        from src.semantic_ranking import get_sbert_model
        model = get_sbert_model()
    
    query_emb = model.encode([query], convert_to_numpy=True)
    scores = util.cos_sim(query_emb, embeddings)[0].cpu().numpy()
    
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score >= min_score:
            t = data["triples"][int(idx)]
            item = t.copy()
            item["semantic_score"] = round(score, 4)
            results.append(item)
    
    return results


def add_triples(page_title, triples):
    data = _load()
    seen = set()
    for t in data["triples"]:
        key = _triple_key(t["triple"])
        seen.add(key)
    added = []
    for t in triples:
        key = _triple_key(t["triple"])
        if key not in seen:
            seen.add(key)
            entry = {
                "triple": t["triple"],
                "linearized": t["linearized"],
                "source": page_title,
                "source_wiki": t.get("source_wiki", page_title),
                "source_sentence": t.get("source_sentence", ""),
                "method": t.get("method", "unknown"),
                "confidence": t.get("confidence", 0.5),
                "negated": t.get("negated", False),
                "entity_types": t.get("entity_types", {}),
            }
            data["triples"].append(entry)
            added.append(entry)
            idx = len(data["triples"]) - 1
            for entity in [t["triple"][0], t["triple"][2]]:
                entity_lower = _normalize_entity(entity)
                if entity_lower not in data["entity_index"]:
                    data["entity_index"][entity_lower] = []
                data["entity_index"][entity_lower].append(idx)
    pk = _normalize_page_key(page_title)
    data["page_index"][pk] = int(data["page_index"].get(pk, 0)) + len(added)
    if added:
        _delete_embeddings()
    _save(data)
    return added


def search_indices_for_term(term, max_results=50):
    """Return triple indices mentioning `term`, using the exact inverted index
    first and falling back to a word-boundary substring scan for entities that
    are embedded inside longer subject/object strings (e.g. 'Barack Obama' in
    'Barack Obama Presidential Center')."""
    data = _load()
    term_norm = _normalize_entity(term)
    if not term_norm:
        return []
    idxs = list(data["entity_index"].get(term_norm, []))
    if len(idxs) < max_results:
        pattern = re.compile(r"\b" + re.escape(term_norm) + r"\b")
        for i, t in enumerate(data["triples"]):
            if len(idxs) >= max_results:
                break
            if i in idxs:
                continue
            subj = _normalize_entity(t["triple"][0])
            obj = _normalize_entity(t["triple"][2])
            if pattern.search(subj) or pattern.search(obj):
                idxs.append(i)
    return idxs[:max_results]


def search_by_entity(entity_text, max_results=50):
    data = _load()
    results = []
    for idx in search_indices_for_term(entity_text, max_results):
        results.append(data["triples"][idx].copy())
    return results


def search(entities, max_per_entity=50):
    results = []
    seen = set()
    for entity in entities:
        if isinstance(entity, dict):
            aliases = entity.get("aliases", [])
            canonical = entity.get("canonical", entity.get("text", ""))
            search_terms = [canonical] + [a for a in aliases if a != canonical]
            text = entity.get("text", "")
        else:
            search_terms = [str(entity)]
            text = str(entity)
        
        for term in search_terms:
            triples = search_by_entity(term, max_results=max_per_entity)
            for t in triples:
                key = _triple_key(t["triple"])
                if key not in seen:
                    seen.add(key)
                    results.append(t)
    return results


def search_hybrid(entity_text, max_results=50, exact_weight=0.5, semantic_weight=0.5):
    """Hybrid search combining exact (inverted index) and semantic (embeddings) search.
    
    Args:
        entity_text: Entity to search for
        max_results: Maximum results to return
        exact_weight: Weight for exact match results (0-1)
        semantic_weight: Weight for semantic match results (0-1)
        
    Returns:
        List of triples with combined scores
    """
    from src.config import get_threshold
    
    exact_results = search_by_entity(entity_text, max_results=max_results)
    
    min_score = get_threshold("cross_encoder_min_score", 0.3)
    semantic_results = search_semantic(entity_text, top_k=max_results, min_score=min_score)
    
    predicate_weights = get_threshold("predicate_weights", {})
    def predicate_weight(triple):
        pred = triple[1].lower().replace(" ", "_")
        return float(predicate_weights.get(pred, predicate_weights.get(triple[1].lower(), 1.0)))

    combined = {}
    
    for r in exact_results:
        key = _triple_key(r["triple"])
        weight = predicate_weight(r["triple"])
        item = r.copy()
        item.update({
            "exact_score": 1.0,
            "semantic_score": 0.0,
            "predicate_weight": weight,
            "combined_score": exact_weight * weight,
        })
        combined[key] = item
    
    for r in semantic_results:
        key = _triple_key(r["triple"])
        sem_score = r.get("semantic_score", 0.0)
        weight = predicate_weight(r["triple"])
        if key in combined:
            combined[key]["semantic_score"] = sem_score
            combined[key]["combined_score"] = (
                exact_weight * combined[key]["exact_score"] + 
                semantic_weight * sem_score
            ) * weight
        else:
            item = r.copy()
            item.update({
                "exact_score": 0.0,
                "semantic_score": sem_score,
                "predicate_weight": weight,
                "combined_score": semantic_weight * sem_score * weight,
            })
            combined[key] = item
    
    sorted_results = sorted(combined.values(), key=lambda x: x["combined_score"], reverse=True)
    return sorted_results[:max_results]


def search_hybrid_multi(entities, max_per_entity=50):
    """Hybrid search for multiple entities."""
    all_results = []
    seen = set()
    
    for entity in entities:
        if isinstance(entity, dict):
            aliases = entity.get("aliases", [])
            canonical = entity.get("canonical", entity.get("text", ""))
            search_terms = [canonical] + [a for a in aliases if a != canonical]
        else:
            search_terms = [str(entity)]
        
        for term in search_terms:
            results = search_hybrid(term, max_results=max_per_entity)
            for r in results:
                key = _triple_key(r["triple"])
                if key not in seen:
                    seen.add(key)
                    all_results.append(r)
    
    all_results.sort(key=lambda x: x["combined_score"], reverse=True)
    return all_results


def get_stats():
    data = _load()
    return {
        "total_triples": len(data["triples"]),
        "pages": len(data["page_index"]),
        "page_index": data["page_index"],
    }


def clear():
    _delete_embeddings()
    _save({"triples": [], "page_index": {}, "entity_index": {}})


def rebuild_from_wiki():
    from src.wiki_kg_retrieval import _load_wiki_triples
    triples = _load_wiki_triples()
    data = {"triples": [], "page_index": {}, "entity_index": {}}
    seen = set()
    for t in triples:
        key = (t["triple"][0].lower(), t["triple"][1].lower(), t["triple"][2].lower())
        if key not in seen:
            seen.add(key)
            idx = len(data["triples"])
            data["triples"].append(t)
            t.setdefault("source_sentence", "")
            t.setdefault("method", "wiki")
            t.setdefault("confidence", 0.5)
            t.setdefault("negated", False)
            t.setdefault("entity_types", {})
            for entity in [t["triple"][0], t["triple"][2]]:
                entity_lower = entity.lower().replace("[[", "").replace("]]", "")
                if entity_lower not in data["entity_index"]:
                    data["entity_index"][entity_lower] = []
                data["entity_index"][entity_lower].append(idx)
    for md_file in Path("data/wiki").glob("*.md"):
        if md_file.name in ("index.md", "log.md"):
            continue
        title = md_file.stem.replace("-", " ").title()
        count = len([t for t in data["triples"] if t.get("source_wiki") == md_file.name])
        pk = _normalize_page_key(title)
        data["page_index"][pk] = int(data["page_index"].get(pk, 0)) + count
    _save(data)
    _delete_embeddings()
    
    if data["triples"]:
        print("      Building triple embeddings...")
        build_triple_embeddings()
    
    return len(data["triples"])
