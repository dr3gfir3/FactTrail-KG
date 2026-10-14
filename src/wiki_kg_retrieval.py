import re
from src.knowledge_graph import search, search_hybrid_multi
from src.config import get_pipeline
from pathlib import Path

WIKI_DIR = Path("data/wiki")


def _load_wiki_triples():
    triples = []
    if not WIKI_DIR.exists():
        return triples
    relation_pattern = re.compile(r'-\s*\(([^|]+)\|([^|]+)\|([^)]+)\)')

    for md_file in WIKI_DIR.glob("*.md"):
        if md_file.name in ["index.md", "log.md"]:
            continue
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
        relations_match = re.search(r'## Relations\n(.*?)(?=\n## |\Z)', content, re.DOTALL)
        if relations_match:
            relations_text = relations_match.group(1)
            for match in relation_pattern.finditer(relations_text):
                subject = match.group(1).strip().replace("[[", "").replace("]]", "")
                predicate = match.group(2).strip()
                obj = match.group(3).strip().replace("[[", "").replace("]]", "")
                triples.append({
                    "triple": [subject, predicate, obj],
                    "linearized": f"{subject} {predicate} {obj}",
                    "source_wiki": md_file.name,
                    "source_sentence": "",
                    "method": "wiki",
                    "confidence": 0.5,
                    "negated": False,
                    "entity_types": {},
                })
    return triples


def retrieve_kg_for_entity(entity_text, wiki_triples, max_triples=50):
    entity_lower = entity_text.lower()
    results = []
    for t in wiki_triples:
        subj = t["triple"][0].lower()
        obj = t["triple"][2].lower()
        if re.search(r'\b' + re.escape(entity_lower) + r'\b', subj.replace("[[", "").replace("]]", "")) or \
           re.search(r'\b' + re.escape(entity_lower) + r'\b', obj.replace("[[", "").replace("]]", "")):
            results.append(t)
            if len(results) >= max_triples:
                break
    return results


def retrieve_wiki_kg(entity_list, max_triples_per_entity=50):
    if get_pipeline("use_local_graph_rag", True):
        try:
            from src.local_graph_rag import retrieve
            results = retrieve(entity_list, max_triples_per_entity)
            if results:
                return results
        except Exception:
            pass

    use_hybrid = get_pipeline("use_hybrid_kg_search", True)
    if use_hybrid:
        return search_hybrid_multi(entity_list, max_per_entity=max_triples_per_entity)
    return search(entity_list, max_per_entity=max_triples_per_entity)


def retrieve_all_wiki_kg(entity_data, max_triples_per_entity=50):
    results = []
    for item in entity_data:
        evidence = retrieve_wiki_kg(item["entities"], max_triples_per_entity)
        results.append({
            "claim_id": item["claim_id"],
            "claim_text": item["claim_text"],
            "kg_evidence": evidence,
        })
    return results
