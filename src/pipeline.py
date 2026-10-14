import sys, os, re, time, json, threading
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def _ignore_auto_conversion_error(args):
    if args.thread and args.thread.name == "Thread-auto_conversion":
        return
    if threading.__excepthook__:
        threading.__excepthook__(args)
threading.excepthook = _ignore_auto_conversion_error

from src.config import get_threshold, get_retrieval, get_pipeline, get_model
from src.eval import DEVICE
from src.entity_extractor import (
    extract_entities, extract_from_noun_chunks, extract_uppercase_words,
    load_nlp, rank_entities_for_retrieval,
)
from src.wiki_kg_retrieval import retrieve_wiki_kg
from src.mpc_server import MPCServer
from src.wiki_compiler import compile_wiki_page, page_exists, get_wiki_summary
from src.knowledge_graph import KG_FILE, rebuild_from_wiki

KG_FILE.parent.mkdir(parents=True, exist_ok=True)
if not KG_FILE.exists() and get_pipeline("auto_rebuild_kg"):
    wiki_mds = [f for f in Path("data/wiki").glob("*.md") if f.name not in ("index.md", "log.md")]
    if wiki_mds:
        n = rebuild_from_wiki()
        print(f"      Rebuilt KG from {len(wiki_mds)} wiki pages: {n} triples")

requests = None
_sbert_cache = [None]
_entity_prototypes_cache = [None]
_cross_encoder_cache = [None]


def _lazy_imports():
    global requests
    if requests is None:
        import requests as _req
        requests = _req


_wiki_warmed_up = False


def _search_wikipedia(query, server, top_k=5):
    global _wiki_warmed_up
    _lazy_imports()
    if not _wiki_warmed_up:
        try:
            requests.get("https://en.wikipedia.org/w/api.php",
                params={"action": "query", "list": "search", "srsearch": "warmup", "format": "json"},
                headers={"User-Agent": "GraphChecking/1.0"}, timeout=1)
        except Exception:
            pass
        _wiki_warmed_up = True
    return server.search(query, top_k=top_k)



def _load_sbert():
    if _sbert_cache[0] is None:
        from src.semantic_ranking import get_sbert_model
        _sbert_cache[0] = get_sbert_model()
    return _sbert_cache[0]


def _get_entity_prototypes(sbert_model):
    if _entity_prototypes_cache[0] is None:
        prototypes = {
            "PERSON": "this is the name of a person",
            "ORG": "this is the name of an organization or company",
            "GPE": "this is the name of a country city or geopolitical entity",
            "LOC": "this is the name of a geographic location",
            "EVENT": "this is the name of a historical event or occurrence",
            "FAC": "this is the name of a building or facility",
            "LAW": "this is the name of a legal document or law",
            "NORP": "this is the name of a nationality or religious group",
            "PRODUCT": "this is the name of a product or vehicle",
            "WORK_OF_ART": "this is the name of a book song movie or work of art",
            "LANGUAGE": "this is the name of a spoken language",
        }
        embeddings = sbert_model.encode(list(prototypes.values()), convert_to_tensor=True)
        _entity_prototypes_cache[0] = dict(zip(prototypes.keys(), embeddings))
    return _entity_prototypes_cache[0]





def _load_cross_encoder():
    if _cross_encoder_cache[0] is None:
        from sentence_transformers import CrossEncoder
        model_name = get_model("cross_encoder", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        _cross_encoder_cache[0] = CrossEncoder(model_name, max_length=512)
    return _cross_encoder_cache[0]


def _run_nli(claim_text, premise, tokenizer, model):
    import torch
    import torch.nn.functional as F
    from src.eval import DEVICE
    inputs = tokenizer(premise, claim_text, return_tensors="pt",
                       truncation=True, max_length=1024, padding=True).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze().tolist()
    nli_id = logits.argmax(dim=-1).item()
    nli_label = model.config.id2label[nli_id]
    id2label = model.config.id2label

    verdict_probs = {"SUPPORTED": 0.0, "REFUTED": 0.0, "NOT ENOUGH INFO": 0.0}
    for idx, prob in enumerate(probs):
        label = id2label[idx].lower()
        if "entail" in label:
            verdict_probs["SUPPORTED"] += prob
        elif "contradict" in label:
            verdict_probs["REFUTED"] += prob
        else:
            verdict_probs["NOT ENOUGH INFO"] += prob

    label_lower = nli_label.lower()
    if "entail" in label_lower:
        verdict = "SUPPORTED"
    elif "contradict" in label_lower:
        verdict = "REFUTED"
    else:
        verdict = "NOT ENOUGH INFO"
    return verdict, nli_label, verdict_probs


def _evidence_text(item):
    return (
        item.get("linearized")
        or item.get("text")
        or item.get("content")
        or item.get("snippet")
        or ""
    ).strip()


def _build_evidence_trace(ranked_evidence, nli_items=None):
    nli_by_text = {}
    for item in nli_items or []:
        text = (item.get("text") or "").strip()
        if text:
            nli_by_text[text] = item

    trace = []
    for idx, item in enumerate(ranked_evidence or [], start=1):
        text = _evidence_text(item)
        if not text:
            continue

        nli = nli_by_text.get(text, {})
        trace.append({
            "rank": idx,
            "source_type": item.get("source_type", "unknown"),
            "source": item.get("source") or item.get("source_wiki") or item.get("title") or "",
            "title": item.get("title") or "",
            "text": text,
            "score": item.get("score"),
            "cross_encoder_score": item.get("cross_encoder_score"),
            "nli_verdict": nli.get("verdict"),
            "nli_confidence": nli.get("confidence"),
            "effective_weight": nli.get("effective_weight"),
        })
    return trace


def _judge_claim(claim_text, kg_evidence, wiki_summaries, premise, ranked_evidence):
    from src.multihop_nli import predict_multihop_nli

    nli_out = predict_multihop_nli(
        claim_text, kg_evidence, wiki_summaries, premise=premise)
    result = {
        "verdict": nli_out["verdict"],
        "confidence": nli_out["confidence"],
        "probs": nli_out["probs"],
        "nli_reasoning": f"RoBERTa multi-hop NLI (conf={nli_out['confidence']})",
    }
    if get_pipeline("use_llm_judge", False):
        from src.llm_judge import judge as llm_judge

        try:
            llm_out = llm_judge(claim_text, ranked_evidence)
        except Exception as exc:
            llm_out = {
                "verdict": None,
                "reasoning": f"LLM error: {exc}",
                "raw_response": "",
            }
        result["llm_judge"] = llm_out
        result["judge_disagreement"] = (
            llm_out.get("verdict") not in (None, result["verdict"]))
    return result




def _fetch_wiki_for_entities(entities, seen_titles, nlp, sbert, extraction_method, claim_text):
    """Return titles of newly compiled Wikipedia pages for missing entities."""
    from src.knowledge_graph import search_by_entity
    _lazy_imports()
    fetched = []
    try:
        with MPCServer() as server:
            request_delay = get_pipeline("wiki_request_delay", 0.3)
            snippet_candidates = get_retrieval("wiki_snippet_candidates", 3)
            full_fetch = get_retrieval("wiki_full_fetch", 3)
            for ent in entities:
                ent_name = ent.get("canonical", ent.get("text", ""))
                aliases = ent.get("aliases", [ent_name])
                if any(search_by_entity(a) for a in aliases):
                    continue
                time.sleep(request_delay)
                results = []
                seen_queries = set()
                for alias in aliases:
                    for query in (f"{claim_text} {alias}", alias):
                        if query.lower() in seen_queries:
                            continue
                        seen_queries.add(query.lower())
                        results.extend(_search_wikipedia(query, server, top_k=snippet_candidates))
                for r in results:
                    title = r.get("title", "")
                    page_id = r.get("id")
                    if title in seen_titles or not page_id:
                        continue
                    seen_titles.add(title)
                    if not page_exists(title):
                        html = server.fetch_clean(int(page_id))
                        if not html or len(html) < 50:
                            html = server.fetch(int(page_id))
                        if html and len(html) > 100:
                            wp = compile_wiki_page(title, html, nlp=nlp, extraction_method=extraction_method)
                            if wp:
                                fetched.append(title)

            if not fetched:
                results = _search_wikipedia(claim_text, server, top_k=snippet_candidates * 5)
                if results:
                    from sentence_transformers import util
                    claim_emb = sbert.encode(claim_text, convert_to_tensor=True)
                    texts = [f"{r.get('title', '')}: {r.get('snippet', '')}" for r in results]
                    text_embs = sbert.encode(texts, convert_to_tensor=True)
                    scores = util.cos_sim(claim_emb, text_embs).squeeze().tolist()
                    if isinstance(scores, float):
                        scores = [scores]
                    scored = []
                    for score, r in zip(scores, results):
                        scored.append((score, r))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    results = [item[1] for item in scored[:full_fetch] if item[0] >= 0.35]
                for r in results:
                    title = r.get("title", "")
                    page_id = r.get("id")
                    if title in seen_titles or not page_id:
                        continue
                    seen_titles.add(title)
                    if not page_exists(title):
                        html = server.fetch_clean(int(page_id))
                        if not html or len(html) < 50:
                            html = server.fetch(int(page_id))
                        if html and len(html) > 100:
                            wp = compile_wiki_page(title, html, nlp=nlp, extraction_method=extraction_method)
                            if wp:
                                fetched.append(title)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Wikipedia fetch failed: %s", e)
    return fetched


def _derive_iteration_entities(ranked_evidence, premise, nlp, sbert, prototypes, exclude_text):
    """Find entities in retrieved evidence for the next retrieval pass."""
    texts = [premise]
    for it in ranked_evidence:
        t = _evidence_text(it)
        if t:
            texts.append(t)
    blob = " ".join(texts)
    if not blob.strip():
        return []

    ents = extract_entities(blob, nlp)
    seen = set(e["text"].lower() for e in ents)
    doc = nlp(blob)
    extract_from_noun_chunks(doc, ents, seen, sbert, prototypes)
    ranked = rank_entities_for_retrieval(ents, blob)

    exclude = set(w.lower() for w in exclude_text.lower().split())
    out = []
    for e in ranked:
        if e.get("role") != "primary":
            continue
        if e["text"].lower() in exclude:
            continue
        out.append({"text": e["text"], "label": e.get("label", "MISC")})
    return out


def verify(claim_text, update_wiki=True, extraction_method=None):
    allowed_extraction_methods = {"regex", "spacy", "hybrid", "llm"}
    if extraction_method not in allowed_extraction_methods:
        extraction_method = None

    pipeline_start = time.time()
    result = {"claim": claim_text, "source": "", "wiki_pages_added": 0,
              "wiki_triples": 0, "premise": "", "verdict": "NOT ENOUGH INFO",
              "confidence": 0.0, "probs": {}, "stages": {},
              "entities": [], "fetched_pages": [],
              "relation_extraction_method": extraction_method or "config"}

    t0 = time.time()
    nlp = load_nlp()
    result["stages"]["nlp_load"] = round(time.time() - t0, 2)

    t0 = time.time()
    from src.entity_linker import apply_coreference, resolve_entities
    
    entities = extract_entities(claim_text, nlp)
    seen = set(e["text"].lower() for e in entities)

    doc = nlp(claim_text)
    sbert = _load_sbert()
    prototypes = _get_entity_prototypes(sbert)
    extract_from_noun_chunks(doc, entities, seen, sbert, prototypes)
    entities.extend(extract_uppercase_words(doc, seen))
    
    claim_text_resolved = apply_coreference(claim_text, entities)
    
    if claim_text_resolved != claim_text:
        entities = extract_entities(claim_text_resolved, nlp)
        seen = set(e["text"].lower() for e in entities)
        
        doc = nlp(claim_text_resolved)
        extract_from_noun_chunks(doc, entities, seen, sbert, prototypes)
        entities.extend(extract_uppercase_words(doc, seen))
    
    ranked_entities = rank_entities_for_retrieval(entities, claim_text_resolved)
    primary_entities = [e for e in ranked_entities if e.get("role") == "primary"]
    scoring_entities = [e for e in ranked_entities if e.get("role") in ("primary", "context")]
    resolved_entities = resolve_entities(primary_entities)
    result["entities"] = [e["text"] for e in ranked_entities]
    result["entity_diagnostics"] = ranked_entities
    result["stages"]["entity_extraction"] = round(time.time() - t0, 2)

    t0 = time.time()
    kg_max_per_entity = get_retrieval("kg_max_per_entity", 50)
    kg_evidence = retrieve_wiki_kg(resolved_entities, max_triples_per_entity=kg_max_per_entity)
    result["stages"]["wiki_kg_query"] = round(time.time() - t0, 2)
    result["wiki_triples"] = len(kg_evidence)

    seen_titles = set()
    if update_wiki:
        fetched = _fetch_wiki_for_entities(
            resolved_entities, seen_titles, nlp, sbert, extraction_method, claim_text_resolved)
        result["fetched_pages"] = fetched
        result["wiki_pages_added"] = len(fetched)
        if fetched:
            t0 = time.time()
            from src.knowledge_graph import build_triple_embeddings
            build_triple_embeddings(_load_sbert())
            result["stages"]["wiki_reload"] = round(time.time() - t0, 2)

    kg_evidence = retrieve_wiki_kg(resolved_entities, max_triples_per_entity=kg_max_per_entity)
    result["wiki_triples"] = len(kg_evidence)
    result["source"] = "wiki" if kg_evidence else "no_evidence"

    t0 = time.time()
    wiki_summaries = []
    for ent in scoring_entities:
        summary = get_wiki_summary(ent["text"] if isinstance(ent, dict) else str(ent))
        if summary:
            ent_name = ent["text"] if isinstance(ent, dict) else str(ent)
            wiki_summaries.append({
                "text": f"Wikipedia summary for {ent_name}: {summary}",
                "score": 0.7, "source_type": "text"
            })
    result["stages"]["wiki_summary"] = round(time.time() - t0, 2)

    t1 = time.time()
    _load_sbert()
    result["stages"]["sbert_load"] = round(time.time() - t1, 2)

    from src.semantic_ranking import rank_and_combine_evidence
    iterative = get_pipeline("iterative_retrieval", True)
    max_iter = int(get_pipeline("max_iterations", 2)) if iterative else 1
    iter_conf = get_threshold("iterative_confidence", 0.5)

    combined_entities = list(resolved_entities)
    ranked_evidence = []

    for iteration in range(max(1, max_iter)):
        t0 = time.time()
        ranked_evidence, premise = rank_and_combine_evidence(
            claim_text, text_evidence=wiki_summaries, kg_evidence=kg_evidence,
            web_evidence=[], top_k=20)
        result["premise"] = premise
        result["evidence_trace"] = _build_evidence_trace(ranked_evidence)
        result["stages"]["evidence_ranking"] = round(time.time() - t0, 2)

        t0 = time.time()
        result.update(_judge_claim(
            claim_text, kg_evidence, wiki_summaries, premise, ranked_evidence))
        result["stages"]["nli_inference"] = round(time.time() - t0, 2)

        if iteration + 1 >= max_iter:
            break
        confident = result["verdict"] != "NOT ENOUGH INFO" and result.get("confidence", 0) >= iter_conf
        if confident:
            break

        extra = _derive_iteration_entities(
            ranked_evidence, premise, nlp, sbert, prototypes, claim_text_resolved)
        if not extra:
            break
        extra_resolved = resolve_entities(extra)
        combined_entities = combined_entities + extra_resolved
        new_fetched = _fetch_wiki_for_entities(
            extra_resolved, seen_titles, nlp, sbert, extraction_method, claim_text_resolved)
        if new_fetched:
            result["fetched_pages"].extend(new_fetched)
            result["wiki_pages_added"] += len(new_fetched)
            t0 = time.time()
            from src.knowledge_graph import build_triple_embeddings
            build_triple_embeddings(_load_sbert())
            result["stages"]["wiki_reload"] = round(time.time() - t0, 2)
        kg_evidence = retrieve_wiki_kg(combined_entities, max_triples_per_entity=kg_max_per_entity)
        result["wiki_triples"] = len(kg_evidence)
        result["source"] = "wiki" if kg_evidence else "no_evidence"

    result["stages"]["total"] = round(time.time() - pipeline_start, 2)
    return result
