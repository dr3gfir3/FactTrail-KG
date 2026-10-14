import math
import re
from sentence_transformers import SentenceTransformer, CrossEncoder, util

_sbert_model = None
_sbert_initialized = False
_cross_encoder_model = None

from src.config import get_model, get_retrieval, get_threshold, get_pipeline


def get_sbert_model(model_name=None):
    global _sbert_model, _sbert_initialized
    if model_name is None:
        model_name = get_model("sbert", "all-MiniLM-L6-v2")
    if _sbert_model is None:
        _sbert_model = SentenceTransformer(model_name)
    if not _sbert_initialized:
        _sbert_model.encode("warmup", convert_to_tensor=True)
        _sbert_initialized = True
    return _sbert_model


def get_cross_encoder(model_name=None):
    global _cross_encoder_model
    if model_name is None:
        model_name = get_model("cross_encoder", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    if _cross_encoder_model is None:
        _cross_encoder_model = CrossEncoder(model_name, max_length=512)
    return _cross_encoder_model


def _to_unit_score(score):
    score = float(score)
    if score >= 0:
        z = math.exp(-score)
        return 1.0 / (1.0 + z)
    z = math.exp(score)
    return z / (1.0 + z)


def _item_text(item):
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("linearized") or item.get("text") or item.get("content") or item.get("snippet") or ""
    return str(item)


def rank_evidence(claim_text, evidence_items, top_k=5):
    """Return the top evidence items with semantic similarity scores."""
    if not evidence_items:
        return []
        
    model = get_sbert_model()
    
    texts = []
    for item in evidence_items:
        text = _item_text(item)
        text = text.replace("[[", "").replace("]]", "")
        texts.append(text)
            
    claim_emb = model.encode(claim_text, convert_to_tensor=True)
    evidence_embs = model.encode(texts, convert_to_tensor=True)
    
    scores = util.cos_sim(claim_emb, evidence_embs).squeeze().tolist()
    
    if isinstance(scores, float):
        scores = [scores]
        
    scored_items = []
    for idx, item in enumerate(evidence_items):
        score = scores[idx]
        if isinstance(item, dict):
            new_item = item.copy()
            new_item["semantic_score"] = round(float(score), 4)
            scored_items.append((new_item, score))
        else:
            scored_items.append(({"text": item, "semantic_score": round(float(score), 4)}, score))
            
    scored_items.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in scored_items[:top_k]]


def rerank_with_cross_encoder(claim_text, evidence_items, top_k=None):
    """Return cross-encoder-ranked items with scores between 0 and 1."""
    if not evidence_items:
        return []
    
    if top_k is None:
        top_k = get_retrieval("final_top_k", 5)
    
    min_score = get_threshold("cross_encoder_min_score", 0.0)
    
    cross_encoder = get_cross_encoder()
    
    pairs = []
    texts = []
    for item in evidence_items:
        text = (
            item.get("linearized") 
            or item.get("text") 
            or item.get("content") 
            or item.get("snippet") 
            or ""
        )
        if item.get("source_type") == "web":
            text = re.sub(r'<[^>]+>', '', text).strip()
        texts.append(text)
        pairs.append([claim_text, text])
    
    if not pairs:
        return []
    
    ce_scores = cross_encoder.predict(pairs, show_progress_bar=False)
    ce_scores = [_to_unit_score(s) for s in ce_scores]
    
    scored_items = []
    for idx, item in enumerate(evidence_items):
        ce_score = ce_scores[idx]
        if ce_score >= min_score:
            new_item = item.copy()
            new_item["cross_encoder_score"] = round(ce_score, 4)
            scored_items.append((new_item, ce_score))
    
    scored_items.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in scored_items[:top_k]]


def rank_and_combine_evidence(claim_text, text_evidence=None, kg_evidence=None, web_evidence=None, top_k=None):
    """Rank all evidence and return the selected items and NLI premise."""
    if top_k is None:
        top_k = get_retrieval("final_top_k", 5)
    
    biencoder_top_k = get_retrieval("biencoder_top_k", 20)
    use_ce = get_pipeline("use_cross_encoder", True)
    
    combined = []
    
    for item in (text_evidence or []):
        if isinstance(item, dict):
            item = item.copy()
            item["source_type"] = "text"
            combined.append(item)
            
    for item in (kg_evidence or []):
        if isinstance(item, dict):
            item = item.copy()
            item["source_type"] = "kg"
            combined.append(item)
            
    for item in (web_evidence or []):
        if isinstance(item, dict):
            item = item.copy()
            item["source_type"] = "web"
            combined.append(item)

    deduped = []
    seen = set()
    for item in combined:
        key = re.sub(r"\s+", " ", _item_text(item).replace("[[", "").replace("]]", "").strip().lower())
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)
    combined = deduped
            
    ranked = rank_evidence(claim_text, combined, top_k=biencoder_top_k)
    
    if use_ce and ranked:
        ranked = rerank_with_cross_encoder(claim_text, ranked, top_k=top_k)
    else:
        ranked = ranked[:top_k]
    
    premise_parts = []
    for item in ranked:
        text = (
            item.get("linearized") 
            or item.get("text") 
            or item.get("content") 
            or item.get("snippet") 
            or ""
        )
        if item.get("source_type") == "web":
            text = re.sub(r'<[^>]+>', '', text).strip()
            
        if text:
            text = text.strip()
            if not text.endswith(('.', '!', '?')):
                text += "."
                
            source_type = item.get("source_type")
            if source_type == "kg":
                text = f"Knowledge Graph: {text}"
            elif source_type == "web":
                title = item.get("title") or "Article"
                text = f"Wikipedia ({title}): {text}"
            elif source_type == "text":
                text = f"Textual Source: {text}"
                
            premise_parts.append(text)
                
    premise = " ".join(premise_parts)
    return ranked, premise

