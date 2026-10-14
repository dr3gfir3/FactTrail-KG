import json
import re
from pathlib import Path
from typing import List, Dict, Set, Optional

from src.config import get_retrieval

REDIRECTS_FILE = Path("data/wiki_redirects.json")
_redirects_cache: Optional[Dict[str, List[str]]] = None


def load_redirects() -> Dict[str, List[str]]:
    global _redirects_cache
    if _redirects_cache is not None:
        return _redirects_cache
    
    if REDIRECTS_FILE.exists():
        with open(REDIRECTS_FILE, "r", encoding="utf-8") as f:
            _redirects_cache = json.load(f)
    else:
        _redirects_cache = {}
    return _redirects_cache


def get_entity_aliases(entity_text: str) -> List[str]:
    """Get all known aliases for an entity from Wikipedia redirects."""
    redirects = load_redirects()
    entity_lower = entity_text.lower().strip()
    
    aliases = [entity_text]
    
    if entity_lower in redirects:
        for target in redirects[entity_lower]:
            if target != entity_text:
                aliases.append(target)
    
    variations = set()
    words = entity_text.split()
    if len(words) > 1:
        variations.add(f"{words[0]} {words[-1]}")
        variations.add(f"{' '.join([w[0] for w in words[:-1]])} {words[-1]}")
        if len(words[-1]) > 3:
            variations.add(words[-1])
    
    for v in variations:
        v_lower = v.lower()
        if v_lower in redirects:
            for target in redirects[v_lower]:
                if target not in aliases:
                    aliases.append(target)
        else:
            aliases.append(v)
    
    return list(dict.fromkeys(aliases))


def expand_entity_query(entity_text: str, max_aliases: int = 5) -> List[str]:
    """Expand an entity query with its known aliases for better KG/Wiki search."""
    aliases = get_entity_aliases(entity_text)
    return aliases[:max_aliases]


def resolve_entities(entities: List[Dict]) -> List[Dict]:
    """Resolve entity mentions to canonical Wikipedia titles using redirects."""
    redirects = load_redirects()
    resolved = []
    
    for ent in entities:
        text = ent.get("text", "") if isinstance(ent, dict) else str(ent)
        text_lower = text.lower()
        
        canonical = text
        if text_lower in redirects and redirects[text_lower]:
            canonical = redirects[text_lower][0]
        
        resolved.append({
            **ent,
            "text": text,
            "canonical": canonical,
            "aliases": get_entity_aliases(text)
        })
    
    return resolved


def build_redirects_from_wiki() -> Dict[str, List[str]]:
    """Rebuild redirect map from current wiki pages."""
    wiki_dir = Path("data/wiki")
    redirects = {}
    
    for md_file in wiki_dir.glob("*.md"):
        if md_file.name in ("index.md", "log.md"):
            continue
        content = md_file.read_text(encoding="utf-8")
        title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
        if title_match:
            title = title_match.group(1)
            slug = md_file.stem
            aliases = [title.lower(), slug.replace('-', ' ').lower()]
            words = title.split()
            if len(words) > 1:
                aliases.append(f"{words[0]} {words[-1]}".lower())
                aliases.append(' '.join([w[0] for w in words]).lower())
            
            for alias in aliases:
                if alias not in redirects:
                    redirects[alias] = []
                if title not in redirects[alias]:
                    redirects[alias].append(title)
    
    REDIRECTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REDIRECTS_FILE, "w", encoding="utf-8") as f:
        json.dump(redirects, f, indent=2, ensure_ascii=False)
    
    global _redirects_cache
    _redirects_cache = redirects
    return redirects


def apply_coreference(claim_text: str, entities: List[Dict]) -> str:
    """Apply coreference resolution to claim text for better entity extraction."""
    if not entities:
        return claim_text
    
    candidates = [e for e in entities if e.get("label") in ("PERSON", "ORG", "GPE", "LOC", "MISC")]
    if not candidates:
        return claim_text
    
    candidates.sort(key=lambda e: e.get("start", 0))
    
    import re
    
    def get_most_recent_entity(pos: int) -> str:
        best_entity = candidates[0]["text"]
        for e in candidates:
            if e.get("end", 0) <= pos:
                best_entity = e["text"]
        return best_entity
    
    def replace_pronoun(match):
        pos = match.start()
        if pos < candidates[0].get("start", 0):
            return match.group(0)
            
        entity_text = get_most_recent_entity(pos)
        
        pronoun = match.group(1).lower()
        if pronoun in ("his", "hers", "their", "its"):
            return entity_text + "'s"
        return entity_text
    
    pronoun_pattern = r'\b(he|she|it|they|him|her|them|his|hers|their|its)\b'
    result = re.sub(pronoun_pattern, replace_pronoun, claim_text, flags=re.IGNORECASE)
    
    return result
