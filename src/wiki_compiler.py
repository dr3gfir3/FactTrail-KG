import re
import datetime
from pathlib import Path
from urllib.parse import quote

from src.entity_extractor import load_nlp

WIKI_DIR = Path("data/wiki")
RAW_DIR = Path("data/raw")
INDEX_FILE = WIKI_DIR / "index.md"
LOG_FILE = WIKI_DIR / "log.md"
SPIDERY_DEPTH = 1


def _init_wiki():
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_FILE.exists():
        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            f.write("# Knowledge Base Index\n\n")
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("# Wiki Log\n\n")


def _strip_html(html_text):
    text = re.sub(r'<table[^>]*?class="[^"]*infobox[^"]*"[^>]*>.*?</table>', ' ', html_text, flags=re.DOTALL)
    text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL)
    text = re.sub(r'<script[^>]*>.*?</script>', ' ', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'hatnote|navbox|shortdescription|noexcerpt|noprint|searchaux|skin-invert|skin-theme-clientpref|TemplateStyles|mw-empty-elt|navigation-not-searchable|mbox-text|reflist|refbegin|refend', '', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


def _clean_wiki_markup(text):
    """Remove Wikipedia template, link, pronunciation, infobox, and reference artifacts."""
    text = re.sub(r'\{\{[^}]*\}\}', ' ', text)
    text = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'\2', text)
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'\|\s*pronunciation:[^|]*', ' ', text)
    text = re.sub(r'\s*\([^)]*pronunciation[^)]*\)\s*', ' ', text)
    text = re.sub(r'\s*\[[^\]]*pronunciation[^\]]*\]\s*', ' ', text)
    text = re.sub(r'\s*IPA[:\s][^.]*\.', ' ', text)
    text = re.sub(r'\|[^a-zA-Z]*', ' ', text)
    text = re.sub(r'\[\d+\]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _is_disambiguation(title, text):
    if "disambiguation" in title.lower():
        return True
    patterns = [
        r'^may refer to',
        r'^can refer to',
        r'^standing for',
        r'^commonly refers to',
    ]
    return any(re.search(p, text.lower()[:300]) for p in patterns)


def _is_valid_predicate(text):
    if len(text) < 2 or len(text) > 45:
        return False
    if re.match(r'^[\s.,;:!?\'"()&\-]+$', text):
        return False
    if re.match(r'^(and|or|the|a|an|of|in|to|for|with|on|at|by|as|from|also)$', text, re.IGNORECASE):
        return False
    if re.match(r'^\s*,?\s*(who|which|that|where|when)\b', text, re.IGNORECASE):
        return False
    stripped = text.strip()
    if stripped.startswith(',') or stripped.endswith(','):
        return False
    if not re.search(r'[a-zA-Z]{3,}', text):
        return False
    return True


def _extract_triples_from_text(text, nlp, title=""):
    doc = nlp(text[:15000])
    triples = []
    seen = set()
    
    from src.entity_extractor import extract_uppercase_words
    
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if len(sent_text) < 20:
            continue
            
        entities = [e for e in doc.ents if e.start >= sent.start and e.end <= sent.end]
        
        token2ent = {}
        ent_types = {}
        seen_lower = set()
        for e in entities:
            if e.label_ == "DATE" and len(e.text) <= 6:
                continue
            seen_lower.add(e.text.lower())
            ent_types[e.text] = e.label_
            for i in range(e.start, e.end):
                token2ent[i] = e.text
                
        for e in extract_uppercase_words(sent, seen_lower):
            span = doc.char_span(e["start"], e["end"])
            if span:
                ent_types[e["text"]] = e["label"]
                for t in span:
                    token2ent[t.i] = e["text"]
                
        for token in sent:
            if token.pos_ == "VERB" or token.dep_ == "ROOT" or token.pos_ == "AUX":
                negated = any(c.dep_ == "neg" for c in token.children)
                
                subjects = []
                for c in token.children:
                    if c.dep_ in ("nsubj", "nsubjpass", "csubj"):
                        if c.i in token2ent:
                            subjects.append(token2ent[c.i])
                        elif c.text.lower() in ("who", "which", "that") and token.dep_ == "relcl":
                            head = token.head
                            if head.i in token2ent:
                                subjects.append(token2ent[head.i])
                        elif c.pos_ == "PRON" and title:
                            subjects.append(title)
                                
                objects = []
                for c in token.children:
                    if c.dep_ in ("dobj", "pobj", "attr", "dative", "oprd"):
                        if c.i in token2ent:
                            objects.append(token2ent[c.i])
                        else:
                            for d in c.subtree:
                                if d.i in token2ent and token2ent[d.i] not in objects:
                                    objects.append(token2ent[d.i])
                    elif c.dep_ == "prep":
                        for d in c.subtree:
                            if d.i in token2ent and token2ent[d.i] not in objects:
                                objects.append(token2ent[d.i])
                                
                if subjects and objects:
                    pred_parts = []
                    pred_norm_parts = []
                    for c in token.lefts:
                        if c.dep_ in ("aux", "auxpass", "neg"):
                            pred_parts.append(c.text)
                            if c.dep_ == "neg":
                                pred_norm_parts.append(c.lemma_.lower())
                                
                    pred_parts.append(token.text)
                    pred_norm_parts.append(token.lemma_.lower())
                    
                    for c in token.rights:
                        if c.dep_ == "prep":
                            pred_parts.append(c.text)
                            pred_norm_parts.append(c.lemma_.lower())
                            
                    predicate = " ".join(pred_parts)
                    predicate_norm = " ".join(pred_norm_parts)
                    
                    for subj in subjects:
                        for obj in objects:
                            if subj.lower() == obj.lower():
                                continue
                            if _is_garbage(subj) or _is_garbage(obj):
                                continue
                            if not _is_valid_predicate(predicate):
                                continue
                                
                            key = (subj.lower(), predicate_norm, obj.lower())
                            if key not in seen:
                                seen.add(key)
                                triples.append({
                                    "triple": [subj, predicate_norm, obj],
                                    "linearized": f"{subj} {predicate} {obj}.",
                                    "source_sentence": sent_text,
                                    "method": "regex",
                                    "confidence": 0.7,
                                    "negated": negated,
                                    "entity_types": {
                                        subj: ent_types.get(subj, "MISC"),
                                        obj: ent_types.get(obj, "MISC"),
                                    },
                                })
                                
        if len(triples) >= 50:
            break
            
    return triples[:50]


def _extract_spacy_triples_from_text(text, nlp, title=""):
    doc = nlp(text[:15000])
    triples = []
    seen = set()

    from src.entity_extractor import extract_uppercase_words

    def sent_entities(sent):
        token2ent = {}
        ent_types = {}
        seen_lower = set()
        for ent in sent.ents:
            if ent.label_ == "DATE" and len(ent.text) <= 6:
                continue
            seen_lower.add(ent.text.lower())
            ent_types[ent.text] = ent.label_
            for token in ent:
                token2ent[token.i] = ent.text

        for ent in extract_uppercase_words(sent, seen_lower):
            span = doc.char_span(ent["start"], ent["end"])
            if span:
                ent_types[ent["text"]] = ent["label"]
                for token in span:
                    token2ent[token.i] = ent["text"]
        return token2ent, ent_types

    def entity_for(token, token2ent):
        if token.i in token2ent:
            return token2ent[token.i]
        for child in token.subtree:
            if child.i in token2ent:
                return token2ent[child.i]
        return None

    def relation_parts(verb):
        left = [c for c in verb.lefts if c.dep_ in ("aux", "auxpass", "neg")]
        right = [c for c in verb.rights if c.dep_ in ("prt", "prep", "agent")]
        display = " ".join([*(c.text for c in left), verb.text, *(c.text for c in right)])
        norm = " ".join([*(c.lemma_.lower() for c in left), verb.lemma_.lower(), *(c.lemma_.lower() for c in right)])
        return display, norm

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if len(sent_text) < 20:
            continue

        token2ent, ent_types = sent_entities(sent)
        for verb in sent:
            if verb.pos_ not in ("VERB", "AUX") and verb.dep_ != "ROOT":
                continue

            subjects = []
            objects = []
            for child in verb.children:
                if child.dep_ in ("nsubj", "nsubjpass", "csubj"):
                    ent = entity_for(child, token2ent)
                    if ent:
                        subjects.append(ent)
                    elif child.text.lower() in ("who", "which", "that") and verb.dep_ == "relcl":
                        ent = entity_for(verb.head, token2ent)
                        if ent:
                            subjects.append(ent)
                    elif child.pos_ == "PRON" and title:
                        subjects.append(title)
                elif child.dep_ in ("dobj", "pobj", "attr", "dative", "oprd", "prep", "agent"):
                    for token in child.subtree:
                        ent = token2ent.get(token.i)
                        if ent and ent not in objects:
                            objects.append(ent)

            predicate, predicate_norm = relation_parts(verb)
            if not subjects or not objects or not _is_valid_predicate(predicate):
                continue

            negated = any(c.dep_ == "neg" for c in verb.children)
            for subj in subjects:
                for obj in objects:
                    if subj.lower() == obj.lower() or _is_garbage(subj) or _is_garbage(obj):
                        continue

                    key = (subj.lower(), predicate_norm, obj.lower())
                    if key in seen:
                        continue

                    seen.add(key)
                    triples.append({
                        "triple": [subj, predicate_norm, obj],
                        "linearized": f"{subj} {predicate} {obj}.",
                        "source_sentence": sent_text,
                        "method": "dependency",
                        "confidence": 0.75,
                        "negated": negated,
                        "entity_types": {
                            subj: ent_types.get(subj, "MISC"),
                            obj: ent_types.get(obj, "MISC"),
                        },
                    })

        if len(triples) >= 50:
            break

    return triples[:50]


_GARBAGE_PATTERN = re.compile(
    r'|'.join([
        r'^\s*mw-', r'^\s*not\(', r'^\s*\.mw-', r'^\s*#', r'^\s*@media',
        r'^\s*\{', r'^\s*\}', r'^\s*\[', r'^\s*table;', r'^\s*max-width',
        r'^\s*screen', r'^\s*html\.', r'^\s*inherit', r'^\s*%7B',
        r'^\s*http', r'^\s*data-mw', r'^\s*class=', r'^\s*style=',
        r'^\d+\.\d+', r'^\d{3,}', r'^[a-z]{2,3}$',
    ])
)


def _is_garbage(text):
    return bool(_GARBAGE_PATTERN.match(text))


def _extract_entities_from_text(text, nlp):
    doc = nlp(text[:10000])
    seen = set()
    entities = []
    for ent in doc.ents:
        key = ent.text.lower()
        if key not in seen and len(ent.text) > 2 and ent.label_ != "DATE":
            if not _is_garbage(ent.text):
                seen.add(key)
                entities.append(ent.text)
    return entities


def page_exists(title):
    slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
    return (WIKI_DIR / f"{slug}.md").exists()


def compile_wiki_page(title, html_content, nlp=None, extraction_method=None):
    _init_wiki()
    if nlp is None:
        nlp = load_nlp()

    from src.config import get_pipeline
    extraction_method = extraction_method or get_pipeline("wiki_extraction_method", "hybrid")

    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')

    is_clean = "\\n" not in html_content[:100] and "<" not in html_content[:100]
    if is_clean:
        plain_text = html_content
    else:
        plain_text = _strip_html(html_content)

    if _is_disambiguation(title, plain_text):
        print(f"      SKIP disambiguation page: {title}")
        return None

    import html as html_mod
    plain_text = html_mod.unescape(plain_text)
    plain_text = _clean_wiki_markup(plain_text)
    plain_text = re.sub(r'\s+', ' ', plain_text).strip()

    if extraction_method == "llm":
        from src.wiki_llm_extractor import extract_with_llm
        try:
            result = extract_with_llm(plain_text, title)
        except Exception as exc:
            print(f"      LLM extraction failed ({exc}); falling back to spacy extraction")
            summary = plain_text[:1000]
            entities = _extract_entities_from_text(plain_text, nlp)
            triples = _extract_spacy_triples_from_text(plain_text, nlp, title)
            result = {
                "wiki_content": "",
                "triples": triples,
                "entities": entities,
                "summary": summary,
            }
    elif extraction_method == "hybrid":
        from src.wiki_llm_extractor import extract_hybrid
        result = extract_hybrid(plain_text, title, nlp)
    elif extraction_method == "spacy":
        summary = plain_text[:1000]
        entities = _extract_entities_from_text(plain_text, nlp)
        triples = _extract_spacy_triples_from_text(plain_text, nlp, title)
        result = {
            "wiki_content": "",
            "triples": triples,
            "entities": entities,
            "summary": summary,
        }
    else:
        summary = plain_text[:1000]
        entities = _extract_entities_from_text(plain_text, nlp)
        triples = _extract_triples_from_text(plain_text, nlp, title)
        result = {
            "wiki_content": "",
            "triples": triples,
            "entities": entities,
            "summary": summary,
        }

    triples = result["triples"]
    entities = result["entities"]
    summary = result.get("summary", plain_text[:1000])

    raw_filename = f"{date_str}-{slug}.md"
    raw_path = RAW_DIR / raw_filename
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(f"---\ntitle: {title}\ndate: {date_str}\nsource_type: wikipedia\n---\n\n")
        f.write(plain_text)

    wiki_filename = f"{slug}.md"
    wiki_path = WIKI_DIR / wiki_filename

    wiki_content = result.get("wiki_content", "")
    if not wiki_content:
        wiki_lines = [f"# {title}", "", "## Summary", summary, "", "## Entities"]
        for ent in entities[:30]:
            wiki_lines.append(f"* [[{ent}]]")
        wiki_lines.append("")
        wiki_lines.append("## Relations")
        for t in triples[:30]:
            subj = t["triple"][0].replace("[[", "").replace("]]", "")
            pred = t["triple"][1]
            obj = t["triple"][2].replace("[[", "").replace("]]", "")
            wiki_lines.append(f"- ([[{subj}]] | {pred} | [[{obj}]]))")
        wiki_content = "\n".join(wiki_lines)

    with open(wiki_path, "w", encoding="utf-8") as f:
        f.write(f"---\nraw_source: ../raw/{raw_filename}\nupdated: {date_str}\n---\n\n")
        f.write(wiki_content)

    with open(INDEX_FILE, "a", encoding="utf-8") as f:
        f.write(f"- [{title}]({wiki_filename}) - Compiled on {date_str}\n")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"## [{date_str}] ingest | {title}\n")

    from src.knowledge_graph import add_triples
    add_triples(title, triples[:30])

    print(f"      Wiki page compiled: {wiki_path.name} ({len(triples)} triples) [{extraction_method}]")
    return wiki_path


def get_wiki_summary(entity_text):
    slug = re.sub(r'[^a-z0-9]+', '-', entity_text.lower()).strip('-')
    path = WIKI_DIR / f"{slug}.md"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        found = None
        for md_file in WIKI_DIR.glob("*.md"):
            if md_file.name in ("index.md", "log.md"):
                continue
            raw = md_file.read_text(encoding="utf-8")
            title_match = re.search(r'^# (.+)$', raw, re.MULTILINE)
            if title_match:
                title_clean = title_match.group(1).lower()
                pattern = r'\b' + re.escape(entity_text.lower()) + r'\b'
                if re.search(pattern, title_clean):
                    found = raw
                    break
        if found is None:
            return ""
        content = found
    summary_match = re.search(r'## Summary\n(.*?)(?=\n## |\Z)', content, re.DOTALL)
    if summary_match:
        text = summary_match.group(1).strip()
        text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
        return text
    return ""


def get_page_entities(title):
    slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
    path = WIKI_DIR / f"{slug}.md"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    entities = re.findall(r'\* \[\[([^\]]+)\]\]', content)
    return entities


def spider_related_pages(title, html_content, server, nlp, max_pages=2, original_entities=None):
    links = re.finditer(r'href="(/wiki/[^"#]+)"', html_content)
    fetched = []
    seen_titles = {title.lower()}
    for match in links:
        link = match.group(1)
        link_title = link.replace("/wiki/", "").replace("_", " ")
        link_title = re.sub(r'\([^)]*\)', '', link_title).strip()
        if link_title.lower() in seen_titles:
            continue
        if ":" in link_title or "disambiguation" in link_title.lower():
            continue
        if original_entities:
            ent_match = any(e.lower() in link_title.lower() or link_title.lower() in e.lower() for e in original_entities)
            if not ent_match:
                continue
        seen_titles.add(link_title.lower())
        if page_exists(link_title):
            continue
        try:
            search_results = server.search(link_title, top_k=1)
            if search_results:
                page_id = search_results[0].get("id")
                page_title = search_results[0].get("title", "")
                if page_id and not page_exists(page_title):
                    content = server.fetch(int(page_id))
                    if content and len(content) > 200:
                        result = compile_wiki_page(page_title, content, nlp=nlp)
                        if result:
                            fetched.append(page_title)
                            print(f"      Spidered: {page_title}")
                            if len(fetched) >= max_pages:
                                break
        except Exception:
            continue
    return fetched
