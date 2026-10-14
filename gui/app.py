import sys, os, json, re, threading, time, copy
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import verify
from src.config import get_pipeline
from src.knowledge_graph import _load as load_kg, KG_FILE, rebuild_from_wiki, clear as clear_kg, add_triples
from src.entity_extractor import (
    extract_entities, extract_from_noun_chunks, extract_uppercase_words,
    load_nlp, rank_entities_for_retrieval,
)
from benchmark_pipeline import bench

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXT = {'.txt', '.pdf', '.docx'}

# Thread-safe global state for running benchmarks in background
benchmark_state = {
    "status": "idle",  # "idle", "running", "completed", "error"
    "current": 0,
    "total": 0,
    "current_claim": "",
    "gold_label": "",
    "predicted_verdict": "",
    "is_correct": True,
    "probs": {},
    "logs": [],
    "error_message": "",
    "results": []
}
benchmark_lock = threading.Lock()
verify_cache = {}
verify_cache_lock = threading.Lock()
VERIFY_CACHE_TTL_SECONDS = 30 * 60
VERIFY_CACHE_MAX_ITEMS = 64
VERIFY_CACHE_FILE = Path(__file__).resolve().parent.parent / "data" / "cache" / "verify_cache.json"
RELATION_EXTRACTION_METHODS = {
    "regex": "Naive regex",
    "spacy": "NER / dependency",
    "hybrid": "Hybrid regex + LLM",
    "llm": "Modern LLM only",
}
NER_PROTOTYPES = {
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


def clean_kg_label(value, max_words=6, max_chars=42):
    text = Path(str(value or "Unknown source")).stem
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        text = "Unknown source"

    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])

    return text[:max_chars].rstrip(" ,;:-") or "Unknown source"


def community_for_triple(triple_entry):
    source = triple_entry.get("source_wiki") or triple_entry.get("source") or "Unknown source"
    return {
        "id": str(source),
        "label": clean_kg_label(source),
    }


def normalized_claim_key(claim_text, relation_mode="config"):
    claim_key = re.sub(r"\s+", " ", claim_text or "").strip().lower()
    judge = get_pipeline("primary_judge", "nli")
    return f"{relation_mode}:{judge}:{claim_key}"


def load_verify_cache_from_disk():
    if verify_cache or not VERIFY_CACHE_FILE.exists():
        return
    try:
        with open(VERIFY_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            verify_cache.update(data)
    except (OSError, json.JSONDecodeError):
        pass


def save_verify_cache_to_disk():
    try:
        VERIFY_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(VERIFY_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(verify_cache, f, ensure_ascii=False)
    except OSError:
        pass


def clear_verify_cache():
    with verify_cache_lock:
        verify_cache.clear()
        try:
            VERIFY_CACHE_FILE.unlink()
        except FileNotFoundError:
            pass


def get_cached_verify_result(claim_text, relation_mode="config"):
    key = normalized_claim_key(claim_text, relation_mode)
    now = time.time()
    with verify_cache_lock:
        load_verify_cache_from_disk()
        cached = verify_cache.get(key)
        if not cached:
            return key, None
        if now - cached["created_at"] > VERIFY_CACHE_TTL_SECONDS:
            verify_cache.pop(key, None)
            save_verify_cache_to_disk()
            return key, None
        result = copy.deepcopy(cached["result"])
    result["cache_hit"] = True
    result["cache_key"] = key
    return key, result


def set_cached_verify_result(cache_key, result):
    with verify_cache_lock:
        if len(verify_cache) >= VERIFY_CACHE_MAX_ITEMS:
            oldest_key = min(verify_cache, key=lambda item: verify_cache[item]["created_at"])
            verify_cache.pop(oldest_key, None)
        verify_cache[cache_key] = {
            "created_at": time.time(),
            "result": copy.deepcopy(result),
        }
        save_verify_cache_to_disk()


def safe_child_path(base_dir, filename):
    base = Path(base_dir).resolve()
    path = (base / filename).resolve()
    try:
        path.relative_to(base)
    except ValueError:
        return None
    return path

def make_progress_callback():
    def progress_callback(current, total, claim, gold, predicted, is_correct, probs):
        with benchmark_lock:
            benchmark_state["current"] = current
            benchmark_state["total"] = total
            benchmark_state["current_claim"] = claim
            benchmark_state["gold_label"] = gold
            benchmark_state["predicted_verdict"] = predicted
            benchmark_state["is_correct"] = is_correct
            benchmark_state["probs"] = probs
            msg = f"[{current}/{total}] Gold: {gold} | Pred: {predicted} | {'OK' if is_correct else 'WRONG'}"
            benchmark_state["logs"].append(msg)
            benchmark_state["results"].append({
                "index": current,
                "claim": claim,
                "gold": gold,
                "predicted": predicted,
                "correct": is_correct,
                "probs": probs
            })
    return progress_callback

def run_benchmark_thread(num_samples, split, seed):
    global benchmark_state
    with benchmark_lock:
        benchmark_state["status"] = "running"
        benchmark_state["current"] = 0
        benchmark_state["total"] = num_samples
        benchmark_state["logs"] = ["Starting benchmark run..."]
        benchmark_state["results"] = []
        benchmark_state["error_message"] = ""

    try:
        cb = make_progress_callback()
        metrics, results = bench(num_samples=num_samples, split=split, seed=seed, progress_callback=cb)
        with benchmark_lock:
            benchmark_state["status"] = "completed"
            benchmark_state["logs"].append(f"Benchmark finished. Accuracy: {metrics.get('accuracy_pct', 0)}%")
    except Exception as e:
        with benchmark_lock:
            benchmark_state["status"] = "error"
            benchmark_state["error_message"] = str(e)
            benchmark_state["logs"].append(f"Error during benchmark: {str(e)}")

def extract_file_text(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.txt':
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    elif ext == '.pdf':
        from pdfminer.high_level import extract_text as pdf_extract
        return pdf_extract(filepath)
    elif ext == '.docx':
        from docx import Document
        doc = Document(filepath)
        return '\n'.join(p.text for p in doc.paragraphs)
    return ''

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify_route():
    claim_text = request.form.get('claim', '').strip()
    relation_mode = request.form.get('relation_mode', 'hybrid').strip().lower()
    if relation_mode not in RELATION_EXTRACTION_METHODS:
        relation_mode = 'hybrid'
    file = request.files.get('file')

    if file and file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXT:
            return jsonify({'error': f'Unsupported file type: {ext}. Use .txt, .pdf, or .docx'}), 400
        filename = secure_filename(file.filename)
        if not filename:
            return jsonify({'error': 'Invalid filename'}), 400
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        save_path = safe_child_path(app.config['UPLOAD_FOLDER'], filename)
        if save_path is None:
            return jsonify({'error': 'Invalid filename'}), 400
        file.save(save_path)
        doc_text = extract_file_text(save_path)
        if not claim_text:
            claim_text = doc_text[:5000]
        else:
            claim_text = claim_text

    if not claim_text:
        return jsonify({'error': 'No claim text provided'}), 400

    cache_key, cached_result = get_cached_verify_result(claim_text, relation_mode)
    if cached_result:
        cached_result['success'] = True
        cached_result['relation_extraction_method'] = relation_mode
        return jsonify(cached_result)

    try:
        result = verify(claim_text, extraction_method=relation_mode)
        result['success'] = True
        result['cache_hit'] = False
        result['cache_key'] = cache_key
        result['relation_extraction_method'] = relation_mode
        result['relation_extraction_label'] = RELATION_EXTRACTION_METHODS[relation_mode]
        set_cached_verify_result(cache_key, result)
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

    return jsonify(result)

@app.route('/api/kg-data')
def kg_data():
    data = load_kg()
    nodes = []
    edges = []
    communities = {}
    added_nodes = set()
    edge_set = set()

    for t in data.get('triples', []):
        s, p, o = t['triple']
        community = community_for_triple(t)
        if community["id"] not in communities:
            communities[community["id"]] = {
                "id": community["id"],
                "label": community["label"],
                "nodes": set(),
                "edge_count": 0,
            }

        for lbl, role in [(s, 'entity'), (o, 'entity')]:
            clean = lbl.replace('[[', '').replace(']]', '')
            communities[community["id"]]["nodes"].add(clean)
            if clean not in added_nodes:
                added_nodes.add(clean)
                grp = 1 if clean == s.replace('[[', '').replace(']]', '') else 2
                nodes.append({'id': clean, 'label': clean, 'group': grp})
        edge_key = (s, p, o)
        if edge_key not in edge_set:
            edge_set.add(edge_key)
            communities[community["id"]]["edge_count"] += 1
            edges.append({
                'from': s.replace('[[', '').replace(']]', ''),
                'to': o.replace('[[', '').replace(']]', ''),
                'label': p,
                'title': t.get('linearized', ''),
                'source': t.get('source_wiki') or t.get('source') or 'Unknown source',
                'community_id': community["id"],
                'community_label': community["label"],
            })

    stats = {
        'total_triples': len(data.get('triples', [])),
        'pages': len(data.get('page_index', {})),
        'page_index': data.get('page_index', {}),
    }

    community_list = [
        {
            "id": community["id"],
            "label": community["label"],
            "node_count": len(community["nodes"]),
            "edge_count": community["edge_count"],
        }
        for community in communities.values()
    ]

    return jsonify({
        'nodes': nodes,
        'edges': edges,
        'communities': community_list,
        'stats': stats,
        'kg_path': str(KG_FILE),
    })

@app.route('/api/wiki-pages')
def list_wiki_pages():
    wiki_dir = Path("data/wiki")
    if not wiki_dir.exists():
        return jsonify([])
    pages = []
    for f in wiki_dir.glob("*.md"):
        if f.name in ("index.md", "log.md"):
            continue
        title = f.stem.replace("-", " ").title()
        try:
            with open(f, "r", encoding="utf-8") as file:
                content = file.read()
            title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
            if title_match:
                title = title_match.group(1).strip()
        except Exception:
            pass
        
        stat = f.stat()
        pages.append({
            "filename": f.name,
            "title": title,
            "size": stat.st_size,
            "modified": stat.st_mtime
        })
    pages.sort(key=lambda x: x["title"])
    return jsonify(pages)

@app.route('/api/wiki-pages/<filename>')
def get_wiki_page(filename):
    wiki_dir = Path("data/wiki")
    filepath = safe_child_path(wiki_dir, filename)
    if filepath is None:
        return jsonify({"error": "Page not found"}), 404
    if not filepath.exists() or filepath.name in ("index.md", "log.md"):
        return jsonify({"error": "Page not found"}), 404
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        summary_match = re.search(r'## Summary\n(.*?)(?=\n## |\Z)', content, re.DOTALL)
        entities_match = re.search(r'## Entities\n(.*?)(?=\n## |\Z)', content, re.DOTALL)
        relations_match = re.search(r'## Relations\n(.*?)(?=\n## |\Z)', content, re.DOTALL)
        
        summary = summary_match.group(1).strip() if summary_match else ""
        
        entities = []
        if entities_match:
            for line in entities_match.group(1).strip().split("\n"):
                ent_match = re.search(r'\* \[\[([^\]]+)\]\]', line)
                if ent_match:
                    entities.append(ent_match.group(1))
                    
        relations = []
        if relations_match:
            for line in relations_match.group(1).strip().split("\n"):
                rel_match = re.search(r'-\s*\(\[\[([^\]]+)\]\]\s*\|\s*([^|]+)\s*\|\s*\[\[([^\]]+)\]\]\)', line)
                if rel_match:
                    relations.append({
                        "subject": rel_match.group(1).strip(),
                        "predicate": rel_match.group(2).strip(),
                        "object": rel_match.group(3).strip()
                    })
                    
        title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else filename
        
        return jsonify({
            "filename": filename,
            "title": title,
            "summary": summary,
            "entities": entities,
            "relations": relations,
            "raw": content
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/kg-manage/rebuild', methods=['POST'])
def rebuild_kg_route():
    try:
        num_triples = rebuild_from_wiki()
        clear_verify_cache()
        return jsonify({"success": True, "triples": num_triples})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/kg-manage/clear', methods=['POST'])
def clear_kg_route():
    try:
        clear_kg()
        clear_verify_cache()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/kg-manage/add-triple', methods=['POST'])
def add_triple_route():
    subj = request.form.get("subject", "").strip()
    pred = request.form.get("predicate", "").strip()
    obj = request.form.get("object", "").strip()
    source = request.form.get("source", "Manual Input").strip()
    
    if not subj or not pred or not obj:
        return jsonify({"success": False, "error": "Subject, predicate, and object are required"}), 400
    
    try:
        triple_data = {
            "triple": [subj, pred, obj],
            "linearized": f"{subj} {pred} {obj}"
        }
        added = add_triples(source, [triple_data])
        if added:
            clear_verify_cache()
        return jsonify({"success": True, "added": len(added) > 0})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/benchmark/list')
def list_benchmarks():
    bench_dir = Path("data/benchmark")
    if not bench_dir.exists():
        return jsonify([])
    runs = []
    for f in bench_dir.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
            metrics = data.get("metrics", {})
            runs.append({
                "filename": f.name,
                "samples": metrics.get("num_samples"),
                "accuracy": metrics.get("accuracy_pct"),
                "timestamp": metrics.get("timestamp"),
                "f1": metrics.get("f1_score"),
                "errors": metrics.get("errors")
            })
        except Exception:
            pass
    runs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return jsonify(runs)

@app.route('/api/benchmark/details/<filename>')
def get_benchmark_details(filename):
    bench_dir = Path("data/benchmark")
    filepath = safe_child_path(bench_dir, filename)
    if filepath is None:
        return jsonify({"error": "Benchmark file not found"}), 404
    if not filepath.exists():
        return jsonify({"error": "Benchmark file not found"}), 404
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/benchmark/run', methods=['POST'])
def run_benchmark():
    with benchmark_lock:
        if benchmark_state["status"] == "running":
            return jsonify({"error": "A benchmark is already running"}), 400
    
    num_samples = int(request.form.get("samples", 10))
    split = request.form.get("split", "train")
    seed = int(request.form.get("seed", 42))
    
    t = threading.Thread(target=run_benchmark_thread, args=(num_samples, split, seed))
    t.daemon = True
    t.start()
    return jsonify({"success": True, "message": "Benchmark started"})

@app.route('/api/benchmark/status')
def get_benchmark_status():
    with benchmark_lock:
        return jsonify(benchmark_state)

@app.route('/api/extract-entities', methods=['POST'])
def extract_entities_route():
    text = request.form.get("text", "").strip()
    if not text:
        return jsonify({"error": "Text is required"}), 400
    
    try:
        nlp = load_nlp()
        doc = nlp(text)

        entities = extract_entities(text, nlp=nlp)
        seen = {entity["text"].lower() for entity in entities}
        noun_chunks = []
        warning = None
        try:
            from src.semantic_ranking import get_sbert_model
            sbert = get_sbert_model()
            embeddings = sbert.encode(list(NER_PROTOTYPES.values()), convert_to_tensor=True)
            prototypes = dict(zip(NER_PROTOTYPES.keys(), embeddings))
            extract_from_noun_chunks(doc, entities, seen, sbert, prototypes, diagnostics=noun_chunks)
        except Exception as e:
            warning = f"SBERT chunk scoring unavailable: {e}"
        entities.extend(extract_uppercase_words(doc, seen))
        ranked_entities = rank_entities_for_retrieval(entities, text)
            
        return jsonify({
            "entities": entities,
            "entity_diagnostics": ranked_entities,
            "noun_chunks": noun_chunks,
            "warning": warning,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/model-status')
def model_status():
    status = {}
    
    # Check spaCy
    try:
        from src.entity_extractor import load_nlp
        nlp = load_nlp()
        status["spacy"] = {"name": "en_core_web_sm", "loaded": True, "status": "Cached & Ready"}
    except Exception as e:
        status["spacy"] = {"name": "en_core_web_sm", "loaded": False, "status": f"Error: {str(e)[:50]}"}
    
    # Check SBERT
    try:
        from src.semantic_ranking import get_sbert_model
        get_sbert_model()
        status["sbert"] = {"name": "all-MiniLM-L6-v2", "loaded": True, "status": "Cached & Ready"}
    except Exception as e:
        status["sbert"] = {"name": "all-MiniLM-L6-v2", "loaded": False, "status": f"Error: {str(e)[:50]}"}
    
    # Check DeBERTa NLI
    try:
        from src.eval import load_model
        load_model("tasksource/deberta-base-long-nli")
        status["deberta"] = {"name": "tasksource/deberta-base-long-nli", "loaded": True, "status": "Cached & Ready"}
    except Exception as e:
        status["deberta"] = {"name": "tasksource/deberta-base-long-nli", "loaded": False, "status": f"Error: {str(e)[:50]}"}
    
    return jsonify(status)

if __name__ == '__main__':
    os.makedirs(os.path.join(os.path.dirname(__file__), 'uploads'), exist_ok=True)
    app.run(debug=True, host='127.0.0.1', port=5000)
