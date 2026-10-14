"""
Recompile all existing wiki pages using the configured extraction method (hybrid via Ollama).

Steps:
  1. Clear the KG so we start fresh (prevents stale triples).
  2. For each wiki/raw pair, re-run compile_wiki_page with the chosen extraction method.
  3. Save progress to data/.recompile_state.json so the job can be resumed.
  4. After all pages are processed, rebuild triple embeddings + qdrant index.
"""

import datetime
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.wiki_compiler import compile_wiki_page, WIKI_DIR, RAW_DIR
from src.knowledge_graph import clear as clear_kg
from src.entity_extractor import load_nlp

STATE_FILE = Path("data/.recompile_state.json")
SKIP_FILES = {"index.md", "log.md"}


def parse_frontmatter(content):
    """Return (frontmatter_dict, body) from a YAML-frontmatter file."""
    parts = content.split("---", 2)
    fm = {}
    body = content.strip()
    if len(parts) >= 3:
        body = parts[2].strip()
        for line in parts[1].strip().splitlines():
            m = re.match(r'^(\w+):\s*(.+)$', line)
            if m:
                fm[m.group(1)] = m.group(2).strip()
    return fm, body


def get_raw_path_from_wiki(wiki_content):
    """Extract the raw_source path from wiki frontmatter.
    Returns absolute path to the raw file, or None."""
    fm, _ = parse_frontmatter(wiki_content)
    raw_rel = fm.get("raw_source", "")
    if raw_rel:
        candidate = (WIKI_DIR / raw_rel).resolve()
        if candidate.exists():
            return candidate
    return None


def find_raw_file_by_slug(slug):
    """Heuristic search for a raw file matching a wiki slug."""
    candidates = []
    for f in RAW_DIR.glob(f"*-{slug}.md"):
        candidates.append(f)
    for f in RAW_DIR.glob(f"-{slug}.md"):  # no date prefix — rare edge case
        if f not in candidates:
            candidates.append(f)
    if candidates:
        candidates.sort()
        return candidates[-1]  # most recent by name
    return None


def get_raw_file_for_wiki(slug, wiki_content):
    """Get the raw file for a wiki page.  Prefers frontmatter reference."""
    rp = get_raw_path_from_wiki(wiki_content)
    if rp:
        return rp
    return find_raw_file_by_slug(slug)


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"completed": [], "total": 0, "failed": []}


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def main():
    print("=" * 60)
    print("  Wiki Page Recompiler (Hybrid: Regex + Ollama/LLM)")
    print("=" * 60)

    # 1. Clear the KG
    print("\n[1/4] Clearing existing KG...")
    clear_kg()
    print("  KG cleared (0 triples).")

    # 2. Load spaCy NLP (for regex extraction baseline)
    print("[2/4] Loading NLP...")
    nlp = load_nlp()
    print("  NLP loaded.")

    # 3. Determine which wiki files need processing
    all_wiki = []
    for f in sorted(WIKI_DIR.glob("*.md")):
        if f.name in SKIP_FILES:
            continue
        slug = f.stem
        wiki_content = f.read_text(encoding="utf-8")
        # Parse title from # Title
        m = re.search(r'^# (.+)$', wiki_content, re.MULTILINE)
        title = m.group(1).strip() if m else ""
        if not title:
            print(f"  SKIP {f.name}: no title found")
            continue
        raw_path = get_raw_file_for_wiki(slug, wiki_content)
        if raw_path is None:
            print(f"  SKIP {f.name}: no raw file found")
            continue
        # Read raw body
        _, raw_body = parse_frontmatter(raw_path.read_text(encoding="utf-8"))
        if not raw_body.strip():
            print(f"  SKIP {f.name}: empty raw body")
            continue
        all_wiki.append((title, raw_body, slug, f.name))

    print(f"  Found {len(all_wiki)} pages to recompile.")

    state = load_state()
    completed_set = set(state.get("completed", []))
    failed_set = set(state.get("failed", []))
    print(f"  Already completed: {len(completed_set)} / failed: {len(failed_set)}")

    t_start = time.time()
    ok_count = len(completed_set)

    for idx, (title, raw_body, slug, fname) in enumerate(all_wiki, 1):
        if fname in completed_set:
            continue

        t_page = time.time()
        try:
            result_path = compile_wiki_page(
                title, raw_body,
                nlp=nlp,
                extraction_method="hybrid",
            )
            elapsed = time.time() - t_page
            if result_path:
                completed_set.add(fname)
                failed_set.discard(fname)
                ok_count = len(completed_set)
                done = ok_count + len(failed_set)
                eta = (time.time() - t_start) / max(done, 1) * (len(all_wiki) - done) if done else 0
                print(f"  [{done}/{len(all_wiki)}] OK  {fname}  ({elapsed:.1f}s  ETA {eta/60:.0f}m)")
            else:
                # disambiguation or empty → don't retry
                completed_set.add(fname)
                print(f"  [{done}/{len(all_wiki)}] SKIP (disambig) {fname}  ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t_page
            print(f"  [{done}/{len(all_wiki)}] ERR {fname}: {e}  ({elapsed:.1f}s)")
            completed_set.add(fname)
            failed_set.add(fname)

        # Save state every page so we can resume after interruption
        state["completed"] = list(completed_set)
        state["failed"] = list(failed_set)
        state["total"] = len(all_wiki)
        save_state(state)

    total_elapsed = time.time() - t_start
    ok_count_final = len(completed_set - failed_set)
    print(f"\n{'='*60}")
    print(f"  Done. Processed {ok_count_final}/{len(all_wiki)} pages in {total_elapsed/60:.1f}m")
    if failed_set:
        print(f"  Failed: {len(failed_set)} page(s)")
        for fn in sorted(failed_set):
            print(f"    - {fn}")
    print(f"{'='*60}")

    # 4. Rebuild triple embeddings and qdrant index
    print("\n[3/4] Rebuilding triple embeddings...")
    from src.knowledge_graph import build_triple_embeddings
    from src.semantic_ranking import get_sbert_model
    t0 = time.time()
    build_triple_embeddings(get_sbert_model())
    print(f"  Triple embeddings built in {time.time()-t0:.1f}s")

    print("\n[4/4] Rebuilding Qdrant index...")
    from src.local_graph_rag import retrieve
    from src.knowledge_graph import _load
    data = _load()
    triples = data.get("triples", [])
    if triples:
        t0 = time.time()
        retrieve([{"canonical": "", "text": "", "aliases": [""]}], max_per_entity=1)
        print(f"  Qdrant index rebuilt in {time.time()-t0:.1f}s  ({len(triples)} triples)")
    else:
        print("  No triples to index.")

    print("\n" + "=" * 60)
    print("  Recompile complete.")
    print(f"  Triples in KG: {len(data.get('triples', []))}")
    print("=" * 60)


if __name__ == "__main__":
    main()
