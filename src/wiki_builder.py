import os
import re
import datetime
from pathlib import Path
import torch
from transformers import pipeline

RAW_DIR = Path("data/raw")
WIKI_DIR = Path("data/wiki")
INDEX_FILE = WIKI_DIR / "index.md"
LOG_FILE = WIKI_DIR / "log.md"

MODEL_ID = "google/gemma-4-E4B-it"

_pipeline = None

def init_wiki():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    WIKI_DIR.mkdir(parents=True, exist_ok=True)

    if not INDEX_FILE.exists():
        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            f.write("# Knowledge Base Index\n\n")

    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("# Wiki Log\n\n")

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        print(f"Loading {MODEL_ID}...")
        device = 0 if torch.cuda.is_available() else -1
        _pipeline = pipeline(
            "text-generation",
            model=MODEL_ID,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device
        )
    return _pipeline

def generate_wiki_content(source_text, title):
    pipe = get_pipeline()
    prompt = f"""You are an expert knowledge curator. Your task is to process the following source text and generate a structured Wiki article in Markdown format.
The Wiki article MUST contain a section called "## Relations" where you extract facts in the format: (Subject | Predicate | Object).
This is crucial as these relations will be used to build a Knowledge Graph.
IMPORTANT: You MUST format all key entities as Obsidian wiki links by enclosing them in double square brackets, e.g., [[Entity Name]]. Do this in the Summary, Entities, and Relations sections. For example: ([[Subject]] | Predicate | [[Object]]).

Source Title: {title}
Source Text:
{source_text}

Output format:
# {title}

## Summary
[Provide a concise summary of the text, using [[Wiki Links]] for key entities]

## Entities
[List key entities found in the text as [[Wiki Links]]]

## Relations
[Extract facts as a list: - ([[Subject]] | Predicate | [[Object]])]
"""
    messages = [
        {"role": "user", "content": prompt},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=512,
        do_sample=False,
    )
    generated_text = outputs[0]["generated_text"][-1]["content"]
    return generated_text

def ingest_source(source_text, title, source_type="dataset"):
    init_wiki()
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
    
    raw_filename = f"{date_str}-{slug}.md"
    raw_path = RAW_DIR / raw_filename
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(f"---\ntitle: {title}\ndate: {date_str}\nsource_type: {source_type}\n---\n\n")
        f.write(source_text)
    
    print(f"Compiling wiki page for: {title}")
    wiki_content = generate_wiki_content(source_text, title)
    
    wiki_filename = f"{slug}.md"
    wiki_path = WIKI_DIR / wiki_filename
    with open(wiki_path, "w", encoding="utf-8") as f:
        f.write(f"---\nraw_source: ../raw/{raw_filename}\nupdated: {date_str}\n---\n\n")
        f.write(wiki_content)
        
    with open(INDEX_FILE, "a", encoding="utf-8") as f:
        f.write(f"- [{title}]({wiki_filename}) - Compiled on {date_str}\n")
        
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"## [{date_str}] ingest | {title}\n")
        
    print(f"Successfully ingested {title} -> {wiki_path}")
    return wiki_path
