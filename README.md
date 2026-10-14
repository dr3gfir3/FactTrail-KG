# FactTrail-KG

FactTrail-KG (formerly TagaP) verifies English-language claims against a local Wikipedia knowledge base. It returns `SUPPORTED`, `REFUTED`, or `NOT ENOUGH INFO`, together with the retrieved evidence and confidence.

## Run it

From the repository root, create an environment and install the Python dependencies:

```powershell
py -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
```

Verify one claim from the command line:

```powershell
venv\Scripts\python.exe verify_claim.py "Paris is the capital of France."
```

Run the web interface, then open [http://127.0.0.1:5000](http://127.0.0.1:5000):

```powershell
venv\Scripts\python.exe gui\app.py
```

Run a HoVer benchmark:

```powershell
venv\Scripts\python.exe benchmark_pipeline.py 10 train
```

Models and spaCy data download on first use. The GUI accepts a claim directly or a `.txt`, `.pdf`, or `.docx` file. Wikipedia access is needed when local evidence is missing.

## Architecture

```
Claim
  -> entity extraction and linking (spaCy)
  -> local retrieval (GraphRAG, hybrid KG search, and wiki summaries)
  -> Wikipedia fetch and compilation when local evidence is missing
  -> SBERT ranking and cross-encoder re-ranking
  -> multi-hop RoBERTa NLI
  -> SUPPORTED / REFUTED / NOT ENOUGH INFO
```

`src/pipeline.py:verify()` is the shared entrypoint for the CLI, benchmark, and Flask GUI.

| Component | Responsibility |
|---|---|
| `src/entity_extractor.py` and `src/entity_linker.py` | Extract, rank, link, and resolve claim entities. |
| `src/wiki_kg_retrieval.py` and `src/local_graph_rag.py` | Retrieve local evidence from the graph store, vectors, BM25, and wiki data. |
| `src/mpc_server.py` and `src/wiki_compiler.py` | Search Wikipedia, compile pages, and add extracted triples to the local store. |
| `src/knowledge_graph.py` | Persist and search triples in `data/knowledge_graph.json`. |
| `src/semantic_ranking.py` | Rank evidence with SBERT and optionally re-rank it with a cross-encoder. |
| `src/multihop_nli.py` | Build an evidence premise and produce the authoritative NLI verdict. |
| `gui/app.py` | Flask interface for verification, knowledge-graph inspection, and benchmarks. |

The pipeline can perform a second retrieval pass when the first verdict is uncertain. The optional LLM judge provides diagnostics only; it does not replace the NLI verdict.

`verify(claim_text, update_wiki=True, extraction_method=None)` returns a dictionary with the verdict, confidence, class probabilities, retrieved entities, evidence trace, fetched pages, and stage timings. Set `update_wiki=False` to use local evidence without fetching new pages.

## Data

| Path | Contents |
|---|---|
| `data/knowledge_graph.json` | Persistent triple store. |
| `data/wiki/*.md` | Compiled Wikipedia pages. |
| `data/raw/*.md` | Raw Wikipedia content. |
| `data/cache/` | Local GUI and retrieval caches. |

If `data/knowledge_graph.json` is absent, the pipeline rebuilds it from `data/wiki/*.md` when `pipeline.auto_rebuild_kg` is enabled in `config.yaml`.

See [Development](docs/development.md) for configuration, tests, and local data handling.
