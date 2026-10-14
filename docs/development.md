# Development

Run commands from the repository root. Paths in `src/` and `config.yaml` are relative to it.

## Configuration

`config.yaml` controls model names, retrieval limits, verdict thresholds, and optional pipeline stages. The default NLI model for verification is `roberta-large-mnli`; `src/eval.py` has a separate default for evaluation. The `wiki_extraction_method` setting selects `regex`, `spacy`, `llm`, or `hybrid`. The LLM path uses the Ollama endpoint and model named in the same file; install Ollama if you want that extraction path. The LLM judge is advisory, while NLI determines the verdict.

## Tests

Run the tests except those marked `slow` with:

```powershell
venv\Scripts\python.exe -m pytest tests -q -m "not slow"
```

Tests marked `slow` load large models. They are included by default. `tests/test_mpc_server.py` uses the live Wikipedia API and needs network access. To include the slow tests, run `venv\Scripts\python.exe -m pytest tests -q`. Model downloads may occur on first use.

## Local data

Verification can fetch Wikipedia pages and update the local graph. `data/raw/` contains fetched text, `data/wiki/` contains compiled pages, and `data/knowledge_graph.json` stores triples. The first two directories are ignored by Git. Without the compiled pages, a missing graph file cannot be rebuilt from local data; verification then needs Wikipedia access to populate it.

The GUI writes a verification cache under `data/cache/`. The local vector index lives under `data/qdrant/`. Benchmarks write results to `data/benchmark/`.
