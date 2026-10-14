def test_build_evidence_trace_merges_ranking_and_nli():
    from src.pipeline import _build_evidence_trace

    ranked = [{
        "linearized": "Albert Einstein developed relativity",
        "source_type": "kg",
        "source_wiki": "Relativity.md",
        "score": 0.88,
    }]
    nli_items = [{
        "text": "Albert Einstein developed relativity",
        "verdict": "SUPPORTED",
        "confidence": 0.92,
        "effective_weight": 0.7,
    }]

    trace = _build_evidence_trace(ranked, nli_items)

    assert trace == [{
        "rank": 1,
        "source_type": "kg",
        "source": "Relativity.md",
        "title": "",
        "text": "Albert Einstein developed relativity",
        "score": 0.88,
        "cross_encoder_score": None,
        "nli_verdict": "SUPPORTED",
        "nli_confidence": 0.92,
        "effective_weight": 0.7,
    }]


def test_nli_verdict_stays_authoritative_when_llm_disagrees(monkeypatch):
    from src import llm_judge, multihop_nli, pipeline

    monkeypatch.setattr(multihop_nli, "predict_multihop_nli", lambda *_, **__: {
        "verdict": "SUPPORTED",
        "confidence": 0.97,
        "probs": {
            "SUPPORTED": 0.97,
            "REFUTED": 0.01,
            "NOT ENOUGH INFO": 0.02,
        },
    })
    monkeypatch.setattr(llm_judge, "judge", lambda *_, **__: {
        "verdict": "REFUTED",
        "reasoning": "Outdated model knowledge",
    })
    monkeypatch.setattr(
        pipeline,
        "get_pipeline",
        lambda name, default=None: True if name == "use_llm_judge" else default,
    )

    result = pipeline._judge_claim(
        "Donald Trump is the 47th president of the USA",
        [],
        [],
        "Donald Trump is the 47th president of the United States.",
        [{"text": "Donald Trump is the 47th president of the United States."}],
    )

    assert result["verdict"] == "SUPPORTED"
    assert result["confidence"] == 0.97
    assert result["probs"]["SUPPORTED"] == 0.97
    assert result["llm_judge"]["verdict"] == "REFUTED"
    assert result["judge_disagreement"] is True


def test_advisory_llm_failure_does_not_replace_nli(monkeypatch):
    from src import llm_judge, multihop_nli, pipeline

    monkeypatch.setattr(multihop_nli, "predict_multihop_nli", lambda *_, **__: {
        "verdict": "SUPPORTED",
        "confidence": 0.97,
        "probs": {
            "SUPPORTED": 0.97,
            "REFUTED": 0.01,
            "NOT ENOUGH INFO": 0.02,
        },
    })
    monkeypatch.setattr(
        llm_judge,
        "judge",
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError("Ollama failed")),
    )
    monkeypatch.setattr(
        pipeline,
        "get_pipeline",
        lambda name, default=None: True if name == "use_llm_judge" else default,
    )

    result = pipeline._judge_claim("Grounded claim", [], [], "Evidence", [])

    assert result["verdict"] == "SUPPORTED"
    assert result["confidence"] == 0.97
    assert result["llm_judge"]["verdict"] is None
    assert result["llm_judge"]["reasoning"] == "LLM error: Ollama failed"


def test_verify_uses_configured_retrieval_knobs(monkeypatch):
    from src import pipeline
    from src import entity_linker
    from src import semantic_ranking
    from src import knowledge_graph

    calls = {"kg_max": [], "wiki_top_k": [], "judge_premises": []}

    class NLP:
        def __call__(self, text):
            return object()

    class Server:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def fetch_clean(self, page_id):
            return "Ada Lovelace wrote notes about the Analytical Engine. " * 3

    class Model:
        def encode(self, texts, convert_to_tensor=False, **_):
            rows = len(texts) if isinstance(texts, list) else 1
            return [[1.0]] * rows

    monkeypatch.setattr(pipeline, "load_nlp", lambda: NLP())
    monkeypatch.setattr(pipeline, "extract_entities", lambda *_: [{"text": "Ada"}])
    monkeypatch.setattr(pipeline, "extract_from_noun_chunks", lambda *_: None)
    monkeypatch.setattr(pipeline, "extract_uppercase_words", lambda *_: [])
    monkeypatch.setattr(entity_linker, "apply_coreference", lambda claim, entities: claim)
    monkeypatch.setattr(entity_linker, "resolve_entities", lambda entities: [{"text": "Ada", "canonical": "Ada", "aliases": ["Ada"]}])
    monkeypatch.setattr(pipeline, "_load_sbert", lambda: Model())
    monkeypatch.setattr(pipeline, "get_retrieval", lambda name, default=None: {
        "kg_max_per_entity": 7,
        "wiki_snippet_candidates": 2,
        "wiki_full_fetch": 1,
    }.get(name, default))
    monkeypatch.setattr(pipeline, "get_pipeline", lambda name, default=None: {
        "wiki_request_delay": 0,
        "structured_nli": True,
    }.get(name, default))
    monkeypatch.setattr(pipeline, "retrieve_wiki_kg", lambda entities, max_triples_per_entity=50: calls["kg_max"].append(max_triples_per_entity) or [])
    monkeypatch.setattr(knowledge_graph, "search_by_entity", lambda _: [])
    monkeypatch.setattr(pipeline, "MPCServer", Server)
    monkeypatch.setattr(pipeline, "_search_wikipedia", lambda query, server, top_k=5: calls["wiki_top_k"].append(top_k) or [{"title": "Ada", "id": 1, "snippet": "Ada Lovelace"}])
    monkeypatch.setattr(pipeline, "page_exists", lambda _: False)
    monkeypatch.setattr(pipeline, "compile_wiki_page", lambda *_, **__: "data/wiki/ada.md")
    monkeypatch.setattr(knowledge_graph, "build_triple_embeddings", lambda *_: None)
    monkeypatch.setattr(pipeline, "get_wiki_summary", lambda _: "")
    monkeypatch.setattr(semantic_ranking, "rank_and_combine_evidence", lambda *_, **__: ([], ""))
    monkeypatch.setattr(pipeline, "_judge_claim", lambda *args: (
        calls["judge_premises"].append(args[3]) or {
            "verdict": "NOT ENOUGH INFO",
            "confidence": 1.0,
            "probs": {
                "SUPPORTED": 0.0,
                "REFUTED": 0.0,
                "NOT ENOUGH INFO": 1.0,
            },
        }
    ))

    pipeline.verify("Ada wrote notes", update_wiki=True)

    assert calls["kg_max"] == [7, 7]
    assert calls["wiki_top_k"][:2] == [2, 2]
    assert calls["judge_premises"] == [""]


def test_verify_fetches_only_primary_entities_but_reports_diagnostics(monkeypatch):
    from src import pipeline
    from src import entity_linker
    from src import semantic_ranking
    from src import knowledge_graph

    resolved_inputs = []
    kg_inputs = []
    wiki_queries = []

    class NLP:
        def __call__(self, text):
            return object()

    class Server:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def fetch_clean(self, page_id):
            return "The James Webb Space Telescope launched from French Guiana on Ariane 5. " * 2

    class Model:
        def encode(self, texts, convert_to_tensor=False, **_):
            rows = len(texts) if isinstance(texts, list) else 1
            return [[1.0]] * rows

    entities = [
        {"text": "James Webb Space Telescope", "label": "PRODUCT", "start": 4, "end": 30},
        {"text": "Ariane 5", "label": "PRODUCT", "start": 46, "end": 54},
        {"text": "French Guiana", "label": "GPE", "start": 67, "end": 80},
        {"text": "French", "label": "NORP", "start": 67, "end": 73},
        {"text": "Guiana", "label": "GPE", "start": 74, "end": 80},
        {"text": "2022", "label": "DATE", "start": 84, "end": 88},
        {"text": "an Ariane 5 rocket", "label": "PRODUCT", "start": 43, "end": 61},
    ]

    def resolve(ents):
        resolved_inputs.append([e["text"] for e in ents])
        return [{**e, "canonical": e["text"], "aliases": [e["text"]]} for e in ents]

    monkeypatch.setattr(pipeline, "load_nlp", lambda: NLP())
    monkeypatch.setattr(pipeline, "extract_entities", lambda *_: list(entities))
    monkeypatch.setattr(pipeline, "extract_from_noun_chunks", lambda *_: None)
    monkeypatch.setattr(pipeline, "extract_uppercase_words", lambda *_: [])
    monkeypatch.setattr(entity_linker, "apply_coreference", lambda claim, ents: claim)
    monkeypatch.setattr(entity_linker, "resolve_entities", resolve)
    monkeypatch.setattr(pipeline, "_load_sbert", lambda: Model())
    monkeypatch.setattr(pipeline, "get_retrieval", lambda name, default=None: {
        "kg_max_per_entity": 7,
        "wiki_snippet_candidates": 1,
        "wiki_full_fetch": 1,
        "max_primary_entities": 5,
        "allow_date_entities": False,
        "min_entity_chars": 3,
    }.get(name, default))
    monkeypatch.setattr(pipeline, "get_pipeline", lambda name, default=None: {
        "wiki_request_delay": 0,
        "structured_nli": True,
    }.get(name, default))
    monkeypatch.setattr(pipeline, "retrieve_wiki_kg", lambda ents, max_triples_per_entity=50: kg_inputs.append([e["text"] for e in ents]) or [])
    monkeypatch.setattr(knowledge_graph, "search_by_entity", lambda _: [])
    monkeypatch.setattr(pipeline, "MPCServer", Server)
    monkeypatch.setattr(pipeline, "_search_wikipedia", lambda query, server, top_k=5: wiki_queries.append(query) or [{"title": query, "id": len(wiki_queries), "snippet": query}])
    monkeypatch.setattr(pipeline, "page_exists", lambda _: False)
    monkeypatch.setattr(pipeline, "compile_wiki_page", lambda *_, **__: "data/wiki/jwst.md")
    monkeypatch.setattr(knowledge_graph, "build_triple_embeddings", lambda *_: None)
    monkeypatch.setattr(pipeline, "get_wiki_summary", lambda _: "")
    monkeypatch.setattr(semantic_ranking, "rank_and_combine_evidence", lambda *_, **__: ([], ""))

    result = pipeline.verify(
        "The James Webb Space Telescope launched on an Ariane 5 rocket from French Guiana in 2022.",
        update_wiki=True,
    )

    assert resolved_inputs[0] == ["James Webb Space Telescope", "French Guiana", "Ariane 5"]
    assert kg_inputs == [
        ["James Webb Space Telescope", "French Guiana", "Ariane 5"],
        ["James Webb Space Telescope", "French Guiana", "Ariane 5"],
    ]
    assert any("James Webb Space Telescope" in q for q in wiki_queries)
    assert any("French Guiana" in q for q in wiki_queries)
    assert any("Ariane 5" in q for q in wiki_queries)
    assert not any(q in {"French", "Guiana", "2022", "an Ariane 5 rocket"} for q in wiki_queries)
    assert {e["text"]: e["role"] for e in result["entity_diagnostics"]}["2022"] == "discarded"
