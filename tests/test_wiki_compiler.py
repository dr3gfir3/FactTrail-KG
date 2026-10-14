"""Tests for the wiki_compiler module, especially the SVO relation extractor."""
import shutil
from pathlib import Path

import src.wiki_compiler as wiki_compiler
from src.wiki_compiler import _extract_spacy_triples_from_text, _extract_triples_from_text
from src.entity_extractor import load_nlp
from src import wiki_llm_extractor

def test_extract_triples_svo():
    nlp = load_nlp()
    # Sentence 1: Standard subject-verb-object
    # Sentence 2: Passive voice + preposition
    # Sentence 3: Relative clause
    text = (
        "Albert Einstein discovered the theory of relativity. "
        "Marie Curie was born in Poland. "
        "Isaac Newton, who wrote Principia, lived in England."
    )
    
    triples = _extract_triples_from_text(text, nlp)
    
    linearized = [t["linearized"] for t in triples]
    
    # We expect things roughly like:
    # Einstein discovered theory of relativity (or similar depending on NER)
    # Marie Curie was born in Poland
    # Isaac Newton wrote Principia
    # Isaac Newton lived in England
    
    # Let's just check that we got some reasonable triples
    assert len(triples) > 0
    
    found_curie = False
    found_newton_wrote = False
    
    for t in triples:
        subj, pred, obj = t["triple"]
        if "Curie" in subj and "bear" in pred and "Poland" in obj:
            found_curie = True
        if "Newton" in subj and "write" in pred and "Principia" in obj:
            found_newton_wrote = True
            
    assert found_curie, f"Missed Marie Curie passive extraction. Got: {linearized}"
    assert found_newton_wrote, f"Missed relative clause extraction. Got: {linearized}"


def test_extract_spacy_triples_dependency_method():
    nlp = load_nlp()
    triples = _extract_spacy_triples_from_text(
        "Barack Obama served as president of the United States.",
        nlp,
    )

    assert any(
        "Obama" in t["triple"][0]
        and "serve" in t["triple"][1]
        and "United States" in t["triple"][2]
        and t["method"] == "dependency"
        for t in triples
    )


def test_phi3_llm_extractor_parses_output(monkeypatch):
    calls = {}

    class FakePipe:
        tokenizer = None

        def __call__(self, prompt, **kwargs):
            calls["prompt"] = prompt
            calls["kwargs"] = kwargs
            return [{
                "generated_text": prompt + """# Marie Curie

## Summary
[[Marie Curie]] discovered [[polonium]].

## Entities
* [[Marie Curie]]
* [[polonium]]

## Relations
- ([[Marie Curie]] | discovered | [[polonium]])
"""
            }]

    monkeypatch.setattr(wiki_llm_extractor, "_get_llm_pipeline", lambda: FakePipe())
    result = wiki_llm_extractor.extract_with_llm("Marie Curie discovered polonium.", "Marie Curie")

    assert "Source Title: Marie Curie" in calls["prompt"]
    assert calls["kwargs"] == {"max_new_tokens": 256, "do_sample": False, "use_cache": False}
    assert result["entities"] == ["Marie Curie", "polonium"]
    assert result["triples"][0]["triple"] == ["Marie Curie", "discovered", "polonium"]


def test_phi3_llm_extractor_retries_empty_relations(monkeypatch):
    calls = []

    class FakePipe:
        tokenizer = None

        def __call__(self, prompt, **kwargs):
            calls.append(prompt)
            if len(calls) == 1:
                return [{
                    "generated_text": prompt + """# Bialetti

## Summary
[[Bialetti]] is an [[Italian]] brand founded by [[Alfonso Bialetti]].

## Entities
* [[Bialetti]]
* [[Italian]]
* [[Alfonso Bialetti]]

## Relations
"""
                }]
            return [{
                "generated_text": prompt + """## Relations
- [[Bialetti]] | founded by | [[Alfonso Bialetti]]
1. ([[Bialetti]] | is | [[Italian]])
"""
            }]

    monkeypatch.setattr(wiki_llm_extractor, "_get_llm_pipeline", lambda: FakePipe())
    result = wiki_llm_extractor.extract_with_llm(
        "Bialetti is an Italian brand founded by Alfonso Bialetti.",
        "Bialetti",
    )

    assert len(calls) == 2
    assert [t["triple"] for t in result["triples"]] == [
        ["Bialetti", "founded by", "Alfonso Bialetti"],
        ["Bialetti", "is", "Italian"],
    ]
    assert calls[0]
    assert "| is |" in result["wiki_content"]


def test_llm_pipeline_uses_configured_model_without_remote_code(monkeypatch):
    calls = {}

    def fake_pipeline(task, **kwargs):
        calls["task"] = task
        calls.update(kwargs)
        return object()

    monkeypatch.setattr(wiki_llm_extractor, "_llm_pipeline", None)
    monkeypatch.setattr(wiki_llm_extractor, "get_model", lambda name, default=None: "Qwen/Qwen3-1.7B" if name == "llm" else default)
    monkeypatch.setattr(wiki_llm_extractor, "pipeline", fake_pipeline)
    monkeypatch.setattr(wiki_llm_extractor.torch.cuda, "is_available", lambda: False)

    try:
        wiki_llm_extractor._get_llm_pipeline()
    finally:
        monkeypatch.setattr(wiki_llm_extractor, "_llm_pipeline", None)

    assert calls["task"] == "text-generation"
    assert calls["model"] == "Qwen/Qwen3-1.7B"
    assert "trust_remote_code" not in calls


def test_llm_prompt_uses_tokenizer_chat_template(monkeypatch):
    calls = {}

    class FakeTokenizer:
        chat_template = "template"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=None):
            assert messages[0]["role"] == "user"
            assert tokenize is False
            assert add_generation_prompt is True
            assert enable_thinking is False
            return "CHAT:" + messages[0]["content"]

    class FakePipe:
        tokenizer = FakeTokenizer()

        def __call__(self, prompt, **kwargs):
            calls["prompt"] = prompt
            return [{"generated_text": prompt + "\n## Relations\n- ([[Ada]] | wrote | [[Notes]])"}]

    monkeypatch.setattr(wiki_llm_extractor, "_get_llm_pipeline", lambda: FakePipe())
    result = wiki_llm_extractor.extract_with_llm("Ada wrote Notes.", "Ada")

    assert calls["prompt"].startswith("CHAT:")
    assert result["triples"][0]["triple"] == ["Ada", "wrote", "Notes"]


def test_llm_relation_parser_allows_weak_be_triples():
    triples = wiki_llm_extractor._parse_llm_relations("""## Relations
- ([[Alfonso Bialetti]] | be | [[Moka Express]])
- ([[Alfonso Bialetti]] | manufactured | [[Moka Express]])
""")

    assert [t["triple"] for t in triples] == [
        ["Alfonso Bialetti", "be", "Moka Express"],
        ["Alfonso Bialetti", "manufactured", "Moka Express"],
    ]


def test_llm_relation_parser_cleans_labels_and_skips_control_noise():
    triples = wiki_llm_extractor._parse_llm_relations("""## Relations
- Subject: [[Marie Curie]] | Predicate: discovered | Object: [[polonium]]
<|end|>
""")

    assert [t["triple"] for t in triples] == [
        ["Marie Curie", "discovered", "polonium"],
    ]


def test_llm_relation_parser_rejects_dangling_entity_words():
    triples = wiki_llm_extractor._parse_llm_relations("""## Relations
- ([[Umberto I of Italy]] | was King of Italy | [[from]])
- ([[Umberto I of Italy]] | reigned from | [[9 January 1878]])
""")

    assert [t["triple"] for t in triples] == [
        ["Umberto I of Italy", "reigned from", "9 January 1878"],
    ]


def test_compile_raw_file_with_llm_only(monkeypatch):
    base = Path("data/test_runtime/wiki_llm_compile")
    if base.exists():
        shutil.rmtree(base)

    raw_file = base / "input.md"
    raw_file.parent.mkdir(parents=True)
    raw_file.write_text("Ada Lovelace wrote notes on the Analytical Engine.", encoding="utf-8")

    raw_dir = base / "raw"
    wiki_dir = base / "wiki"
    monkeypatch.setattr(wiki_compiler, "RAW_DIR", raw_dir)
    monkeypatch.setattr(wiki_compiler, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(wiki_compiler, "INDEX_FILE", wiki_dir / "index.md")
    monkeypatch.setattr(wiki_compiler, "LOG_FILE", wiki_dir / "log.md")

    monkeypatch.setattr("src.knowledge_graph.KG_FILE", base / "knowledge_graph.json")
    monkeypatch.setattr("src.knowledge_graph.KG_EMBEDDINGS_FILE", base / "kg" / "triple_embeddings.npy")
    monkeypatch.setattr("src.knowledge_graph.KG_EMBEDDINGS_INDEX_FILE", base / "kg" / "triple_embeddings_index.json")
    from src.knowledge_graph import _invalidate_cache
    _invalidate_cache()

    def regex_should_not_run(*args, **kwargs):
        raise AssertionError("regex extractor should not run in llm mode")

    monkeypatch.setattr(wiki_compiler, "_extract_triples_from_text", regex_should_not_run)
    monkeypatch.setattr(wiki_compiler, "_extract_spacy_triples_from_text", regex_should_not_run)

    class FakePipe:
        def __call__(self, prompt, **kwargs):
            return [{
                "generated_text": prompt + """# Ada Lovelace

## Summary
[[Ada Lovelace]] wrote [[Notes]].

## Entities
* [[Ada Lovelace]]
* [[Notes]]
* [[Analytical Engine]]

## Relations
- ([[Ada Lovelace]] | wrote | [[Notes]])
- ([[Notes]] | described | [[Analytical Engine]])
"""
            }]

    monkeypatch.setattr(wiki_llm_extractor, "_get_llm_pipeline", lambda: FakePipe())

    try:
        wiki_path = wiki_compiler.compile_wiki_page(
            "Ada Lovelace",
            raw_file.read_text(encoding="utf-8"),
            nlp=object(),
            extraction_method="llm",
        )

        content = Path(wiki_path).read_text(encoding="utf-8")
        assert wiki_path == wiki_dir / "ada-lovelace.md"
        assert "- ([[Ada Lovelace]] | wrote | [[Notes]])" in content

        from src.knowledge_graph import search_by_entity
        assert search_by_entity("ada lovelace")[0]["method"] == "llm"
    finally:
        shutil.rmtree(base, ignore_errors=True)
