import sys, os, json, io, tempfile, pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="module")
def app():
    from gui.app import app as flask_app
    flask_app.config.update({
        "TESTING": True,
        "UPLOAD_FOLDER": tempfile.mkdtemp(),
    })
    with flask_app.test_client() as client:
        yield client


class TestGUIAPI:
    def test_index_returns_html(self, app):
        resp = app.get("/")
        assert resp.status_code == 200
        assert b"TagaP" in resp.data
        assert b"Wiki Explorer" in resp.data
        assert b"vis-network" in resp.data or b"vis" in resp.data

    def test_kg_data_endpoint(self, app):
        resp = app.get("/api/kg-data")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "nodes" in data
        assert "edges" in data
        assert "communities" in data
        assert "stats" in data
        assert isinstance(data["nodes"], list)
        assert isinstance(data["edges"], list)
        assert isinstance(data["communities"], list)
        assert "total_triples" in data["stats"]
        assert "pages" in data["stats"]
        if data["edges"]:
            assert "community_id" in data["edges"][0]
            assert "community_label" in data["edges"][0]

    def test_kg_community_label_is_short_and_readable(self, app):
        from gui.app import clean_kg_label

        label = clean_kg_label("this-is-a-very-long-wikipedia-page-name-with-too-many-words.md")
        assert len(label) <= 42
        assert label == "this is a very long wikipedia"

    def test_verify_empty_claim_returns_error(self, app):
        resp = app.post("/verify", data={"claim": ""})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_verify_with_simpsons_claim(self, app):
        resp = app.post("/verify", data={"claim": "The Simpsons is an American animated sitcom"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data.get("success", True) is True
        assert "verdict" in data
        assert data["verdict"] in ("SUPPORTED", "REFUTED", "NOT ENOUGH INFO")
        assert "confidence" in data
        assert "probs" in data
        assert "entities" in data
        assert len(data["entities"]) > 0
        assert "stages" in data
        assert "premise" in data

    def test_verify_prob_structure(self, app):
        resp = app.post("/verify", data={"claim": "Albert Einstein developed the theory of relativity"})
        assert resp.status_code == 200
        data = resp.get_json()
        probs = data["probs"]
        for label in ("SUPPORTED", "REFUTED", "NOT ENOUGH INFO"):
            assert label in probs
            assert 0.0 <= probs[label] <= 1.0
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01

    def test_verify_entity_extraction(self, app):
        resp = app.post("/verify", data={"claim": "Tesla is an electric car company founded by Elon Musk"})
        assert resp.status_code == 200
        data = resp.get_json()
        entities = [e.lower() for e in data["entities"]]
        assert any("tesla" in e for e in entities)

    def test_verify_rejects_unsupported_file(self, app):
        resp = app.post("/verify", data={"claim": "test", "file": (io.BytesIO(b"test"), "test.exe")})
        assert resp.status_code == 400

    def test_verify_handles_txt_file_upload(self, app):
        content = b"Mount Everest is the highest mountain on Earth"
        resp = app.post("/verify", data={
            "claim": "",
            "file": (io.BytesIO(content), "claim.txt")
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert "verdict" in data

    def test_safe_child_path_rejects_traversal(self, app):
        from gui.app import safe_child_path

        base = Path(app.application.config["UPLOAD_FOLDER"])
        assert safe_child_path(base, "..\\outside.txt") is None

    def test_upload_filename_is_sanitized(self, app, monkeypatch):
        import gui.app as gui_app

        monkeypatch.setattr(gui_app, "verify", lambda claim, **kwargs: {
            "verdict": "NOT ENOUGH INFO",
            "confidence": 1.0,
            "probs": {},
        })
        upload_dir = Path(app.application.config["UPLOAD_FOLDER"])

        resp = app.post("/verify", data={
            "claim": "",
            "file": (io.BytesIO(b"claim text"), "..\\evil.txt")
        })

        assert resp.status_code == 200
        assert (upload_dir / "evil.txt").exists()
        assert not (upload_dir.parent / "evil.txt").exists()

    def test_verify_route_uses_cache_for_repeated_claim(self, app, monkeypatch):
        import gui.app as gui_app

        calls = {"count": 0}

        def fake_verify(claim, **kwargs):
            calls["count"] += 1
            return {
                "claim": claim,
                "verdict": "SUPPORTED",
                "confidence": 0.9,
                "probs": {"SUPPORTED": 0.9, "REFUTED": 0.0, "NOT ENOUGH INFO": 0.1},
                "entities": ["Cache Test"],
                "stages": {"total": 0.01},
                "premise": "Cached premise",
            }

        gui_app.clear_verify_cache()
        monkeypatch.setattr(gui_app, "verify", fake_verify)

        first = app.post("/verify", data={"claim": "Cache test claim"})
        second = app.post("/verify", data={"claim": "  Cache   test claim  "})

        assert first.status_code == 200
        assert second.status_code == 200
        assert calls["count"] == 1
        assert first.get_json()["cache_hit"] is False
        assert second.get_json()["cache_hit"] is True

    def test_verify_cache_key_includes_primary_judge(self, monkeypatch):
        import gui.app as gui_app

        monkeypatch.setattr(
            gui_app, "get_pipeline",
            lambda name, default=None: "nli",
            raising=False,
        )
        nli_key = gui_app.normalized_claim_key("Same claim", "llm")
        monkeypatch.setattr(
            gui_app, "get_pipeline",
            lambda name, default=None: "llm",
            raising=False,
        )
        llm_key = gui_app.normalized_claim_key("Same claim", "llm")

        assert nli_key != llm_key

    def test_verify_cache_survives_memory_clear(self, app, monkeypatch, tmp_path):
        import gui.app as gui_app

        monkeypatch.setattr(gui_app, "VERIFY_CACHE_FILE", tmp_path / "verify_cache.json")
        gui_app.clear_verify_cache()

        key = gui_app.normalized_claim_key("Persisted claim", "hybrid")
        gui_app.set_cached_verify_result(key, {
            "claim": "Persisted claim",
            "verdict": "SUPPORTED",
            "confidence": 0.9,
        })
        gui_app.verify_cache.clear()

        _, cached = gui_app.get_cached_verify_result("Persisted claim", "hybrid")

        assert cached["cache_hit"] is True
        assert cached["verdict"] == "SUPPORTED"

    def test_verify_contract_includes_evidence_trace(self, app, monkeypatch):
        import gui.app as gui_app

        gui_app.clear_verify_cache()
        monkeypatch.setattr(gui_app, "verify", lambda claim, **kwargs: {
            "claim": claim,
            "verdict": "SUPPORTED",
            "confidence": 0.92,
            "probs": {"SUPPORTED": 0.92, "REFUTED": 0.02, "NOT ENOUGH INFO": 0.06},
            "entities": ["Albert Einstein"],
            "entity_diagnostics": [{"text": "Albert Einstein", "label": "PERSON", "role": "primary"}],
            "stages": {"entity_extraction": 0.01, "evidence_ranking": 0.02, "total": 0.03},
            "premise": "Knowledge Graph: Albert Einstein developed relativity.",
            "evidence_trace": [{
                "rank": 1,
                "source_type": "kg",
                "source": "Relativity.md",
                "text": "Albert Einstein developed relativity",
                "score": 0.88,
                "nli_verdict": "SUPPORTED",
                "nli_confidence": 0.92,
            }],
        })

        resp = app.post("/verify", data={"claim": "Albert Einstein developed relativity"})
        data = resp.get_json()

        assert resp.status_code == 200
        assert data["verdict"] == "SUPPORTED"
        assert data["entity_diagnostics"][0]["role"] == "primary"
        assert data["evidence_trace"][0]["source_type"] == "kg"
        assert data["evidence_trace"][0]["nli_verdict"] == "SUPPORTED"

    def test_verify_route_passes_relation_mode(self, app, monkeypatch):
        import gui.app as gui_app

        captured = {}

        def fake_verify(claim, **kwargs):
            captured["extraction_method"] = kwargs.get("extraction_method")
            return {
                "claim": claim,
                "verdict": "SUPPORTED",
                "confidence": 0.8,
                "probs": {"SUPPORTED": 0.8, "REFUTED": 0.1, "NOT ENOUGH INFO": 0.1},
                "entities": [],
                "stages": {"total": 0.01},
                "premise": "Premise",
            }

        gui_app.clear_verify_cache()
        monkeypatch.setattr(gui_app, "verify", fake_verify)

        resp = app.post("/verify", data={"claim": "Mode test", "relation_mode": "llm"})
        data = resp.get_json()

        assert resp.status_code == 200
        assert captured["extraction_method"] == "llm"
        assert data["relation_extraction_method"] == "llm"

    def test_wiki_page_rejects_traversal(self, app):
        resp = app.get("/api/wiki-pages/..%5Cconfig.yaml")
        assert resp.status_code == 404

    def test_benchmark_details_rejects_traversal(self, app):
        resp = app.get("/api/benchmark/details/..%5Cconfig.yaml")
        assert resp.status_code == 404
