import json
from types import SimpleNamespace

import pytest

from benchmark.wiki.src.core.full_document_inputs import prepare_full_document_texts, read_full_document
from benchmark.wiki.src.core.vector_store import VikingStoreWrapper
from benchmark.wiki.src.pipeline import BenchmarkPipeline


def test_original_text_keeps_tail_and_does_not_create_cache(tmp_path):
    source = tmp_path / "source.txt"
    text = "First\n" + "full text\n" * 5000 + "Last\n"
    source.write_text(text)
    assert read_full_document(source, tmp_path / "cache") == text
    assert not (tmp_path / "cache").exists()


def test_pdf_cache_reuses_text_and_invalidates_changed_source(tmp_path, monkeypatch):
    source, cache_dir = tmp_path / "source.pdf", tmp_path / "cache"
    source.write_bytes(b"pdf-one")
    calls = []

    def extract(_parser, path):
        calls.append(path)
        return path.read_bytes().decode() + " full text"

    monkeypatch.setattr("benchmark.wiki.src.core.full_document_inputs.PDFParser.extract_text", extract)
    assert read_full_document(source, cache_dir) == "pdf-one full text"
    assert read_full_document(source, cache_dir) == "pdf-one full text"
    assert len(calls) == 1
    source.write_bytes(b"pdf-two")
    assert read_full_document(source, cache_dir) == "pdf-two full text"
    assert len(calls) == 2
    assert len(list(cache_dir.iterdir())) == 1


def test_source_mapping_uses_document_ids_not_list_order(tmp_path):
    paths = [tmp_path / name for name in ["b.txt", "a.txt"]]
    for p in paths:
        p.write_text(p.stem + " original")
    client = SimpleNamespace(read=lambda uri: json.dumps({"documents": [
        {"doc_id": "a", "title": "A", "relative_uri": "nested/a"},
        {"doc_id": "b", "title": "B", "relative_uri": "b"},
    ]}))
    documents = [SimpleNamespace(sample_id=p.stem, doc_path=str(p)) for p in paths]
    assert prepare_full_document_texts(client, ["viking://resources/docs"], documents, tmp_path / "cache") == {
        "viking://resources/docs/nested/a": "a original",
        "viking://resources/docs/b": "b original",
    }
    with pytest.raises(ValueError, match="not mapped.*a"):
        prepare_full_document_texts(client, ["viking://resources/docs"], documents[:1], tmp_path / "cache")


def test_nodes_only_build_needs_no_original_files(tmp_path):
    store = VikingStoreWrapper.__new__(VikingStoreWrapper)
    store.store_path = str(tmp_path)
    seen = {}

    def build(**kwargs):
        seen.update(kwargs)
        return {"status": "success"}

    store.client = SimpleNamespace(build_wiki=build)
    store.build_wiki(["viking://resources/docs"], card_input_mode="full_document", build_stage="nodes")
    assert seen["full_document_texts"] is None
    assert seen["build_stage"] == "nodes"


def test_build_cards_uses_preserved_manifest_without_import(tmp_path):
    manifest = tmp_path / "imported_resources.json"
    manifest.write_text(json.dumps({"resource_uris": ["viking://resources/docs"]}))
    docs = [SimpleNamespace(sample_id="a", doc_path=str(tmp_path / "a.txt"))]
    seen = {}

    def build(**kwargs):
        seen.update(kwargs)
        return {"status": "success", "time": 1}

    config = {
        "paths": {"output_dir": str(tmp_path / "output"), "resource_manifest": str(manifest), "doc_output_dir": str(tmp_path)},
        "execution": {"wiki_card_input_mode": "full_document", "wiki_max_source_input_chars": 5000},
    }
    pipe = BenchmarkPipeline(
        config, adapter=SimpleNamespace(data_prepare=lambda path: docs),
        vector_db=SimpleNamespace(build_wiki=build), llm=None,
    )
    pipe.run_build_wiki("cards")
    assert seen["resource_uris"] == ["viking://resources/docs"]
    assert seen["source_documents"] == docs
    assert seen["max_source_input_chars"] == 5000
    assert not (tmp_path / "output" / "imported_resources.json").exists()
