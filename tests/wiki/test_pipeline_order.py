import json

import pytest

from openviking.wiki.config import WikiConfig
from openviking.wiki.content_loader import WikiContentLoader
from openviking.wiki.llm import WikiLLMRunner
from openviking.wiki.pipeline import WikiPipeline
from openviking.wiki.schemas import ResourceDocument, SourceSection, WikiResourceInput
from openviking.wiki.writer import WikiVikingFSWriter

from .fakes import FakeClient, FakeMixedVLM, FakeVLM


def _doc(index: int) -> ResourceDocument:
    content = f"# Paper {index}\n\nContent about question answering."
    return ResourceDocument(
        doc_id=f"doc_{index}", resource_uri=f"viking://resources/doc_{index}/",
        title=f"Paper {index}", content_or_structure=content,
        source_sections=[SourceSection(section_uri=f"viking://resources/doc_{index}/", content=content)],
    )


def _wiki_input(doc: ResourceDocument) -> WikiResourceInput:
    return WikiResourceInput(doc_id=doc.doc_id, resource_uri=doc.resource_uri, title=doc.title)


def _card_content(index: int) -> dict:
    return {"summary": f"Paper {index} discusses question answering.", "candidate_topics": ["question answering"]}


def _call(name: str, **arguments: object) -> dict:
    return {"name": name, "arguments": arguments}


class FakeContentLoader:
    def __init__(self, docs: list[ResourceDocument]):
        self.docs_by_id = {doc.doc_id: doc for doc in docs}

    async def load_document(self, doc: WikiResourceInput, **_: object) -> ResourceDocument:
        return self.docs_by_id[doc.doc_id]


def _writer(client: FakeClient, config: WikiConfig) -> WikiVikingFSWriter:
    return WikiVikingFSWriter(viking_fs=client, vikingdb=object(), ctx=object(), config=config, content_writer=client)


@pytest.mark.asyncio
async def test_pipeline_generates_layer_content_before_next_agent_layer():
    docs = [_doc(index) for index in range(1, 4)]
    fake_vlm = FakeMixedVLM(
        [
            *[_card_content(index) for index in range(1, 4)],
            {"documents": [{"title": "High-Level Knowledge", "content": "Synthesized QA knowledge."}]},
            {"summary": "Question answering node synthesis."},
        ],
        [
            [_call("create_node", node_id="question_answering", title="Question Answering", scope="QA methods and evaluation.", card_ids=["doc_1", "doc_2", "doc_3"])],
            [_call("finish_layer")],
            [_call("finish_layer")],
        ],
    )
    client, config = FakeClient(), WikiConfig()
    artifacts = await WikiPipeline(writer=_writer(client, config), config=config, llm=WikiLLMRunner(fake_vlm)).run_from_inputs(
        [_wiki_input(doc) for doc in docs], content_loader=FakeContentLoader(docs)
    )

    assert "viking://wiki/nodes/question_answering/0001.md" in client.writes
    assert "viking://wiki/nodes/question_answering/card.json" in client.writes
    node_card = json.loads(client.writes["viking://wiki/nodes/question_answering/card.json"])
    assert node_card["scope"] == "QA methods and evaluation."
    assert "candidate_topics" not in node_card
    assert '"build_stage": "all"' in client.writes["viking://wiki/run/config.json"]
    assert len(artifacts.nodes) == 1
    assert len(fake_vlm.tool_calls) == 3
    assert '"card_id": "question_answering"' in fake_vlm.tool_calls[2]
    assert '"scope": "QA methods and evaluation."' in fake_vlm.tool_calls[2]
    assert '"candidate_topics"' not in fake_vlm.tool_calls[2]
    assert '"card_id": "doc_3"' not in fake_vlm.tool_calls[2]


@pytest.mark.asyncio
async def test_pipeline_stops_when_agent_creates_no_directory_nodes():
    docs = [_doc(index) for index in range(1, 4)]
    fake_vlm = FakeMixedVLM([_card_content(index) for index in range(1, 4)], [[_call("finish_layer")]])
    client, config = FakeClient(), WikiConfig()
    artifacts = await WikiPipeline(writer=_writer(client, config), config=config, llm=WikiLLMRunner(fake_vlm)).run_from_inputs(
        [_wiki_input(doc) for doc in docs], content_loader=FakeContentLoader(docs)
    )

    assert artifacts.nodes == []
    assert artifacts.node_contexts == []
    assert len(fake_vlm.tool_calls) == 1


@pytest.mark.asyncio
async def test_pipeline_cards_stage_persists_cards_without_calling_agent():
    docs = [_doc(index) for index in range(1, 4)]
    fake_vlm = FakeVLM([_card_content(index) for index in range(1, 4)])
    client, config = FakeClient(), WikiConfig()
    artifacts = await WikiPipeline(writer=_writer(client, config), config=config, llm=WikiLLMRunner(fake_vlm)).run_from_inputs(
        [_wiki_input(doc) for doc in docs], content_loader=FakeContentLoader(docs), build_stage="cards"
    )

    assert len(artifacts.cards) == 3
    assert artifacts.nodes == []
    assert "viking://wiki/cards/doc_1.card.json" in client.writes


@pytest.mark.asyncio
async def test_pipeline_nodes_stage_reuses_persisted_cards():
    docs = [_doc(index) for index in range(1, 4)]
    client = FakeClient()
    for index in range(1, 4):
        client.writes[f"viking://wiki/cards/doc_{index}.card.json"] = json.dumps({
            "doc_id": f"doc_{index}", "resource_uri": f"viking://resources/doc_{index}/",
            "title": f"Paper {index}", **_card_content(index),
        })
    fake_vlm = FakeMixedVLM(
        [
            {"documents": [{"title": "High-Level Knowledge", "content": "Synthesized QA knowledge."}]},
            {"summary": "Question answering node synthesis."},
        ],
        [
            [_call("create_node", node_id="question_answering", title="Question Answering", scope="QA methods and evaluation.", card_ids=["doc_1", "doc_2", "doc_3"])],
            [_call("finish_layer")],
            [_call("finish_layer")],
        ],
    )
    config = WikiConfig()
    artifacts = await WikiPipeline(writer=_writer(client, config), config=config, llm=WikiLLMRunner(fake_vlm)).run_from_inputs(
        [_wiki_input(doc) for doc in docs], content_loader=FakeContentLoader(docs), build_stage="nodes"
    )

    assert len(artifacts.nodes) == 1
    assert all(f"viking://wiki/cards/doc_{index}.card.json" in client.writes for index in range(1, 4))
    assert ("viking://wiki/nodes/", True) in client.removed
    assert '"build_stage": "nodes"' in client.writes["viking://wiki/run/config.json"]


@pytest.mark.asyncio
async def test_full_document_cards_keep_entire_original_without_reading_chunks():
    uri = "viking://resources/paper"
    full_text = "# Original title\n" + "Evidence.\n" * 4000 + "FINAL_RESULT_42"
    fake_vlm = FakeVLM([_card_content(1)])
    client, config = FakeClient(), WikiConfig()
    artifacts = await WikiPipeline(
        writer=_writer(client, config), config=config, llm=WikiLLMRunner(fake_vlm)
    ).run_from_inputs(
        [WikiResourceInput(doc_id="paper", resource_uri=uri, title="Paper")],
        content_loader=WikiContentLoader(object(), object(), object()),
        card_input_mode="full_document", full_document_texts={uri: full_text},
        max_card_input_chars=1000, build_stage="cards",
    )
    assert len(artifacts.cards) == 1
    assert json.dumps(full_text) in fake_vlm.calls[0]
    assert "...(truncated)" not in fake_vlm.calls[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("texts", [None, {"viking://resources/wrong": "text"}, {"viking://resources/paper": " "}])
async def test_full_document_missing_source_fails_before_model_call(texts):
    fake_vlm = FakeVLM([])
    client, config = FakeClient(), WikiConfig()
    with pytest.raises(ValueError, match="full_document_texts|missing or empty"):
        await WikiPipeline(
            writer=_writer(client, config), config=config, llm=WikiLLMRunner(fake_vlm)
        ).run_from_inputs(
            [WikiResourceInput(doc_id="paper", resource_uri="viking://resources/paper", title="Paper")],
            content_loader=WikiContentLoader(object(), object(), object()),
            card_input_mode="full_document", full_document_texts=texts, build_stage="cards",
        )
    assert fake_vlm.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["summary", "raw_chunk", "full_document"])
async def test_pipeline_keeps_node_source_budget_independent_of_card_budget(mode, monkeypatch):
    from .test_content_loader import FakeVikingFS

    class SourceFS(FakeVikingFS):
        async def read_file(self, uri, **kwargs):
            return "source evidence " * 300

    class SummaryDB:
        async def get_context_by_uri(self, **kwargs):
            return [{"abstract": "summary " * 600}]

    doc = _wiki_input(_doc(1))
    captured = {}
    client, config = FakeClient(), WikiConfig()
    pipeline = WikiPipeline(
        writer=_writer(client, config), config=config, llm=WikiLLMRunner(FakeVLM([_card_content(1)]))
    )

    async def capture_sources(cards, artifacts, source_docs, **kwargs):
        captured.update(source_docs)
        return artifacts

    monkeypatch.setattr(pipeline, "_run_from_cards", capture_sources)
    await pipeline.run_from_inputs(
        [doc], content_loader=WikiContentLoader(SourceFS(), SummaryDB(), object()),
        card_input_mode=mode, max_card_input_chars=1000, max_source_input_chars=12000,
        full_document_texts={doc.resource_uri: "Original uncut text"} if mode == "full_document" else None,
    )
    assert len(captured[doc.doc_id].source_sections) == 2
    assert all(s.content == ("source evidence " * 300).strip() for s in captured[doc.doc_id].source_sections)
