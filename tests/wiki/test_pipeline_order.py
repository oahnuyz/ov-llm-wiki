import pytest

from openviking.wiki.config import WikiConfig
from openviking.wiki.llm import WikiLLMRunner
from openviking.wiki.pipeline import WikiPipeline
from openviking.wiki.schemas import ResourceDocument, SourceSection, WikiResourceInput
from openviking.wiki.writer import WikiVikingFSWriter

from .fakes import FakeClient, FakeVLM


def _doc(index: int) -> ResourceDocument:
    content = f"# Paper {index}\n\nContent about question answering."
    return ResourceDocument(
        doc_id=f"doc_{index}",
        resource_uri=f"viking://resources/doc_{index}/",
        title=f"Paper {index}",
        content_or_structure=content,
        source_sections=[SourceSection(section_uri=f"viking://resources/doc_{index}/", content=content)],
    )


def _wiki_input(doc: ResourceDocument) -> WikiResourceInput:
    return WikiResourceInput(doc_id=doc.doc_id, resource_uri=doc.resource_uri, title=doc.title)


def _card_content(index: int) -> dict:
    return {
        "summary": f"Paper {index} discusses question answering.",
        "main_points": ["QA method"],
        "important_terms": ["question answering"],
        "candidate_topics": ["question answering"],
    }


class FakeContentLoader:
    def __init__(self, docs: list[ResourceDocument]):
        self.docs_by_id = {doc.doc_id: doc for doc in docs}

    async def load_document(self, doc: WikiResourceInput, **_: object) -> ResourceDocument:
        return self.docs_by_id[doc.doc_id]


@pytest.mark.asyncio
async def test_pipeline_generates_layer_content_before_next_aggregation():
    docs = [_doc(index) for index in range(1, 4)]
    fake_vlm = FakeVLM(
        [
            *[_card_content(index) for index in range(1, 4)],
            {
                "operations": [
                    {
                        "op": "create_candidate",
                        "candidate_ref": "qa",
                        "title": "Question Answering",
                        "scope": "QA methods and evaluation.",
                        "card_ids": ["doc_1", "doc_2", "doc_3"],
                    }
                ]
            },
            {"documents": [{"title": "High-Level Knowledge", "content": "Synthesized QA knowledge."}]},
            {
                "summary": "Question answering node synthesis.",
                "main_points": ["QA synthesis"],
                "important_terms": ["question answering"],
                "candidate_topics": ["question answering systems"],
            },
            {"operations": []},
        ]
    )
    client = FakeClient()
    config = WikiConfig()
    writer = WikiVikingFSWriter(
        viking_fs=client,
        vikingdb=object(),
        ctx=object(),
        config=config,
        content_writer=client,
    )

    artifacts = await WikiPipeline(
        writer=writer,
        config=config,
        llm=WikiLLMRunner(fake_vlm),
    ).run_from_inputs(
        [_wiki_input(doc) for doc in docs],
        content_loader=FakeContentLoader(docs),
    )

    assert "viking://wiki/nodes/question_answering/documents/0001.md" in client.writes
    assert "viking://wiki/nodes/question_answering/card.json" in client.writes
    assert len(fake_vlm.calls) == 7


@pytest.mark.asyncio
async def test_pipeline_stops_when_a_layer_has_only_pending_candidates():
    docs = [_doc(index) for index in range(1, 4)]
    fake_vlm = FakeVLM([*[_card_content(index) for index in range(1, 4)], {"operations": []}])
    client = FakeClient()
    config = WikiConfig()
    writer = WikiVikingFSWriter(
        viking_fs=client,
        vikingdb=object(),
        ctx=object(),
        config=config,
        content_writer=client,
    )

    artifacts = await WikiPipeline(
        writer=writer,
        config=config,
        llm=WikiLLMRunner(fake_vlm),
    ).run_from_inputs(
        [_wiki_input(doc) for doc in docs],
        content_loader=FakeContentLoader(docs),
    )

    assert artifacts.nodes == []
    assert artifacts.node_contexts == []
    assert len(fake_vlm.calls) == 4
