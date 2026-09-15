import pytest

from openviking.wiki.cards import DocumentCardGenerator
from openviking.wiki.llm import WikiLLMRunner
from openviking.wiki.schemas import NodeDocument, ResourceDocument, WikiNode

from .fakes import FakeVLM


def _card_content_response(index: int) -> dict:
    return {
        "title": f"Readable Paper {index}",
        "summary": f"Paper {index} discusses question answering.",
        "candidate_topics": ["question answering"],
    }


@pytest.mark.asyncio
async def test_document_card_retries_invalid_json_result_with_same_prompt():
    fake_vlm = FakeVLM([None, _card_content_response(1)])
    generator = DocumentCardGenerator(WikiLLMRunner(fake_vlm))

    card = await generator.generate(
        [
            ResourceDocument(
                doc_id="OARW_1",
                resource_uri="viking://resources/OARW_1/",
                title="garbled_7f8d",
                content_or_structure="# Paper 1\n\nContent about question answering.",
            )
        ]
    )

    assert card[0].doc_id == "OARW_1"
    assert card[0].title == "Readable Paper 1"
    assert "# Wiki Card: Readable Paper 1" in card[0].markdown
    assert card[0].summary == "Paper 1 discusses question answering."
    assert len(fake_vlm.calls) == 2
    assert fake_vlm.calls[0] == fake_vlm.calls[1]


@pytest.mark.asyncio
async def test_missing_generated_title_retries_instead_of_using_source_filename():
    missing_title = {"summary": "Evidence", "candidate_topics": ["Topic"]}
    fake_vlm = FakeVLM([missing_title, _card_content_response(1)])
    cards = await DocumentCardGenerator(WikiLLMRunner(fake_vlm)).generate([
        ResourceDocument(doc_id="paper", resource_uri="viking://resources/paper",
                         title="garbled_file_836", content_or_structure="Paper text")
    ])
    assert cards[0].title == "Readable Paper 1"
    assert len(fake_vlm.calls) == 2
    assert set(fake_vlm.schemas[0]["required"]) == {"title", "summary", "candidate_topics"}


@pytest.mark.asyncio
async def test_node_card_uses_wiki_node_uri_and_node_card_step():
    fake_vlm = FakeVLM(
        [
            {
                "summary": "Question answering node synthesis.",
            }
        ]
    )
    generator = DocumentCardGenerator(WikiLLMRunner(fake_vlm))

    card = await generator.generate_node_card(
        WikiNode(
            node_id="question_answering",
            title="Question Answering",
            depth=1,
            scope="QA methods and evaluation.",
        ),
        [NodeDocument(document_id="0001", content="# QA\n\nSynthesized QA knowledge.")],
        resource_uri="viking://wiki/nodes/question_answering/",
    )

    assert card.doc_id == "question_answering"
    assert card.title == "Question Answering"
    assert card.resource_uri == "viking://wiki/nodes/question_answering/"
    assert card.summary == "Question answering node synthesis."
    assert card.scope == "QA methods and evaluation."
    assert "candidate_topics" not in card.model_dump()
    assert set(fake_vlm.schemas[0]["properties"]) == {"summary"}
    assert fake_vlm.calls
