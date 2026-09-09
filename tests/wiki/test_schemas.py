import pytest
from pydantic import ValidationError

from openviking.wiki.schemas import CreateNodeToolArgs, DocumentCard, NodeCard, WikiNode


def test_document_card_requires_candidate_topics():
    with pytest.raises(ValidationError):
        DocumentCard(
            doc_id="OARW_1", resource_uri="viking://resources/OARW_1/", title="Title",
            summary="Summary", candidate_topics=[],
        )


def test_node_card_uses_scope_without_candidate_topics():
    card = NodeCard(
        doc_id="question_answering", resource_uri="viking://wiki/nodes/question_answering/",
        title="Question Answering", summary="Summary", scope="QA methods.",
    )
    assert card.resource_uri == "viking://wiki/nodes/question_answering/"
    assert "candidate_topics" not in card.model_dump()


def test_document_card_rejects_wiki_node_uri():
    with pytest.raises(ValidationError):
        DocumentCard(
            doc_id="question_answering",
            resource_uri="viking://wiki/nodes/question_answering/",
            title="Question Answering",
            summary="Summary",
            candidate_topics=["QA"],
        )


def test_document_card_rejects_removed_redundant_fields():
    with pytest.raises(ValidationError):
        DocumentCard(
            doc_id="paper_1", resource_uri="viking://resources/paper_1/", title="Paper 1",
            summary="Summary", candidate_topics=["Topic"], main_points=["Rejected"],
        )


def test_create_node_tool_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        CreateNodeToolArgs.model_validate({
            "node_id": "topic", "title": "Topic", "scope": "Scope", "card_ids": ["doc_1"], "reason": "extra",
        })


def test_node_id_must_be_snake_case():
    with pytest.raises(ValidationError):
        WikiNode(node_id="Question Answering", title="Question Answering", depth=1, scope="QA papers")
