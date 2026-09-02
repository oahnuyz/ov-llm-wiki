import pytest
from pydantic import ValidationError

from openviking.wiki.schemas import (
    CandidateOperationsResponse,
    DocumentCard,
    WikiNode,
)


def test_document_card_requires_candidate_topics():
    with pytest.raises(ValidationError):
        DocumentCard(
            doc_id="OARW_1",
            resource_uri="viking://resources/OARW_1/",
            title="Title",
            summary="Summary",
            main_points=["Point"],
            candidate_topics=[],
        )


def test_document_card_allows_wiki_node_uri():
    card = DocumentCard(
        doc_id="question_answering",
        resource_uri="viking://wiki/nodes/question_answering/",
        title="Question Answering",
        summary="Summary",
        main_points=["Point"],
        candidate_topics=["Parent topic"],
    )

    assert card.resource_uri == "viking://wiki/nodes/question_answering/"


def test_candidate_operations_reject_unknown_fields():
    with pytest.raises(ValidationError):
        CandidateOperationsResponse.model_validate(
            {
                "operations": [
                    {
                        "op": "create_candidate",
                        "candidate_ref": "topic",
                        "title": "Topic",
                        "scope": "Topic scope",
                        "card_ids": ["doc_1"],
                        "reason": "extra",
                    }
                ]
            }
        )


def test_candidate_operations_require_operations_field():
    with pytest.raises(ValidationError):
        CandidateOperationsResponse.model_validate({})


def test_node_id_must_be_snake_case():
    with pytest.raises(ValidationError):
        WikiNode(
            node_id="Question Answering",
            title="Question Answering",
            depth=1,
            scope="QA papers",
        )
