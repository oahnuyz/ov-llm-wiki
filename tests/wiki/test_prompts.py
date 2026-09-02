import pytest

from openviking.prompts.manager import PromptManager
from openviking.wiki.prompts import (
    build_candidate_aggregation_prompt,
    build_document_card_prompt,
    build_node_card_prompt,
    build_node_documents_prompt,
)
from openviking.wiki.schemas import (
    AggregationCardView,
    CandidateMemberView,
    CandidateView,
    DocumentCard,
    NodeDocument,
    ResourceDocument,
    WikiNode,
)


@pytest.mark.parametrize(
    "prompt_id",
    [
        "wiki.document_card",
        "wiki.candidate_aggregation",
        "wiki.node_card",
        "wiki.node_documents",
    ],
)
def test_wiki_prompt_templates_render(prompt_id: str):
    rendered = PromptManager().render(prompt_id, {"input_json": '{"example": true}'})

    assert '{"example": true}' in rendered


def test_wiki_prompt_template_requires_input_json():
    with pytest.raises(ValueError, match="input_json"):
        PromptManager().render("wiki.document_card", {})


def test_document_card_prompt_uses_only_semantic_input_fields():
    prompt = build_document_card_prompt(
        ResourceDocument(
            doc_id="paper_1",
            resource_uri="viking://resources/paper_1/",
            title="Paper 1",
            content_or_structure="semantic content",
            metadata={
                "card_input_mode": "summary",
                "missing_summary_uris": ["viking://resources/missing"],
                "root_uri": "viking://resources/root",
            },
        )
    )

    assert '"content_or_structure": "semantic content"' in prompt
    assert '"card_input_mode": "summary"' in prompt
    assert "paper_1" not in prompt
    assert "Paper 1" not in prompt
    assert '"doc_id"' not in prompt
    assert "root_uri" not in prompt


def test_candidate_aggregation_prompt_contains_full_current_cards_and_compact_history():
    prompt = build_candidate_aggregation_prompt(
        [
            CandidateView(
                candidate_id="candidate_0001",
                title="Topic",
                scope="Topic scope",
                cards=[CandidateMemberView(card_id="old_1", summary="Old summary")],
                status="pending",
            )
        ],
        [
            AggregationCardView(
                card_id="new_1",
                title="New card",
                summary="New summary",
                main_points=["Point"],
                important_terms=["term"],
                candidate_topics=["Topic"],
            )
        ],
    )

    assert '"existing_candidates"' in prompt
    assert '"card_id": "old_1"' in prompt
    assert '"summary": "Old summary"' in prompt
    assert '"card_id": "new_1"' in prompt
    assert '"main_points"' in prompt
    assert '"important_terms"' in prompt
    assert "Return exactly one JSON object with the field operations." in prompt


def test_node_documents_prompt_uses_only_node_boundary_and_source_sections():
    prompt = build_node_documents_prompt(
        WikiNode(
            node_id="question_answering",
            title="Question Answering",
            depth=1,
            scope="QA methods and evaluation.",
        ),
        [
            {
                "source_id": "paper_1",
                "sections": [
                    {
                        "section_uri": "viking://resources/paper_1/abstract",
                        "content": "Question answering evidence.",
                    }
                ],
            }
        ],
    )

    assert '"title": "Question Answering"' in prompt
    assert '"scope": "QA methods and evaluation."' in prompt
    assert '"source_documents"' in prompt
    assert "Question answering evidence." in prompt
    assert '"node_id"' not in prompt
    assert '"depth"' not in prompt
    assert '"source_refs"' not in prompt


def test_node_card_prompt_uses_node_boundary_and_generated_documents():
    prompt = build_node_card_prompt(
        WikiNode(
            node_id="question_answering",
            title="Question Answering",
            depth=2,
            scope="QA methods and evaluation.",
        ),
        [
            NodeDocument(
                document_id="0001",
                title="Retrieval QA",
                content="Retrieval child document.",
            )
        ],
    )

    assert '"title": "Question Answering"' in prompt
    assert '"scope": "QA methods and evaluation."' in prompt
    assert "Retrieval child document." in prompt
    assert '"node_id"' not in prompt
    assert '"document_id"' not in prompt
