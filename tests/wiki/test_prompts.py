import pytest

from openviking.prompts.manager import PromptManager
from openviking.wiki.prompts import (
    build_document_card_prompt,
    build_node_aggregation_agent_prompt,
    build_node_card_prompt,
    build_node_documents_prompt,
)
from openviking.wiki.schemas import AggregationCardView, NodeDocument, ResourceDocument, WikiNode


@pytest.mark.parametrize("prompt_id", [
    "wiki.document_card", "wiki.node_aggregation_agent", "wiki.node_card", "wiki.node_documents",
])
def test_wiki_prompt_templates_render(prompt_id: str):
    assert '{"example": true}' in PromptManager().render(
        prompt_id, {"input_json": '{"example": true}', "has_more_batches": False}
    )


def test_document_card_prompt_uses_only_semantic_input_fields():
    prompt = build_document_card_prompt(ResourceDocument(
        doc_id="paper_1", resource_uri="viking://resources/paper_1/", title="Paper 1",
        content_or_structure="semantic content", metadata={"card_input_mode": "summary", "root_uri": "hidden"},
    ))
    assert '"content_or_structure": "semantic content"' in prompt
    assert '"card_input_mode": "summary"' in prompt
    assert "paper_1" not in prompt
    assert "root_uri" not in prompt
    assert "compact evidence units" in prompt
    assert "representative named work" in prompt


def test_aggregation_agent_prompt_contains_compact_nodes_and_full_unassigned_cards():
    prompt = build_node_aggregation_agent_prompt(
        [{"node_id": "topic", "title": "Topic", "scope": "Topic scope", "cards": [{
            "card_id": "old_1", "title": "Old", "candidate_topics": ["Old topic"],
        }]}],
        [AggregationCardView(card_id="new_1", title="New", summary="New summary", candidate_topics=["Topic"])],
        ["create_node: invalid node_id"],
        has_more_batches=True, read_summaries={},
    )
    assert '"existing_nodes"' in prompt
    assert '"unassigned_cards"' in prompt
    assert '"old_1"' in prompt and '"new_1"' in prompt
    assert "calls execute in order" in prompt
    assert "finish" in prompt
    assert "Coherence test" in prompt
    assert "More new cards will arrive later" in prompt
    assert "There are no later batches" not in prompt
    assert "drug delivery with biopharmaceutical manufacturing" in prompt
    state, feedback = prompt.split("Previous turn error feedback (tool_errors):")
    assert "create_node: invalid node_id" not in state
    assert "create_node: invalid node_id" in feedback
    assert prompt.endswith("call finish.")


def test_aggregation_prompt_omits_error_feedback_when_no_errors():
    prompt = build_node_aggregation_agent_prompt([], [], [], has_more_batches=False, read_summaries={})
    assert "Previous turn error feedback" not in prompt
    assert "There are no later batches" in prompt
    assert "More new cards will arrive later" not in prompt


def test_node_documents_prompt_uses_only_node_boundary_and_source_sections():
    prompt = build_node_documents_prompt(
        WikiNode(node_id="question_answering", title="Question Answering", depth=1, scope="QA methods."),
        [{"source_id": "paper_1", "sections": [{"section_uri": "viking://resources/paper_1/abstract", "content": "QA evidence."}]}],
    )
    assert '"title": "Question Answering"' in prompt
    assert '"role": "leaf_directory"' in prompt
    assert '"child_count": 0' in prompt
    assert "QA evidence." in prompt
    assert '"node_id"' not in prompt
    assert "detailed scientific synthesis" in prompt
    assert "Prefer complete coverage over" in prompt


def test_node_documents_prompt_marks_parent_directory_authoritatively():
    prompt = build_node_documents_prompt(
        WikiNode(
            node_id="language_systems",
            title="Language Systems",
            depth=2,
            scope="Language system methods.",
            child_node_ids=["question_answering"],
        ),
        [{"source_id": "question_answering", "sections": [{"content": "Child synthesis."}]}],
    )
    assert '"role": "parent_directory"' in prompt
    assert '"child_count": 1' in prompt
    assert "navigation and synthesis document" in prompt


def test_node_card_prompt_uses_node_boundary_and_generated_documents():
    prompt = build_node_card_prompt(
        WikiNode(node_id="question_answering", title="Question Answering", depth=2, scope="QA methods."),
        [NodeDocument(document_id="0001", title="Retrieval QA", content="Retrieved evidence.")],
    )
    assert "Retrieved evidence." in prompt
    assert '"document_id"' not in prompt
    assert '"role"' not in prompt
    assert "JSON object with the field summary" in prompt


def test_node_card_prompt_is_independent_of_directory_role():
    leaf_prompt = build_node_card_prompt(
        WikiNode(node_id="language_system", title="Language System", depth=1, scope="Methods."),
        [NodeDocument(document_id="0001", content="Knowledge.")],
    )
    parent_prompt = build_node_card_prompt(
        WikiNode(
            node_id="language_system",
            title="Language System",
            depth=2,
            scope="Methods.",
            child_node_ids=["question_answering"],
        ),
        [NodeDocument(document_id="0001", content="Knowledge.")],
    )
    assert leaf_prompt == parent_prompt
