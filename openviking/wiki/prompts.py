"""Prompt builders for Wiki generation."""

from __future__ import annotations

import json

from openviking.prompts.manager import PromptManager

from .schemas import (
    AggregationCardView,
    NodeDocument,
    ResourceDocument,
    WikiNode,
)

_PROMPT_MANAGER = PromptManager()


def build_document_card_prompt(doc: ResourceDocument) -> str:
    metadata = {
        key: value
        for key, value in (doc.metadata or {}).items()
        if key in {"card_input_mode", "missing_summary_uris"}
    }
    payload = {
        "content_or_structure": doc.content_or_structure,
        "metadata": metadata,
    }
    return _render_wiki_prompt("wiki.document_card", payload)


def build_node_aggregation_agent_prompt(
    existing_nodes: list[dict],
    unassigned_cards: list[AggregationCardView],
    tool_errors: list[str],
) -> str:
    """Build one full-layer tool-calling aggregation agent turn."""
    inputs = {
        "existing_nodes": existing_nodes,
        "unassigned_cards": [
            card.model_dump(mode="json", exclude_none=True) for card in unassigned_cards
        ],
    }
    prompt = _render_wiki_prompt("wiki.node_aggregation_agent", inputs)
    if tool_errors:
        prompt = (
            f"{prompt.rstrip()}\n\nPrevious turn error feedback (tool_errors):\n"
            f"{json.dumps(tool_errors, ensure_ascii=False, indent=2)}\n"
            "The layer is not finished. Correct the errors using the current state above. "
            "Return structured function calls, not explanatory text. If no useful edit "
            "remains, call finish_layer."
        )
    return prompt


def build_node_card_prompt(node: WikiNode, documents: list[NodeDocument]) -> str:
    inputs = {
        "node": node.model_dump(include={"title", "scope"}, mode="json"),
        "documents": [
            document.model_dump(include={"title", "content"}, mode="json")
            for document in documents
        ],
    }
    return _render_wiki_prompt("wiki.node_card", inputs)


def build_node_documents_prompt(
    node: WikiNode,
    source_documents: list[dict],
) -> str:
    node_role = "parent_directory" if node.child_node_ids else "leaf_directory"
    inputs = {
        "node": {
            **node.model_dump(include={"title", "scope"}, mode="json"),
            "role": node_role,
            "child_count": len(node.child_node_ids),
        },
        "source_documents": source_documents,
    }
    return _render_wiki_prompt("wiki.node_documents", inputs)


def _render_wiki_prompt(prompt_id: str, payload: object, **extra_vars: object) -> str:
    variables = {
        "input_json": json.dumps(payload, ensure_ascii=False, indent=2),
        **extra_vars,
    }
    return _PROMPT_MANAGER.render(prompt_id, variables)
