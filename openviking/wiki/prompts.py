"""Prompt builders for Wiki generation."""

from __future__ import annotations

import json

from openviking.prompts.manager import PromptManager

from .schemas import (
    AggregationCardView,
    CandidateView,
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


def build_candidate_aggregation_prompt(
    existing_candidates: list[CandidateView],
    current_batch_cards: list[AggregationCardView],
) -> str:
    """Build the strict ordered-operation prompt for one aggregation batch."""
    inputs = {
        "existing_candidates": [candidate.model_dump(mode="json") for candidate in existing_candidates],
        "current_batch_cards": [card.model_dump(mode="json") for card in current_batch_cards],
    }
    return _render_wiki_prompt("wiki.candidate_aggregation", inputs)


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
    inputs = {
        "node": node.model_dump(include={"title", "scope"}, mode="json"),
        "source_documents": source_documents,
    }
    return _render_wiki_prompt("wiki.node_documents", inputs)


def _render_wiki_prompt(prompt_id: str, payload: object, **extra_vars: object) -> str:
    variables = {
        "input_json": json.dumps(payload, ensure_ascii=False, indent=2),
        **extra_vars,
    }
    return _PROMPT_MANAGER.render(prompt_id, variables)
