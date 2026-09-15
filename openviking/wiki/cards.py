"""Generate Wiki source cards."""

from __future__ import annotations

import asyncio
import logging

from pydantic import ValidationError

from .llm import WikiLLMRunner
from .prompts import build_document_card_prompt, build_node_card_prompt
from .schemas import (
    DocumentCard,
    DocumentCardContent,
    NodeCard,
    NodeCardContent,
    NodeDocument,
    ResourceDocument,
    WikiNode,
)

logger = logging.getLogger(__name__)


class DocumentCardGenerator:
    def __init__(self, llm: WikiLLMRunner, max_concurrent: int = 10):
        self.llm = llm
        self.max_concurrent = max(1, max_concurrent)

    async def generate(self, docs: list[ResourceDocument]) -> list[DocumentCard]:
        sem = asyncio.Semaphore(self.max_concurrent)
        cards: list[DocumentCard | None] = [None] * len(docs)

        async def _generate_card_at_index(index: int, doc: ResourceDocument) -> None:
            async with sem:
                try:
                    cards[index] = await self._generate_card(doc)
                except Exception as exc:
                    raise RuntimeError(
                        f"Document card failed: {doc.doc_id} ({doc.resource_uri}), "
                        f"input_chars={len(doc.content_or_structure)}: {exc}"
                    ) from exc

        await asyncio.gather(
            *[_generate_card_at_index(index, doc) for index, doc in enumerate(docs)]
        )
        if any(card is None for card in cards):
            raise RuntimeError("document card generation did not produce all cards")
        return [card for card in cards if card is not None]

    async def _generate_card(self, doc: ResourceDocument) -> DocumentCard:
        prompt = build_document_card_prompt(doc)
        return await self._generate_card_from_prompt(
            prompt=prompt,
            step="doc_card",
            retry_step="doc_card_retry",
            doc_id=doc.doc_id,
            resource_uri=doc.resource_uri,
        )

    async def generate_node_card(
        self,
        node: WikiNode,
        documents: list[NodeDocument],
        *,
        resource_uri: str,
    ) -> NodeCard:
        prompt = build_node_card_prompt(node, documents)
        content = await self._complete_card_content(
            prompt=prompt,
            step="node_card",
            retry_step="node_card_retry",
            doc_id=node.node_id,
            content_model=NodeCardContent,
        )
        card = NodeCard.model_validate(
            {
                **content.model_dump(mode="json"),
                "doc_id": node.node_id,
                "resource_uri": resource_uri,
                "title": node.title,
                "scope": node.scope,
            }
        )
        return card.model_copy(update={"markdown": render_card_markdown(card)})

    async def _generate_card_from_prompt(
        self,
        *,
        prompt: str,
        step: str,
        retry_step: str,
        doc_id: str,
        resource_uri: str,
    ) -> DocumentCard:
        content = await self._complete_card_content(
            prompt=prompt,
            step=step,
            retry_step=retry_step,
            doc_id=doc_id,
            content_model=DocumentCardContent,
        )
        card = DocumentCard.model_validate(
            {
                **content.model_dump(mode="json"),
                "doc_id": doc_id,
                "resource_uri": resource_uri,
            }
        )
        if not card.markdown:
            card = card.model_copy(update={"markdown": render_card_markdown(card)})
        return card

    async def _complete_card_content(
        self,
        *,
        prompt: str,
        step: str,
        retry_step: str,
        doc_id: str,
        content_model: type[DocumentCardContent] | type[NodeCardContent],
    ) -> DocumentCardContent | NodeCardContent:
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                result = await self.llm.complete_json(
                    step=step if attempt == 1 else retry_step,
                    prompt=prompt,
                    schema=content_model.model_json_schema(),
                )
                return content_model.model_validate(result)
            except (RuntimeError, ValidationError) as exc:
                last_error = exc
                if attempt == 3:
                    raise
                logger.info(
                    "[Wiki] Retrying card generation for doc_id=%s step=%s attempt=%d/3",
                    doc_id,
                    step,
                    attempt,
                )
        else:
            assert last_error is not None
            raise last_error


def render_card_markdown(card: DocumentCard | NodeCard) -> str:
    if isinstance(card, NodeCard):
        return f"""# Wiki Card: {card.title}

## Source Info

- Source URI: {card.resource_uri}

## Scope

{card.scope}

## Summary

{card.summary}
"""
    topics = "\n".join(f"- {item}" for item in card.candidate_topics)
    return f"""# Wiki Card: {card.title}

## Source Info

- Source URI: {card.resource_uri}

## Summary

{card.summary}

## Candidate Wiki Topics

{topics}
"""
