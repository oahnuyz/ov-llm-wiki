"""Generate Wiki node documents."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import TypeVar

from pydantic import ValidationError

from .llm import WikiLLMRunner
from .prompts import build_node_documents_prompt
from .schemas import (
    NodeDocument,
    NodeDocumentContent,
    NodeDocumentsResponse,
    WikiNode,
)

logger = logging.getLogger(__name__)
MAX_VALIDATION_ATTEMPTS = 3
MAX_LOGGED_EXTRA_FIELDS = 5
MAX_LOGGED_FIELD_NAME_CHARS = 80
# A short declared ``content`` paired with undeclared fields is a strong
# signal that a gateway split one intended document across sibling JSON keys.
# Normal short documents without undeclared fields remain valid.
MIN_CONTENT_CHARS_WITH_EXTRA_FIELDS = 300
T = TypeVar("T")


class NodeContentGenerator:
    def __init__(self, llm: WikiLLMRunner):
        self.llm = llm

    async def generate_node_documents(
        self,
        node: WikiNode,
        source_documents: list[dict],
    ) -> list[NodeDocument]:
        prompt = build_node_documents_prompt(
            node,
            source_documents,
        )
        return await _complete_with_validation_retry(
            self.llm,
            step="node_documents",
            prompt=prompt,
            schema=NodeDocumentsResponse.model_json_schema(),
            node_id=node.node_id,
            validate=lambda result: self._parse_node_documents_result(
                node,
                result,
            ),
        )

    def _parse_node_documents_result(
        self,
        node: WikiNode,
        result: dict,
    ) -> list[NodeDocument]:
        response = NodeDocumentsResponse.model_validate(
            _drop_undeclared_node_document_fields(node.node_id, result)
        )
        documents = _build_node_documents(response.documents)
        if not documents:
            raise RuntimeError(f"node_documents for {node.node_id} is empty")
        return documents


def _drop_undeclared_node_document_fields(node_id: str, result: dict) -> dict:
    """Remove undeclared fields and discard individually invalid documents.

    Some OpenAI-compatible gateways advertise JSON Schema support but still
    return extra object properties or incomplete list items. Projecting extra
    properties is safe, and dropping an ordinary invalid item lets the remaining
    valid documents survive. A short item with undeclared fields is treated as a
    likely fragmented structured output and rejects the whole completion so the
    caller retries. A response must still contain at least one complete document.
    """
    documents = result.get("documents")
    if not isinstance(documents, list):
        return result

    top_level_extras = _summarize_extra_fields(set(result) - {"documents"})
    if top_level_extras:
        logger.warning(
            "[Wiki] Dropping undeclared node_documents fields node_id=%s fields=%s",
            node_id,
            top_level_extras,
        )

    projected_documents = []
    allowed_document_fields = {"title", "content"}
    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            logger.warning(
                "[Wiki] Dropping invalid node_documents item "
                "node_id=%s document_index=%d reason=not_an_object",
                node_id,
                index,
            )
            continue
        undeclared_fields = set(document) - allowed_document_fields
        extras = _summarize_extra_fields(undeclared_fields)
        if extras:
            logger.warning(
                "[Wiki] Dropping undeclared node_documents item fields "
                "node_id=%s document_index=%d fields=%s",
                node_id,
                index,
                extras,
            )
        projected = {
            key: value for key, value in document.items() if key in allowed_document_fields
        }
        content = projected.get("content")
        if (
            undeclared_fields
            and isinstance(content, str)
            and len(content.strip()) < MIN_CONTENT_CHARS_WITH_EXTRA_FIELDS
        ):
            logger.warning(
                "[Wiki] Dropping invalid node_documents item "
                "node_id=%s document_index=%d reason=short_content_with_extra_fields "
                "content_chars=%d threshold=%d",
                node_id,
                index,
                len(content.strip()),
                MIN_CONTENT_CHARS_WITH_EXTRA_FIELDS,
            )
            raise RuntimeError(
                f"node_documents for {node_id} contains a likely fragmented "
                f"document at index {index}: content has {len(content.strip())} characters "
                f"(< {MIN_CONTENT_CHARS_WITH_EXTRA_FIELDS}) with undeclared fields {extras}"
            )
        try:
            NodeDocumentContent.model_validate(projected)
        except ValidationError as exc:
            invalid_fields = sorted(
                {
                    str(error["loc"][0])
                    for error in exc.errors()
                    if error.get("loc")
                }
            )
            logger.warning(
                "[Wiki] Dropping invalid node_documents item "
                "node_id=%s document_index=%d fields=%s",
                node_id,
                index,
                invalid_fields,
            )
            continue
        projected_documents.append(projected)
    return {"documents": projected_documents}


def _summarize_extra_fields(fields: set[str]) -> list[str]:
    ordered = sorted(fields)
    summarized = [
        name
        if len(name) <= MAX_LOGGED_FIELD_NAME_CHARS
        else f"{name[:MAX_LOGGED_FIELD_NAME_CHARS]}..."
        for name in ordered[:MAX_LOGGED_EXTRA_FIELDS]
    ]
    omitted = len(ordered) - len(summarized)
    if omitted:
        summarized.append(f"... and {omitted} more")
    return summarized


def _build_node_documents(document_contents: list) -> list[NodeDocument]:
    return [
        NodeDocument.model_validate(
            {
                **document.model_dump(mode="json"),
                "document_id": f"{index:04d}",
            }
        )
        for index, document in enumerate(document_contents, start=1)
    ]


async def _complete_with_validation_retry(
    llm: WikiLLMRunner,
    *,
    step: str,
    prompt: str,
    schema: dict,
    node_id: str,
    validate: Callable[[dict], T],
) -> T:
    last_error: Exception | None = None
    retry_prompt = prompt
    for attempt in range(1, MAX_VALIDATION_ATTEMPTS + 1):
        try:
            result = await llm.complete_json(
                step=step,
                prompt=retry_prompt,
                schema=schema,
            )
            return validate(result)
        except (RuntimeError, ValidationError) as exc:
            last_error = exc
            if attempt == MAX_VALIDATION_ATTEMPTS:
                break
            error = (
                json.dumps(exc.errors(include_input=False, include_url=False), default=str)
                if isinstance(exc, ValidationError)
                else str(exc)
            )
            # Keep the original evidence intact and append only the latest failure.
            retry_prompt = (
                f"{prompt.rstrip()}\n\nPrevious response validation error "
                f"(attempt {attempt}/{MAX_VALIDATION_ATTEMPTS}, node_id={node_id}):\n"
                f"{error[:2000]}\n"
                "Regenerate the complete response as valid JSON matching the supplied schema. "
                "Return at least one complete document. Each document may contain only title "
                "and content; put all Markdown prose inside the content string and correctly "
                "escape quotes and newlines. Do not split prose into extra fields."
            )
            logger.info(
                "[Wiki] Retrying %s for node_id=%s after validation failure attempt=%d/%d",
                step,
                node_id,
                attempt,
                MAX_VALIDATION_ATTEMPTS,
            )
    assert last_error is not None
    raise last_error
