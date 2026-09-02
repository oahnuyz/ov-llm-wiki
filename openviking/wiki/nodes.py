"""Discover one Wiki layer through streaming candidate operations."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from pydantic import ValidationError

from .config import WikiConfig
from .llm import WikiLLMRunner
from .prompts import build_candidate_aggregation_prompt
from .schemas import (
    AggregationCardView,
    AssignCardsOperation,
    CandidateMemberView,
    CandidateOperationsResponse,
    CandidateView,
    CreateCandidateOperation,
    DocumentCard,
    MergeCandidatesOperation,
    RenameCandidateOperation,
    SourceAssignmentItem,
    SourceAssignmentResponse,
    SplitCandidateOperation,
    UpdateScopeOperation,
    WikiNode,
)
from .uri import sanitize_node_id

logger = logging.getLogger(__name__)
MAX_VALIDATION_ATTEMPTS = 3
T = TypeVar("T")


@dataclass
class CandidateState:
    candidate_id: str
    title: str
    scope: str
    card_ids: list[str]

    @property
    def status(self) -> str:
        return "pending" if len(set(self.card_ids)) == 1 else "provisional"

    def copy(self) -> "CandidateState":
        return CandidateState(
            candidate_id=self.candidate_id,
            title=self.title,
            scope=self.scope,
            card_ids=list(self.card_ids),
        )


@dataclass(frozen=True)
class NodeDiscoveryResult:
    nodes: list[WikiNode]
    source_assignments: SourceAssignmentResponse


class NodeDiscoveryRunner:
    def __init__(self, llm: WikiLLMRunner, config: WikiConfig):
        self.llm = llm
        self.config = config
        self._next_candidate_number = 1

    async def discover_layer(
        self,
        cards: list[DocumentCard],
        *,
        depth: int,
        reserved_node_ids: set[str] | None = None,
    ) -> NodeDiscoveryResult:
        """Aggregate one layer in sequential card batches and materialize viable nodes."""
        if not cards:
            return NodeDiscoveryResult(
                nodes=[],
                source_assignments=SourceAssignmentResponse(assignments=[]),
            )

        cards_by_id = {card.doc_id: card for card in cards}
        if len(cards_by_id) != len(cards):
            raise RuntimeError("current aggregation layer contains duplicate card IDs")

        self._next_candidate_number = 1
        candidates: dict[str, CandidateState] = {}
        batch_size = max(1, self.config.limits.aggregation_batch_size)
        total_batches = (len(cards) + batch_size - 1) // batch_size
        for start in range(0, len(cards), batch_size):
            batch = cards[start : start + batch_size]
            prompt = build_candidate_aggregation_prompt(
                self._candidate_views(candidates, cards_by_id),
                [self._aggregation_card_view(card) for card in batch],
            )
            candidates = await _complete_with_validation_retry(
                self.llm,
                step="candidate_aggregation",
                prompt=prompt,
                schema=CandidateOperationsResponse.model_json_schema(),
                validate=lambda result, current=candidates, current_batch=batch, final=(start + len(batch) == len(cards)): self._apply_batch_result(
                    result,
                    current,
                    cards_by_id,
                    current_batch,
                    require_reduction=final,
                ),
            )
            logger.info(
                "[Wiki] Aggregated depth=%d batch=%d/%d cards=%d candidates=%d",
                depth,
                start // batch_size + 1,
                total_batches,
                len(batch),
                len(candidates),
            )

        provisional = [candidate for candidate in candidates.values() if candidate.status == "provisional"]
        if not provisional:
            return NodeDiscoveryResult(
                nodes=[],
                source_assignments=SourceAssignmentResponse(assignments=[]),
            )

        nodes = self._build_nodes(
            provisional,
            depth,
            reserved_node_ids=reserved_node_ids or set(),
        )
        assignments = [
            SourceAssignmentItem(
                node_id=node.node_id,
                source_ids=list(dict.fromkeys(candidate.card_ids)),
                support_scope=candidate.scope,
            )
            for node, candidate in zip(nodes, provisional, strict=True)
        ]
        return NodeDiscoveryResult(
            nodes=nodes,
            source_assignments=SourceAssignmentResponse(assignments=assignments),
        )

    def _apply_batch_result(
        self,
        result: dict,
        candidates: dict[str, CandidateState],
        cards_by_id: dict[str, DocumentCard],
        current_batch: list[DocumentCard],
        *,
        require_reduction: bool = False,
    ) -> dict[str, CandidateState]:
        response = CandidateOperationsResponse.model_validate(result)
        working = {candidate_id: candidate.copy() for candidate_id, candidate in candidates.items()}
        local_refs: dict[str, str] = {}

        def resolve_candidate_id(reference: str) -> str:
            if reference in working:
                return reference
            candidate_id = local_refs.get(reference)
            if candidate_id and candidate_id in working:
                return candidate_id
            raise RuntimeError(f"operation references unknown candidate: {reference}")

        def validate_card_ids(card_ids: list[str]) -> list[str]:
            unique_ids = list(dict.fromkeys(card_ids))
            unknown = [card_id for card_id in unique_ids if card_id not in cards_by_id]
            if unknown:
                raise RuntimeError(f"operation references unknown card IDs: {unknown}")
            return unique_ids

        def register_ref(candidate_ref: str, candidate_id: str) -> None:
            if candidate_ref in local_refs or candidate_ref in working or candidate_ref == candidate_id:
                raise RuntimeError(f"duplicate or ambiguous candidate_ref: {candidate_ref}")
            local_refs[candidate_ref] = candidate_id

        for operation in response.operations:
            if isinstance(operation, CreateCandidateOperation):
                card_ids = validate_card_ids(operation.card_ids)
                candidate_id = self._new_candidate_id(working)
                register_ref(operation.candidate_ref, candidate_id)
                working[candidate_id] = CandidateState(
                    candidate_id=candidate_id,
                    title=operation.title,
                    scope=operation.scope,
                    card_ids=card_ids,
                )
                continue

            if isinstance(operation, AssignCardsOperation):
                candidate_id = resolve_candidate_id(operation.candidate_id)
                card_ids = validate_card_ids(operation.card_ids)
                candidate = working[candidate_id]
                candidate.card_ids = list(dict.fromkeys([*candidate.card_ids, *card_ids]))
                continue

            if isinstance(operation, RenameCandidateOperation):
                candidate_id = resolve_candidate_id(operation.candidate_id)
                working[candidate_id].title = operation.title
                continue

            if isinstance(operation, UpdateScopeOperation):
                candidate_id = resolve_candidate_id(operation.candidate_id)
                working[candidate_id].scope = operation.scope
                continue

            if isinstance(operation, MergeCandidatesOperation):
                target_id = resolve_candidate_id(operation.target_candidate_id)
                source_ids = [resolve_candidate_id(item) for item in operation.source_candidate_ids]
                if target_id in source_ids or len(set(source_ids)) != len(source_ids):
                    raise RuntimeError("merge_candidates requires distinct source and target candidates")
                target = working[target_id]
                merged_ids = list(target.card_ids)
                for source_id in source_ids:
                    merged_ids.extend(working[source_id].card_ids)
                target.card_ids = list(dict.fromkeys(merged_ids))
                for source_id in source_ids:
                    del working[source_id]
                continue

            if isinstance(operation, SplitCandidateOperation):
                candidate_id = resolve_candidate_id(operation.candidate_id)
                original_ids = set(working[candidate_id].card_ids)
                group_refs = [group.candidate_ref for group in operation.groups]
                if len(set(group_refs)) != len(group_refs):
                    raise RuntimeError("split_candidate group candidate_ref values must be unique")

                pending_groups: list[tuple[str, CandidateState]] = []
                covered_ids: set[str] = set()
                for group in operation.groups:
                    group_ids = validate_card_ids(group.card_ids)
                    if not set(group_ids).issubset(original_ids):
                        raise RuntimeError("split_candidate groups may only contain original candidate cards")
                    covered_ids.update(group_ids)
                    occupied = {**working, **{state.candidate_id: state for _, state in pending_groups}}
                    new_id = self._new_candidate_id(occupied)
                    pending_groups.append(
                        (
                            group.candidate_ref,
                            CandidateState(
                                candidate_id=new_id,
                                title=group.title,
                                scope=group.scope,
                                card_ids=group_ids,
                            ),
                        )
                    )
                if covered_ids != original_ids:
                    raise RuntimeError("split_candidate must preserve every original card")

                del working[candidate_id]
                for candidate_ref, state in pending_groups:
                    register_ref(candidate_ref, state.candidate_id)
                    working[state.candidate_id] = state
                continue

            raise RuntimeError(f"unsupported candidate operation: {operation.op}")

        assigned_ids = {
            card_id
            for candidate in working.values()
            for card_id in candidate.card_ids
        }
        for card in current_batch:
            if card.doc_id in assigned_ids:
                continue
            candidate_id = self._new_candidate_id(working)
            working[candidate_id] = CandidateState(
                candidate_id=candidate_id,
                title=card.title,
                scope=f"Knowledge specifically covered by {card.title}.",
                card_ids=[card.doc_id],
            )
        if require_reduction:
            provisional_count = sum(
                candidate.status == "provisional" for candidate in working.values()
            )
            if provisional_count >= len(cards_by_id):
                raise RuntimeError(
                    "aggregation must reduce the number of provisional candidates below the input card count"
                )
        return working

    def _new_candidate_id(self, candidates: dict[str, CandidateState]) -> str:
        while True:
            candidate_id = f"candidate_{self._next_candidate_number:04d}"
            self._next_candidate_number += 1
            if candidate_id not in candidates:
                return candidate_id

    @staticmethod
    def _aggregation_card_view(card: DocumentCard) -> AggregationCardView:
        return AggregationCardView(
            card_id=card.doc_id,
            title=card.title,
            summary=card.summary,
            main_points=card.main_points,
            important_terms=card.important_terms,
            candidate_topics=card.candidate_topics,
        )

    @staticmethod
    def _candidate_views(
        candidates: dict[str, CandidateState],
        cards_by_id: dict[str, DocumentCard],
    ) -> list[CandidateView]:
        return [
            CandidateView(
                candidate_id=candidate.candidate_id,
                title=candidate.title,
                scope=candidate.scope,
                cards=[
                    CandidateMemberView(card_id=card_id, summary=cards_by_id[card_id].summary)
                    for card_id in candidate.card_ids
                ],
                status=candidate.status,
            )
            for candidate in candidates.values()
        ]

    @staticmethod
    def _build_nodes(
        candidates: list[CandidateState],
        depth: int,
        *,
        reserved_node_ids: set[str],
    ) -> list[WikiNode]:
        used_ids = set(reserved_node_ids)
        nodes: list[WikiNode] = []
        for candidate in candidates:
            base_id = sanitize_node_id(candidate.title)
            node_id = base_id
            suffix = 2
            while node_id in used_ids:
                node_id = f"{base_id}_{suffix}"
                suffix += 1
            used_ids.add(node_id)
            nodes.append(
                WikiNode(
                    node_id=node_id,
                    title=candidate.title,
                    depth=depth,
                    scope=candidate.scope,
                )
            )
        return nodes


async def _complete_with_validation_retry(
    llm: WikiLLMRunner,
    *,
    step: str,
    prompt: str,
    schema: dict,
    validate: Callable[[dict], T],
) -> T:
    last_error: Exception | None = None
    for attempt in range(1, MAX_VALIDATION_ATTEMPTS + 1):
        try:
            result = await llm.complete_json(
                step=step,
                prompt=prompt,
                schema=schema,
            )
            return validate(result)
        except (RuntimeError, ValidationError) as exc:
            last_error = exc
            if attempt == MAX_VALIDATION_ATTEMPTS:
                break
            logger.info(
                "[Wiki] Retrying %s after validation failure attempt=%d/%d",
                step,
                attempt,
                MAX_VALIDATION_ATTEMPTS,
            )
    assert last_error is not None
    raise last_error
