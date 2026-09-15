"""Discover one Wiki layer through sequential batches of tool-calling agent loops."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from .config import WikiConfig
from .llm import WikiLLMRunner, _tool_response_payload
from .prompts import build_node_aggregation_agent_prompt
from .schemas import (
    AddCardsToolArgs,
    AggregationCardView,
    AggregationMemberView,
    AggregationNodeView,
    CreateNodeToolArgs,
    DocumentCard,
    FinishToolArgs,
    MergeNodesToolArgs,
    NodeCard,
    ReadSummaryToolArgs,
    RemoveCardsToolArgs,
    RenameNodeToolArgs,
    SourceAssignmentItem,
    SourceAssignmentResponse,
    SplitNodeToolArgs,
    UpdateNodeScopeToolArgs,
    WikiNode,
)

logger = logging.getLogger(__name__)


@dataclass
class AggregationNodeState:
    node_id: str
    title: str
    scope: str
    card_ids: list[str]


@dataclass(frozen=True)
class NodeDiscoveryResult:
    nodes: list[WikiNode]
    source_assignments: SourceAssignmentResponse


class NodeDiscoveryRunner:
    """Share directory state across independent card-batch agent loops."""

    def __init__(self, llm: WikiLLMRunner, config: WikiConfig):
        self.llm = llm
        self.config = config
        self.aggregation_logs: list[dict[str, Any]] = []

    async def discover_layer(
        self,
        cards: list[DocumentCard | NodeCard],
        *,
        depth: int,
        reserved_node_ids: set[str] | None = None,
        on_turn_complete: Callable[[dict], Awaitable[None]] | None = None,
    ) -> NodeDiscoveryResult:
        if not cards:
            return NodeDiscoveryResult([], SourceAssignmentResponse(assignments=[]))
        if len({card.doc_id for card in cards}) != len(cards):
            raise RuntimeError("current aggregation layer contains duplicate card IDs")

        reserved_ids = set(reserved_node_ids or set())
        nodes: dict[str, AggregationNodeState] = {}
        introduced: dict[str, DocumentCard | NodeCard] = {}
        pending: dict[str, DocumentCard | NodeCard] = {}
        membership_counts: dict[str, int] = {}
        batch_size = max(1, self.config.limits.aggregation_batch_size)
        max_turns = max(1, self.config.limits.aggregation_agent_max_turns)

        for batch_index, start in enumerate(range(0, len(cards), batch_size), start=1):
            new_cards = cards[start : start + batch_size]
            for card in new_cards:
                introduced[card.doc_id] = card
                pending[card.doc_id] = card
                membership_counts[card.doc_id] = 0
            has_more_batches = start + batch_size < len(cards)
            read_summaries: dict[str, str] = {}
            tool_errors: list[str] = []

            for turn in range(1, max_turns + 1):
                prompt = build_node_aggregation_agent_prompt(
                    self._node_views(nodes, introduced),
                    [self._card_view(card) for card in pending.values()],
                    tool_errors,
                    has_more_batches=has_more_batches,
                    read_summaries=read_summaries,
                )
                response = await self.llm.complete_tool_calls(
                    step="node_aggregation_agent", prompt=prompt, tools=_AGGREGATION_TOOLS
                )
                state_before = self._state_summary(nodes, pending)
                turn_errors: list[str] = []
                executed_calls: list[dict[str, Any]] = []
                end_reason: str | None = None
                if not response.tool_calls:
                    turn_errors.append(
                        f"No structured tool calls were returned (finish_reason={response.finish_reason}). "
                        "Continue editing with function calls, or call finish if no useful edit remains."
                    )

                for call in response.tool_calls:
                    record = {"id": call.id, "name": call.name, "arguments": call.arguments}
                    try:
                        if call.name == "finish":
                            FinishToolArgs.model_validate(call.arguments)
                            executed_calls.append(record)
                            end_reason = "finish"
                            break
                        if call.name == "read_summary":
                            args = ReadSummaryToolArgs.model_validate(call.arguments)
                            card_ids = self._valid_card_ids(args.card_ids, introduced)
                            if any(membership_counts[card_id] == 0 for card_id in card_ids):
                                raise RuntimeError("read_summary may only read directory member cards")
                            for card_id in card_ids:
                                read_summaries[card_id] = introduced[card_id].summary
                            record["result"] = [
                                {"card_id": card_id, "summary": read_summaries[card_id]}
                                for card_id in card_ids
                            ]
                        else:
                            deltas = self._execute_tool_call(
                                call.name, call.arguments, nodes, introduced, reserved_ids
                            )
                            # Only cards affected by this successful edit are examined.
                            for card_id, delta in deltas.items():
                                if not delta:
                                    continue
                                membership_counts[card_id] += delta
                                if membership_counts[card_id] == 0:
                                    pending[card_id] = introduced[card_id]
                                    read_summaries.pop(card_id, None)
                                else:
                                    pending.pop(card_id, None)
                        executed_calls.append(record)
                    except (RuntimeError, ValidationError, ValueError) as exc:
                        turn_errors.append(f"{call.name}: {exc}")

                # Evaluate forced exits after executing the whole turn, never mid-edit.
                if end_reason is None:
                    if turn == max_turns:
                        end_reason = "max_turns"
                    elif has_more_batches and len(pending) < 5:
                        end_reason = "few_unassigned"
                log_record = {
                    "step": "node_aggregation_agent",
                    "depth": depth,
                    "batch_index": batch_index,
                    "turn": turn,
                    "has_more_batches": has_more_batches,
                    "finish_reason": response.finish_reason,
                    "tool_calls": executed_calls,
                    "tool_errors": turn_errors,
                    "state_before": state_before,
                    "state_after": self._state_summary(nodes, pending),
                    "finished": end_reason is not None,
                    "batch_end_reason": end_reason,
                    "response": _tool_response_payload(response),
                }
                self.aggregation_logs.append(log_record)
                if on_turn_complete is not None:
                    await on_turn_complete(log_record)
                logger.info(
                    "[Wiki] Aggregation depth=%d batch=%d turn=%d nodes=%d unassigned=%d end=%s",
                    depth, batch_index, turn, len(nodes), len(pending), end_reason,
                )
                if end_reason == "max_turns":
                    logger.warning(
                        "[Wiki] Aggregation depth=%d batch=%d reached %d turns; "
                        "%d unassigned cards %s",
                        depth, batch_index, max_turns, len(pending),
                        "carry into the next batch" if has_more_batches else "remain unassigned in this layer",
                    )
                if end_reason is not None:
                    break
                tool_errors = turn_errors

        materialized = self._split_oversized_nodes(
            list(nodes.values()), depth=depth, reserved_node_ids=reserved_ids
        )
        if not materialized:
            return NodeDiscoveryResult([], SourceAssignmentResponse(
                assignments=[], unassigned_source_ids=list(pending),
            ))
        wiki_nodes = self._build_nodes(materialized, depth, reserved_ids)
        assignments = [
            SourceAssignmentItem(
                node_id=node.node_id,
                source_ids=list(dict.fromkeys(state.card_ids)),
                support_scope=state.scope,
            )
            for node, state in zip(wiki_nodes, materialized, strict=True)
        ]
        return NodeDiscoveryResult(
            nodes=wiki_nodes,
            source_assignments=SourceAssignmentResponse(
                assignments=assignments,
                unassigned_source_ids=list(pending),
            ),
        )

    def _execute_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
        nodes: dict[str, AggregationNodeState],
        cards_by_id: dict[str, DocumentCard | NodeCard],
        reserved_ids: set[str],
    ) -> dict[str, int]:
        """Apply a validated edit and return membership deltas for affected cards only."""
        if name == "create_node":
            args = CreateNodeToolArgs.model_validate(arguments)
            if args.node_id in nodes or args.node_id in reserved_ids:
                raise RuntimeError(f"node_id already exists or is reserved: {args.node_id}")
            nodes[args.node_id] = AggregationNodeState(
                args.node_id, args.title, args.scope,
                self._valid_card_ids(args.card_ids, cards_by_id, minimum=2),
            )
            return {card_id: 1 for card_id in nodes[args.node_id].card_ids}
        if name == "add_cards":
            args = AddCardsToolArgs.model_validate(arguments)
            node = self._node(nodes, args.node_id)
            valid_ids = self._valid_card_ids(args.card_ids, cards_by_id)
            existing = set(node.card_ids)
            added = [card_id for card_id in valid_ids if card_id not in existing]
            node.card_ids.extend(added)
            return {card_id: 1 for card_id in added}
        if name == "remove_cards":
            args = RemoveCardsToolArgs.model_validate(arguments)
            node = self._node(nodes, args.node_id)
            removed_ids = self._valid_card_ids(args.card_ids, cards_by_id)
            removed = set(removed_ids)
            if not removed.issubset(node.card_ids):
                raise RuntimeError("remove_cards may only remove cards assigned to the node")
            remaining = [card_id for card_id in node.card_ids if card_id not in removed]
            if len(remaining) < 2:
                raise RuntimeError("remove_cards may not leave a directory node with fewer than two cards")
            node.card_ids = remaining
            return {card_id: -1 for card_id in removed_ids}
        if name == "merge_nodes":
            args = MergeNodesToolArgs.model_validate(arguments)
            target = self._node(nodes, args.target_node_id)
            source_ids = list(dict.fromkeys(args.source_node_ids))
            if args.target_node_id in source_ids:
                raise RuntimeError("merge_nodes source_node_ids may not contain target_node_id")
            source_nodes = [self._node(nodes, source_id) for source_id in source_ids]
            merged_card_ids = list(dict.fromkeys([
                *target.card_ids,
                *(card_id for source in source_nodes for card_id in source.card_ids),
            ]))
            deltas = Counter(merged_card_ids)
            deltas.subtract(target.card_ids)
            for source in source_nodes:
                deltas.subtract(source.card_ids)
            target.card_ids = merged_card_ids
            for source_id in source_ids:
                del nodes[source_id]
            return dict(deltas)
        if name == "split_node":
            args = SplitNodeToolArgs.model_validate(arguments)
            original = self._node(nodes, args.node_id)
            new_ids = [group.node_id for group in args.groups]
            if len(set(new_ids)) != len(new_ids):
                raise RuntimeError("split_node group node_ids must be unique")
            if any(node_id in nodes and node_id != args.node_id for node_id in new_ids):
                raise RuntimeError("split_node group node_id already exists")
            if any(node_id in reserved_ids for node_id in new_ids):
                raise RuntimeError("split_node group node_id is reserved")
            original_ids = set(original.card_ids)
            groups: list[AggregationNodeState] = []
            covered: set[str] = set()
            for group in args.groups:
                card_ids = self._valid_card_ids(group.card_ids, cards_by_id, minimum=2)
                if not set(card_ids).issubset(original_ids):
                    raise RuntimeError("split_node groups may only contain cards from the original node")
                covered.update(card_ids)
                groups.append(AggregationNodeState(group.node_id, group.title, group.scope, card_ids))
            if covered != original_ids:
                raise RuntimeError("split_node must preserve every original card")
            deltas = Counter(card_id for group in groups for card_id in group.card_ids)
            deltas.subtract(original.card_ids)
            del nodes[args.node_id]
            nodes.update({group.node_id: group for group in groups})
            return dict(deltas)
        if name == "rename_node":
            args = RenameNodeToolArgs.model_validate(arguments)
            self._node(nodes, args.node_id).title = args.title
            return {}
        if name == "update_node_scope":
            args = UpdateNodeScopeToolArgs.model_validate(arguments)
            self._node(nodes, args.node_id).scope = args.scope
            return {}
        raise RuntimeError(f"unsupported aggregation tool: {name}")

    @staticmethod
    def _node(nodes: dict[str, AggregationNodeState], node_id: str) -> AggregationNodeState:
        node = nodes.get(node_id)
        if node is None:
            raise RuntimeError(f"tool references unknown node: {node_id}")
        return node

    @staticmethod
    def _valid_card_ids(
        card_ids: list[str],
        cards_by_id: dict[str, DocumentCard | NodeCard],
        *,
        minimum: int = 1,
    ) -> list[str]:
        unique_ids = list(dict.fromkeys(card_ids))
        unknown = [card_id for card_id in unique_ids if card_id not in cards_by_id]
        if unknown:
            raise RuntimeError(f"tool references unknown card IDs: {unknown}")
        if len(unique_ids) < minimum:
            raise RuntimeError(f"operation requires at least {minimum} distinct cards")
        return unique_ids

    def _split_oversized_nodes(
        self,
        nodes: list[AggregationNodeState],
        *,
        depth: int,
        reserved_node_ids: set[str],
    ) -> list[AggregationNodeState]:
        chunk_size = max(1, int(self.config.limits.max_cards_per_node))
        split: list[AggregationNodeState] = []
        used_ids = {*reserved_node_ids, *(node.node_id for node in nodes)}
        for node in nodes:
            card_ids = list(dict.fromkeys(node.card_ids))
            if len(card_ids) <= chunk_size:
                split.append(node)
                continue
            step, cursor, part = max(1, chunk_size - 1), 0, 1
            while cursor < len(card_ids):
                chunk = card_ids[cursor : cursor + chunk_size]
                node_id = f"{node.node_id}_d{depth}_{part}"
                while node_id in used_ids:
                    part += 1
                    node_id = f"{node.node_id}_d{depth}_{part}"
                used_ids.add(node_id)
                split.append(AggregationNodeState(node_id, f"{node.title}_{part}", node.scope, chunk))
                if cursor + chunk_size >= len(card_ids):
                    break
                cursor += step
                part += 1
        return split

    @staticmethod
    def _build_nodes(
        states: list[AggregationNodeState], depth: int, reserved_node_ids: set[str]
    ) -> list[WikiNode]:
        used = set(reserved_node_ids)
        nodes: list[WikiNode] = []
        for state in states:
            if state.node_id in used:
                raise RuntimeError(f"materialized node_id is reserved: {state.node_id}")
            used.add(state.node_id)
            nodes.append(WikiNode(node_id=state.node_id, title=state.title, depth=depth, scope=state.scope))
        return nodes

    @staticmethod
    def _member_view(card: DocumentCard | NodeCard) -> AggregationMemberView:
        common = {
            "card_id": card.doc_id,
            "title": card.title,
        }
        if isinstance(card, NodeCard):
            return AggregationMemberView(**common, scope=card.scope)
        return AggregationMemberView(**common, candidate_topics=card.candidate_topics)

    @classmethod
    def _card_view(cls, card: DocumentCard | NodeCard) -> AggregationCardView:
        return AggregationCardView(
            **cls._member_view(card).model_dump(exclude_none=True), summary=card.summary
        )

    def _node_views(
        self,
        nodes: dict[str, AggregationNodeState],
        cards_by_id: dict[str, DocumentCard | NodeCard],
    ) -> list[dict]:
        return [
            AggregationNodeView(
                node_id=node.node_id, title=node.title, scope=node.scope,
                cards=[self._member_view(cards_by_id[card_id]) for card_id in node.card_ids],
            ).model_dump(mode="json", exclude_none=True)
            for node in nodes.values()
        ]

    @staticmethod
    def _state_summary(
        nodes: dict[str, AggregationNodeState],
        pending: dict[str, DocumentCard | NodeCard],
    ) -> dict[str, Any]:
        return {
            "node_ids": list(nodes), "node_count": len(nodes),
            "unassigned_card_ids": list(pending),
        }


def _tool(name: str, description: str, model: type) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": model.model_json_schema(),
        },
    }


_AGGREGATION_TOOLS = [
    _tool("create_node", "Create a coherent directory node from two or more cards.", CreateNodeToolArgs),
    _tool("add_cards", "Add cards to an existing directory node; this may create DAG overlap.", AddCardsToolArgs),
    _tool("remove_cards", "Remove cards while keeping at least two in the node.", RemoveCardsToolArgs),
    _tool("merge_nodes", "Merge existing directory nodes into the target node.", MergeNodesToolArgs),
    _tool("split_node", "Replace one mixed directory node with coherent nodes.", SplitNodeToolArgs),
    _tool("rename_node", "Rename an existing directory node.", RenameNodeToolArgs),
    _tool("update_node_scope", "Update an existing directory node scope.", UpdateNodeScopeToolArgs),
    _tool("read_summary", "Read summaries of existing directory member cards in this layer.", ReadSummaryToolArgs),
    _tool("finish", "Finish this batch; leave unclear cards unassigned. Call alone.", FinishToolArgs),
]
