"""Discover one Wiki layer through a full-state tool-calling agent loop."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from .config import WikiConfig
from .llm import WikiLLMRunner, _tool_response_payload
from .prompts import build_node_aggregation_agent_prompt
from .schemas import (
    AddCardsToolArgs,
    AggregationCardView,
    AggregationNodeView,
    CreateNodeToolArgs,
    DocumentCard,
    FinishLayerToolArgs,
    MergeNodesToolArgs,
    NodeCard,
    RemoveCardsToolArgs,
    RenameNodeToolArgs,
    SourceAssignmentItem,
    SourceAssignmentResponse,
    SplitNodeToolArgs,
    UpdateNodeScopeToolArgs,
    WikiNode,
)

logger = logging.getLogger(__name__)
MAX_CONSECUTIVE_NO_PROGRESS_TURNS = 3


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
    """Run a stateful function-calling agent over one complete Wiki layer."""

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
        cards_by_id = {card.doc_id: card for card in cards}
        if len(cards_by_id) != len(cards):
            raise RuntimeError("current aggregation layer contains duplicate card IDs")

        reserved_ids = set(reserved_node_ids or set())
        nodes: dict[str, AggregationNodeState] = {}
        tool_errors: list[str] = []
        consecutive_no_progress = 0
        max_turns = max(1, self.config.limits.aggregation_agent_max_turns)

        for turn in range(1, max_turns + 1):
            prompt = build_node_aggregation_agent_prompt(
                self._node_views(nodes, cards_by_id),
                self._unassigned_card_views(nodes, cards_by_id),
                tool_errors,
            )
            response = await self.llm.complete_tool_calls(
                step="node_aggregation_agent", prompt=prompt, tools=_AGGREGATION_TOOLS
            )
            state_before = self._state_summary(nodes, cards_by_id)
            nodes_before = deepcopy(nodes)
            turn_errors: list[str] = []
            executed_calls: list[dict[str, Any]] = []
            finished = False
            if not response.tool_calls:
                turn_errors.append(
                    f"No structured tool calls were returned (finish_reason={response.finish_reason}). "
                    "This does not finish the layer. Continue editing with function calls, "
                    "or call finish_layer if no useful edit remains."
                )

            for call in response.tool_calls:
                record = {"id": call.id, "name": call.name, "arguments": call.arguments}
                try:
                    if call.name == "finish_layer":
                        FinishLayerToolArgs.model_validate(call.arguments)
                        executed_calls.append(record)
                        finished = True
                        break
                    self._execute_tool_call(
                        call.name, call.arguments, nodes, cards_by_id, reserved_ids
                    )
                    executed_calls.append(record)
                except (RuntimeError, ValidationError, ValueError) as exc:
                    turn_errors.append(f"{call.name}: {exc}")

            made_progress = nodes != nodes_before
            consecutive_no_progress = (
                0 if made_progress or finished else consecutive_no_progress + 1
            )
            if consecutive_no_progress:
                turn_errors.append(
                    f"No node state change for {consecutive_no_progress} consecutive turn(s); "
                    f"the run fails after {MAX_CONSECUTIVE_NO_PROGRESS_TURNS}. "
                    "Make a useful edit or explicitly call finish_layer."
                )
            log_record = {
                "step": "node_aggregation_agent",
                "depth": depth,
                "turn": turn,
                "finish_reason": response.finish_reason,
                "tool_calls": executed_calls,
                "tool_errors": turn_errors,
                "state_before": state_before,
                "state_after": self._state_summary(nodes, cards_by_id),
                "finished": finished,
                "made_progress": made_progress,
                "consecutive_no_progress": consecutive_no_progress,
                # Persist the returned text and all calls, including rejected calls,
                # through the existing per-turn callback even if a later stage fails.
                "response": _tool_response_payload(response),
            }
            self.aggregation_logs.append(log_record)
            if on_turn_complete is not None:
                await on_turn_complete(log_record)
            logger.info(
                "[Wiki] Aggregation agent depth=%d turn=%d nodes=%d unassigned=%d calls=%d finished=%s",
                depth, turn, len(nodes), len(self._unassigned_card_ids(nodes, cards_by_id)),
                len(executed_calls), finished,
            )
            if finished:
                break
            if consecutive_no_progress >= MAX_CONSECUTIVE_NO_PROGRESS_TURNS:
                raise RuntimeError(
                    f"aggregation agent made no progress for {consecutive_no_progress} "
                    f"consecutive turns at depth={depth}; finish_layer was not called"
                )
            tool_errors = turn_errors
        else:
            raise RuntimeError(f"aggregation agent exceeded max turns ({max_turns}) at depth={depth}")

        materialized = self._split_oversized_nodes(
            list(nodes.values()), depth=depth, reserved_node_ids=reserved_ids
        )
        if not materialized:
            return NodeDiscoveryResult([], SourceAssignmentResponse(assignments=[]))
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
                unassigned_source_ids=self._unassigned_card_ids(nodes, cards_by_id),
            ),
        )

    def _execute_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
        nodes: dict[str, AggregationNodeState],
        cards_by_id: dict[str, DocumentCard | NodeCard],
        reserved_ids: set[str],
    ) -> None:
        if name == "create_node":
            args = CreateNodeToolArgs.model_validate(arguments)
            if args.node_id in nodes or args.node_id in reserved_ids:
                raise RuntimeError(f"node_id already exists or is reserved: {args.node_id}")
            nodes[args.node_id] = AggregationNodeState(
                args.node_id, args.title, args.scope,
                self._valid_card_ids(args.card_ids, cards_by_id, minimum=2),
            )
            return
        if name == "add_cards":
            args = AddCardsToolArgs.model_validate(arguments)
            node = self._node(nodes, args.node_id)
            node.card_ids = list(dict.fromkeys([
                *node.card_ids, *self._valid_card_ids(args.card_ids, cards_by_id)
            ]))
            return
        if name == "remove_cards":
            args = RemoveCardsToolArgs.model_validate(arguments)
            node = self._node(nodes, args.node_id)
            removed = set(self._valid_card_ids(args.card_ids, cards_by_id))
            if not removed.issubset(node.card_ids):
                raise RuntimeError("remove_cards may only remove cards assigned to the node")
            remaining = [card_id for card_id in node.card_ids if card_id not in removed]
            if len(remaining) < 2:
                raise RuntimeError("remove_cards may not leave a directory node with fewer than two cards")
            node.card_ids = remaining
            return
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
            target.card_ids = merged_card_ids
            for source_id in source_ids:
                del nodes[source_id]
            return
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
            del nodes[args.node_id]
            nodes.update({group.node_id: group for group in groups})
            return
        if name == "rename_node":
            args = RenameNodeToolArgs.model_validate(arguments)
            self._node(nodes, args.node_id).title = args.title
            return
        if name == "update_node_scope":
            args = UpdateNodeScopeToolArgs.model_validate(arguments)
            self._node(nodes, args.node_id).scope = args.scope
            return
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
    def _card_view(card: DocumentCard | NodeCard) -> AggregationCardView:
        common = {
            "card_id": card.doc_id,
            "title": card.title,
            "summary": card.summary,
        }
        if isinstance(card, NodeCard):
            return AggregationCardView(**common, scope=card.scope)
        return AggregationCardView(**common, candidate_topics=card.candidate_topics)

    def _node_views(
        self,
        nodes: dict[str, AggregationNodeState],
        cards_by_id: dict[str, DocumentCard | NodeCard],
    ) -> list[dict]:
        return [
            AggregationNodeView(
                node_id=node.node_id, title=node.title, scope=node.scope,
                cards=[self._card_view(cards_by_id[card_id]) for card_id in node.card_ids],
            ).model_dump(mode="json", exclude_none=True)
            for node in nodes.values()
        ]

    def _unassigned_card_views(
        self,
        nodes: dict[str, AggregationNodeState],
        cards_by_id: dict[str, DocumentCard | NodeCard],
    ) -> list[AggregationCardView]:
        return [self._card_view(cards_by_id[card_id]) for card_id in self._unassigned_card_ids(nodes, cards_by_id)]

    @staticmethod
    def _unassigned_card_ids(
        nodes: dict[str, AggregationNodeState],
        cards_by_id: dict[str, DocumentCard | NodeCard],
    ) -> list[str]:
        assigned = {card_id for node in nodes.values() for card_id in node.card_ids}
        return [card_id for card_id in cards_by_id if card_id not in assigned]

    def _state_summary(
        self,
        nodes: dict[str, AggregationNodeState],
        cards_by_id: dict[str, DocumentCard | NodeCard],
    ) -> dict[str, Any]:
        return {
            "node_ids": list(nodes), "node_count": len(nodes),
            "unassigned_card_ids": self._unassigned_card_ids(nodes, cards_by_id),
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
    _tool("finish_layer", "Finish the aggregation layer after all useful edits are complete.", FinishLayerToolArgs),
]
