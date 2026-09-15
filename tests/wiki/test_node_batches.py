import json

import pytest

from openviking.wiki.config import WikiConfig, WikiGenerationLimits
from openviking.wiki.llm import WikiLLMRunner
from openviking.wiki.nodes import AggregationNodeState, NodeDiscoveryRunner
from openviking.wiki.schemas import NodeCard

from .fakes import FakeToolVLM
from .test_nodes import _call, _card


def _create(node_id, *indices):
    return _call("create_node", node_id=node_id, title=node_id, scope=f"Scope of {node_id}",
                 card_ids=[f"card_{index}" for index in indices])


def _state(prompt):
    return json.JSONDecoder().raw_decode(prompt.split("Current state:\n", 1)[1].lstrip())[0]


def _pending(state):
    return [card["card_id"] for card in state["unassigned_cards"]]


async def _run(responses, count, **limits):
    fake = FakeToolVLM(responses)
    runner = NodeDiscoveryRunner(WikiLLMRunner(fake), WikiConfig(limits=WikiGenerationLimits(**limits)))
    result = await runner.discover_layer([_card(i) for i in range(1, count + 1)], depth=1)
    return runner, result, [_state(prompt) for prompt in fake.calls]


def test_batch_defaults():
    assert WikiGenerationLimits().aggregation_batch_size == 25
    assert WikiGenerationLimits().aggregation_agent_max_turns == 40


@pytest.mark.asyncio
async def test_batch_takes_25_new_cards_and_carries_all_leftovers():
    runner, result, states = await _run([[_call("finish")]] * 3, 51)
    assert [len(s["unassigned_cards"]) for s in states] == [25, 50, 51]
    assert [s["has_more_batches"] for s in states] == [True, True, False]
    assert all("batch_index" not in s and "batch_number" not in s for s in states)
    assert [r["batch_index"] for r in runner.aggregation_logs] == [1, 2, 3]
    assert result.nodes == []
    assert result.source_assignments.unassigned_source_ids == [f"card_{i}" for i in range(1, 52)]


@pytest.mark.asyncio
async def test_automatic_exit_runs_after_all_calls_and_last_batch_requires_finish():
    runner, result, states = await _run([
        [_create("topic", *range(1, 23)), _call("update_node_scope", node_id="topic", scope="Corrected scope")],
        [_call("add_cards", node_id="topic", card_ids=[f"card_{i}" for i in range(23, 27)])],
        [_call("finish")],
    ], 26)
    assert [r["batch_end_reason"] for r in runner.aggregation_logs] == ["few_unassigned", None, "finish"]
    assert _pending(states[1]) == ["card_23", "card_24", "card_25", "card_26"]
    assert states[1]["existing_nodes"][0]["scope"] == "Corrected scope"
    assert all("summary" not in c for c in states[1]["existing_nodes"][0]["cards"])
    assert all("summary" in c for c in states[1]["unassigned_cards"])
    assert _pending(states[2]) == []
    assert result.source_assignments.unassigned_source_ids == []


@pytest.mark.asyncio
async def test_exactly_five_pending_does_not_trigger_automatic_exit():
    runner, _, states = await _run([
        [_create("topic", *range(1, 21))], [_call("finish")], [_call("finish")],
    ], 26)
    assert len(states[1]["unassigned_cards"]) == 5
    assert runner.aggregation_logs[0]["batch_end_reason"] is None
    assert [r["batch_index"] for r in runner.aggregation_logs] == [1, 1, 2]


@pytest.mark.asyncio
async def test_read_summaries_persist_deduplicate_and_clear_on_next_batch():
    _, _, states = await _run([
        [_create("topic", 1, 2)],
        [_call("read_summary", card_ids=["card_1", "card_1"])],
        [_call("rename_node", node_id="topic", title="Renamed")],
        [_call("read_summary", card_ids=["card_1"])],
        [_call("finish")], [_call("finish")],
    ], 31)
    expected = [{"card_id": "card_1", "summary": "Summary 1"}]
    assert [s["read_summaries"] for s in states[2:5]] == [expected] * 3
    assert states[5]["read_summaries"] == []
    assert "card_1" not in _pending(states[5]) and "card_2" not in _pending(states[5])
    assert len(states[5]["unassigned_cards"]) == 29


@pytest.mark.asyncio
async def test_previous_batch_member_returns_to_current_pending_and_loses_read_record():
    _, result, states = await _run([
        [_create("topic", *range(1, 23))],
        [_call("read_summary", card_ids=["card_1"])],
        [_call("remove_cards", node_id="topic", card_ids=["card_1"])],
        [_call("finish")],
    ], 26)
    assert states[2]["read_summaries"] == [{"card_id": "card_1", "summary": "Summary 1"}]
    assert states[3]["read_summaries"] == []
    assert _pending(states[3]) == ["card_23", "card_24", "card_25", "card_26", "card_1"]
    assert result.source_assignments.unassigned_source_ids == _pending(states[3])


@pytest.mark.asyncio
async def test_overlap_merge_split_and_duplicate_add_keep_membership_counts_correct():
    _, _, states = await _run([
        [_create("left", 1, 2, 3), _create("right", 1, 4, 5)],
        [_call("read_summary", card_ids=["card_1"])],
        [_call("merge_nodes", target_node_id="left", source_node_ids=["right"])],
        [_call("split_node", node_id="left", groups=[
            {"node_id": "a", "title": "A", "scope": "A", "card_ids": ["card_1", "card_2", "card_3"]},
            {"node_id": "b", "title": "B", "scope": "B", "card_ids": ["card_1", "card_4", "card_5"]},
        ])],
        [_call("add_cards", node_id="a", card_ids=["card_1", "card_1"]),
         _call("remove_cards", node_id="a", card_ids=["card_1"])],
        [_call("remove_cards", node_id="b", card_ids=["card_1"])],
        [_call("finish")],
    ], 6)
    for state in states[2:6]:
        assert _pending(state) == ["card_6"]
        assert state["read_summaries"] == [{"card_id": "card_1", "summary": "Summary 1"}]
    assert _pending(states[6]) == ["card_6", "card_1"]
    assert states[6]["read_summaries"] == []


@pytest.mark.asyncio
async def test_read_record_is_removed_even_if_card_is_reassigned_in_same_turn():
    _, _, states = await _run([
        [_create("topic", 1, 2, 3)],
        [_call("read_summary", card_ids=["card_1"])],
        [_call("remove_cards", node_id="topic", card_ids=["card_1"]),
         _call("add_cards", node_id="topic", card_ids=["card_1"])],
        [_call("finish")],
    ], 3)
    assert states[2]["read_summaries"]
    assert states[3]["read_summaries"] == [] and _pending(states[3]) == []


@pytest.mark.asyncio
async def test_future_pending_and_current_directory_ids_cannot_be_read_as_members():
    runner, _, states = await _run([
        [_create("future", 1, 26), _create("topic", 1, 2),
         _call("read_summary", card_ids=["card_1", "card_3"]),
         _call("read_summary", card_ids=["card_26"]),
         _call("read_summary", card_ids=["topic"])],
        [_call("finish")], [_call("finish")],
    ], 26)
    assert states[1]["read_summaries"] == []
    assert [n["node_id"] for n in states[1]["existing_nodes"]] == ["topic"]
    assert len(runner.aggregation_logs[0]["tool_errors"]) == 4
    assert [c["name"] for c in runner.aggregation_logs[0]["tool_calls"]] == ["create_node"]


@pytest.mark.asyncio
async def test_40_turn_limit_carries_pending_then_ends_last_batch_without_extra_batches(monkeypatch):
    warnings = []
    monkeypatch.setattr("openviking.wiki.nodes.logger.warning", lambda fmt, *args: warnings.append(fmt % args))
    runner, result, states = await _run([[]] * 80, 26)
    assert len(states) == 80
    assert len(states[40]["unassigned_cards"]) == 26
    endings = [r for r in runner.aggregation_logs if r["finished"]]
    assert [(r["batch_index"], r["turn"], r["batch_end_reason"]) for r in endings] == [
        (1, 40, "max_turns"), (2, 40, "max_turns"),
    ]
    assert len(result.source_assignments.unassigned_source_ids) == 26
    assert any("carry into the next batch" in warning for warning in warnings)
    assert any("remain unassigned in this layer" in warning for warning in warnings)
    assert all("consecutive_no_progress" not in r and "made_progress" not in r for r in runner.aggregation_logs)


@pytest.mark.asyncio
async def test_finish_does_not_execute_following_calls_and_limit_keeps_last_turn_edits():
    runner, result, _ = await _run([
        [_call("finish"), _create("ignored", 1, 2)],
    ], 2)
    assert result.nodes == [] and runner.aggregation_logs[0]["batch_end_reason"] == "finish"
    runner, result, _ = await _run([[_create("kept", 1, 2)]], 3, aggregation_agent_max_turns=1)
    assert result.nodes[0].node_id == "kept"
    assert result.source_assignments.unassigned_source_ids == ["card_3"]
    assert runner.aggregation_logs[0]["batch_end_reason"] == "max_turns"


@pytest.mark.asyncio
async def test_upper_layer_members_keep_scope_and_read_the_lower_node_card_summary():
    cards = [NodeCard(doc_id=f"card_{i}", resource_uri=f"viking://wiki/nodes/card_{i}",
                      title=f"Node {i}", scope=f"Scope {i}", summary=f"Summary {i}") for i in [1, 2]]
    fake = FakeToolVLM([[_create("parent", 1, 2)], [_call("read_summary", card_ids=["card_1"])], [_call("finish")]])
    runner = NodeDiscoveryRunner(WikiLLMRunner(fake), WikiConfig())
    await runner.discover_layer(cards, depth=2)
    members = _state(fake.calls[1])["existing_nodes"][0]["cards"]
    assert members[0] == {"card_id": "card_1", "title": "Node 1", "scope": "Scope 1"}
    assert _state(fake.calls[2])["read_summaries"] == [{"card_id": "card_1", "summary": "Summary 1"}]


def test_edit_deltas_do_not_scan_unrelated_nodes_or_all_cards():
    class NoScan(dict):
        def __iter__(self):
            pytest.fail("Edit scanned the whole mapping")

        def values(self):
            pytest.fail("Edit scanned all values")

        def items(self):
            pytest.fail("Edit scanned all items")

    runner = NodeDiscoveryRunner(object(), WikiConfig())
    cards = NoScan({f"card_{i}": _card(i) for i in range(1, 1001)})
    nodes = NoScan({"topic": AggregationNodeState("topic", "Topic", "Scope", ["card_1", "card_2", "card_3"])})
    assert runner._execute_tool_call("remove_cards", {"node_id": "topic", "card_ids": ["card_1"]}, nodes, cards, set()) == {"card_1": -1}
    assert runner._execute_tool_call("add_cards", {"node_id": "topic", "card_ids": ["card_2", "card_4"]}, nodes, cards, set()) == {"card_4": 1}
    assert runner._execute_tool_call("rename_node", {"node_id": "topic", "title": "Renamed"}, nodes, cards, set()) == {}
    assert runner._execute_tool_call("create_node", {
        "node_id": "other", "title": "Other", "scope": "Other scope", "card_ids": ["card_4", "card_5"],
    }, nodes, cards, set()) == {"card_4": 1, "card_5": 1}
    assert runner._execute_tool_call("merge_nodes", {
        "target_node_id": "topic", "source_node_ids": ["other"],
    }, nodes, cards, set()) == {"card_2": 0, "card_3": 0, "card_4": -1, "card_5": 0}
    assert runner._execute_tool_call("split_node", {
        "node_id": "topic", "groups": [
            {"node_id": "a", "title": "A", "scope": "A", "card_ids": ["card_2", "card_3", "card_4"]},
            {"node_id": "b", "title": "B", "scope": "B", "card_ids": ["card_4", "card_5"]},
        ],
    }, nodes, cards, set()) == {"card_2": 0, "card_3": 0, "card_4": 1, "card_5": 0}
