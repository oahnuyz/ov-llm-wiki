import pytest

from openviking.models.vlm.base import ToolCall, VLMResponse
from openviking.wiki.config import WikiConfig, WikiGenerationLimits
from openviking.wiki.llm import WikiLLMRunner
from openviking.wiki.nodes import NodeDiscoveryRunner
from openviking.wiki.schemas import DocumentCard

from .fakes import FakeToolVLM


def _card(index: int) -> DocumentCard:
    return DocumentCard(
        doc_id=f"card_{index}",
        resource_uri=f"viking://resources/card_{index}",
        title=f"Card {index}",
        summary=f"Summary {index}",
        candidate_topics=["Topic"],
    )


def _call(name: str, **arguments: object) -> dict:
    return {"name": name, "arguments": arguments}


def test_default_aggregation_agent_turn_limit_is_fifty():
    assert WikiGenerationLimits().aggregation_agent_max_turns == 50


def test_default_aggregation_agent_max_tokens_is_12288():
    assert WikiGenerationLimits().aggregation_agent_max_tokens == 12288


def test_default_max_cards_per_node_is_one_thousand():
    assert WikiGenerationLimits().max_cards_per_node == 1000


@pytest.mark.asyncio
async def test_agent_receives_full_layer_then_materializes_created_nodes():
    fake_vlm = FakeToolVLM(
        [
            [_call("create_node", node_id="shared_topic", title="Shared Topic", scope="Shared scope", card_ids=["card_1", "card_2"])],
            [_call("finish_layer")],
        ]
    )
    runner = NodeDiscoveryRunner(WikiLLMRunner(fake_vlm), WikiConfig())

    result = await runner.discover_layer([_card(1), _card(2), _card(3)], depth=1)

    assert len(fake_vlm.calls) == 2
    assert fake_vlm.calls[0].count('"card_id"') == 3
    assert result.nodes[0].node_id == "shared_topic"
    assert result.source_assignments.assignments[0].source_ids == ["card_1", "card_2"]
    assert result.source_assignments.unassigned_source_ids == ["card_3"]


@pytest.mark.asyncio
async def test_agent_may_finish_without_forcing_any_aggregation():
    fake_vlm = FakeToolVLM([[_call("finish_layer")]])
    runner = NodeDiscoveryRunner(WikiLLMRunner(fake_vlm), WikiConfig())

    result = await runner.discover_layer([_card(1), _card(2)], depth=1)

    assert result.nodes == []
    assert result.source_assignments.assignments == []


@pytest.mark.asyncio
async def test_sequential_tool_calls_preserve_dag_overlap():
    fake_vlm = FakeToolVLM(
        [[
            _call("create_node", node_id="topic_a", title="Topic A", scope="A scope", card_ids=["card_1", "card_2"]),
            _call("create_node", node_id="topic_b", title="Topic B", scope="B scope", card_ids=["card_1", "card_3"]),
        ], [_call("finish_layer")]]
    )
    runner = NodeDiscoveryRunner(WikiLLMRunner(fake_vlm), WikiConfig())

    result = await runner.discover_layer([_card(1), _card(2), _card(3)], depth=1)

    assert [item.source_ids for item in result.source_assignments.assignments] == [
        ["card_1", "card_2"], ["card_1", "card_3"]
    ]


@pytest.mark.asyncio
async def test_invalid_tool_call_is_returned_to_agent_for_correction():
    fake_vlm = FakeToolVLM(
        [
            [_call("add_cards", node_id="missing", card_ids=["card_1"])],
            [_call("create_node", node_id="topic", title="Topic", scope="Topic scope", card_ids=["card_1", "card_2"])],
            [_call("finish_layer")],
        ]
    )
    runner = NodeDiscoveryRunner(WikiLLMRunner(fake_vlm), WikiConfig())

    result = await runner.discover_layer([_card(1), _card(2)], depth=1)

    assert result.nodes[0].node_id == "topic"
    assert "unknown node" in fake_vlm.calls[1]
    assert runner.aggregation_logs[0]["tool_errors"]


@pytest.mark.asyncio
async def test_invalid_merge_does_not_partially_change_nodes():
    fake_vlm = FakeToolVLM(
        [[
            _call("create_node", node_id="left", title="Left", scope="Left scope", card_ids=["card_1", "card_2"]),
            _call("create_node", node_id="right", title="Right", scope="Right scope", card_ids=["card_3", "card_4"]),
        ], [
            _call("merge_nodes", target_node_id="left", source_node_ids=["right", "missing"]),
        ], [_call("finish_layer")]]
    )
    runner = NodeDiscoveryRunner(WikiLLMRunner(fake_vlm), WikiConfig())

    result = await runner.discover_layer([_card(index) for index in range(1, 5)], depth=1)

    assert [item.node_id for item in result.source_assignments.assignments] == ["left", "right"]
    assert [item.source_ids for item in result.source_assignments.assignments] == [
        ["card_1", "card_2"], ["card_3", "card_4"]
    ]
    assert "unknown node" in fake_vlm.calls[2]


@pytest.mark.asyncio
async def test_oversized_nodes_keep_existing_sliding_window_materialization():
    fake_vlm = FakeToolVLM(
        [[
            _call("create_node", node_id="topic", title="Topic", scope="Topic scope", card_ids=["card_1", "card_2", "card_3", "card_4"]),
        ], [_call("finish_layer")]]
    )
    runner = NodeDiscoveryRunner(
        WikiLLMRunner(fake_vlm), WikiConfig(limits=WikiGenerationLimits(max_cards_per_node=3))
    )

    result = await runner.discover_layer([_card(index) for index in range(1, 5)], depth=1)

    assert [node.node_id for node in result.nodes] == ["topic_d1_1", "topic_d1_2"]
    assert [item.source_ids for item in result.source_assignments.assignments] == [
        ["card_1", "card_2", "card_3"], ["card_3", "card_4"]
    ]


@pytest.mark.asyncio
async def test_split_ids_are_unique_across_depths_when_topic_names_repeat():
    fake_vlm = FakeToolVLM(
        [[
            _call("create_node", node_id="topic", title="Topic", scope="Topic scope", card_ids=["card_1", "card_2", "card_3", "card_4"]),
        ], [_call("finish_layer")]]
    )
    runner = NodeDiscoveryRunner(
        WikiLLMRunner(fake_vlm), WikiConfig(limits=WikiGenerationLimits(max_cards_per_node=3))
    )

    result = await runner.discover_layer(
        [_card(index) for index in range(1, 5)],
        depth=2,
        reserved_node_ids={"topic_d1_1", "topic_d1_2"},
    )

    assert [node.node_id for node in result.nodes] == ["topic_d2_1", "topic_d2_2"]


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason,content", [("length", "Long explanation"), ("stop", None)])
async def test_zero_calls_retry_with_feedback_at_prompt_end(finish_reason, content):
    fake_vlm = FakeToolVLM([
        VLMResponse(content=content, finish_reason=finish_reason),
        [_call("create_node", node_id="topic", title="Topic", scope="Scope", card_ids=["card_1", "card_2"])],
        [_call("finish_layer")],
    ])
    runner = NodeDiscoveryRunner(WikiLLMRunner(fake_vlm), WikiConfig())
    result = await runner.discover_layer([_card(1), _card(2)], depth=2)
    assert result.nodes[0].node_id == "topic"
    assert runner.aggregation_logs[0]["finished"] is False
    assert runner.aggregation_logs[0]["response"]["content"] == content
    feedback = fake_vlm.calls[1].split("Previous turn error feedback (tool_errors):")[1]
    assert f"finish_reason={finish_reason}" in feedback
    assert "No structured tool calls" in feedback
    assert fake_vlm.calls[1].endswith("call finish_layer.")
    assert "Previous turn error feedback" not in fake_vlm.calls[2]


@pytest.mark.asyncio
async def test_length_response_preserves_valid_calls_and_logs_rejected_calls():
    fake_vlm = FakeToolVLM([
        VLMResponse(
            content="Partial response",
            finish_reason="length",
            usage={"completion_tokens": 12288},
            tool_calls=[
                ToolCall("ok", "create_node", dict(node_id="topic", title="Topic", scope="Scope", card_ids=["card_1", "card_2"])),
                ToolCall("bad", "create_node", {"raw": '{"node_id":'}),
            ],
        ),
        [_call("finish_layer")],
    ])
    runner = NodeDiscoveryRunner(WikiLLMRunner(fake_vlm), WikiConfig())
    result = await runner.discover_layer([_card(1), _card(2)], depth=1)
    assert result.nodes[0].node_id == "topic"
    first = runner.aggregation_logs[0]
    assert first["finished"] is False and first["made_progress"] is True
    assert len(first["tool_calls"]) == 1
    assert len(first["response"]["tool_calls"]) == 2
    assert first["response"]["usage"]["completion_tokens"] == 12288


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [[], [_call("finish_layer", unexpected=True)]])
async def test_three_no_progress_turns_fail_after_last_turn_is_persisted(response):
    fake_vlm = FakeToolVLM([response] * 3)
    runner = NodeDiscoveryRunner(WikiLLMRunner(fake_vlm), WikiConfig())
    persisted = []

    async def save(record):
        persisted.append(record)

    with pytest.raises(RuntimeError, match="no progress for 3 consecutive turns at depth=2"):
        await runner.discover_layer([_card(1), _card(2)], depth=2, on_turn_complete=save)
    assert len(fake_vlm.calls) == len(persisted) == 3
    assert [r["consecutive_no_progress"] for r in persisted] == [1, 2, 3]
    assert all(not r["finished"] for r in persisted)


@pytest.mark.asyncio
async def test_scope_edits_reset_stall_counter_but_successful_noops_do_not():
    create = _call("create_node", node_id="topic", title="Topic", scope="Scope", card_ids=["card_1", "card_2"])
    change = _call("update_node_scope", node_id="topic", scope="Revised scope")
    fake_vlm = FakeToolVLM([[create], [], [], [change], [change], [change], [_call("finish_layer")]])
    runner = NodeDiscoveryRunner(WikiLLMRunner(fake_vlm), WikiConfig())
    result = await runner.discover_layer([_card(1), _card(2)], depth=1)
    assert result.nodes[0].scope == "Revised scope"
    assert [r["consecutive_no_progress"] for r in runner.aggregation_logs] == [0, 1, 2, 0, 1, 2, 0]


@pytest.mark.asyncio
async def test_successful_noop_calls_cannot_avoid_stall_limit():
    create = _call("create_node", node_id="topic", title="Topic", scope="Scope", card_ids=["card_1", "card_2"])
    noop = _call("add_cards", node_id="topic", card_ids=["card_1"])
    fake_vlm = FakeToolVLM([[create], [noop], [noop], [noop]])
    runner = NodeDiscoveryRunner(WikiLLMRunner(fake_vlm), WikiConfig())
    with pytest.raises(RuntimeError, match="no progress for 3 consecutive turns"):
        await runner.discover_layer([_card(1), _card(2)], depth=1)
    assert runner.aggregation_logs[-1]["consecutive_no_progress"] == 3
