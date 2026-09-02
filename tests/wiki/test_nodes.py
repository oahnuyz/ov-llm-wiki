import pytest

from openviking.wiki.config import WikiConfig, WikiGenerationLimits
from openviking.wiki.llm import WikiLLMRunner
from openviking.wiki.nodes import NodeDiscoveryRunner
from openviking.wiki.schemas import DocumentCard

from .fakes import FakeVLM


def _card(index: int) -> DocumentCard:
    return DocumentCard(
        doc_id=f"card_{index}",
        resource_uri=f"viking://resources/card_{index}",
        title=f"Card {index}",
        summary=f"Summary {index}",
        main_points=[f"Point {index}"],
        important_terms=[f"term_{index}"],
        candidate_topics=["Topic"],
    )


@pytest.mark.asyncio
async def test_streaming_aggregation_processes_batches_in_order_and_keeps_only_provisional_nodes():
    cards = [_card(index) for index in range(1, 17)]
    fake_vlm = FakeVLM(
        [
            {
                "operations": [
                    {
                        "op": "create_candidate",
                        "candidate_ref": "topic",
                        "title": "Topic",
                        "scope": "Topic scope",
                        "card_ids": [f"card_{index}" for index in range(1, 3)],
                    }
                ]
            },
            {
                "operations": [
                    {
                        "op": "assign_cards",
                        "candidate_id": "candidate_0001",
                        "card_ids": ["card_16"],
                    }
                ]
            },
        ]
    )
    runner = NodeDiscoveryRunner(
        WikiLLMRunner(fake_vlm),
        WikiConfig(limits=WikiGenerationLimits(aggregation_batch_size=15)),
    )

    result = await runner.discover_layer(cards, depth=1)

    assert len(fake_vlm.calls) == 2
    assert len(result.nodes) == 1
    assert result.nodes[0].title == "Topic"
    assert result.source_assignments.assignments[0].source_ids == ["card_1", "card_2", "card_16"]
    assert fake_vlm.calls[0].count('"card_id"') == 15
    assert fake_vlm.calls[1].count('"card_id"') == 16


@pytest.mark.asyncio
async def test_all_pending_candidates_stop_without_materializing_nodes():
    fake_vlm = FakeVLM([{"operations": []}])
    runner = NodeDiscoveryRunner(WikiLLMRunner(fake_vlm), WikiConfig())

    result = await runner.discover_layer([_card(1)], depth=1)

    assert result.nodes == []
    assert result.source_assignments.assignments == []


@pytest.mark.asyncio
async def test_invalid_operation_retries_with_same_prompt():
    fake_vlm = FakeVLM(
        [
            {"operations": [{"op": "assign_cards", "candidate_id": "unknown", "card_ids": ["card_1"]}]},
            {
                "operations": [
                    {
                        "op": "create_candidate",
                        "candidate_ref": "topic",
                        "title": "Topic",
                        "scope": "Topic scope",
                        "card_ids": ["card_1", "card_2"],
                    }
                ]
            },
        ]
    )
    runner = NodeDiscoveryRunner(WikiLLMRunner(fake_vlm), WikiConfig())

    result = await runner.discover_layer([_card(1), _card(2)], depth=1)

    assert len(fake_vlm.calls) == 2
    assert fake_vlm.calls[0] == fake_vlm.calls[1]
    assert result.nodes[0].title == "Topic"


def test_split_and_merge_preserve_card_membership():
    cards = [_card(index) for index in range(1, 4)]
    by_id = {card.doc_id: card for card in cards}
    runner = NodeDiscoveryRunner(WikiLLMRunner(FakeVLM([])), WikiConfig())
    state = runner._apply_batch_result(
        {
            "operations": [
                {
                    "op": "create_candidate",
                    "candidate_ref": "a",
                    "title": "A",
                    "scope": "A scope",
                    "card_ids": ["card_1", "card_2"],
                },
                {
                    "op": "create_candidate",
                    "candidate_ref": "b",
                    "title": "B",
                    "scope": "B scope",
                    "card_ids": ["card_3"],
                },
                {
                    "op": "merge_candidates",
                    "target_candidate_id": "candidate_0001",
                    "source_candidate_ids": ["candidate_0002"],
                },
            ]
        },
        {},
        by_id,
        cards,
    )
    state = runner._apply_batch_result(
        {
            "operations": [
                {
                    "op": "split_candidate",
                    "candidate_id": "candidate_0001",
                    "groups": [
                        {
                            "candidate_ref": "x",
                            "title": "X",
                            "scope": "X scope",
                            "card_ids": ["card_1", "card_3"],
                        },
                        {
                            "candidate_ref": "y",
                            "title": "Y",
                            "scope": "Y scope",
                            "card_ids": ["card_2", "card_3"],
                        },
                    ],
                }
            ]
        },
        state,
        by_id,
        [],
    )

    assert sorted(card for candidate in state.values() for card in candidate.card_ids) == [
        "card_1",
        "card_2",
        "card_3",
        "card_3",
    ]
