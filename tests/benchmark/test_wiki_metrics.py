import pytest

from benchmark.wiki.src.core.metrics import MetricsCalculator


@pytest.mark.parametrize("usage, expected", [
    ({"prompt_tokens": 100, "completion_tokens": 20, "total_input_tokens": 0,
      "llm_output_tokens": 0, "total_tokens": 120}, (100, 20, 120)),
    ({"input_tokens": 3, "output_tokens": 4}, (3, 4, 7)),
    ({"total_input_tokens": 5, "llm_output_tokens": 6}, (5, 6, 11)),
    ({"prompt_tokens": 0, "total_input_tokens": 9}, (0, 0, 0)),
    ({"prompt_tokens": None, "input_tokens": 2}, (2, 0, 2)),
    ({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 17}, (10, 5, 17)),
    ({}, (0, 0, 0)),
])
def test_qa_token_usage(usage, expected):
    assert MetricsCalculator.qa_token_usage(usage) == expected


def test_average_uses_successful_queries_and_cumulative_counts():
    assert MetricsCalculator.average_qa_tokens([
        {"token_usage": {"prompt_tokens": 100, "completion_tokens": 20}},
        {"token_usage": {"prompt_tokens": 200, "completion_tokens": 40}},
        {"generation_failed": True, "token_usage": {"prompt_tokens": 999}},
    ]) == {"Average Input Tokens": 150, "Average Output Tokens": 30, "Average Total Tokens": 180}
    assert all(value == 0 for value in MetricsCalculator.average_qa_tokens([]).values())
    assert all(value == 0 for value in MetricsCalculator.average_qa_tokens([
        {"generation_failed": True, "token_usage": {"total_tokens": 123}}
    ]).values())


@pytest.mark.parametrize("text", ["None of the methods improved.", "nonetheless", "none"])
def test_none_does_not_trigger_refusal_override(text):
    assert not MetricsCalculator.check_refusal(text)


def test_explicit_refusal_is_still_detected():
    assert MetricsCalculator.check_refusal("There is no information about that.")
