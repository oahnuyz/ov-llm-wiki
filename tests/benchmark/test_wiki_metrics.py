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
    ({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 0}, (10, 5, 0)),
    ({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
      "retrieval_embedding_tokens": 3}, (13, 5, 18)),
    ({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 18,
      "llm_total_tokens": 15, "retrieval_embedding_tokens": 3}, (13, 5, 18)),
])
def test_qa_token_usage(usage, expected):
    assert MetricsCalculator.qa_token_usage(usage) == expected


def test_average_uses_successful_queries_and_cumulative_counts():
    assert MetricsCalculator.average_qa_tokens([
        {"token_usage": {"prompt_tokens": 100, "completion_tokens": 20}},
        {"token_usage": {"prompt_tokens": 200, "completion_tokens": 40}},
        {"generation_failed": True, "token_usage": {"prompt_tokens": 999}},
    ]) == {"Average Input Tokens": 150, "Average Output Tokens": 30, "Average Total Tokens": 180,
           "Average Embedding Tokens": 0, "Average LLM Tokens": 180,
           "Queries Missing Embedding Usage": 0, "Queries Missing LLM Usage": 2}
    assert all(value == 0 for value in MetricsCalculator.average_qa_tokens([]).values())
    assert all(value == 0 for value in MetricsCalculator.average_qa_tokens([
        {"generation_failed": True, "token_usage": {"total_tokens": 123}}
    ]).values())


def test_embedding_is_included_once_and_failed_queries_stay_excluded():
    report = MetricsCalculator.average_qa_tokens([
        {"token_usage": {"prompt_tokens": 10, "completion_tokens": 5,
                         "llm_total_tokens": 15, "total_tokens": 18,
                         "retrieval_embedding_tokens": 3,
                         "retrieval_embedding_usage_complete": True}},
        {"token_usage": {"prompt_tokens": 20, "completion_tokens": 10,
                         "total_tokens": 30, "retrieval_embedding_tokens": 7}},
        {"generation_failed": True, "token_usage": {"retrieval_embedding_tokens": 999}},
    ])
    assert report["Average Embedding Tokens"] == 5
    assert report["Average LLM Tokens"] == 22.5
    assert report["Average Input Tokens"] == 20
    assert report["Average Total Tokens"] == 27.5


@pytest.mark.parametrize("record", [
    {"token_usage": {"total_tokens": 15, "retrieval_embedding_tokens": 0},
     "vikingbot": {"tools_used_names": ["openviking_search"]}},
    {"token_usage": {"total_tokens": 15, "retrieval_embedding_usage_complete": False}},
])
def test_missing_embedding_usage_keeps_numeric_total_and_marks_gap(record):
    report = MetricsCalculator.average_qa_tokens([record])
    assert report["Queries Missing Embedding Usage"] == 1
    assert report["Average Total Tokens"] == 15
    assert report["Average Embedding Tokens"] == 0
    assert report["Average LLM Tokens"] == 15
