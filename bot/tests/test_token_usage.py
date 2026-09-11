from types import SimpleNamespace

import pytest

from openviking.utils.token_usage import parse_llm_usage
from vikingbot.utils.token_usage import record_embedding_usage, record_llm_usage


@pytest.mark.parametrize('raw', [None, {}, [], 'bad', {'prompt_tokens': None},
                                     {'total_tokens': 'invalid'}, {'total_tokens': -1},
                                     {'total_tokens': float('nan')}])
def test_missing_llm_usage_preserves_prior_and_subsequent_counts(raw):
    usage = {'llm_usage_complete': True}
    valid = {'prompt_tokens': 10, 'completion_tokens': 3, 'total_tokens': 13}
    record_llm_usage(usage, valid, iteration=1)
    record_llm_usage(usage, raw, iteration=2)
    record_llm_usage(usage, valid, iteration=3)
    assert (usage['prompt_tokens'], usage['completion_tokens'], usage['total_tokens']) == (20, 6, 26)
    assert usage['llm_usage_complete'] is False
    assert len(usage['usage_issues']) == 1
    assert usage['usage_issues'][0]['iteration'] == 2


def test_partial_llm_usage_keeps_known_components_and_explicit_total():
    usage = {'llm_usage_complete': True}
    record_llm_usage(usage, {'prompt_tokens': 10}, iteration=1)
    record_llm_usage(usage, {'prompt_tokens': None, 'completion_tokens': 3, 'total_tokens': 8}, iteration=2)
    assert (usage['prompt_tokens'], usage['completion_tokens'], usage['total_tokens']) == (10, 3, 18)
    assert usage['usage_issues'][0]['fields'] == ['completion_tokens', 'total_tokens']
    assert usage['usage_issues'][1]['fields'] == ['prompt_tokens']


@pytest.mark.parametrize('raw', [None, {}, {'summary': None}, {'summary': {'tokens': None}},
                                     {'summary': {'tokens': {'embedding': {'total': 'bad'}}}}])
def test_missing_embedding_usage_keeps_other_search_calls(raw):
    usage = {'retrieval_embedding_usage_complete': True}
    valid = {'summary': {'tokens': {'embedding': {'total': 7}}}}
    record_embedding_usage(usage, valid)
    record_embedding_usage(usage, raw)
    record_embedding_usage(usage, valid)
    assert usage['retrieval_embedding_tokens'] == 14
    assert usage['retrieval_embedding_usage_complete'] is False
    assert len(usage['usage_issues']) == 1


def test_provider_embedding_gap_is_not_hidden_by_numeric_telemetry_total():
    usage = {'retrieval_embedding_tokens': 7, 'retrieval_embedding_usage_complete': True}
    record_embedding_usage(usage, {'summary': {'tokens': {'embedding': {'total': 4, 'usage_missing': 1}}}})
    assert usage['retrieval_embedding_tokens'] == 11
    assert usage['retrieval_embedding_usage_complete'] is False


@pytest.mark.parametrize('raw', [
    {'prompt_tokens': 10, 'completion_tokens': None, 'prompt_tokens_details': {'cached_tokens': 'bad'}},
    SimpleNamespace(prompt_tokens=10, prompt_tokens_details=SimpleNamespace(cached_tokens=None)),
])
def test_provider_parser_preserves_missing_primary_fields(raw):
    assert parse_llm_usage(raw) == {'prompt_tokens': 10}
