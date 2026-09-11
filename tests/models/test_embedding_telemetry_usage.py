# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

from types import SimpleNamespace

from openviking.models.embedder.openai_embedders import OpenAIDenseEmbedder
from openviking.models.embedder.volcengine_embedders import VolcengineDenseEmbedder
from openviking.telemetry.backends.memory import MemoryOperationTelemetry
from openviking.telemetry.context import bind_telemetry


def _usage(prompt_tokens: int, total_tokens: int):
    return SimpleNamespace(prompt_tokens=prompt_tokens, total_tokens=total_tokens)


def test_openai_dense_embedder_reports_embedding_telemetry_usage(monkeypatch):
    response = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])],
        usage=_usage(prompt_tokens=9, total_tokens=9),
    )

    fake_client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kwargs: response))
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: fake_client)

    telemetry = MemoryOperationTelemetry(operation="search.find", enabled=True)
    with bind_telemetry(telemetry):
        embedder = OpenAIDenseEmbedder(
            model_name="text-embedding-3-small",
            api_key="test",
            dimension=3,
        )
        result = embedder.embed("hello")

    assert result.dense_vector == [0.1, 0.2, 0.3]
    summary = telemetry.finish().summary
    assert summary["tokens"]["embedding"] == {"total": 9}
    assert summary["tokens"]["total"] == 9


def test_volcengine_dense_embedder_reports_embedding_telemetry_usage(monkeypatch):
    response = SimpleNamespace(
        data=SimpleNamespace(embedding=[0.4, 0.5, 0.6]),
        usage=_usage(prompt_tokens=16, total_tokens=16),
    )

    fake_client = SimpleNamespace(
        multimodal_embeddings=SimpleNamespace(create=lambda **kwargs: response),
    )
    monkeypatch.setattr(
        "volcenginesdkarkruntime.Ark",
        lambda **kwargs: fake_client,
    )

    telemetry = MemoryOperationTelemetry(operation="resources.add_resource", enabled=True)
    with bind_telemetry(telemetry):
        embedder = VolcengineDenseEmbedder(
            model_name="doubao-embedding-vision-251215",
            api_key="test",
            input_type="multimodal",
            dimension=3,
        )
        result = embedder.embed("hello")

    assert result.dense_vector == [0.4, 0.5, 0.6]
    summary = telemetry.finish().summary
    assert summary["tokens"]["embedding"] == {"total": 16}
    assert summary["tokens"]["total"] == 16


def test_volcengine_dense_embedder_reports_embedding_telemetry_usage_from_dict_usage(
    monkeypatch,
):
    response = SimpleNamespace(
        data=SimpleNamespace(embedding=[0.4, 0.5, 0.6]),
        usage={
            "prompt_tokens": 16,
            "prompt_tokens_details": {"image_tokens": 0, "text_tokens": 16},
            "total_tokens": 16,
        },
    )

    fake_client = SimpleNamespace(
        multimodal_embeddings=SimpleNamespace(create=lambda **kwargs: response),
    )
    monkeypatch.setattr(
        "volcenginesdkarkruntime.Ark",
        lambda **kwargs: fake_client,
    )

    telemetry = MemoryOperationTelemetry(operation="search.find", enabled=True)
    with bind_telemetry(telemetry):
        embedder = VolcengineDenseEmbedder(
            model_name="doubao-embedding-vision-251215",
            api_key="test",
            input_type="multimodal",
            dimension=3,
        )
        result = embedder.embed("hello")

    assert result.dense_vector == [0.4, 0.5, 0.6]
    summary = telemetry.finish().summary
    assert summary["tokens"]["embedding"] == {"total": 16}


def test_missing_embedding_usage_returns_vectors_and_retains_known_usage(monkeypatch):
    responses = iter([
        SimpleNamespace(data=[SimpleNamespace(embedding=[0.1])], usage=_usage(9, 9)),
        SimpleNamespace(data=[SimpleNamespace(embedding=[0.2])], usage=None),
        SimpleNamespace(data=[SimpleNamespace(embedding=[0.3])], usage={'prompt_tokens': 4, 'total_tokens': 'bad'}),
        SimpleNamespace(data=[SimpleNamespace(embedding=[0.4])], usage=_usage(3, 3)),
    ])
    client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kwargs: next(responses)))
    monkeypatch.setattr('openai.OpenAI', lambda **kwargs: client)
    telemetry = MemoryOperationTelemetry(operation='search.search', enabled=True)
    with bind_telemetry(telemetry):
        embedder = OpenAIDenseEmbedder(model_name='text-embedding-3-small', api_key='test', dimension=1)
        vectors = [embedder.embed('hello').dense_vector for _ in range(4)]
    assert vectors == [[0.1], [0.2], [0.3], [0.4]]
    assert telemetry.finish().summary['tokens']['embedding'] == {'total': 16, 'usage_missing': 2}
