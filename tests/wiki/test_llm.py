import pytest
from pydantic import BaseModel

from openviking.wiki.llm import WikiLLMRunner

from .fakes import FakeToolVLM, FakeVLM, FlakyVLM


def test_wiki_llm_runner_scopes_max_tokens_to_aggregation_agent(monkeypatch):
    configs = []

    class FakeStructuredVLM:
        def __init__(self, vlm_config):
            configs.append(vlm_config)

    monkeypatch.setattr("openviking.wiki.llm.StructuredVLM", FakeStructuredVLM)

    runner = WikiLLMRunner(
        vlm_config={"model": "demo", "max_tokens": 4096},
        aggregation_agent_max_tokens=12288,
    )

    assert configs == [
        {"model": "demo", "max_retries": 0},
        {"model": "demo", "max_retries": 0, "max_tokens": 12288},
    ]
    assert runner.vlm is not runner.aggregation_agent_vlm


@pytest.mark.asyncio
async def test_wiki_llm_runner_uses_schema_argument_without_prompt_schema_shape():
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }
    fake_vlm = FakeVLM([{"ok": True}])
    runner = WikiLLMRunner(fake_vlm)

    result = await runner.complete_json(
        step="demo",
        prompt="Business prompt only.",
        schema=schema,
    )

    assert result == {"ok": True}
    assert fake_vlm.calls == ["Business prompt only."]
    assert fake_vlm.schemas == [schema]
    assert fake_vlm.schema_names == ["wiki_demo"]
    assert "Return only JSON matching this shape" not in runner.log.prompts[0].prompt
    assert runner.log.prompts[0].schema_name == "wiki_demo"
    assert runner.log.prompts[0].schema_hash is not None


@pytest.mark.asyncio
async def test_wiki_llm_runner_retries_network_failures_three_total_attempts():
    runner = WikiLLMRunner(FlakyVLM([{"ok": True}], failures=2))
    result = await runner.complete_json(step="demo", prompt="p", schema={"type": "object"})
    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_wiki_llm_runner_stops_after_three_network_attempts():
    runner = WikiLLMRunner(FlakyVLM([], failures=3))
    with pytest.raises(ConnectionError):
        await runner.complete_json(step="demo", prompt="p", schema={"type": "object"})


@pytest.mark.asyncio
async def test_wiki_llm_runner_returns_tool_calls_without_json_parsing():
    fake_vlm = FakeToolVLM([[{"name": "finish", "arguments": {}}]])
    response = await WikiLLMRunner(fake_vlm).complete_tool_calls(
        step="agent",
        prompt="p",
        tools=[{"type": "function", "function": {"name": "finish", "parameters": {}}}],
    )

    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].name == "finish"


@pytest.mark.asyncio
async def test_tool_call_log_serializes_provider_usage_models():
    class PromptTokensDetails(BaseModel):
        cached_tokens: int

    fake_vlm = FakeToolVLM([[{"name": "finish", "arguments": {}}]])
    original_complete = fake_vlm.complete_tools_async

    async def complete_with_usage(**kwargs):
        response = await original_complete(**kwargs)
        response.usage = {
            "prompt_tokens": 12,
            "prompt_tokens_details": PromptTokensDetails(cached_tokens=3),
        }
        return response

    fake_vlm.complete_tools_async = complete_with_usage
    runner = WikiLLMRunner(fake_vlm)

    await runner.complete_tool_calls(
        step="agent",
        prompt="p",
        tools=[{"type": "function", "function": {"name": "finish", "parameters": {}}}],
    )

    assert runner.log.raw_outputs[0].raw_output["usage"] == {
        "prompt_tokens": 12,
        "prompt_tokens_details": {"cached_tokens": 3},
    }
