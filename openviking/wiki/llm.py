"""LLM wrapper for Wiki structured calls."""

from __future__ import annotations

import hashlib
import json
import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from openviking.models.vlm.base import VLMResponse
from openviking.models.vlm.llm import StructuredVLM

logger = logging.getLogger(__name__)
MAX_NETWORK_ATTEMPTS = 3
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0


@dataclass
class LLMCallRecord:
    step: str
    prompt_version: str
    input_hash: str
    prompt: str
    schema_name: str | None = None
    schema_hash: str | None = None


@dataclass
class LLMOutputRecord:
    step: str
    output_hash: str
    raw_output: Any


@dataclass
class WikiLLMRunLog:
    prompts: list[LLMCallRecord] = field(default_factory=list)
    raw_outputs: list[LLMOutputRecord] = field(default_factory=list)


class WikiLLMRunner:
    def __init__(
        self,
        vlm: Any | None = None,
        vlm_config: dict[str, Any] | None = None,
        request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        aggregation_agent_max_tokens: int = 12288,
    ):
        self.aggregation_agent_max_tokens = max(1, int(aggregation_agent_max_tokens))
        if vlm is not None:
            self.vlm = vlm
            self.aggregation_agent_vlm = vlm
        else:
            # Wiki owns the retry policy. Disable provider-internal retries so
            # one request cannot silently expand to hundreds of attempts.
            config = dict(vlm_config or {})
            config["max_retries"] = 0
            # Non-agent Wiki calls intentionally use the provider's default
            # output limit. The aggregation agent gets its own provider so its
            # larger limit cannot leak into card or node-document generation.
            config.pop("max_tokens", None)
            self.vlm = StructuredVLM(vlm_config=config)
            aggregation_config = dict(config)
            aggregation_config["max_tokens"] = self.aggregation_agent_max_tokens
            self.aggregation_agent_vlm = StructuredVLM(vlm_config=aggregation_config)
        self.request_timeout = max(1.0, float(request_timeout_seconds))
        self.log = WikiLLMRunLog()

    async def complete_json(
        self,
        step: str,
        prompt: str,
        schema: dict[str, Any],
        prompt_version: str | None = None,
    ) -> dict[str, Any]:
        prompt_version = prompt_version or f"{step}_v1"
        schema_name = f"wiki_{step}"
        self.log.prompts.append(
            LLMCallRecord(
                step=step,
                prompt_version=prompt_version,
                input_hash=_hash_text(prompt),
                prompt=prompt,
                schema_name=schema_name,
                schema_hash=_hash_text(json.dumps(schema, ensure_ascii=False, sort_keys=True)),
            )
        )

        result = None
        for attempt in range(1, MAX_NETWORK_ATTEMPTS + 1):
            try:
                request = self.vlm.complete_json_async(
                    prompt=prompt,
                    schema=schema,
                    schema_name=schema_name,
                )
                result = await asyncio.wait_for(request, timeout=self.request_timeout)
                break
            except Exception:
                if attempt == MAX_NETWORK_ATTEMPTS:
                    logger.exception(
                        "[Wiki] LLM request failed after %d/%d attempts step=%s",
                        attempt,
                        MAX_NETWORK_ATTEMPTS,
                        step,
                    )
                    raise
                logger.warning(
                    "[Wiki] Retrying LLM request step=%s attempt=%d/%d",
                    step,
                    attempt,
                    MAX_NETWORK_ATTEMPTS,
                )
                await asyncio.sleep(2 ** (attempt - 1))
        if result is None:
            raise RuntimeError(f"LLM step {step} returned no parseable JSON")
        if not isinstance(result, dict):
            raise RuntimeError(f"LLM step {step} must return a JSON object")

        self.log.raw_outputs.append(
            LLMOutputRecord(
                step=step,
                output_hash=_hash_text(json.dumps(result, ensure_ascii=False, sort_keys=True)),
                raw_output=result,
            )
        )
        return result

    async def complete_tool_calls(
        self,
        *,
        step: str,
        prompt: str,
        tools: list[dict[str, Any]],
    ) -> VLMResponse:
        """Request one tool-calling agent turn with the Wiki retry policy."""
        self.log.prompts.append(
            LLMCallRecord(
                step=step,
                prompt_version=f"{step}_v1",
                input_hash=_hash_text(prompt),
                prompt=prompt,
                schema_name=f"wiki_{step}_tools",
                schema_hash=_hash_text(json.dumps(tools, ensure_ascii=False, sort_keys=True)),
            )
        )

        result: Any = None
        for attempt in range(1, MAX_NETWORK_ATTEMPTS + 1):
            try:
                request = self.aggregation_agent_vlm.complete_tools_async(
                    prompt=prompt,
                    tools=tools,
                    tool_choice="required",
                )
                result = await asyncio.wait_for(request, timeout=self.request_timeout)
                break
            except Exception:
                if attempt == MAX_NETWORK_ATTEMPTS:
                    logger.exception(
                        "[Wiki] LLM request failed after %d/%d attempts step=%s",
                        attempt,
                        MAX_NETWORK_ATTEMPTS,
                        step,
                    )
                    raise
                logger.warning(
                    "[Wiki] Retrying LLM request step=%s attempt=%d/%d",
                    step,
                    attempt,
                    MAX_NETWORK_ATTEMPTS,
                )
                await asyncio.sleep(2 ** (attempt - 1))

        if not isinstance(result, VLMResponse):
            raise RuntimeError(f"LLM step {step} did not return a tool-call response")
        self.log.raw_outputs.append(
            LLMOutputRecord(
                step=step,
                output_hash=_hash_text(json.dumps(_tool_response_payload(result), ensure_ascii=False, sort_keys=True)),
                raw_output=_tool_response_payload(result),
            )
        )
        return result


def _hash_text(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _tool_response_payload(response: VLMResponse) -> dict[str, Any]:
    return {
        "content": response.content,
        "finish_reason": response.finish_reason,
        "usage": _json_compatible(response.usage),
        "tool_calls": [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in response.tool_calls
        ],
    }


def _json_compatible(value: Any) -> Any:
    """Convert provider SDK response models into values accepted by json.dumps."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_compatible(model_dump(mode="json"))
    return str(value)
