from __future__ import annotations

from typing import Any

from openviking.models.vlm.base import ToolCall, VLMResponse


class FakeVLM:
    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.calls: list[str] = []
        self.schemas: list[dict | None] = []
        self.schema_names: list[str | None] = []

    async def complete_json_async(
        self,
        prompt: str = "",
        schema: dict | None = None,
        schema_name: str | None = None,
        **_: Any,
    ) -> dict:
        if not self.responses:
            raise AssertionError("FakeVLM has no remaining responses")
        self.calls.append(prompt)
        self.schemas.append(schema)
        self.schema_names.append(schema_name)
        return self.responses.pop(0)


class FlakyVLM(FakeVLM):
    def __init__(self, responses: list[dict[str, Any]], failures: int):
        super().__init__(responses)
        self.failures = failures

    async def complete_json_async(self, *args: Any, **kwargs: Any) -> dict:
        if self.failures:
            self.failures -= 1
            raise ConnectionError("temporary network failure")
        return await super().complete_json_async(*args, **kwargs)


class FakeToolVLM:
    def __init__(self, responses: list[list[dict[str, Any]] | VLMResponse]):
        self.responses = list(responses)
        self.calls: list[str] = []
        self.tools: list[list[dict[str, Any]] | None] = []

    async def complete_tools_async(
        self,
        prompt: str = "",
        tools: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> VLMResponse:
        if not self.responses:
            raise AssertionError("FakeToolVLM has no remaining responses")
        self.calls.append(prompt)
        self.tools.append(tools)
        calls = self.responses.pop(0)
        if isinstance(calls, VLMResponse):
            return calls
        return VLMResponse(
            tool_calls=[
                ToolCall(
                    id=str(call.get("id", f"call_{index}")),
                    name=call["name"],
                    arguments=call.get("arguments", {}),
                )
                for index, call in enumerate(calls, start=1)
            ],
            finish_reason="tool_calls" if calls else "stop",
        )


class FakeMixedVLM(FakeVLM):
    def __init__(self, json_responses: list[dict[str, Any]], tool_responses: list[list[dict[str, Any]]]):
        super().__init__(json_responses)
        self.tool_responses = list(tool_responses)
        self.tool_calls: list[str] = []

    async def complete_tools_async(
        self,
        prompt: str = "",
        tools: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> VLMResponse:
        if not self.tool_responses:
            raise AssertionError("FakeMixedVLM has no remaining tool responses")
        self.tool_calls.append(prompt)
        calls = self.tool_responses.pop(0)
        return VLMResponse(
            tool_calls=[
                ToolCall(
                    id=str(call.get("id", f"call_{index}")),
                    name=call["name"],
                    arguments=call.get("arguments", {}),
                )
                for index, call in enumerate(calls, start=1)
            ],
            finish_reason="tool_calls" if calls else "stop",
        )


class FakeClient:
    def __init__(self):
        self.mkdirs: list[str] = []
        self.writes: dict[str, str] = {}
        self.removed: list[tuple[str, bool]] = []
        self.links: list[tuple[str, str, str]] = []

    async def mkdir(self, uri: str, *_: Any, **__: Any) -> None:
        self.mkdirs.append(uri)

    async def write(self, uri: str, content: str, *_: Any, **__: Any) -> dict[str, Any]:
        self.writes[uri] = content
        return {}

    async def write_file(self, uri: str, content: str, *_: Any, **__: Any) -> None:
        self.writes[uri] = content

    async def link(
        self,
        from_uri: str,
        to_uri: str,
        reason: str = "",
        **__: Any,
    ) -> None:
        self.links.append((from_uri, to_uri, reason))

    async def read_file(self, uri: str, *_: Any, **__: Any) -> str:
        if uri not in self.writes:
            raise FileNotFoundError(uri)
        return self.writes[uri]

    async def exists(self, uri: str, *_: Any, **__: Any) -> bool:
        return uri in self.writes or uri in self.mkdirs or any(
            path.startswith(uri) for path in self.writes
        )

    async def rm(self, uri: str, *, recursive: bool = False, **__: Any) -> dict[str, Any]:
        self.removed.append((uri, recursive))
        if recursive:
            self.writes = {path: content for path, content in self.writes.items() if not path.startswith(uri)}
            self.mkdirs = [path for path in self.mkdirs if not path.startswith(uri)]
        else:
            self.writes.pop(uri, None)
        return {}
