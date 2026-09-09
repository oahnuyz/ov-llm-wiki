import os

os.environ["OPENVIKING_CONFIG_FILE"] = "/tmp/openviking-missing-test.conf"
os.environ["OPENVIKING_CLI_CONFIG_FILE"] = "/tmp/openviking-cli-missing-test.conf"

import pytest

from vikingbot.agent.tools.base import ToolContext
from vikingbot.agent.tools.ov_file import (
    VikingGrepTool,
    VikingListTool,
    VikingMultiReadTool,
    VikingSearchTool,
)


class FakeVikingClient:
    def __init__(self, *, actor_peer_id: str | None = None, sender_fanout: bool = False):
        self.actor_peer_id = actor_peer_id
        self.sender_fanout = sender_fanout
        self.search_calls = []

    def should_sender_fanout(self) -> bool:
        return self.sender_fanout

    def _memory_target_uri(self, user_id: str | None) -> str:
        if user_id:
            return f"viking://user/{user_id}/memories/"
        return "viking://user/memories/"

    def build_current_memory_target_uris(
        self, *, peer_ids: list[str] | None = None, include_self: bool = False
    ) -> list[str]:
        return [f"viking://user/peers/{peer_id}/memories/" for peer_id in peer_ids or []]

    async def search(self, query: str, **kwargs):
        self.search_calls.append({"query": query, **kwargs})
        return {
            "resources": [
                {
                    "uri": kwargs.get("target_uri") or "",
                    "abstract": "matched",
                    "score": 1.0,
                }
            ],
            "memories": [],
            "skills": [],
        }

    async def close(self) -> None:
        pass


class FakeVikingSearchTool(VikingSearchTool):
    def __init__(self, client: FakeVikingClient):
        super().__init__()
        self.client = client

    async def _get_client(self, tool_context: ToolContext):
        return self.client

    async def _release_client(self, tool_context: ToolContext, client) -> None:
        pass


class FakeReadClient:
    def __init__(self):
        self.read_calls = []

    async def read_content(self, uri: str, level: str = "read") -> str:
        self.read_calls.append((uri, level))
        return f"content for {uri}"

    async def close(self) -> None:
        pass


class FakeVikingMultiReadTool(VikingMultiReadTool):
    def __init__(self, client: FakeReadClient):
        super().__init__()
        self.client = client

    async def _get_client(self, tool_context: ToolContext):
        return self.client

    async def _release_client(self, tool_context: ToolContext, client) -> None:
        pass


class FakeListClient:
    def __init__(self):
        self.list_calls = []

    async def list_resources(self, path: str, recursive: bool = False):
        self.list_calls.append((path, recursive))
        if path == "viking://wiki/nodes/parent":
            entries = [
                {
                    "name": "child",
                    "size": 0,
                    "uri": f"{path}/child",
                    "isDir": True,
                },
                {
                    "name": "0001.md",
                    "size": 100,
                    "uri": f"{path}/0001.md",
                    "isDir": False,
                },
                {
                    "name": "sources",
                    "size": 0,
                    "uri": f"{path}/sources",
                    "isDir": True,
                },
                {
                    "name": "card.md",
                    "size": 80,
                    "uri": f"{path}/card.md",
                    "isDir": False,
                },
                {
                    "name": "card.json",
                    "size": 120,
                    "uri": f"{path}/card.json",
                    "isDir": False,
                },
            ]
            if recursive:
                entries.extend(
                    [
                        {
                            "name": "paper.ref.json",
                            "size": 40,
                            "uri": f"{path}/sources/paper.ref.json",
                            "isDir": False,
                        },
                        {
                            "name": "card.json",
                            "size": 120,
                            "uri": f"{path}/child/card.json",
                            "isDir": False,
                        },
                        {
                            "name": "0001.md",
                            "size": 90,
                            "uri": f"{path}/child/0001.md",
                            "isDir": False,
                        },
                    ]
                )
            return entries
        return []


class FakeVikingListTool(VikingListTool):
    def __init__(self, client: FakeListClient):
        super().__init__()
        self.client = client

    async def _get_client(self, tool_context: ToolContext):
        return self.client

    async def _release_client(self, tool_context: ToolContext, client) -> None:
        pass


class FakeGrepClient:
    def __init__(self, *, actor_peer_id: str | None = None):
        self.actor_peer_id = actor_peer_id
        self.grep_calls = []

    def _memory_target_uri(self, user_id: str | None) -> str:
        if user_id:
            return f"viking://user/{user_id}/memories/"
        return "viking://user/memories/"

    def build_current_memory_target_uris(
        self, *, peer_ids: list[str] | None = None, include_self: bool = False
    ) -> list[str]:
        return [f"viking://user/peers/{peer_id}/memories/" for peer_id in peer_ids or []]

    async def grep(
        self, uri: str, pattern: str, case_insensitive: bool = False, user_id: str | None = None
    ) -> dict:
        self.grep_calls.append(
            {
                "uri": uri,
                "pattern": pattern,
                "case_insensitive": case_insensitive,
                "user_id": user_id,
            }
        )
        return {
            "matches": [
                {
                    "uri": "viking://wiki/nodes/steam/0001.md",
                    "line": 1,
                    "content": "steam creator support",
                },
                {
                    "uri": "viking://wiki/nodes/steam_workshop/0001.md",
                    "line": 2,
                    "content": "steam workshop support",
                },
            ]
            if uri == "viking://wiki/nodes" or uri == "viking://wiki/nodes/steam"
            else []
        }

    async def close(self) -> None:
        pass


class FakeVikingGrepTool(VikingGrepTool):
    def __init__(self, client: FakeGrepClient):
        super().__init__()
        self.client = client

    async def _get_client(self, tool_context: ToolContext):
        return self.client

    async def _release_client(self, tool_context: ToolContext, client) -> None:
        pass


@pytest.mark.asyncio
async def test_openviking_search_includes_wiki_nodes_for_actor_peer_default_target():
    client = FakeVikingClient(actor_peer_id="cli-user")
    tool = FakeVikingSearchTool(client)

    await tool.execute(ToolContext(actor_peer_id="cli-user"), query="steam")

    target_uris = [call["target_uri"] for call in client.search_calls]
    assert "viking://resources/" in target_uris
    assert "viking://wiki/nodes" in target_uris
    assert "viking://user/memories/" in target_uris
    assert "viking://user/skills/" in target_uris


@pytest.mark.asyncio
async def test_openviking_grep_includes_wiki_nodes_for_default_target():
    client = FakeGrepClient()
    tool = FakeVikingGrepTool(client)

    result = await tool.execute(ToolContext(), pattern="steam")

    target_uris = [call["uri"] for call in client.grep_calls]
    assert "viking://resources/" in target_uris
    assert "viking://wiki/nodes" in target_uris
    assert "viking://user/memories/" in target_uris
    assert "viking://user/skills/" in target_uris
    assert "viking://wiki/nodes/steam/0001.md" in result
    assert "viking://wiki/nodes/steam_workshop/0001.md" in result


@pytest.mark.asyncio
async def test_openviking_grep_keeps_explicit_wiki_node_target_for_client_api():
    client = FakeGrepClient()
    tool = FakeVikingGrepTool(client)

    result = await tool.execute(
        ToolContext(),
        pattern="steam",
        uri="viking://wiki/nodes/steam",
    )

    target_uris = [call["uri"] for call in client.grep_calls]
    assert target_uris == ["viking://wiki/nodes/steam"]
    assert "viking://wiki/nodes/steam/0001.md" in result
    assert "viking://wiki/nodes/steam_workshop/0001.md" in result


@pytest.mark.asyncio
async def test_openviking_search_includes_wiki_nodes_for_sender_fanout_default_target():
    client = FakeVikingClient(sender_fanout=True)
    tool = FakeVikingSearchTool(client)

    await tool.execute(
        ToolContext(memory_owner_user_ids=["owner-a"]),
        query="steam",
    )

    target_calls = [(call["target_uri"], call.get("user_id")) for call in client.search_calls]
    assert ("viking://resources/", None) in target_calls
    assert ("viking://wiki/nodes", None) in target_calls
    assert ("viking://user/owner-a/memories/", "owner-a") in target_calls
    assert ("viking://user/owner-a/skills/", "owner-a") in target_calls


@pytest.mark.asyncio
async def test_openviking_multi_read_supports_wiki_node_documents():
    client = FakeReadClient()
    tool = FakeVikingMultiReadTool(client)
    uri = "viking://wiki/nodes/steam_community/0001.md"

    result = await tool.execute(ToolContext(), uris=[uri])

    assert client.read_calls == [(uri, "read")]
    assert f"--- START OF {uri} ---" in result
    assert f"content for {uri}" in result
    assert f"--- END OF {uri} ---" in result


@pytest.mark.asyncio
async def test_configured_root_limits_default_search_to_wiki(monkeypatch):
    monkeypatch.setenv("VIKINGBOT_OPENVIKING_ROOT_URI", "viking://wiki/nodes")
    client = FakeVikingClient(actor_peer_id="cli-user")
    tool = FakeVikingSearchTool(client)

    await tool.execute(ToolContext(actor_peer_id="cli-user"), query="steam")

    assert [call["target_uri"] for call in client.search_calls] == ["viking://wiki/nodes"]


@pytest.mark.asyncio
async def test_configured_root_rejects_resource_search_and_read(monkeypatch):
    monkeypatch.setenv("VIKINGBOT_OPENVIKING_ROOT_URI", "viking://wiki/nodes")
    search_client = FakeVikingClient()
    search_tool = FakeVikingSearchTool(search_client)
    read_client = FakeReadClient()
    read_tool = FakeVikingMultiReadTool(read_client)

    search_result = await search_tool.execute(
        ToolContext(), query="steam", target_uri="viking://resources/paper"
    )
    read_result = await read_tool.execute(
        ToolContext(), uris=["viking://resources/paper/overview.md"]
    )

    assert "outside the configured OpenViking retrieval root" in search_result
    assert "outside the configured OpenViking retrieval root" in read_result
    assert search_client.search_calls == []
    assert read_client.read_calls == []


@pytest.mark.asyncio
async def test_wiki_list_exposes_node_content_but_hides_cards_and_sources():
    client = FakeListClient()
    tool = FakeVikingListTool(client)

    result = await tool.execute(ToolContext(), uri="viking://wiki/nodes/parent")

    assert "viking://wiki/nodes/parent/child" in result
    assert "viking://wiki/nodes/parent/0001.md" in result
    assert "viking://wiki/nodes/parent/sources" not in result
    assert "viking://wiki/nodes/parent/card.md" not in result
    assert "viking://wiki/nodes/parent/card.json" not in result
    assert "logical_type" not in result


@pytest.mark.asyncio
async def test_recursive_wiki_list_hides_source_descendants_and_nested_cards():
    client = FakeListClient()
    tool = FakeVikingListTool(client)

    result = await tool.execute(
        ToolContext(), uri="viking://wiki/nodes/parent", recursive=True
    )

    assert "viking://wiki/nodes/parent/child/0001.md" in result
    assert "viking://wiki/nodes/parent/sources" not in result
    assert "paper.ref.json" not in result
    assert "viking://wiki/nodes/parent/child/card.json" not in result
