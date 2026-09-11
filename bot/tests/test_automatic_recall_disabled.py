from unittest.mock import AsyncMock

import pytest

from vikingbot.agent.memory import MemoryStore
from vikingbot.openviking_mount.ov_server import VikingClient


@pytest.mark.asyncio
async def test_disabled_recall_never_creates_retrieval_client(tmp_path, monkeypatch):
    monkeypatch.setenv("VIKINGBOT_AUTOMATIC_RECALL", "0")
    create = AsyncMock(side_effect=AssertionError("Automatic recall must not call the server"))
    monkeypatch.setattr(VikingClient, "create", create)
    store = MemoryStore(tmp_path)
    assert await store.get_viking_memory_context("question", "shared", "") == ""
    assert await store.get_viking_experience_reminder("question", "shared") == ("", [])
    # Write-triggered recall uses this same experience entry point.
    assert await store.get_viking_experience_context("question", "shared") == ""
    create.assert_not_called()


def test_regular_bot_retains_recall_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("VIKINGBOT_AUTOMATIC_RECALL", raising=False)
    assert MemoryStore(tmp_path).automatic_recall_enabled is True
