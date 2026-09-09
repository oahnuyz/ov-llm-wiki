import json

import pytest

from openviking.wiki.config import WikiConfig
from openviking.wiki.writer import WikiVikingFSWriter

from .fakes import FakeClient


@pytest.mark.asyncio
async def test_writer_writes_json_and_jsonl():
    client = FakeClient()
    writer = WikiVikingFSWriter(
        viking_fs=client,
        vikingdb=object(),
        ctx=object(),
        config=WikiConfig(),
        content_writer=client,
    )

    await writer.write_json("viking://wiki/nodes.json", {"nodes": []})
    await writer.write_jsonl("viking://wiki/run/raw_outputs.jsonl", [{"step": "a"}, {"step": "b"}])

    assert json.loads(client.writes["viking://wiki/nodes.json"]) == {"nodes": []}
    assert client.writes["viking://wiki/run/raw_outputs.jsonl"].splitlines() == [
        '{"step": "a"}',
        '{"step": "b"}',
    ]
    assert await writer.read_jsonl("viking://wiki/run/raw_outputs.jsonl") == [
        {"step": "a"},
        {"step": "b"},
    ]


@pytest.mark.asyncio
async def test_writer_ensure_dirs_uses_viking_wiki_root():
    client = FakeClient()
    writer = WikiVikingFSWriter(
        viking_fs=client,
        vikingdb=object(),
        ctx=object(),
        config=WikiConfig(),
        content_writer=client,
    )

    await writer.ensure_dirs(["question_answering"])

    assert "viking://wiki/" in client.mkdirs
    assert "viking://wiki/nodes/question_answering/" in client.mkdirs
    assert "viking://wiki/nodes/question_answering/sources/" in client.mkdirs
    assert all("/documents/" not in uri and "/children/" not in uri for uri in client.mkdirs)
    assert all("corpus" not in uri for uri in client.mkdirs)


@pytest.mark.asyncio
async def test_reset_node_outputs_can_preserve_build_checkpoints():
    client = FakeClient()
    client.writes["viking://wiki/build/layers/depth_1/node/context.json"] = "{}"
    client.writes["viking://wiki/nodes/node/card.json"] = "{}"
    client.writes["viking://wiki/run/aggregation_operations.jsonl"] = "{}\n"
    writer = WikiVikingFSWriter(
        viking_fs=client,
        vikingdb=object(),
        ctx=object(),
        config=WikiConfig(),
        content_writer=client,
    )

    await writer.reset_node_outputs(preserve_build=True)

    assert "viking://wiki/build/layers/depth_1/node/context.json" in client.writes
    assert "viking://wiki/nodes/node/card.json" not in client.writes
    assert "viking://wiki/run/aggregation_operations.jsonl" not in client.writes
