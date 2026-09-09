import json

import pytest

from openviking.wiki.config import WikiConfig
from openviking.wiki.llm import WikiLLMRunner
from openviking.wiki.pipeline import (
    WikiPipeline,
    _assign_parent_node_links,
    _build_node_tree_paths,
    _with_child_node_ids_from_refs_for_layer,
)
from openviking.wiki.schemas import (
    GeneratedNodeContext,
    NodeCard,
    NodeDocument,
    PipelineArtifacts,
    SourceAssignmentResult,
    SourceRef,
    WikiNode,
)
from openviking.wiki.writer import WikiVikingFSWriter

from .fakes import FakeClient, FakeVLM


def test_parent_node_records_all_child_node_ids():
    parent = _node("parent", depth=2)
    result = _assignment_result("parent", ["child_a", "child_b"])

    updated = _with_child_node_ids_from_refs_for_layer([parent], result)

    assert updated[0].child_node_ids == ["child_a", "child_b"]


def test_child_node_can_have_multiple_parent_nodes():
    child = _node("child_a", depth=1)
    parents = [
        _node("parent_a", depth=2).model_copy(update={"child_node_ids": ["child_a"]}),
        _node("parent_b", depth=2).model_copy(update={"child_node_ids": ["child_a"]}),
    ]

    linked_nodes = _assign_parent_node_links([child], parents)

    assert linked_nodes[0].parent_node_ids == ["parent_a", "parent_b"]


def test_first_parent_becomes_primary_nested_directory():
    config = WikiConfig()
    nodes = [
        _node("child", depth=1).model_copy(
            update={"parent_node_ids": ["parent_a", "parent_b"]}
        ),
        _node("parent_a", depth=2),
        _node("parent_b", depth=2),
    ]

    placed, paths = _build_node_tree_paths(nodes, config)

    child = next(node for node in placed if node.node_id == "child")
    assert child.primary_parent_id == "parent_a"
    assert paths["child"] == "viking://wiki/nodes/parent_a/child/"
    assert paths["parent_b"] == "viking://wiki/nodes/parent_b/"


@pytest.mark.asyncio
async def test_materialization_links_secondary_parent_to_primary_child_path():
    client = FakeClient()
    config = WikiConfig()
    writer = WikiVikingFSWriter(
        viking_fs=client,
        vikingdb=object(),
        ctx=object(),
        config=config,
        content_writer=client,
    )
    pipeline = WikiPipeline(
        writer=writer,
        config=config,
        llm=WikiLLMRunner(FakeVLM([])),
    )
    nodes = [
        _node("child", depth=1).model_copy(
            update={"parent_node_ids": ["parent_a", "parent_b"]}
        ),
        _node("parent_a", depth=2).model_copy(update={"child_node_ids": ["child"]}),
        _node("parent_b", depth=2).model_copy(update={"child_node_ids": ["child"]}),
    ]
    contexts = [
        _context(nodes[0], [_source_ref("source_doc", ref_type="document")]),
        _context(nodes[1], [_source_ref("child")]),
        _context(nodes[2], [_source_ref("child")]),
    ]
    artifacts = PipelineArtifacts(nodes=nodes, node_contexts=contexts)

    await pipeline._materialize_node_tree(
        artifacts,
        document_cards=[],
        unassigned_source_ids=[],
    )

    child_uri = "viking://wiki/nodes/parent_a/child/"
    assert f"{child_uri}card.json" in client.writes
    assert f"{child_uri}0001.md" in client.writes
    assert client.links == [
        (
            "viking://wiki/nodes/parent_b/",
            child_uri,
            "wiki_child:secondary_parent:child",
        )
    ]
    node_index = json.loads(client.writes["viking://wiki/node_index.json"])["nodes"]
    assert node_index["child"] == {
        "primary_parent_id": "parent_a",
        "primary_uri": child_uri,
        "document_uris": [f"{child_uri}0001.md"],
        "secondary_parent_ids": ["parent_b"],
        "depth": 1,
        "aggregation_order": 0,
    }


@pytest.mark.asyncio
async def test_layer_context_generation_reuses_matching_build_checkpoint():
    client = FakeClient()
    config = WikiConfig()
    writer = WikiVikingFSWriter(
        viking_fs=client,
        vikingdb=object(),
        ctx=object(),
        config=config,
        content_writer=client,
    )
    pipeline = WikiPipeline(
        writer=writer,
        config=config,
        llm=WikiLLMRunner(FakeVLM([])),
    )
    node = _node("child", depth=1)
    refs = [_source_ref("source_doc", ref_type="document")]
    context = _context(node, refs)
    client.writes[
        "viking://wiki/build/layers/depth_1/child/context.json"
    ] = json.dumps(context.model_dump(mode="json"))

    contexts = await pipeline._generate_layer_contexts(
        [node],
        SourceAssignmentResult(source_refs_by_node={node.node_id: refs}),
        {},
        depth=1,
    )

    assert contexts == [context]


def _assignment_result(node_id: str, child_node_ids: list[str]) -> SourceAssignmentResult:
    refs = [_source_ref(child_node_id) for child_node_id in child_node_ids]
    return SourceAssignmentResult(source_refs_by_node={node_id: refs})


def _node(node_id: str, depth: int) -> WikiNode:
    return WikiNode(
        node_id=node_id,
        title=node_id.replace("_", " ").title(),
        depth=depth,
        scope="Supported topic.",
    )


def _source_ref(doc_id: str, *, ref_type: str = "wiki_node") -> SourceRef:
    prefix = "resources" if ref_type == "document" else "wiki/nodes"
    return SourceRef(
        ref_id=doc_id,
        ref_type=ref_type,
        doc_id=doc_id,
        resource_uri=f"viking://{prefix}/{doc_id}/",
        card_uri=f"viking://{prefix}/{doc_id}/card.md",
        title=doc_id,
        support_scope="Supports node.",
    )


def _context(node: WikiNode, refs: list[SourceRef]) -> GeneratedNodeContext:
    resource_uri = f"viking://wiki/build/layers/depth_{node.depth}/{node.node_id}/"
    return GeneratedNodeContext(
        node=node,
        card=NodeCard(
            doc_id=node.node_id,
            resource_uri=resource_uri,
            title=node.title,
            summary=f"Summary for {node.title}.",
            scope=node.scope,
        ),
        documents=[
            NodeDocument(
                document_id="0001",
                title="High-Level Knowledge",
                content=f"Knowledge for {node.title}.",
            )
        ],
        source_refs=refs,
    )
