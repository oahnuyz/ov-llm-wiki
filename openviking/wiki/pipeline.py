"""Wiki generation orchestrator."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from typing import Literal

from .assignments import SourceRefBuilder
from .cards import DocumentCardGenerator, render_card_markdown
from .config import WikiConfig
from .content_loader import WikiCardInputMode, WikiContentLoader
from .documents import NodeContentGenerator
from .llm import WikiLLMRunner
from .nodes import NodeDiscoveryRunner
from .schemas import (
    DocumentCard,
    GeneratedNodeContext,
    NodeCard,
    PipelineArtifacts,
    ResourceDocument,
    SourceAssignmentResult,
    SourceRef,
    WikiNode,
    WikiResourceInput,
)
from .uri import (
    build_dir,
    build_node_root_uri,
    card_json_uri,
    card_md_uri,
    child_node_root_uri,
    node_card_json_uri_at,
    node_card_md_uri_at,
    node_document_uri_at,
    node_root_uri,
    node_sources_dir_at,
    run_dir,
    wiki_root,
)
from .writer import WikiVikingFSWriter

logger = logging.getLogger(__name__)
_RESERVED_NODE_PATH_NAMES = {"sources"}
_SENSITIVE_CONFIG_KEYS = {
    "api_key",
    "apikey",
    "api-key",
    "access_key",
    "access-key",
    "secret_key",
    "secret-key",
    "secret",
    "token",
    "authorization",
    "password",
}


class WikiPipeline:
    def __init__(
        self,
        writer: WikiVikingFSWriter,
        config: WikiConfig | None = None,
        llm: WikiLLMRunner | None = None,
    ):
        self.config = config or WikiConfig()
        self.llm = llm or WikiLLMRunner(
            vlm_config=self.config.vlm_config,
            request_timeout_seconds=self.config.limits.llm_request_timeout_seconds,
            aggregation_agent_max_tokens=self.config.limits.aggregation_agent_max_tokens,
        )
        self.writer = writer
        self.card_generator = DocumentCardGenerator(
            self.llm,
            max_concurrent=self.config.limits.max_concurrent_cards,
        )
        self.node_discovery = NodeDiscoveryRunner(self.llm, self.config)
        self.source_ref_builder = SourceRefBuilder(self.config)
        self.content_generator = NodeContentGenerator(self.llm)

    async def run_from_inputs(
        self,
        docs: list[WikiResourceInput],
        *,
        content_loader: WikiContentLoader,
        card_input_mode: WikiCardInputMode | str = WikiCardInputMode.SUMMARY,
        max_card_input_chars: int = 20000,
        build_stage: Literal["all", "cards", "nodes"] = "all",
    ) -> PipelineArtifacts:
        if not docs:
            raise ValueError("Wiki pipeline requires at least one resource document")
        if build_stage not in {"all", "cards", "nodes"}:
            raise ValueError("build_stage must be one of: all, cards, nodes")

        artifacts = PipelineArtifacts()
        await self.writer.ensure_dirs()

        if build_stage == "nodes":
            cards = await self._load_persisted_cards(docs)
            logger.info("[Wiki] Loaded %d persisted document cards", len(cards))
            source_docs = await self._load_documents(
                docs,
                content_loader=content_loader,
                mode=WikiCardInputMode.RAW_CHUNK,
                max_card_input_chars=max_card_input_chars,
            )
            await self.writer.reset_node_outputs()
            return await self._run_from_cards(
                cards,
                artifacts,
                {doc.doc_id: doc for doc in source_docs},
                write_cards=False,
                build_stage="nodes",
            )

        logger.info(
            "[Wiki] Generating document cards for %d docs from %s inputs",
            len(docs),
            card_input_mode,
        )

        input_mode = WikiCardInputMode(card_input_mode)
        resource_docs = await self._load_documents(
            docs,
            content_loader=content_loader,
            mode=input_mode,
            max_card_input_chars=max_card_input_chars,
        )
        source_docs_task = (
            None
            if input_mode == WikiCardInputMode.RAW_CHUNK or build_stage == "cards"
            else asyncio.create_task(
                self._load_documents(
                    docs,
                    content_loader=content_loader,
                    mode=WikiCardInputMode.RAW_CHUNK,
                    max_card_input_chars=max_card_input_chars,
                )
            )
        )
        try:
            cards = await self.card_generator.generate(resource_docs)
        except Exception:
            if source_docs_task is not None:
                source_docs_task.cancel()
                await asyncio.gather(source_docs_task, return_exceptions=True)
            raise
        source_docs = resource_docs if source_docs_task is None else await source_docs_task
        logger.info("[Wiki] Generated %d document cards", len(cards))
        await self.writer.reset_card_outputs()
        await self._write_cards(cards)
        artifacts.cards = list(cards)
        if build_stage == "cards":
            await self._write_run_records(build_stage="cards")
            logger.info("[Wiki] Completed document card generation: cards=%d", len(cards))
            return artifacts
        return await self._run_from_cards(
            cards,
            artifacts,
            {doc.doc_id: doc for doc in source_docs},
            write_cards=False,
            build_stage="all",
        )

    async def _load_documents(
        self,
        docs: list[WikiResourceInput],
        *,
        content_loader: WikiContentLoader,
        mode: WikiCardInputMode,
        max_card_input_chars: int,
    ) -> list[ResourceDocument]:
        max_concurrent = max(1, self.config.limits.max_concurrent_cards)
        sem = asyncio.Semaphore(max_concurrent)
        results: list[ResourceDocument | None] = [None] * len(docs)

        async def _load_one(index: int, doc: WikiResourceInput) -> None:
            async with sem:
                results[index] = await content_loader.load_document(
                    doc,
                    mode=mode,
                    max_card_input_chars=max_card_input_chars,
                )

        await asyncio.gather(*[_load_one(index, doc) for index, doc in enumerate(docs)])
        if any(result is None for result in results):
            raise RuntimeError("resource document loading did not produce all documents")
        return [result for result in results if result is not None]

    async def _load_persisted_cards(
        self,
        docs: list[WikiResourceInput],
    ) -> list[DocumentCard]:
        cards: list[DocumentCard] = []
        for doc in docs:
            uri = card_json_uri(self.config, doc.doc_id)
            try:
                payload = await self.writer.read_json(uri)
            except Exception as exc:
                raise RuntimeError(
                    f"persisted Wiki card is missing or invalid for doc_id={doc.doc_id}; "
                    "run the cards build stage first"
                ) from exc
            card = DocumentCard.model_validate(payload)
            if card.doc_id != doc.doc_id:
                raise RuntimeError(
                    f"persisted Wiki card ID mismatch: expected {doc.doc_id}, got {card.doc_id}"
                )
            cards.append(card)
        return cards

    async def _run_from_cards(
        self,
        cards: list[DocumentCard],
        artifacts: PipelineArtifacts,
        resource_documents_by_id: dict[str, ResourceDocument],
        *,
        write_cards: bool = True,
        build_stage: Literal["all", "nodes"] = "all",
    ) -> PipelineArtifacts:
        all_cards: list[DocumentCard | NodeCard] = list(cards)
        source_documents_by_id = dict(resource_documents_by_id)
        artifacts.cards = all_cards
        if write_cards:
            await self._write_cards(cards)

        all_nodes: list[WikiNode] = []
        all_source_refs_by_node: dict[str, list[SourceRef]] = {}
        all_unassigned_source_ids: list[str] = []
        all_contexts: list[GeneratedNodeContext] = []
        current_layer_cards: list[DocumentCard | NodeCard] = list(cards)
        reserved_node_ids = {card.doc_id for card in cards} | _RESERVED_NODE_PATH_NAMES

        depth = 1
        while current_layer_cards:
            source_cards = current_layer_cards
            logger.info(
                "[Wiki] Discovering depth=%d nodes from %d current-layer cards",
                depth,
                len(source_cards),
            )
            discovery = await self.node_discovery.discover_layer(
                source_cards,
                depth=depth,
                reserved_node_ids=reserved_node_ids,
                on_turn_complete=lambda _record: self.writer.write_jsonl(
                    f"{run_dir(self.config)}aggregation_operations.jsonl",
                    self.node_discovery.aggregation_logs,
                ),
            )
            layer_nodes = discovery.nodes
            layer_nodes = [
                node.model_copy(update={"aggregation_order": index})
                for index, node in enumerate(layer_nodes)
            ]
            logger.info(
                "[Wiki] Depth=%d discovered %d directory nodes",
                depth,
                len(layer_nodes),
            )
            if not layer_nodes:
                logger.info("[Wiki] Depth=%d produced no directory nodes; stopping", depth)
                break

            logger.info(
                "[Wiki] Building source refs for %d nodes from %d source cards",
                len(layer_nodes),
                len(source_cards),
            )
            assignment_result = SourceAssignmentResult(
                source_refs_by_node=self.source_ref_builder.build_refs_by_node(
                    discovery.source_assignments.assignments,
                    source_cards,
                ),
                unassigned_source_ids=discovery.source_assignments.unassigned_source_ids,
            )
            logger.info(
                "[Wiki] Depth=%d produced %d source refs",
                depth,
                sum(len(refs) for refs in assignment_result.source_refs_by_node.values()),
            )

            layer_nodes = _with_child_node_ids_from_refs_for_layer(layer_nodes, assignment_result)
            if depth > 1:
                all_nodes = _assign_parent_node_links(all_nodes, layer_nodes)

            all_nodes.extend(layer_nodes)
            reserved_node_ids.update(node.node_id for node in layer_nodes)
            artifacts.nodes = all_nodes
            await self.writer.write_internal_json(
                f"{build_dir(self.config)}nodes.json",
                {"nodes": all_nodes},
            )

            all_source_refs_by_node.update(assignment_result.source_refs_by_node)
            all_unassigned_source_ids.extend(assignment_result.unassigned_source_ids)
            artifacts.source_refs_by_node = all_source_refs_by_node
            await self.writer.write_internal_json(
                f"{build_dir(self.config)}source_assignments.json",
                {
                    "source_refs_by_node": all_source_refs_by_node,
                    "unassigned_source_ids": all_unassigned_source_ids,
                },
            )

            layer_contexts = await self._generate_layer_contexts(
                layer_nodes,
                assignment_result,
                source_documents_by_id,
                depth=depth,
            )
            all_contexts.extend(layer_contexts)
            current_layer_cards = [context.card for context in layer_contexts]
            all_cards.extend(current_layer_cards)
            source_documents_by_id.update(
                {
                    context.node.node_id: _resource_document_for_node(context)
                    for context in layer_contexts
                }
            )
            logger.info(
                "[Wiki] Depth=%d generated %d node contexts (total=%d)",
                depth,
                len(layer_contexts),
                len(all_contexts),
            )

            artifacts.node_contexts = all_contexts
            artifacts.cards = all_cards

            depth += 1

        if len(all_contexts) != len(all_nodes):
            raise RuntimeError(
                "node aggregation did not produce a complete context for every materialized node"
            )
        await self.writer.write_internal_json(
            f"{build_dir(self.config)}manifest.json",
            {
                "status": "ready_for_materialization",
                "nodes": len(all_nodes),
                "depths": max((node.depth for node in all_nodes), default=0),
            },
        )
        logger.info("[Wiki] Materializing %d nodes into the directory tree", len(all_nodes))
        await self._materialize_node_tree(
            artifacts,
            document_cards=list(cards),
            unassigned_source_ids=all_unassigned_source_ids,
        )
        await self.writer.write_internal_json(
            f"{build_dir(self.config)}manifest.json",
            {
                "status": "complete",
                "nodes": len(artifacts.nodes),
                "depths": max((node.depth for node in artifacts.nodes), default=0),
            },
        )
        await self._write_run_records(build_stage=build_stage)
        logger.info(
            "[Wiki] Completed wiki generation: cards=%d nodes=%d contexts=%d wiki_root=%s",
            len(artifacts.cards),
            len(artifacts.nodes),
            len(artifacts.node_contexts),
            wiki_root(self.config),
        )
        return artifacts

    async def _generate_layer_contexts(
        self,
        nodes: list[WikiNode],
        assignment_result: SourceAssignmentResult,
        source_documents_by_id: dict[str, ResourceDocument],
        *,
        depth: int,
    ) -> list[GeneratedNodeContext]:
        max_concurrent = max(1, self.config.limits.max_concurrent_nodes)
        sem = asyncio.Semaphore(max_concurrent)
        contexts: list[GeneratedNodeContext | None] = [None] * len(nodes)
        logger.info(
            "[Wiki] Depth=%d generating %d node contexts with max_concurrent=%d",
            depth,
            len(nodes),
            max_concurrent,
        )

        async def _generate_one(index: int, node: WikiNode) -> None:
            async with sem:
                reusable = await self._load_reusable_node_context(
                    node,
                    assignment_result,
                    depth=depth,
                )
                if reusable is not None:
                    contexts[index] = reusable
                    logger.info("[Wiki] Depth=%d reused node context: %s", depth, node.node_id)
                    return
                logger.info("[Wiki] Depth=%d generating node context: %s", depth, node.node_id)
                contexts[index] = await self._generate_node_context(
                    node,
                    assignment_result,
                    source_documents_by_id,
                )
                await self.writer.write_internal_json(
                    f"{build_node_root_uri(self.config, depth, node.node_id)}context.json",
                    contexts[index],
                )
                logger.info("[Wiki] Depth=%d generated node context: %s", depth, node.node_id)

        await asyncio.gather(*[_generate_one(index, node) for index, node in enumerate(nodes)])
        if any(context is None for context in contexts):
            raise RuntimeError("node context generation did not produce all contexts")
        return [context for context in contexts if context is not None]

    async def _load_reusable_node_context(
        self,
        node: WikiNode,
        assignment_result: SourceAssignmentResult,
        *,
        depth: int,
    ) -> GeneratedNodeContext | None:
        expected_refs = assignment_result.source_refs_by_node.get(node.node_id)
        if not expected_refs:
            return None
        staging_uri = build_node_root_uri(self.config, depth, node.node_id)
        try:
            payload = await self.writer.read_json(f"{staging_uri}context.json")
            context = GeneratedNodeContext.model_validate(payload)
        except Exception:
            return None
        if (
            context.node != node
            or context.source_refs != expected_refs
            or context.card.resource_uri != staging_uri
        ):
            return None
        return context

    async def _generate_node_context(
        self,
        node: WikiNode,
        assignment_result: SourceAssignmentResult,
        source_documents_by_id: dict[str, ResourceDocument],
    ) -> GeneratedNodeContext:
        source_refs = assignment_result.source_refs_by_node.get(node.node_id)
        if not source_refs:
            raise RuntimeError(f"active node {node.node_id} has no source refs")

        source_documents = _source_documents_for_refs(source_refs, source_documents_by_id)
        documents = await self.content_generator.generate_node_documents(
            node,
            source_documents,
        )
        staging_uri = build_node_root_uri(self.config, node.depth, node.node_id)
        card = await self.card_generator.generate_node_card(
            node,
            documents,
            resource_uri=staging_uri,
        )

        context = GeneratedNodeContext(
            node=node,
            card=card,
            documents=documents,
            source_refs=source_refs,
        )
        return context

    async def _materialize_node_tree(
        self,
        artifacts: PipelineArtifacts,
        *,
        document_cards: list[DocumentCard],
        unassigned_source_ids: list[str],
    ) -> None:
        placed_nodes, path_by_id = _build_node_tree_paths(artifacts.nodes, self.config)
        contexts_by_id = {context.node.node_id: context for context in artifacts.node_contexts}
        if set(contexts_by_id) != set(path_by_id):
            raise RuntimeError("cannot materialize an incomplete Wiki node graph")

        max_concurrent = max(1, self.config.limits.max_concurrent_nodes)
        sem = asyncio.Semaphore(max_concurrent)
        materialized_contexts: list[GeneratedNodeContext | None] = [None] * len(placed_nodes)

        async def _materialize_one(index: int, node: WikiNode) -> None:
            async with sem:
                node_uri = path_by_id[node.node_id]
                context = contexts_by_id[node.node_id]
                source_refs = _refs_with_materialized_node_uris(
                    context.source_refs,
                    path_by_id,
                )
                await self.writer.ensure_node_uri_dirs(node_uri)
                await self._write_source_refs_at(node_uri, source_refs)
                card = context.card.model_copy(update={"resource_uri": node_uri, "markdown": ""})
                card = card.model_copy(update={"markdown": render_card_markdown(card)})
                document_abstract = node.scope
                for document in context.documents:
                    await self.writer.write_text(
                        node_document_uri_at(node_uri, document.document_id),
                        document.content,
                        abstract=document_abstract,
                    )
                await self._write_node_card_at(node_uri, card)
                materialized_contexts[index] = GeneratedNodeContext(
                    node=node,
                    card=card,
                    documents=context.documents,
                    source_refs=source_refs,
                )

        await asyncio.gather(
            *[_materialize_one(index, node) for index, node in enumerate(placed_nodes)]
        )
        if any(context is None for context in materialized_contexts):
            raise RuntimeError("directory tree materialization did not produce every node")

        for node in placed_nodes:
            node_uri = path_by_id[node.node_id]
            for secondary_parent_id in node.parent_node_ids[1:]:
                from_uri = path_by_id[secondary_parent_id]
                await self.writer.link_node(
                    from_uri,
                    node_uri,
                    reason=f"wiki_child:secondary_parent:{node.node_id}",
                )

        contexts = [context for context in materialized_contexts if context is not None]
        source_refs_by_node = {
            context.node.node_id: context.source_refs for context in contexts
        }
        node_index = {
            node.node_id: {
                "primary_parent_id": node.primary_parent_id,
                "primary_uri": path_by_id[node.node_id],
                "document_uris": [
                    node_document_uri_at(path_by_id[node.node_id], document.document_id)
                    for document in contexts_by_id[node.node_id].documents
                ],
                "secondary_parent_ids": node.parent_node_ids[1:],
                "depth": node.depth,
                "aggregation_order": node.aggregation_order,
            }
            for node in placed_nodes
        }
        await self.writer.write_json(f"{wiki_root(self.config)}nodes.json", {"nodes": placed_nodes})
        await self.writer.write_json(
            f"{wiki_root(self.config)}source_assignments.json",
            {
                "source_refs_by_node": source_refs_by_node,
                "unassigned_source_ids": unassigned_source_ids,
            },
        )
        await self.writer.write_json(
            f"{wiki_root(self.config)}node_index.json",
            {"nodes": node_index},
        )
        artifacts.nodes = placed_nodes
        artifacts.node_contexts = contexts
        artifacts.source_refs_by_node = source_refs_by_node
        artifacts.cards = [*document_cards, *[context.card for context in contexts]]

    async def _write_cards(self, cards: list[DocumentCard]) -> None:
        for card in cards:
            await self.writer.write_text(card_md_uri(self.config, card.doc_id), card.markdown)
            await self.writer.write_json(card_json_uri(self.config, card.doc_id), card)

    async def _write_node_card_at(self, node_uri: str, card: NodeCard) -> None:
        await self.writer.write_text(node_card_md_uri_at(node_uri), card.markdown)
        await self.writer.write_json(node_card_json_uri_at(node_uri), card)

    async def _write_source_refs_at(self, node_uri: str, source_refs: list[SourceRef]) -> None:
        for source_ref in source_refs:
            await self.writer.write_json(
                f"{node_sources_dir_at(node_uri)}{source_ref.doc_id}.ref.json",
                source_ref,
            )

    async def _write_run_records(self, *, build_stage: str) -> None:
        run_root = run_dir(self.config)
        run_config = {
            "pipeline_version": self.config.pipeline_version,
            "build_stage": build_stage,
            "model_config": _redact_sensitive_config(self.config.vlm_config or {}),
            "limits": asdict(self.config.limits),
        }
        await self.writer.write_json(f"{run_root}config.json", run_config)
        await self.writer.write_jsonl(f"{run_root}prompts.jsonl", self.llm.log.prompts)
        await self.writer.write_jsonl(f"{run_root}raw_outputs.jsonl", self.llm.log.raw_outputs)
        await self.writer.write_jsonl(
            f"{run_root}aggregation_operations.jsonl",
            self.node_discovery.aggregation_logs,
        )
        await self.writer.write_text(
            f"{run_root}logs.md",
            "# Wiki Run Logs\n\nGeneration completed without pipeline-level errors.\n",
        )


def _source_documents_for_refs(
    source_refs: list[SourceRef],
    source_documents_by_id: dict[str, ResourceDocument],
) -> list[dict]:
    source_documents: list[dict] = []
    for source_ref in source_refs:
        resource_document = source_documents_by_id.get(source_ref.doc_id)
        if not resource_document:
            raise RuntimeError(f"node source ref has no loaded source document: {source_ref.doc_id}")
        if not resource_document.source_sections:
            raise RuntimeError(f"node source ref has no source sections: {source_ref.doc_id}")
        source_documents.append(
            {
                "source_id": source_ref.doc_id,
                "sections": [
                    section.model_dump(mode="json")
                    for section in resource_document.source_sections
                ],
            }
        )
    return source_documents


def _redact_sensitive_config(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: "***REDACTED***" if str(key).lower() in _SENSITIVE_CONFIG_KEYS else _redact_sensitive_config(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive_config(item) for item in value]
    return value


def _with_child_node_ids_from_refs(
    node: WikiNode,
    assignment_result: SourceAssignmentResult,
) -> WikiNode:
    child_node_ids = [
        ref.doc_id
        for ref in assignment_result.source_refs_by_node.get(node.node_id, [])
        if ref.ref_type == "wiki_node"
    ]
    if not child_node_ids:
        return node
    return node.model_copy(update={"child_node_ids": child_node_ids})


def _with_child_node_ids_from_refs_for_layer(
    nodes: list[WikiNode],
    assignment_result: SourceAssignmentResult,
) -> list[WikiNode]:
    return [_with_child_node_ids_from_refs(node, assignment_result) for node in nodes]


def _assign_parent_node_links(
    nodes: list[WikiNode],
    parent_nodes: list[WikiNode],
) -> list[WikiNode]:
    parent_ids_by_child_id: dict[str, list[str]] = {}
    for parent in parent_nodes:
        for child_node_id in parent.child_node_ids:
            parent_ids_by_child_id.setdefault(child_node_id, []).append(parent.node_id)

    updated_nodes: list[WikiNode] = []
    for node in nodes:
        parent_ids = parent_ids_by_child_id.get(node.node_id)
        if not parent_ids:
            updated_nodes.append(node)
            continue
        updated_nodes.append(
            node.model_copy(
                update={"parent_node_ids": list(dict.fromkeys([*node.parent_node_ids, *parent_ids]))}
            )
        )
    return updated_nodes


def _build_node_tree_paths(
    nodes: list[WikiNode],
    config: WikiConfig,
) -> tuple[list[WikiNode], dict[str, str]]:
    nodes_by_id = {node.node_id: node for node in nodes}
    if len(nodes_by_id) != len(nodes):
        raise RuntimeError("Wiki node graph contains duplicate node IDs")

    placed_nodes: list[WikiNode] = []
    for node in nodes:
        unknown_parent_ids = [
            parent_id for parent_id in node.parent_node_ids if parent_id not in nodes_by_id
        ]
        if unknown_parent_ids:
            raise RuntimeError(
                f"Wiki node {node.node_id} references unknown parents: {unknown_parent_ids}"
            )
        primary_parent_id = node.parent_node_ids[0] if node.parent_node_ids else ""
        placed_nodes.append(node.model_copy(update={"primary_parent_id": primary_parent_id}))

    placed_by_id = {node.node_id: node for node in placed_nodes}
    path_by_id: dict[str, str] = {}
    visiting: set[str] = set()

    def _path_for(node_id: str) -> str:
        existing = path_by_id.get(node_id)
        if existing:
            return existing
        if node_id in visiting:
            raise RuntimeError(f"Wiki node graph contains a primary-parent cycle at {node_id}")
        visiting.add(node_id)
        node = placed_by_id[node_id]
        if node.primary_parent_id:
            parent = placed_by_id[node.primary_parent_id]
            if parent.depth <= node.depth:
                raise RuntimeError(
                    f"Wiki parent {parent.node_id} must be above child {node.node_id}"
                )
            path = child_node_root_uri(_path_for(parent.node_id), node.node_id)
        else:
            path = node_root_uri(config, node.node_id)
        visiting.remove(node_id)
        path_by_id[node_id] = path
        return path

    for node in placed_nodes:
        _path_for(node.node_id)
    return placed_nodes, path_by_id


def _refs_with_materialized_node_uris(
    source_refs: list[SourceRef],
    path_by_id: dict[str, str],
) -> list[SourceRef]:
    refs: list[SourceRef] = []
    for source_ref in source_refs:
        if source_ref.ref_type != "wiki_node":
            refs.append(source_ref)
            continue
        node_uri = path_by_id.get(source_ref.doc_id)
        if not node_uri:
            raise RuntimeError(
                f"Wiki source ref has no materialized node path: {source_ref.doc_id}"
            )
        refs.append(
            source_ref.model_copy(
                update={
                    "resource_uri": node_uri,
                    "card_uri": node_card_md_uri_at(node_uri),
                }
            )
        )
    return refs


def _resource_document_for_node(context: GeneratedNodeContext) -> ResourceDocument:
    node_uri = context.card.resource_uri
    return ResourceDocument(
        doc_id=context.node.node_id,
        resource_uri=node_uri,
        title=context.node.title,
        content_or_structure="\n\n".join(document.content for document in context.documents),
        source_sections=[
            {
                "section_uri": node_document_uri_at(node_uri, document.document_id),
                "content": document.content,
            }
            for document in context.documents
        ],
        metadata={"source_type": "wiki_node"},
    )
