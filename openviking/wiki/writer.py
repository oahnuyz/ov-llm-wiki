"""vikingfs writer for Wiki assets."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

from pydantic import BaseModel

from openviking.server.identity import RequestContext
from openviking.storage import VikingDBManager
from openviking.storage.content_write import ContentWriteCoordinator
from openviking.storage.viking_fs import VikingFS
from openviking_cli.exceptions import NotFoundError

from . import uri as wiki_uri
from .config import WikiConfig


class WikiVikingFSWriter:
    def __init__(
        self,
        *,
        viking_fs: VikingFS,
        vikingdb: VikingDBManager,
        ctx: RequestContext,
        config: WikiConfig,
        content_writer: Any | None = None,
    ):
        self.viking_fs = viking_fs
        self.ctx = ctx
        self.config = config
        self._writer = content_writer or ContentWriteCoordinator(viking_fs=viking_fs, vikingdb=vikingdb)

    async def ensure_dirs(self, node_ids: list[str] | None = None) -> None:
        """创建必要的目录"""
        dirs = [
            wiki_uri.wiki_root(self.config),
            wiki_uri.cards_dir(self.config),
            wiki_uri.nodes_dir(self.config),
            wiki_uri.run_dir(self.config),
        ]
        for node_id in node_ids or []:
            dirs.extend(
                [
                    wiki_uri.node_root_uri(self.config, node_id),
                    wiki_uri.node_sources_dir(self.config, node_id),
                ]
            )

        for directory in dirs:
            await self.viking_fs.mkdir(directory, exist_ok=True, ctx=self.ctx)

    async def write_text(self, uri: str, content: str, *, abstract: str | None = None) -> None:
        try:
            await self._writer.write(
                uri=uri,
                content=content,
                abstract=abstract,
                mode="create",
                wait=True,
                ctx=self.ctx,
            )
        except Exception as exc:
            if not isinstance(exc, NotFoundError) and "exist" not in str(exc).lower():
                raise
            await self._writer.write(
                uri=uri,
                content=content,
                abstract=abstract,
                mode="replace",
                wait=True,
                ctx=self.ctx,
            )

    async def write_json(self, uri: str, payload: Any) -> None:
        content = json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2)
        await self.write_text(uri, content)

    async def write_jsonl(self, uri: str, rows: list[Any]) -> None:
        content = "\n".join(json.dumps(_to_jsonable(row), ensure_ascii=False) for row in rows)
        if content:
            content += "\n"
        await self.write_text(uri, content)

    async def write_internal_json(self, uri: str, payload: Any) -> None:
        """Persist non-indexed build state directly in VikingFS."""
        content = json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2)
        await self.viking_fs.write_file(uri, content, ctx=self.ctx)

    async def ensure_node_uri_dirs(self, node_uri: str) -> None:
        dirs = [
            node_uri,
            wiki_uri.node_sources_dir_at(node_uri),
        ]
        for directory in dirs:
            await self.viking_fs.mkdir(directory, exist_ok=True, ctx=self.ctx)

    async def link_node(self, from_uri: str, to_uri: str, *, reason: str) -> None:
        await self.viking_fs.link(from_uri, to_uri, reason, ctx=self.ctx)

    async def read_json(self, uri: str) -> Any:
        content = await self.viking_fs.read_file(uri, ctx=self.ctx)
        return json.loads(str(content))

    async def read_jsonl(self, uri: str) -> list[Any]:
        content = await self.viking_fs.read_file(uri, ctx=self.ctx)
        return [json.loads(line) for line in str(content).splitlines() if line.strip()]

    async def reset_card_outputs(self) -> None:
        """Remove persisted cards and all downstream outputs for a fresh card run."""
        await self._remove_if_present(wiki_uri.cards_dir(self.config), recursive=True)
        await self.reset_node_outputs()
        await self.ensure_dirs()

    async def reset_node_outputs(self, *, preserve_build: bool = False) -> None:
        """Remove node/run outputs while preserving reusable document cards."""
        await self._remove_if_present(wiki_uri.nodes_dir(self.config), recursive=True)
        if not preserve_build:
            await self._remove_if_present(wiki_uri.build_dir(self.config), recursive=True)
        await self._remove_if_present(f"{wiki_uri.wiki_root(self.config)}nodes.json")
        await self._remove_if_present(f"{wiki_uri.wiki_root(self.config)}source_assignments.json")
        await self._remove_if_present(f"{wiki_uri.wiki_root(self.config)}node_index.json")
        await self._remove_if_present(wiki_uri.run_dir(self.config), recursive=True)
        await self.ensure_dirs()

    async def _remove_if_present(self, uri: str, *, recursive: bool = False) -> None:
        if not await self.viking_fs.exists(uri, ctx=self.ctx):
            return
        await self.viking_fs.rm(uri, recursive=recursive, ctx=self.ctx)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value
