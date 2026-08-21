import os
import time
from typing import List
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from adapters.base import StandardDoc, StandardSample
import tiktoken
import openviking as ov


class VikingStoreWrapper:
    SOURCE_TOKENIZER = "cl100k_base"
    SOURCE_TEXT_EXTENSIONS = {
        ".json",
        ".markdown",
        ".md",
        ".text",
        ".txt",
        ".yaml",
        ".yml",
    }

    def __init__(self, store_path: str):
        self.store_path = store_path
        if not os.path.exists(store_path):
            os.makedirs(store_path)
        
        self.client = ov.SyncOpenViking(path=store_path)
        
        try:
            self.enc = tiktoken.get_encoding(self.SOURCE_TOKENIZER)
        except Exception as e:
            print(f"[Warning] tiktoken init failed: {e}")
            self.enc = None

    def count_tokens(self, text: str) -> int:
        if not text or not self.enc:
            return 0
        return len(self.enc.encode(str(text)))

    def _count_ingested_source_tokens(self, resource_uris: List[str]) -> int:
        """Count searchable source text after parsing, excluding derived summaries."""
        if not self.enc:
            raise RuntimeError(
                f"Cannot count source document tokens: tokenizer {self.SOURCE_TOKENIZER!r} "
                "is unavailable"
            )

        total_tokens = 0
        counted_uris: set[str] = set()
        for root_uri in dict.fromkeys(resource_uris):
            entries = self.client.ls(
                root_uri,
                recursive=True,
                output="original",
                show_all_hidden=False,
                node_limit=None,
                level_limit=None,
            )
            for entry in entries:
                if entry.get("isDir", False):
                    continue
                uri = str(entry.get("uri", ""))
                suffix = Path(str(entry.get("name", uri))).suffix.lower()
                if not uri or uri in counted_uris or suffix not in self.SOURCE_TEXT_EXTENSIONS:
                    continue
                total_tokens += self.count_tokens(self.client.read(uri))
                counted_uris.add(uri)
        return total_tokens

    def ingest(
        self,
        samples: List[StandardDoc],
        monitor=None,
        ingest_mode="per_file",
    ) -> dict:
        start_time = time.time()
        total_input_tokens = 0
        total_output_tokens = 0
        total_embedding_tokens = 0
        resource_uris: list[str] = []
        
        if not samples:
            return {
                "time": time.time() - start_time,
                "input_tokens": 0,
                "output_tokens": 0,
                "embedding_tokens": 0,
                "source_documents": 0,
                "source_document_tokens": 0,
                "source_tokenizer": self.SOURCE_TOKENIZER,
                "resource_uris": [],
            }
        
        if ingest_mode == "directory":
            doc_paths = [os.path.abspath(s.doc_path) for s in samples]
            common_ancestor = None
            if doc_paths:
                try:
                    common_ancestor = os.path.commonpath(doc_paths)
                except ValueError:
                    common_ancestor = None
            
            if common_ancestor:
                result = self.client.add_resource(
                    common_ancestor,
                    wait=True,
                    telemetry=True,
                )
                if result.get("root_uri"):
                    resource_uris.append(result["root_uri"])
                telemetry = result.get("telemetry", {})
                summary = telemetry.get("summary", {})
                tokens = summary.get("tokens", {})
                llm_tokens = tokens.get("llm", {})
                embedding_tokens = tokens.get("embedding", {})
                total_input_tokens = llm_tokens.get("input", 0)
                total_output_tokens = llm_tokens.get("output", 0)
                total_embedding_tokens = embedding_tokens.get("total", 0)
            else:
                for sample in samples:
                    result = self.client.add_resource(
                        sample.doc_path,
                        wait=True,
                        telemetry=True,
                    )
                    if result.get("root_uri"):
                        resource_uris.append(result["root_uri"])
                    telemetry = result.get("telemetry", {})
                    summary = telemetry.get("summary", {})
                    tokens = summary.get("tokens", {})
                    llm_tokens = tokens.get("llm", {})
                    embedding_tokens = tokens.get("embedding", {})
                    total_input_tokens += llm_tokens.get("input", 0)
                    total_output_tokens += llm_tokens.get("output", 0)
                    total_embedding_tokens += embedding_tokens.get("total", 0)
        else:
            for sample in samples:
                result = self.client.add_resource(
                    sample.doc_path,
                    wait=True,
                    telemetry=True,
                )
                if result.get("root_uri"):
                    resource_uris.append(result["root_uri"])
                telemetry = result.get("telemetry", {})
                summary = telemetry.get("summary", {})
                tokens = summary.get("tokens", {})
                llm_tokens = tokens.get("llm", {})
                embedding_tokens = tokens.get("embedding", {})
                total_input_tokens += llm_tokens.get("input", 0)
                total_output_tokens += llm_tokens.get("output", 0)
                total_embedding_tokens += embedding_tokens.get("total", 0)

        insertion_time = time.time() - start_time
        source_document_tokens = self._count_ingested_source_tokens(resource_uris)

        return {
            "time": insertion_time,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "embedding_tokens": total_embedding_tokens,
            "source_documents": len({os.path.abspath(sample.doc_path) for sample in samples}),
            "source_document_tokens": source_document_tokens,
            "source_tokenizer": self.SOURCE_TOKENIZER,
            "resource_uris": resource_uris,
        }

    def build_wiki(
        self,
        resource_uris: list[str],
        card_input_mode: str = "summary",
        max_card_input_chars: int = 20000,
    ) -> dict:
        start_time = time.time()
        if not resource_uris:
            return {
                "time": time.time() - start_time,
                "status": "skipped",
                "resource_uris": [],
            }
        result = self.client.build_wiki(
            resource_uris=resource_uris,
            card_input_mode=card_input_mode,
            max_card_input_chars=max_card_input_chars,
        )
        result["time"] = time.time() - start_time
        return result

    def retrieve(self, query: str, topk: int, target_uri: str = "viking://resources"):
        """Execute retrieval"""
        return self.client.find(query=query, limit=topk, target_uri=target_uri)

    def read_resource(self, uri: str) -> str:
        """Read resource content"""
        return str(self.client.read(uri))

    def clear(self):
        """Clear the store"""
        self.client.rm("viking://resources", recursive=True)
        try:
            self.client.clear_wiki()
        except Exception:
            pass

    def close(self):
        """Release the underlying OpenViking client if supported."""
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
