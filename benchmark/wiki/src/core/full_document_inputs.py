"""Load original files for Wiki cards without using the ingested chunks."""

import hashlib
import json
from pathlib import Path

from openviking.parse.parsers.pdf import PDFParser
from openviking.wiki.document_manifest import document_manifest_uri, wiki_inputs_from_manifest
from openviking.wiki.schemas import ResourceDocumentDraft


def read_full_document(path: Path, cache_dir: Path) -> str:
    if path.suffix.lower() in {".txt", ".text", ".md", ".markdown", ".mdown", ".mkd"}:
        return path.read_text(encoding="utf-8")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Unsupported full-document input: {path}")

    # Include a format version so changes to the text extraction can invalidate the cache.
    source_hash = hashlib.sha256(b"pdf-text-v1\0" + path.read_bytes()).hexdigest()
    cache_path = cache_dir / (hashlib.sha256(str(path.resolve()).encode()).hexdigest() + ".json")
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached["source_hash"] == source_hash:
            return cached["text"]

    text = PDFParser().extract_text(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"source_path": str(path), "source_hash": source_hash, "text": text}, ensure_ascii=False),
        encoding="utf-8",
    )
    return text


def prepare_full_document_texts(client, resource_uris, documents, cache_dir: Path) -> dict[str, str]:
    """Match dataset document IDs to the resource manifest, never directory-list order."""
    paths = {doc.sample_id: Path(doc.doc_path) for doc in documents}
    if len(paths) != len(documents):
        raise ValueError("Duplicate document IDs in full-document sources")
    inputs = []
    for root in resource_uris:
        manifest = json.loads(client.read(document_manifest_uri(root)))
        drafts = [ResourceDocumentDraft.model_validate(item) for item in manifest["documents"]]
        if not drafts:
            raise ValueError(f"No document boundaries in {document_manifest_uri(root)}")
        inputs.extend(wiki_inputs_from_manifest(root, drafts))
    missing = {doc.doc_id for doc in inputs} - paths.keys()
    if missing:
        raise ValueError(f"Original source files are not mapped for document IDs: {sorted(missing)}")
    return {
        doc.resource_uri: read_full_document(paths[doc.doc_id], cache_dir)
        for doc in inputs
    }
