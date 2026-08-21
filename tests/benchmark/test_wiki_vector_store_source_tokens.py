from benchmark.wiki.src.core.vector_store import VikingStoreWrapper


class _Encoding:
    def encode(self, text):
        return text.split()


class _Client:
    def __init__(self):
        self.read_calls = []

    def ls(self, uri, **kwargs):
        assert kwargs == {
            "recursive": True,
            "output": "original",
            "show_all_hidden": False,
            "node_limit": None,
            "level_limit": None,
        }
        return [
            {"isDir": True, "name": "section", "uri": f"{uri}/section"},
            {"isDir": False, "name": "body.md", "uri": f"{uri}/body.md"},
            {"isDir": False, "name": "notes.txt", "uri": f"{uri}/notes.txt"},
            {"isDir": False, "name": "figure.png", "uri": f"{uri}/figure.png"},
        ]

    def read(self, uri):
        self.read_calls.append(uri)
        return {
            "viking://resources/doc/body.md": "one two three",
            "viking://resources/doc/notes.txt": "four five",
        }[uri]


def test_count_ingested_source_tokens_counts_only_visible_text_bodies():
    store = VikingStoreWrapper.__new__(VikingStoreWrapper)
    store.enc = _Encoding()
    store.client = _Client()

    count = store._count_ingested_source_tokens(
        ["viking://resources/doc", "viking://resources/doc"]
    )

    assert count == 5
    assert store.client.read_calls == [
        "viking://resources/doc/body.md",
        "viking://resources/doc/notes.txt",
    ]


def test_empty_ingest_reports_zero_source_document_tokens():
    store = VikingStoreWrapper.__new__(VikingStoreWrapper)
    store.enc = _Encoding()

    stats = store.ingest([])

    assert stats["source_documents"] == 0
    assert stats["source_document_tokens"] == 0
    assert stats["source_tokenizer"] == "cl100k_base"
