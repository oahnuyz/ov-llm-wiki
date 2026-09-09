import json
import sys
from pathlib import Path


WIKI_BENCHMARK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WIKI_BENCHMARK_DIR))

from src.vikingbot_runner import (  # noqa: E402
    _build_vikingbot_input_message,
    _load_wiki_directory_catalog,
)


def test_catalog_contains_complete_dag_uris_scopes_and_documents(tmp_path):
    wiki_dir = tmp_path / "viking" / "default" / "wiki"
    wiki_dir.mkdir(parents=True)
    (wiki_dir / "nodes.json").write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "node_id": "parent",
                        "title": "Parent",
                        "depth": 2,
                        "scope": "Parent scope.",
                        "parent_node_ids": [],
                    },
                    {
                        "node_id": "child",
                        "title": "Child",
                        "depth": 1,
                        "scope": "Child scope.",
                        "parent_node_ids": ["parent"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (wiki_dir / "node_index.json").write_text(
        json.dumps(
            {
                "nodes": {
                    "parent": {
                        "primary_uri": "viking://wiki/nodes/parent/",
                        "primary_parent_id": "",
                        "secondary_parent_ids": [],
                    },
                    "child": {
                        "primary_uri": "viking://wiki/nodes/parent/child/",
                        "primary_parent_id": "parent",
                        "secondary_parent_ids": [],
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    catalog = _load_wiki_directory_catalog(str(tmp_path))

    assert "Complete Wiki directory DAG (2 nodes)." in catalog
    assert "scope=Parent scope." in catalog
    assert 'children=["child"]' in catalog
    assert 'parents=["parent"]' in catalog
    assert "viking://wiki/nodes/parent/child" in catalog
    assert "document_uri is <uri>/0001.md" in catalog


def test_first_agent_message_includes_catalog_and_scoped_search_instruction():
    message = _build_vikingbot_input_message(
        "What is the result?",
        "Complete Wiki directory DAG (1 nodes).\n{node catalog}",
        openviking_root_uri="viking://wiki/nodes",
    )

    assert "restricted to viking://wiki/nodes" in message
    assert "target_uri set to those node URIs" in message
    assert "Wiki node document URI" in message
    assert "{node catalog}" in message
    assert message.endswith("Question: What is the result?")
