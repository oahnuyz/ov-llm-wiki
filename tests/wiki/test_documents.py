import pytest

from openviking.wiki.documents import (
    NodeContentGenerator,
    _drop_undeclared_node_document_fields,
    _summarize_extra_fields,
)
from openviking.wiki.llm import WikiLLMRunner
from openviking.wiki.schemas import WikiNode

from .fakes import FakeVLM


def test_extra_field_log_summary_is_bounded():
    summary = _summarize_extra_fields({"z" * 200, "a", "b", "c", "d", "e"})

    assert len(summary) == 6
    assert summary[-1] == "... and 1 more"
    assert max(len(field) for field in summary[:-1]) <= 83


@pytest.mark.asyncio
async def test_node_documents_retries_empty_documents_with_error_at_prompt_end():
    fake_vlm = FakeVLM(
        [
            {"documents": []},
            {
                "documents": [
                    {
                        "title": "Node Synthesis",
                        "content": "# Node Synthesis\n\nValid content.",
                    }
                ],
            },
        ]
    )
    generator = NodeContentGenerator(WikiLLMRunner(fake_vlm))

    documents = await generator.generate_node_documents(
        WikiNode(
            node_id="question_answering",
            title="Question Answering",
            depth=1,
            scope="QA methods and evaluation.",
        ),
        [
            {
                "source_id": "OARW_1",
                "sections": [
                    {
                        "section_uri": "viking://resources/OARW_1/abstract",
                        "content": "Question answering evidence.",
                    }
                ],
            }
        ],
    )

    assert documents[0].document_id == "0001"
    assert documents[0].content == "# Node Synthesis\n\nValid content."
    assert len(fake_vlm.calls) == 2
    assert fake_vlm.calls[1].startswith(fake_vlm.calls[0].rstrip())
    feedback = fake_vlm.calls[1][len(fake_vlm.calls[0].rstrip()):]
    assert "node_documents for question_answering is empty" in feedback
    assert "Return at least one complete document" in feedback
    assert fake_vlm.schemas[0] == fake_vlm.schemas[1]


@pytest.mark.asyncio
async def test_node_documents_drops_only_undeclared_fields_before_strict_validation():
    long_content = "# Node Synthesis\n\n" + ("Concrete supported detail. " * 30)
    fake_vlm = FakeVLM(
        [
            {
                "unexpected": "ignored",
                "documents": [
                    {
                        "title": "Node Synthesis",
                        "content": long_content,
                        "offset": 12,
                        "platforms": ["example"],
                    }
                ],
            }
        ]
    )
    generator = NodeContentGenerator(WikiLLMRunner(fake_vlm))

    documents = await generator.generate_node_documents(
        WikiNode(
            node_id="question_answering",
            title="Question Answering",
            depth=1,
            scope="QA methods and evaluation.",
        ),
        [{"source_id": "doc_1", "sections": []}],
    )

    assert documents[0].model_dump() == {
        "title": "Node Synthesis",
        "content": long_content.strip(),
        "document_id": "0001",
    }
    assert len(fake_vlm.calls) == 1


@pytest.mark.asyncio
async def test_node_documents_retries_short_content_split_across_extra_fields():
    fake_vlm = FakeVLM(
        [
            {
                "documents": [
                    {
                        "title": "Fragmented Node",
                        "content": "# Fragmented Node\n\nOne incomplete sentence.",
                        "mechanisms": "The missing long mechanism section.",
                        "evidence": "The missing long evidence section.",
                    }
                ]
            },
            {
                "documents": [
                    {
                        "title": "Complete Node",
                        "content": "# Complete Node\n\nA complete retry result.",
                    }
                ]
            },
        ]
    )
    generator = NodeContentGenerator(WikiLLMRunner(fake_vlm))

    documents = await generator.generate_node_documents(
        WikiNode(
            node_id="question_answering",
            title="Question Answering",
            depth=1,
            scope="QA methods and evaluation.",
        ),
        [{"source_id": "doc_1", "sections": []}],
    )

    assert documents[0].title == "Complete Node"
    assert len(fake_vlm.calls) == 2
    assert fake_vlm.calls[1].startswith(fake_vlm.calls[0].rstrip())
    feedback = fake_vlm.calls[1][len(fake_vlm.calls[0].rstrip()):]
    assert "document at index 0" in feedback
    assert "< 300" in feedback
    assert "mechanisms" in feedback and "evidence" in feedback
    assert "escape quotes and newlines" in feedback


@pytest.mark.parametrize("length", [299, 300, 499])
def test_fragmented_document_threshold_uses_trimmed_300_characters(length):
    item = {"title": "Node", "content": "  " + "x" * length + "\n", "extra": "text"}
    if length < 300:
        with pytest.raises(RuntimeError, match="< 300"):
            _drop_undeclared_node_document_fields("node", {"documents": [item]})
    else:
        result = _drop_undeclared_node_document_fields("node", {"documents": [item]})
        assert result["documents"] == [{"title": "Node", "content": item["content"]}]


def test_short_document_without_extra_fields_remains_valid():
    item = {"title": "Node", "content": "Short but valid."}
    assert _drop_undeclared_node_document_fields("node", {"documents": [item]}) == {
        "documents": [item]
    }


@pytest.mark.asyncio
async def test_validation_retries_keep_only_latest_feedback_and_stop_after_three_attempts():
    fake_vlm = FakeVLM([
        {"documents": []},
        {"documents": [{"content": "Short", "extra": "fragment"}]},
        {"documents": []},
    ])
    generator = NodeContentGenerator(WikiLLMRunner(fake_vlm))
    with pytest.raises(RuntimeError, match="is empty"):
        await generator.generate_node_documents(
            WikiNode(node_id="node", title="Node", depth=1, scope="Scope"), [],
        )
    assert len(fake_vlm.calls) == 3
    third = fake_vlm.calls[2]
    assert third.startswith(fake_vlm.calls[0].rstrip())
    assert third.count("Previous response validation error") == 1
    assert "attempt 2/3" in third
    assert "likely fragmented" in third
    assert "node_documents for node is empty" not in third
