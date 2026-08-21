import json
from types import SimpleNamespace

from benchmark.wiki.src.pipeline import BenchmarkPipeline
import benchmark.wiki.src.pipeline as pipeline_module


class _Monitor:
    def worker_start(self):
        pass

    def worker_end(self, **kwargs):
        pass

    def get_status_dict(self):
        return {}


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def _pipeline(tmp_path):
    pipeline = BenchmarkPipeline.__new__(BenchmarkPipeline)
    pipeline.config = {
        "dataset_name": "cost_test",
        "execution": {"mode": "vikingbot", "max_workers": 1, "max_queries": None},
    }
    pipeline.output_dir = str(tmp_path)
    pipeline.generated_file = str(tmp_path / "generated_answers.json")
    pipeline.eval_file = str(tmp_path / "qa_eval_detailed_results.json")
    pipeline.report_file = str(tmp_path / "benchmark_metrics_report.json")
    pipeline.resource_manifest_file = str(tmp_path / "imported_resources.json")
    pipeline.metrics_summary = {
        "insertion": {"time": 0, "input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0},
        "deletion": {"time": 0, "input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0},
    }
    pipeline.monitor = _Monitor()
    pipeline.logger = _Logger()
    return pipeline


def test_vikingbot_query_cost_includes_all_agent_and_search_api_tokens(tmp_path, monkeypatch):
    pipeline = _pipeline(tmp_path)
    monkeypatch.setattr(
        pipeline_module,
        "run_vikingbot_query",
        lambda **kwargs: {
            "answer": "answer",
            "total_time_sec": 12.5,
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
            "tools_used": [
                {
                    "tool_name": "openviking_search",
                    "metadata": {
                        "telemetry_collected": True,
                        "api_token_usage": {
                            "llm_input_tokens": 2,
                            "llm_output_tokens": 3,
                            "embedding_tokens": 5,
                        }
                    },
                },
                {
                    "tool_name": "openviking_search",
                    "metadata": {
                        "telemetry_collected": True,
                        "api_token_usage": {
                            "llm_input_tokens": 7,
                            "llm_output_tokens": 11,
                            "embedding_tokens": 13,
                        }
                    },
                },
            ],
            "trace": [],
        },
    )
    qa = SimpleNamespace(
        question="question",
        gold_answers=["answer"],
        category="test",
        evidence=[],
    )

    result = pipeline._process_vikingbot_task({"id": 0, "sample_id": "s", "qa": qa})

    assert result["retrieval"]["latency_sec"] == 12.5
    assert result["token_usage"] == {
        "total_input_tokens": 109,
        "llm_output_tokens": 34,
        "retrieval_embedding_tokens": 18,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "agent_prompt_tokens": 100,
        "agent_completion_tokens": 20,
        "agent_total_tokens": 120,
        "search_llm_input_tokens": 9,
        "search_llm_output_tokens": 14,
        "retrieval_total_tokens": 161,
        "total_tokens": 161,
    }


def test_generation_report_contains_total_and_average_retrieval_cost(tmp_path):
    pipeline = _pipeline(tmp_path)
    qa = SimpleNamespace(question="q", gold_answers=["a"], category="test", evidence=[])
    pipeline.adapter = SimpleNamespace(
        load_and_transform=lambda: [SimpleNamespace(sample_id="s", qa_pairs=[qa, qa])]
    )
    results = iter(
        [
            {
                "_global_index": 0,
                "retrieval": {"latency_sec": 10.0},
                "token_usage": {
                    "agent_prompt_tokens": 100,
                    "agent_completion_tokens": 20,
                    "search_llm_input_tokens": 0,
                    "search_llm_output_tokens": 0,
                    "retrieval_embedding_tokens": 5,
                },
            },
            {
                "_global_index": 1,
                "retrieval": {"latency_sec": 14.0},
                "token_usage": {
                    "agent_prompt_tokens": 200,
                    "agent_completion_tokens": 40,
                    "search_llm_input_tokens": 0,
                    "search_llm_output_tokens": 0,
                    "retrieval_embedding_tokens": 10,
                },
            },
        ]
    )
    pipeline._process_vikingbot_task = lambda task: next(results)

    pipeline.run_generation()

    report = json.loads((tmp_path / "benchmark_metrics_report.json").read_text())
    assert report["Query Efficiency (Average Per Query)"] == {
        "Average Retrieval Time (s)": 12.0,
        "Average Retrieval Token Cost": 187.5,
    }
    assert report["Query Efficiency (Total Dataset)"]["Total Retrieval Token Cost"] == 375


def test_insertion_report_contains_total_time_and_token_cost(tmp_path):
    pipeline = _pipeline(tmp_path)
    pipeline.config["paths"] = {"doc_output_dir": str(tmp_path / "docs")}
    pipeline.config["execution"]["ingest_mode"] = "per_file"
    pipeline.adapter = SimpleNamespace(data_prepare=lambda doc_dir: [SimpleNamespace()])
    pipeline.db = SimpleNamespace(
        ingest=lambda *args, **kwargs: {
            "time": 30.0,
            "input_tokens": 100,
            "output_tokens": 20,
            "embedding_tokens": 80,
            "source_documents": 2,
            "source_document_tokens": 1234,
            "source_tokenizer": "cl100k_base",
            "resource_uris": ["viking://resources/test"],
        }
    )

    pipeline.run_import()

    report = json.loads((tmp_path / "benchmark_metrics_report.json").read_text())
    insertion = report["Insertion Efficiency (Total Dataset)"]
    assert insertion["Total Insertion Time (s)"] == 30.0
    assert insertion["Total Source Documents"] == 2
    assert insertion["Total Source Document Tokens"] == 1234
    assert insertion["Source Tokenizer"] == "cl100k_base"
    assert insertion["Total Insertion Token Cost"] == 200


def test_evaluation_report_contains_normalized_accuracy(tmp_path):
    pipeline = _pipeline(tmp_path)
    generated = {
        "results": [
            {"_global_index": 0, "metrics": {"Recall": 0.0}},
            {"_global_index": 1, "metrics": {"Recall": 0.0}},
        ]
    }
    (tmp_path / "generated_answers.json").write_text(json.dumps(generated))

    def evaluate(item):
        score = 4.0 if item["_global_index"] == 0 else 2.0
        return {
            **item,
            "metrics": {"Recall": 0.0, "F1": 0.0, "Accuracy": score},
        }

    pipeline._process_evaluation_task = evaluate
    pipeline.run_evaluation()

    report = json.loads((tmp_path / "benchmark_metrics_report.json").read_text())
    assert report["Performance Metrics"]["Normalized Accuracy (0-1)"] == 0.75
