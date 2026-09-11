import json

from benchmark.wiki.src.vikingbot_runner import _loads_vikingbot_json


def test_benchmark_disables_automatic_recall_without_disabling_wiki_tools(tmp_path, monkeypatch):
    from benchmark.wiki.src.vikingbot_runner import _build_vikingbot_env

    config = tmp_path / "ov.conf"
    config.write_text("{}")
    monkeypatch.setenv("VIKINGBOT_AUTOMATIC_RECALL", "1")
    env = _build_vikingbot_env(str(config), openviking_root_uri="viking://wiki/nodes")
    assert env["VIKINGBOT_AUTOMATIC_RECALL"] == "0"
    assert env["VIKINGBOT_OPENVIKING_ROOT_URI"] == "viking://wiki/nodes"


def test_loads_vikingbot_json_preserves_valid_latex_backslashes():
    payload = {"text": r"state posterior $\\pi _{jm}$", "trace": []}

    parsed = _loads_vikingbot_json(json.dumps(payload, ensure_ascii=False))

    assert parsed["text"] == payload["text"]


def test_loads_vikingbot_json_repairs_invalid_answer_backslashes():
    raw = r'{"text":"bad latex \q stays readable","trace":[]}'

    parsed = _loads_vikingbot_json(raw)

    assert parsed["text"] == r"bad latex \q stays readable"
