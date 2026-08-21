import asyncio

from openviking_cli.client.http import AsyncHTTPClient


class _Response:
    is_success = True
    status_code = 200
    text = ""

    def json(self):
        return {
            "status": "ok",
            "result": {"resources": [], "memories": [], "skills": [], "total": 0},
            "telemetry": {
                "summary": {
                    "tokens": {
                        "llm": {"input": 0, "output": 0},
                        "embedding": {"total": 7},
                    }
                }
            },
        }


def test_search_preserves_requested_telemetry(monkeypatch):
    client = AsyncHTTPClient(url="http://127.0.0.1:1933")

    async def request(*args, **kwargs):
        return _Response()

    monkeypatch.setattr(client, "_request", request)
    result = asyncio.run(client.search(query="test", telemetry=True))

    assert result["telemetry"]["summary"]["tokens"]["embedding"]["total"] == 7
