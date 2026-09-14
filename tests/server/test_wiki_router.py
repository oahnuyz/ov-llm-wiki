from types import SimpleNamespace

import httpx

from openviking import AsyncOpenViking
from openviking.wiki.router import BuildWikiRequest, ClearWikiRequest, build_wiki, clear_wiki
from openviking_cli.client.http import AsyncHTTPClient


async def test_build_wiki_router_calls_service(monkeypatch):
    seen = {}

    async def fake_build_wiki(**kwargs):
        seen.update(kwargs)
        return {
            "status": "success",
            "wiki_root_uri": kwargs["wiki_root_uri"],
            "resource_uris": kwargs["resource_uris"],
        }

    service = SimpleNamespace(wiki=SimpleNamespace(build_wiki=fake_build_wiki))
    monkeypatch.setattr("openviking.wiki.router.get_service", lambda: service)

    body = await build_wiki(
        BuildWikiRequest(
            resource_uris=["viking://resources/demo"],
            wiki_root_uri="viking://wiki/",
            card_input_mode="summary",
            max_card_input_chars=20000,
            build_stage="cards",
        ),
        _ctx=object(),
    )

    assert body["result"]["wiki_root_uri"] == "viking://wiki/"
    assert seen["resource_uris"] == ["viking://resources/demo"]
    assert seen["card_input_mode"] == "summary"
    assert seen["build_stage"] == "cards"


async def test_clear_wiki_router_calls_service(monkeypatch):
    seen = {}

    async def fake_clear_wiki(**kwargs):
        seen.update(kwargs)
        return {
            "status": "success",
            "wiki_root_uri": kwargs["wiki_root_uri"],
            "cleared": False,
            "missing": True,
        }

    service = SimpleNamespace(wiki=SimpleNamespace(clear_wiki=fake_clear_wiki))
    monkeypatch.setattr("openviking.wiki.router.get_service", lambda: service)

    body = await clear_wiki(
        ClearWikiRequest(wiki_root_uri="viking://wiki/"),
        _ctx=object(),
    )

    assert body["result"]["missing"] is True
    assert seen["wiki_root_uri"] == "viking://wiki/"


async def test_full_document_survives_public_client_http_and_router(monkeypatch):
    seen = {}

    async def service_build(**kwargs):
        seen.update(kwargs)
        return {"status": "success"}

    monkeypatch.setattr(
        "openviking.wiki.router.get_service",
        lambda: SimpleNamespace(wiki=SimpleNamespace(build_wiki=service_build)),
    )

    async def request(method, path, *, json):
        assert (method, path) == ("POST", "/api/v1/wiki/build")
        response = await build_wiki(BuildWikiRequest(**json), _ctx=object())
        return httpx.Response(200, json=response)

    http_client = AsyncHTTPClient.__new__(AsyncHTTPClient)
    http_client._request = request
    client = AsyncOpenViking.__new__(AsyncOpenViking)
    client._client = http_client
    client._initialized = True
    text = "正文\n" * 10000 + "Final evidence"
    result = await client.build_wiki(
        ["viking://resources/docs"], card_input_mode="full_document", build_stage="cards",
        full_document_texts={"viking://resources/docs/a": text}, max_source_input_chars=4500,
    )
    assert result["status"] == "success"
    assert seen["full_document_texts"] == {"viking://resources/docs/a": text}
    assert seen["max_source_input_chars"] == 4500
    assert seen["build_stage"] == "cards"
