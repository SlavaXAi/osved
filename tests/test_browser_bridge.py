from __future__ import annotations

import http.client
import json
import threading

import pytest

from app.collectors.browser_bridge import MAX_BODY_BYTES, BrowserBridgeServer
from app.collectors.browser_queue import BrowserTaskQueue

EXTENSION_ID = "a" * 32
TOKEN = "s" * 43


@pytest.fixture
def bridge(tmp_path):
    queue = BrowserTaskQueue(tmp_path / "bridge.sqlite")
    queue.initialize()
    server = BrowserBridgeServer(queue, extension_id=EXTENSION_ID, pairing_token=TOKEN, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def call(server, path, payload=None, *, method="POST", headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    actual = {
        "Origin": f"chrome-extension://{EXTENSION_ID}",
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
    }
    actual.update(headers or {})
    connection.request(method, path, body=json.dumps(payload or {}).encode(), headers=actual)
    response = connection.getresponse()
    data = response.read()
    result = (response.status, json.loads(data) if data else {}, dict(response.getheaders()))
    connection.close()
    return result


def test_health_and_claim_stage_round_trip(bridge):
    assert bridge.server_address[0] == "127.0.0.1"
    status, payload, _ = call(bridge, "/v1/health", method="GET")
    assert status == 200
    assert payload["capability"] == "x_visible_for_you_read_only"
    task_id = bridge.queue.enqueue("x-for-you", task_key="test")
    status, payload, _ = call(bridge, "/v1/tasks/claim")
    assert status == 200
    task = payload["task"]
    snapshot = {
        "feed_mode": "for_you",
        "posts": [
            {
                "id": "123456",
                "author": "example",
                "text": "Test release",
                "published_at": "2026-09-07T08:00:00Z",
                "links": [],
            }
        ],
    }
    body = {"task_id": task_id, "claim_token": task["claim_token"], "result": snapshot}
    status, payload, _ = call(bridge, "/v1/tasks/result", body)
    assert status == 200
    assert payload == {"status": "staged", "replayed": False}
    assert call(bridge, "/v1/tasks/result", body)[1]["replayed"]
    assert bridge.queue.ready_results()[0]["result"]["posts"][0]["text"] == "Test release"


@pytest.mark.parametrize(
    "headers,status",
    [
        ({"Origin": "https://evil.example"}, 403),
        ({"Origin": "null"}, 403),
        ({"Origin": ""}, 403),
        ({"Origin": f"chrome-extension://{'b' * 32}"}, 403),
        ({"Authorization": "Bearer wrong"}, 401),
        ({"Host": "evil.example"}, 403),
        ({"Host": "localhost"}, 403),
    ],
)
def test_origin_host_and_pairing_token_are_required(bridge, headers, status):
    assert call(bridge, "/v1/tasks/claim", headers=headers)[0] == status


def test_no_arbitrary_urls_actions_or_payloads(bridge):
    assert call(bridge, "/v1/fetch", {"url": "http://example.com"})[0] == 404
    assert call(bridge, "/v1/tasks/claim", {"execute": "document.cookie"})[0] == 409
    assert call(bridge, "/v1/tasks/claim", method="PUT")[0] == 405
    assert call(bridge, "/v1/tasks/claim", headers={"Content-Type": "text/plain"})[0] == 415
    oversized = {"Content-Length": str(MAX_BODY_BYTES + 1)}
    assert call(bridge, "/v1/tasks/claim", headers=oversized)[0] == 413


def test_cors_is_exact_extension_only(bridge):
    status, _, headers = call(
        bridge,
        "/v1/tasks/claim",
        method="OPTIONS",
        headers={
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert status == 204
    assert headers["Access-Control-Allow-Origin"] == f"chrome-extension://{EXTENSION_ID}"
    assert headers["Access-Control-Allow-Origin"] != "*"
    assert "Access-Control-Allow-Credentials" not in headers
    assert (
        call(
            bridge,
            "/v1/tasks/claim",
            method="OPTIONS",
            headers={
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "cookie",
            },
        )[0]
        == 403
    )


def test_invalid_extension_id_and_short_token_rejected(tmp_path):
    queue = BrowserTaskQueue(tmp_path / "browser.sqlite")
    with pytest.raises(ValueError, match="extension ID"):
        BrowserBridgeServer(queue, extension_id="https://evil.example", port=0)
    with pytest.raises(ValueError, match="Pairing token"):
        BrowserBridgeServer(queue, extension_id=EXTENSION_ID, pairing_token="short", port=0)
