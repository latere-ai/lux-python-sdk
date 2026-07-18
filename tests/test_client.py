"""Tests for the lux Python SDK against an in-process fake gateway."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import luxsdk

OK_RESPONSE = {
    "id": "msg_1",
    "model": "claude-sonnet-5",
    "blocks": [{"type": "text", "text": "hello"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 3, "output_tokens": 2},
}

STREAM_BODY = (
    'event: message_start\ndata: {"type":"message_start","id":"msg_1","model":"m","index":0,'
    '"usage":{"input_tokens":3,"output_tokens":0}}\n\n'
    'event: block_start\ndata: {"type":"block_start","index":0,"block":{"type":"text"}}\n\n'
    'event: text_delta\ndata: {"type":"text_delta","index":0,"delta":"hel"}\n\n'
    "event: ping\ndata: {}\n\n"  # unknown frame: skipped
    "data: [DONE]\n\n"  # unnamed frame: skipped
    'event: text_delta\ndata: {"index":0,"delta":"lo"}\n\n'  # type filled from name
    'event: block_stop\ndata: {"type":"block_stop","index":0}\n\n'
    'event: message_delta\ndata: {"type":"message_delta","index":0,"stop_reason":"end_turn",'
    '"usage":{"input_tokens":3,"output_tokens":2}}\n\n'
    'event: message_stop\ndata: {"type":"message_stop","index":0}'  # unterminated final frame
)


class _Handler(BaseHTTPRequestHandler):
    # Each test assigns `respond(handler) -> None`.
    respond = staticmethod(lambda h: None)
    last: dict = {}

    def do_POST(self):  # noqa: N802 (http.server naming)
        length = int(self.headers.get("Content-Length", 0))
        _Handler.last = {
            "path": self.path,
            "auth": self.headers.get("Authorization", ""),
            "body": json.loads(self.rfile.read(length) or b"{}"),
        }
        _Handler.respond(self)

    def log_message(self, *args):  # silence
        pass


@pytest.fixture(scope="module")
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _json_response(h, status=200, payload=None, headers=None):
    raw = json.dumps(payload or {}).encode()
    h.send_response(status)
    h.send_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        h.send_header(k, v)
    h.send_header("Content-Length", str(len(raw)))
    h.end_headers()
    h.wfile.write(raw)


def _sse_response(h, body, headers=None):
    raw = body.encode()
    h.send_response(200)
    h.send_header("Content-Type", "text/event-stream; charset=utf-8")
    for k, v in (headers or {}).items():
        h.send_header(k, v)
    h.send_header("Content-Length", str(len(raw)))
    h.end_headers()
    h.wfile.write(raw)


def test_generate(server):
    _Handler.respond = lambda h: _json_response(
        h, payload=OK_RESPONSE, headers={"X-Lux-Compat-Loss": "top_k,thinking"}
    )
    c = luxsdk.Client(server + "/", api_key="lux_k1")
    res = c.generate(
        model="claude-sonnet-5",
        messages=[luxsdk.user_text("hi"), luxsdk.assistant_text("prior")],
        stream=True,  # must be forced off
    )
    assert _Handler.last["path"] == "/lux/v1/generate"
    assert _Handler.last["auth"] == "Bearer lux_k1"
    assert _Handler.last["body"]["stream"] is False
    assert res.blocks[0]["text"] == "hello"
    assert res.stop_reason == "end_turn"
    assert res.usage["input_tokens"] == 3
    assert res.loss == ["top_k", "thinking"]


def test_generate_error_envelope(server):
    _Handler.respond = lambda h: _json_response(
        h,
        status=429,
        payload={
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "slow down", "request_id": "req_9"},
        },
    )
    c = luxsdk.Client(server)
    with pytest.raises(luxsdk.Error) as exc:
        c.generate(model="m", messages=[luxsdk.user_text("x")])
    e = exc.value
    assert (e.status, e.code, e.message, e.request_id) == (
        429,
        "rate_limit_error",
        "slow down",
        "req_9",
    )
    assert "rate_limit_error" in str(e)


def test_generate_opaque_error(server):
    def respond(h):
        raw = b"upstream fell over"
        h.send_response(502)
        h.send_header("Content-Length", str(len(raw)))
        h.end_headers()
        h.wfile.write(raw)

    _Handler.respond = respond
    with pytest.raises(luxsdk.Error) as exc:
        luxsdk.Client(server).generate(model="m", messages=[luxsdk.user_text("x")])
    assert exc.value.status == 502
    assert exc.value.code == ""
    assert "upstream fell over" in exc.value.message


def test_token_source_wins(server):
    _Handler.respond = lambda h: _json_response(h, payload=OK_RESPONSE)
    c = luxsdk.Client(server, api_key="static", token_source=lambda: "jwt-1")
    c.generate(model="m", messages=[luxsdk.user_text("x")])
    assert _Handler.last["auth"] == "Bearer jwt-1"


def test_count_tokens(server):
    _Handler.respond = lambda h: _json_response(h, payload={"input_tokens": 42})
    tc = luxsdk.Client(server, api_key="k").count_tokens(
        model="m", messages=[luxsdk.user_text("hi")]
    )
    assert _Handler.last["path"] == "/lux/v1/count_tokens"
    assert (tc.input_tokens, tc.estimated) == (42, False)


def test_count_tokens_estimated(server):
    _Handler.respond = lambda h: _json_response(
        h, payload={"input_tokens": 7}, headers={"X-Lux-Compat-Estimated": "true"}
    )
    tc = luxsdk.Client(server).count_tokens(model="m", messages=[luxsdk.user_text("x")])
    assert (tc.input_tokens, tc.estimated) == (7, True)


def test_stream(server):
    _Handler.respond = lambda h: _sse_response(
        h, STREAM_BODY, headers={"X-Lux-Compat-Loss": "top_k"}
    )
    st = luxsdk.Client(server, api_key="k").stream(model="m", messages=[luxsdk.user_text("x")])
    assert _Handler.last["body"]["stream"] is True
    assert st.loss == ["top_k"]
    text, types = "", []
    with st:
        for ev in st:
            types.append(ev["type"])
            if ev["type"] == "text_delta":
                text += ev["delta"]
    assert text == "hello"
    assert types == [
        "message_start",
        "block_start",
        "text_delta",
        "text_delta",
        "block_stop",
        "message_delta",
        "message_stop",
    ]


def test_stream_mid_stream_error(server):
    body = (
        'event: message_start\ndata: {"type":"message_start","id":"m1","index":0}\n\n'
        'event: error\ndata: {"type":"error","error":{"type":"overloaded_error","message":"busy"}}\n\n'
    )
    _Handler.respond = lambda h: _sse_response(h, body)
    st = luxsdk.Client(server).stream(model="m", messages=[luxsdk.user_text("x")])
    seen = []
    with pytest.raises(luxsdk.StreamError) as exc:
        for ev in st:
            seen.append(ev)
    assert exc.value.code == "overloaded_error"
    assert len(seen) == 1


def test_stream_opaque_error_frame(server):
    _Handler.respond = lambda h: _sse_response(h, "event: error\ndata: it broke\n\n")
    st = luxsdk.Client(server).stream(model="m", messages=[luxsdk.user_text("x")])
    with pytest.raises(luxsdk.StreamError) as exc:
        list(st)
    assert "it broke" in str(exc.value)


def test_stream_rejects_non_sse(server):
    _Handler.respond = lambda h: _json_response(h, payload=OK_RESPONSE)
    with pytest.raises(luxsdk.Error):
        luxsdk.Client(server).stream(model="m", messages=[luxsdk.user_text("x")])


def test_stream_error_status(server):
    _Handler.respond = lambda h: _json_response(
        h, status=403, payload={"type": "error", "error": {"type": "permission_error", "message": "no"}}
    )
    with pytest.raises(luxsdk.Error) as exc:
        luxsdk.Client(server).stream(model="m", messages=[luxsdk.user_text("x")])
    assert exc.value.status == 403
