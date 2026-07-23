"""A hung gateway must not hang the caller.

Every stall the client can meet has its own test: headers that never
arrive, a body that stalls after fast headers, and an idle gap between
stream frames. All three must surface as :class:`luxsdk.Error`, never as
a bare ``TimeoutError``, and streaming must keep its unbounded default so
model think-time is not mistaken for a dead connection.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import luxsdk

OK_BODY = json.dumps(
    {
        "id": "msg_1",
        "model": "m",
        "blocks": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }
).encode()

# Longer than any timeout under test, so a stalled call can only end by
# the client giving up.
STALL = 5.0


class _Handler(BaseHTTPRequestHandler):
    """Each test assigns ``respond(handler) -> None``."""

    respond = staticmethod(lambda h: None)
    last: dict = {}

    def do_POST(self):  # noqa: N802 (http.server naming)
        length = int(self.headers.get("Content-Length", 0))
        _Handler.last = {
            "path": self.path,
            "body": json.loads(self.rfile.read(length) or b"{}"),
        }
        try:
            _Handler.respond(self)
        except OSError:
            # The client gave up and closed the socket mid-write, which
            # is the point of these tests.
            pass

    def log_message(self, *args):  # silence the default stderr log
        pass


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.daemon_threads = True
    # Teardown must not join the handler thread that is still sleeping.
    srv.block_on_close = False
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _send_json(h, body=OK_BODY):
    h.send_response(200)
    h.send_header("Content-Type", "application/json")
    h.send_header("Content-Length", str(len(body)))
    h.end_headers()
    h.wfile.write(body)


def _stall_before_headers(h):
    time.sleep(STALL)
    _send_json(h)


def _stall_after_headers(h):
    h.send_response(200)
    h.send_header("Content-Type", "application/json")
    h.send_header("Content-Length", str(len(OK_BODY)))
    h.end_headers()
    h.wfile.flush()  # headers land immediately; only the body stalls
    time.sleep(STALL)
    h.wfile.write(OK_BODY)


def _sse(h, frames, gap):
    h.send_response(200)
    h.send_header("Content-Type", "text/event-stream")
    h.end_headers()
    h.wfile.flush()
    for i, frame in enumerate(frames):
        if i:
            time.sleep(gap)
        h.wfile.write(frame.encode())
        h.wfile.flush()


STREAM_FRAMES = [
    'event: message_start\ndata: {"type":"message_start","id":"msg_1","model":"m","index":0}\n\n',
    'event: text_delta\ndata: {"type":"text_delta","index":0,"delta":"hello"}\n\n',
    'event: message_stop\ndata: {"type":"message_stop","index":0}\n\n',
]


def test_client_accepts_a_timeout(server):
    """The bound is a constructor argument, not a request field."""
    _Handler.respond = staticmethod(_send_json)
    c = luxsdk.Client(server, timeout=5.0)
    assert c.generate(model="m", messages=[luxsdk.user_text("hi")]).id == "msg_1"


def test_generate_fails_fast_when_gateway_stalls(server):
    """Headers that never arrive end the call at the bound."""
    _Handler.respond = staticmethod(_stall_before_headers)
    c = luxsdk.Client(server)
    start = time.monotonic()
    with pytest.raises(luxsdk.Error):
        c.generate(timeout=0.5, model="m", messages=[luxsdk.user_text("hi")])
    assert time.monotonic() - start < 2.0


def test_count_tokens_fails_fast_when_gateway_stalls(server):
    _Handler.respond = staticmethod(_stall_before_headers)
    c = luxsdk.Client(server, timeout=0.5)
    start = time.monotonic()
    with pytest.raises(luxsdk.Error):
        c.count_tokens(model="m", messages=[luxsdk.user_text("hi")])
    assert time.monotonic() - start < 2.0


def test_slow_body_surfaces_as_error(server):
    """Fast headers then a stalled body: the read path times out too, and
    it must arrive as a client error rather than a bare TimeoutError."""
    _Handler.respond = staticmethod(_stall_after_headers)
    c = luxsdk.Client(server, timeout=0.5)
    start = time.monotonic()
    with pytest.raises(luxsdk.Error):
        c.generate(model="m", messages=[luxsdk.user_text("hi")])
    assert time.monotonic() - start < 2.0


def test_timeout_not_sent_in_body(server):
    """The bound is a client concern; the gateway never sees it."""
    _Handler.respond = staticmethod(_send_json)
    c = luxsdk.Client(server, timeout=5.0)
    c.generate(timeout=3.0, model="m", messages=[luxsdk.user_text("hi")])
    assert "timeout" not in _Handler.last["body"]


def test_stream_has_no_idle_bound_by_default(server):
    """A quiet gap between frames is model think-time, not a stall: the
    client's unary bound must not cut a healthy live stream."""
    _Handler.respond = staticmethod(
        lambda h: _sse(h, STREAM_FRAMES, gap=1.0)  # gap > the client's 0.5s bound
    )
    c = luxsdk.Client(server, timeout=0.5)
    events = list(c.stream(model="m", messages=[luxsdk.user_text("hi")]))
    assert [e["type"] for e in events] == ["message_start", "text_delta", "message_stop"]


def test_stream_honors_an_explicit_timeout(server):
    """Opting in bounds the idle gap, and the gap surfaces as an Error."""
    _Handler.respond = staticmethod(lambda h: _sse(h, STREAM_FRAMES, gap=STALL))
    c = luxsdk.Client(server)
    start = time.monotonic()
    with pytest.raises(luxsdk.Error):
        list(c.stream(timeout=0.5, model="m", messages=[luxsdk.user_text("hi")]))
    assert time.monotonic() - start < 2.0
