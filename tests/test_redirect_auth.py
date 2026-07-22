"""Redirects must not carry the bearer to another origin."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import luxsdk


class _Sink(BaseHTTPRequestHandler):
    """Records what a redirect target would have received."""

    received: "list[dict[str, str]]" = []

    def _record(self) -> None:
        type(self).received.append(
            {
                "method": self.command,
                "auth": self.headers.get("Authorization", ""),
            }
        )
        body = json.dumps({"id": "r", "model": "m", "blocks": []}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = _record
    do_POST = _record

    def log_message(self, *a):  # noqa: D102 - silence the default stderr log
        pass


def _redirector(location: str):
    class Redirect(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(301)
            self.send_header("Location", location)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *a):
            pass

    return Redirect


def _serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def sink():
    _Sink.received = []
    server = _serve(_Sink)
    yield server, _Sink
    server.shutdown()
    server.server_close()


@pytest.fixture
def redirector(sink):
    server, _ = sink
    host, port = server.server_address[0], server.server_address[1]
    src = _serve(_redirector(f"http://{host}:{port}/lux/v1/generate"))
    yield src
    src.shutdown()
    src.server_close()


def _client(server):
    host, port = server.server_address[0], server.server_address[1]
    return luxsdk.Client(base_url=f"http://{host}:{port}", api_key="lux_secret")


def test_bearer_not_forwarded_across_origins(sink, redirector):
    _, handler = sink
    try:
        _client(redirector).generate(model="m", messages=[])
    except luxsdk.Error:
        pass
    assert not handler.received, (
        f"bearer leaked to the redirect target: {handler.received[0]}"
    )


def test_redirect_surfaces_as_error(sink, redirector):
    with pytest.raises(luxsdk.Error) as excinfo:
        _client(redirector).generate(model="m", messages=[])
    err = excinfo.value
    assert err.status == 301
    assert "/lux/v1/generate" in err.message, err.message
