"""Python client for the Latere Lux gateway's native dialect.

One request/response/stream shape for every model Lux routes, spoken at
``POST {base}/lux/v1/generate``. Standard library only (urllib); dicts
with the wire's snake_case keys go in, dicts come out.

    import luxsdk

    c = luxsdk.Client("https://lux.latere.ai", api_key=os.environ["LUX_API_KEY"])
    res = c.generate(model="claude-sonnet-5", messages=[luxsdk.user_text("Hello")])
    print(res.blocks[0]["text"], res.usage)

    for ev in c.stream(model="claude-sonnet-5", messages=[luxsdk.user_text("Hi")]):
        if ev["type"] == "text_delta":
            print(ev["delta"], end="")
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

__all__ = [
    "Client",
    "Error",
    "StreamError",
    "Result",
    "TokenCount",
    "Stream",
    "user_text",
    "assistant_text",
    "ENV_BASE_URL",
    "ENV_API_KEY",
    "DEFAULT_BASE_URL",
]

_GENERATE_PATH = "/lux/v1/generate"
_COUNT_TOKENS_PATH = "/lux/v1/count_tokens"
_LOSS_HEADER = "X-Lux-Compat-Loss"
_ESTIMATED_HEADER = "X-Lux-Compat-Estimated"
_COST_TAG_HEADER = "Lux-Cost-Tag"

#: Gateway base, e.g. ``https://lux.latere.ai``. Deliberately not
#: ``LUX_API_URL``: that is the ``latere`` CLI's own target, and one
#: variable steering both would let ``eval "$(latere lux env --compat
#: lux)"`` silently retarget the CLI from a subshell.
ENV_BASE_URL = "LUX_BASE_URL"
#: Carries exactly what ``Authorization: Bearer`` carries: a ``lux_*``
#: virtual key, or a Latere Auth identity/actor token.
ENV_API_KEY = "LUX_API_KEY"
#: Used when neither an explicit base URL nor :data:`ENV_BASE_URL` is set.
DEFAULT_BASE_URL = "https://lux.latere.ai"


def _format_cost_tags(tags: "dict[str, str] | str | None") -> str:
    """Serialize cost tags to the ``Lux-Cost-Tag`` wire form: sorted
    ``key=value`` pairs joined by commas, no spaces. A pre-formatted
    string passes through unchanged; an empty/None value yields ""."""
    if not tags:
        return ""
    if isinstance(tags, str):
        return tags
    return ",".join(f"{k}={tags[k]}" for k in sorted(tags))

_VALID_EVENTS = {
    "message_start",
    "block_start",
    "text_delta",
    "args_delta",
    "thinking_delta",
    "signature_delta",
    "block_stop",
    "message_delta",
    "message_stop",
}


def user_text(text: str) -> dict[str, Any]:
    """One-block user turn."""
    return {"role": "user", "blocks": [{"type": "text", "text": text}]}


def assistant_text(text: str) -> dict[str, Any]:
    """One-block assistant turn."""
    return {"role": "assistant", "blocks": [{"type": "text", "text": text}]}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuses every redirect.

    Returning ``None`` from ``redirect_request`` makes urllib fall
    through to ``HTTPDefaultErrorHandler``, so the 3xx reaches the
    caller as an ``HTTPError`` and is decoded like any other non-2xx
    response. Following it would leak the bearer to another origin and
    turn the POST into a bodyless GET.
    """

    def redirect_request(self, *args, **kwargs):
        return None


class Error(Exception):
    """A non-2xx gateway response, decoded from the error envelope."""

    def __init__(self, status: int, code: str, message: str, request_id: str = ""):
        self.status = status
        self.code = code
        self.message = message
        self.request_id = request_id
        label = f"lux: {status} {code}: {message}" if code else f"lux: {status}: {message}"
        super().__init__(label)


class StreamError(Exception):
    """A mid-stream ``event: error`` frame from the gateway."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        label = f"lux: stream error ({code}): {message}" if code else f"lux: stream error: {message}"
        super().__init__(label)


@dataclass
class Result:
    """A completed non-streaming call.

    ``loss`` lists request fields the backend dialect could not
    represent (empty when the target speaks the full IR).
    """

    id: str
    model: str
    blocks: list[dict[str, Any]]
    stop_reason: str
    usage: dict[str, Any]
    stop_sequence: str = ""
    loss: list[str] = field(default_factory=list)


@dataclass
class TokenCount:
    """A count_tokens answer. ``estimated`` marks a heuristic count for
    targets with no native tokenizer endpoint."""

    input_tokens: int
    estimated: bool


class Stream:
    """A live event stream: iterate for events (dicts in the IR event
    grammar); iteration ends after ``message_stop``. ``loss`` lists
    request fields the backend dialect could not represent. Close (or
    exhaust) the stream to release the connection; usable as a context
    manager."""

    def __init__(self, resp: Any, loss: list[str]):
        self._resp = resp
        self.loss = loss

    def __iter__(self) -> Iterator[dict[str, Any]]:
        try:
            for frame in _frames(self._resp):
                ev = _parse_frame(frame)
                if ev is not None:
                    yield ev
        finally:
            self.close()

    def close(self) -> None:
        self._resp.close()

    def __enter__(self) -> "Stream":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _frames(resp: Any) -> Iterator[str]:
    buf = b""
    while True:
        chunk = resp.read1(8192) if hasattr(resp, "read1") else resp.read(8192)
        if not chunk:
            break
        buf += chunk
        while (sep := buf.find(b"\n\n")) >= 0:
            yield buf[:sep].decode("utf-8")
            buf = buf[sep + 2 :]
    # A final unterminated frame still parses (stream cut early).
    if buf.strip():
        yield buf.decode("utf-8")


def _parse_frame(frame: str) -> dict[str, Any] | None:
    name = ""
    data_lines: list[str] = []
    for line in frame.split("\n"):
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    data = "\n".join(data_lines)
    if name == "error":
        try:
            wire = json.loads(data)
            err = wire.get("error") or {}
            if err.get("type") or err.get("message"):
                raise StreamError(err.get("type", ""), err.get("message", ""))
        except (json.JSONDecodeError, AttributeError):
            pass
        raise StreamError("", data)
    if name not in _VALID_EVENTS:
        return None  # unknown frames are skipped (forward compatibility)
    ev = json.loads(data)
    ev.setdefault("type", name)
    return ev


class Client:
    """Calls one Lux deployment.

    ``api_key`` is a static bearer (a Lux virtual key). ``token_source``
    supplies a per-call bearer (e.g. a rotating JWT) and wins over
    ``api_key``.

    ``cost_tags`` attributes every call's cost to named dimensions
    within the caller's own spend (e.g. ``{"tenant": "acme"}``); a
    per-call ``cost_tags`` overrides this default.

    Both connection values fall back to the environment when omitted, so
    a process configured by ``eval "$(latere lux env --compat lux)"``
    needs neither::

        c = luxsdk.Client()  # LUX_BASE_URL + LUX_API_KEY

    Explicit arguments always win: the environment fills only what the
    caller left unset, so setting ``LUX_BASE_URL`` in a shell can never
    redirect a program that passed its own. An omitted credential stays
    empty rather than defaulting to unauthenticated, so a misspelled
    variable fails at the gateway instead of becoming an anonymous call.

    Redirects are not followed: urllib's redirect handler re-sends the
    ``Authorization`` header to whatever host a 3xx names and rewrites
    POST to GET, dropping the body. A 3xx is surfaced as an
    :class:`Error` carrying the status and the ``Location`` instead.
    This applies to the default opener only; a caller-supplied
    ``opener`` follows redirects again, an explicit opt-out.
    """

    def __init__(
        self,
        base_url: str = "",
        *,
        api_key: str = "",
        token_source: Callable[[], str] | None = None,
        cost_tags: "dict[str, str] | str | None" = None,
        opener: Any = None,
    ):
        base_url = base_url or os.environ.get(ENV_BASE_URL, "") or DEFAULT_BASE_URL
        if not api_key and token_source is None:
            api_key = os.environ.get(ENV_API_KEY, "")
        self._base = base_url.rstrip("/")
        self._api_key = api_key
        self._token_source = token_source
        self._cost_tags = cost_tags
        self._opener = opener or urllib.request.build_opener(_NoRedirect)

    def generate(
        self, *, cost_tags: "dict[str, str] | str | None" = None, **request: Any
    ) -> Result:
        """Non-streaming call; the stream flag is overridden off."""
        resp = self._post(_GENERATE_PATH, {**request, "stream": False}, cost_tags)
        with resp:
            body = json.loads(resp.read())
            return Result(
                id=body.get("id", ""),
                model=body.get("model", ""),
                blocks=body.get("blocks") or [],
                stop_reason=body.get("stop_reason", ""),
                stop_sequence=body.get("stop_sequence", "") or "",
                usage=body.get("usage") or {},
                loss=_parse_loss(resp),
            )

    def count_tokens(
        self, *, cost_tags: "dict[str, str] | str | None" = None, **request: Any
    ) -> TokenCount:
        """Token count without spending output tokens; no spend gates run."""
        resp = self._post(_COUNT_TOKENS_PATH, {**request, "stream": False}, cost_tags)
        with resp:
            body = json.loads(resp.read())
            return TokenCount(
                input_tokens=int(body["input_tokens"]),
                estimated=resp.headers.get(_ESTIMATED_HEADER) == "true",
            )

    def stream(
        self, *, cost_tags: "dict[str, str] | str | None" = None, **request: Any
    ) -> Stream:
        """Streaming call; the stream flag is overridden on."""
        resp = self._post(_GENERATE_PATH, {**request, "stream": True}, cost_tags)
        ctype = resp.headers.get("Content-Type", "")
        if not ctype.startswith("text/event-stream"):
            resp.close()
            raise Error(resp.status, "", f"expected an event stream, got {ctype!r}")
        return Stream(resp, _parse_loss(resp))

    def _post(
        self,
        path: str,
        payload: dict[str, Any],
        cost_tags: "dict[str, str] | str | None" = None,
    ) -> Any:
        headers = {"Content-Type": "application/json"}
        bearer = self._token_source() if self._token_source else self._api_key
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        tags = cost_tags if cost_tags is not None else self._cost_tags
        formatted = _format_cost_tags(tags)
        if formatted:
            headers[_COST_TAG_HEADER] = formatted
        req = urllib.request.Request(
            self._base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            return self._opener.open(req)
        except urllib.error.HTTPError as e:
            raise _decode_error(e) from None


def _parse_loss(resp: Any) -> list[str]:
    v = resp.headers.get(_LOSS_HEADER)
    return v.split(",") if v else []


def _decode_error(e: urllib.error.HTTPError) -> Error:
    raw = e.read()
    try:
        wire = json.loads(raw)
        err = wire.get("error") or {}
        if err.get("type") or err.get("message"):
            return Error(
                e.code,
                err.get("type", ""),
                err.get("message", ""),
                err.get("request_id", ""),
            )
    except (json.JSONDecodeError, AttributeError):
        pass
    message = raw.decode("utf-8", "replace").strip()
    # A redirect body is empty; the Location is the whole signal.
    location = e.headers.get("Location") if e.headers else None
    if location:
        message = f"{message} (Location: {location})".lstrip()
    return Error(e.code, "", message)
