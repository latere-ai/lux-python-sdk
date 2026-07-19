# luxsdk (Python)

The Python client for [Latere Lux](https://lux.latere.ai)'s native
dialect: one request/response/stream shape for every model Lux routes.
Standard library only. No dependencies to install alongside it.
Messages, blocks, and events are dicts with the wire's snake_case
keys, verbatim.

## Install

```
pip install luxsdk
```

```python
import luxsdk

c = luxsdk.Client("https://lux.latere.ai", api_key=os.environ["LUX_API_KEY"])

res = c.generate(
    model="claude-sonnet-5",
    max_tokens=256,
    messages=[luxsdk.user_text("Hello")],
)
print(res.blocks[0]["text"], res.usage)
```

## Streaming

```python
with c.stream(model="claude-sonnet-5", messages=[luxsdk.user_text("Hi")]) as st:
    for ev in st:
        if ev["type"] == "text_delta":
            print(ev["delta"], end="")
```

The stream grammar is the gateway IR's, verbatim:

```
message_start (block_start (text_delta|args_delta|thinking_delta|signature_delta)* block_stop)* message_delta message_stop
```

Assemble a streamed tool call from `block_start` (id, name) plus
`args_delta` fragments, closed by that index's `block_stop`. `usage`
appears on `message_start` (input side) and `message_delta` (output
side); accumulate both. Iteration ends after `message_stop`; a
mid-stream gateway failure raises `luxsdk.StreamError`.

## Token counting

```python
tc = c.count_tokens(model="claude-sonnet-5", messages=[luxsdk.user_text("Hi")])
# tc.input_tokens; tc.estimated is True when the target has no native tokenizer
```

## Auth

`api_key` is a static bearer (a Lux virtual key). `token_source` is a
zero-argument callable supplying a per-call bearer (e.g. a rotating
JWT) and wins over `api_key`.

## Cost attribution

`cost_tags` attributes a call's cost to named dimensions within your
own spend, sent as the `Lux-Cost-Tag` header. It never changes who is
billed or what the key can reach. Pass a `dict` (serialized to sorted
`key=value` pairs) or a pre-formatted string:

```python
c = luxsdk.Client("https://lux.latere.ai", api_key=key, cost_tags={"tenant": "acme"})

# Per call; overrides the client default.
res = c.generate(
    model="claude-sonnet-5",
    messages=[luxsdk.user_text("Hi")],
    cost_tags={"tenant": "acme", "project": "web"},  # sent as project=web,tenant=acme
)
```

The gateway validates the value and rejects a malformed one with a
`400`; the SDK passes it through untouched.

## Errors and loss

Non-2xx responses raise `luxsdk.Error` with `status`, `code`,
`message`, and `request_id` in the retryable type vocabulary
(`rate_limit_error`, `overloaded_error`, ...). Request fields the
target dialect cannot represent are never silently dropped: they
arrive as `result.loss` / `stream.loss` from the `X-Lux-Compat-Loss`
header.

## Tests

```
python -m pytest tests/
```
