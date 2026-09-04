#!/usr/bin/env python3
"""Dependency-free Anthropic-compatible bridge for local OpenAI/vLLM.

Purpose: let Hyperloom/Claude-oriented orchestration talk to a local Qwen model
served by vLLM, without installing a proxy globally. This is experimental and
intentionally scoped to the API surface needed by the Hyperloom experiment.

Supported:
- GET /health
- POST /v1/messages (non-streaming and buffered Anthropic SSE)
- POST /v1/messages/count_tokens (best-effort conservative estimate)
- text content
- Anthropic tool definitions -> OpenAI function tools
- tool_use / tool_result round-tripping

The upstream OpenAI request is intentionally non-streaming even when the client
asks for SSE. We then emit a valid Anthropic event stream from the completed
response. That is slower for first-token delivery, but deterministic and enough
for a compatibility experiment.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

DEFAULT_UPSTREAM = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
DEFAULT_MODEL = os.environ.get("UPSTREAM_MODEL", "").strip()
BRIDGE_TOKEN = os.environ.get("BRIDGE_TOKEN", "local-hyperloom")


def _json_request(url: str, payload: dict[str, Any], timeout: float = 300.0) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _discover_model(base_url: str) -> str:
    req = urllib.request.Request(base_url.rstrip("/") + "/models", method="GET")
    with urllib.request.urlopen(req, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    data = payload.get("data") or []
    if not data or not isinstance(data[0], dict) or not data[0].get("id"):
        raise RuntimeError("upstream /models returned no model id")
    return str(data[0]["id"])


def _text_from_blocks(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            parts.append(str(block.get("text") or ""))
        elif kind == "thinking":
            continue
    return "\n".join(p for p in parts if p)


def _anthropic_messages_to_openai(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    system = payload.get("system")
    if system:
        system_text = _text_from_blocks(system)
        if system_text:
            result.append({"role": "system", "content": system_text})

    for msg in payload.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            result.append({"role": role, "content": str(content or "")})
            continue

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                text_parts.append(str(block.get("text") or ""))
            elif kind == "tool_use":
                tool_calls.append({
                    "id": str(block.get("id") or f"toolu_{uuid.uuid4().hex}"),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name") or "tool"),
                        "arguments": json.dumps(block.get("input") or {}, separators=(",", ":")),
                    },
                })
            elif kind == "tool_result":
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": str(block.get("tool_use_id") or ""),
                    "content": _text_from_blocks(block.get("content", "")),
                })

        if role == "assistant":
            assistant: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts) or None}
            if tool_calls:
                assistant["tool_calls"] = tool_calls
            result.append(assistant)
        else:
            text = "\n".join(text_parts)
            if text:
                result.append({"role": role, "content": text})
            result.extend(tool_results)
    return result


def _anthropic_tools_to_openai(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for tool in payload.get("tools") or []:
        if not isinstance(tool, dict) or not tool.get("name"):
            continue
        result.append({
            "type": "function",
            "function": {
                "name": str(tool["name"]),
                "description": str(tool.get("description") or ""),
                "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
            },
        })
    return result


def _finish_reason(reason: str, has_tools: bool) -> str:
    if has_tools:
        return "tool_use"
    if reason == "length":
        return "max_tokens"
    return "end_turn"


def _openai_to_anthropic(response: dict[str, Any], requested_model: str) -> dict[str, Any]:
    choices = response.get("choices") or [{}]
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    content: list[dict[str, Any]] = []
    text = message.get("content")
    if text:
        content.append({"type": "text", "text": str(text)})
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        raw_args = fn.get("arguments") or "{}"
        try:
            parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            parsed_args = {"_raw": str(raw_args)}
        content.append({
            "type": "tool_use",
            "id": str(call.get("id") or f"toolu_{uuid.uuid4().hex}"),
            "name": str(fn.get("name") or "tool"),
            "input": parsed_args or {},
        })
    if not content:
        content = [{"type": "text", "text": ""}]
    usage = response.get("usage") or {}
    return {
        "id": "msg_" + uuid.uuid4().hex,
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": content,
        "stop_reason": _finish_reason(str(choice.get("finish_reason") or "stop"), any(b["type"] == "tool_use" for b in content)),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


def _build_openai_payload(payload: dict[str, Any], model: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model": model,
        "messages": _anthropic_messages_to_openai(payload),
        "max_tokens": int(payload.get("max_tokens") or 1024),
        "temperature": float(payload.get("temperature") if payload.get("temperature") is not None else 0.0),
        "stream": False,
    }
    tools = _anthropic_tools_to_openai(payload)
    if tools:
        out["tools"] = tools
        choice = payload.get("tool_choice")
        if isinstance(choice, dict):
            kind = choice.get("type")
            if kind == "any":
                out["tool_choice"] = "required"
            elif kind == "auto":
                out["tool_choice"] = "auto"
            elif kind == "tool" and choice.get("name"):
                out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    return out


def _sse_events(message: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    base = {k: v for k, v in message.items() if k != "content"}
    base["content"] = []
    base["stop_reason"] = None
    events: list[tuple[str, dict[str, Any]]] = [("message_start", {"type": "message_start", "message": base})]
    for idx, block in enumerate(message["content"]):
        if block.get("type") == "text":
            events.append(("content_block_start", {"type": "content_block_start", "index": idx, "content_block": {"type": "text", "text": ""}}))
            events.append(("content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {"type": "text_delta", "text": block.get("text", "")}}))
            events.append(("content_block_stop", {"type": "content_block_stop", "index": idx}))
        elif block.get("type") == "tool_use":
            events.append(("content_block_start", {"type": "content_block_start", "index": idx, "content_block": {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}}))
            events.append(("content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {"type": "input_json_delta", "partial_json": json.dumps(block.get("input") or {}, separators=(",", ":"))}}))
            events.append(("content_block_stop", {"type": "content_block_stop", "index": idx}))
    events.append(("message_delta", {"type": "message_delta", "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None}, "usage": {"output_tokens": message["usage"]["output_tokens"]}}))
    events.append(("message_stop", {"type": "message_stop"}))
    return events


class Handler(BaseHTTPRequestHandler):
    server_version = "HyperloomR9700AnthropicBridge/0.2"

    def log_message(self, fmt: str, *args: Any) -> None:
        print("bridge:", fmt % args, flush=True)

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/health", "/v1/health"):
            try:
                model = DEFAULT_MODEL or _discover_model(DEFAULT_UPSTREAM)
                self._write_json(200, {"ok": True, "upstream": DEFAULT_UPSTREAM, "model": model})
            except Exception as exc:  # noqa: BLE001
                self._write_json(503, {"ok": False, "error": type(exc).__name__, "detail": str(exc)[:300]})
            return
        self._write_json(404, {"type": "error", "error": {"type": "not_found_error", "message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except Exception as exc:  # noqa: BLE001
            self._write_json(400, {"type": "error", "error": {"type": "invalid_request_error", "message": str(exc)}})
            return

        if self.path.endswith("/messages/count_tokens"):
            # Compatibility-only estimate. It is deliberately labelled as such;
            # execution metrics continue to come from vLLM/InferenceX.
            text = json.dumps(payload, ensure_ascii=False)
            self._write_json(200, {"input_tokens": max(1, len(text) // 4)})
            return
        if not self.path.endswith("/messages"):
            self._write_json(404, {"type": "error", "error": {"type": "not_found_error", "message": "not found"}})
            return

        requested_model = str(payload.get("model") or "local-qwen")
        try:
            upstream_model = DEFAULT_MODEL or _discover_model(DEFAULT_UPSTREAM)
            openai_payload = _build_openai_payload(payload, upstream_model)
            upstream = _json_request(DEFAULT_UPSTREAM + "/chat/completions", openai_payload)
            message = _openai_to_anthropic(upstream, requested_model)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            self._write_json(502, {"type": "error", "error": {"type": "api_error", "message": f"upstream HTTP {exc.code}: {detail}"}})
            return
        except Exception as exc:  # noqa: BLE001
            self._write_json(502, {"type": "error", "error": {"type": "api_error", "message": f"{type(exc).__name__}: {exc}"}})
            return

        if not payload.get("stream"):
            self._write_json(200, message)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for event, data in _sse_events(message):
            wire = f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode("utf-8")
            self.wfile.write(wire)
            self.wfile.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"status": "ready", "host": args.host, "port": args.port, "upstream": DEFAULT_UPSTREAM, "model": DEFAULT_MODEL or "auto"}), flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
