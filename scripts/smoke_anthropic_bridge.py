#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PORT = 4010
BASE = f"http://127.0.0.1:{PORT}"
MODEL = "qwen3-coder-r9700"


def http_json(url: str, payload: dict | None = None, headers: dict | None = None, timeout: float = 30.0):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode())


def wait_ready(timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            status, _ = http_json(f"{BASE}/health/liveliness", timeout=3)
            if status == 200:
                return
        except Exception as exc:
            last = repr(exc)
        time.sleep(1)
    raise RuntimeError(f"bridge_not_ready: {last}")


def main() -> int:
    env = os.environ.copy()
    cmd = [
        "litellm",
        "--config",
        str(ROOT / "config.yaml"),
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
    ]
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        wait_ready()
        payload = {
            "model": MODEL,
            "max_tokens": 64,
            "temperature": 0,
            "messages": [{"role": "user", "content": "Reply with exactly: R9700 BRIDGE PASS"}],
        }
        headers = {
            "content-type": "application/json",
            "x-api-key": "local-hyperloom-bridge",
            "anthropic-version": "2023-06-01",
        }
        status, body = http_json(f"{BASE}/v1/messages", payload, headers, timeout=90)
        text = "".join(block.get("text", "") for block in body.get("content", []) if isinstance(block, dict))
        ok = status == 200 and "R9700 BRIDGE PASS" in text
        print(json.dumps({
            "ok": ok,
            "status": status,
            "endpoint": "/v1/messages",
            "model": MODEL,
            "response_type": body.get("type"),
            "response_model": body.get("model"),
            "text": text,
        }, indent=2))
        return 0 if ok else 4
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        out, err = proc.communicate()
        if proc.returncode not in (0, -15, -9):
            print(out[-3000:], file=sys.stderr)
            print(err[-3000:], file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
