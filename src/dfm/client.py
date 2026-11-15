"""Minimal OpenRouter chat-completions client (standard library only)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field

from .paths import REPO_ROOT

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_RETRY_CODES = {402, 429, 502, 503, 504}


def load_dotenv(path=None) -> None:
    """Read ``KEY=VALUE`` lines from ``.env`` without overriding the environment."""
    env_path = path or (REPO_ROOT / ".env")
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class Reply:
    content: str
    reasoning: str = ""
    provider: str | None = None
    usage: dict = field(default_factory=dict)


def _text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text") or "") if isinstance(item, dict) else str(item)
            for item in content
            if isinstance(item, str) or (isinstance(item, dict) and item.get("type") in {"text", "output_text"})
        )
    return ""


def _reasoning(message: dict, payload: dict) -> str:
    chunks = [message[k] for k in ("reasoning", "reasoning_content") if isinstance(message.get(k), str) and message[k].strip()]
    details = message.get("reasoning_details") or payload.get("reasoning_details")
    if isinstance(details, list):
        for item in details:
            if isinstance(item, dict) and (item.get("text") or item.get("summary")):
                chunks.append(str(item.get("text") or item.get("summary")))
            elif isinstance(item, str):
                chunks.append(item)
    return "\n\n".join(chunks)


def _usage(payload: dict) -> dict:
    usage = payload.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    cost = usage.get("cost")
    if cost is None and isinstance(usage.get("cost_details"), dict):
        cost = usage["cost_details"].get("upstream_inference_cost") or usage["cost_details"].get("total_cost")
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "reasoning_tokens": int(details.get("reasoning_tokens") or usage.get("reasoning_tokens") or 0),
        "cost_usd": float(cost) if cost is not None else None,
    }


class OpenRouter:
    """Chat client bound to one model configuration.

    ``model_cfg`` holds ``model``, ``reasoning``, ``temperature``, ``top_p``,
    ``max_tokens`` and an optional ``provider`` routing block, as in
    ``configs/*.json``.
    """

    def __init__(self, model_cfg: dict, api_key: str | None = None, *, timeout_s: float = 600, max_retries: int = 4):
        load_dotenv()
        self.cfg = model_cfg
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not self.api_key:
            raise SystemExit("OPENROUTER_API_KEY is not set (export it or add it to .env)")
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def _payload(self, messages: list[dict]) -> bytes:
        cfg = self.cfg
        payload = {
            "model": cfg["model"],
            "messages": messages,
            "reasoning": cfg.get("reasoning") or {"enabled": True},
            "usage": {"include": True},
            "provider": cfg.get("provider") or {"allow_fallbacks": True, "data_collection": "deny"},
        }
        for key in ("temperature", "top_p", "max_tokens"):
            if cfg.get(key) is not None:
                payload[key] = cfg[key]
        return json.dumps(payload).encode("utf-8")

    def _post(self, body: bytes) -> dict:
        request = urllib.request.Request(
            OPENROUTER_URL,
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def chat(self, messages: list[dict]) -> Reply:
        body = self._payload(messages)
        last: Exception | None = None
        for attempt in range(self.max_retries):
            pool = ThreadPoolExecutor(max_workers=1)
            try:
                payload = pool.submit(self._post, body).result(timeout=self.timeout_s + 2)
                message = ((payload.get("choices") or [{}])[0]).get("message") or {}
                return Reply(_text(message), _reasoning(message, payload), payload.get("provider"), _usage(payload))
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code not in _RETRY_CODES:
                    detail = exc.read().decode("utf-8", errors="replace")[:800]
                    raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
                time.sleep(120.0 if exc.code == 402 else min(90.0, 5.0 * 2**attempt))
            except (FuturesTimeout, urllib.error.URLError, TimeoutError) as exc:
                last = exc
                time.sleep(min(60.0, 2.0**attempt))
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
        raise RuntimeError(f"request failed after {self.max_retries} attempts: {last}")
