"""Minimal OpenRouter chat client (stdlib only) with retries, cost tracking and a hard budget cap."""
from __future__ import annotations

import http.client
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API_URL = "https://openrouter.ai/api/v1/chat/completions"


class BudgetExceeded(RuntimeError):
    pass


class LLMError(RuntimeError):
    pass


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key:
        raise LLMError("OPENROUTER_API_KEY missing (set it in the environment or in .env)")
    return key


class Budget:
    def __init__(self, limit_usd: float, log_path: Path | None = None):
        self.limit, self.spent = limit_usd, 0.0
        self.by_model: dict[str, float] = {}
        self.lock = threading.Lock()
        self.log_path = log_path

    def check(self):
        if self.spent >= self.limit:
            raise BudgetExceeded(f"budget of ${self.limit:.2f} reached (spent ${self.spent:.2f})")

    def add(self, model: str, cost: float, usage: dict, tag: str):
        with self.lock:
            self.spent += cost
            self.by_model[model] = self.by_model.get(model, 0.0) + cost
            if self.log_path:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"t": round(time.time(), 1), "model": model, "tag": tag, "cost": cost,
                                        "prompt_tokens": usage.get("prompt_tokens"),
                                        "completion_tokens": usage.get("completion_tokens")}) + "\n")


BUDGET = Budget(float(os.environ.get("REPLAYRCA_BUDGET", "5")))
_MODELS: dict[str, dict] = {}
_MOD_LOCK = threading.Lock()


def _model_info(model: str) -> dict:
    """OpenRouter model metadata (fetched once per process)."""
    with _MOD_LOCK:
        for attempt in range(5):
            if _MODELS:
                break
            try:
                req = urllib.request.Request("https://openrouter.ai/api/v1/models",
                                             headers={"Authorization": f"Bearer {_api_key()}"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    for m in json.loads(r.read().decode())["data"]:
                        _MODELS[m["id"]] = m
            except (urllib.error.URLError, http.client.HTTPException, OSError, json.JSONDecodeError) as e:
                if attempt == 4:
                    raise LLMError(f"cannot fetch the OpenRouter model list: {e}")
                time.sleep(2 ** attempt)
    if model not in _MODELS and "/" in model and not model.startswith("baseline/"):
        raise LLMError(f"unknown OpenRouter model id: {model}")
    return _MODELS.get(model, {})


def supports_images(model: str) -> bool:
    return "image" in (_model_info(model).get("architecture", {}).get("input_modalities") or ["text"])


OUTPUT_BUDGET = 64000  # max output tokens (reasoning + answer) for every model at a pinned effort


def top_effort(model: str) -> str | None:
    """Highest effort that fits the shared 64k output budget. Efforts are listed strongest first.
    Anthropic turns effort into a thinking budget of 95% (max/xhigh) or 80% (high) of max_tokens and fills it:
    at max, Sonnet 5.5 thought for 93k tokens and leaves no room for an answer within 64k, so its top is 'high'."""
    efforts = (_model_info(model).get("reasoning") or {}).get("supported_efforts") or []
    if model.startswith("anthropic/") and "high" in efforts:
        return "high"
    return efforts[0] if efforts else None


def reasoning_extra(model: str, effort: str) -> dict | None:
    return {"reasoning": {"effort": effort}} if effort else None


def output_budget(model: str, effort: str) -> int:
    """max_tokens for a call: the shared budget at a pinned effort; an explicit Anthropic max/xhigh needs more room
    because the thinking budget is 95% of max_tokens."""
    if not effort:
        return 24000
    if model.startswith("anthropic/") and effort in ("max", "xhigh"):
        return 100000
    return OUTPUT_BUDGET


def resolve_effort(model: str, effort: str) -> str:
    """'model' -> 'model@effort'; effort 'top' picks each model's own maximum. Explicit @effort is kept."""
    if not effort or "@" in model or "/" not in model:
        return model
    base, sep, variant = model.partition("#")
    level = top_effort(base) if effort == "top" else effort
    return f"{base}@{level}{sep}{variant}" if level else model


def chat(model: str, messages: list[dict], *, tools: list[dict] | None = None, temperature: float | None = None,
         max_tokens: int = 4096, extra: dict | None = None, tag: str = "", retries: int = 6) -> dict:
    """Returns {"message": assistant_message_dict, "cost": float, "usage": dict}."""
    BUDGET.check()
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "usage": {"include": True}}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if temperature is not None:
        body["temperature"] = temperature
    if extra:
        body.update(extra)
    data = json.dumps(body).encode()
    headers = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json",
               "X-Title": "ReplayRCA benchmark"}
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(API_URL, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=300) as r:
                resp = json.loads(r.read().decode())
            if "error" in resp:
                raise LLMError(str(resp["error"])[:500])
            choice = resp["choices"][0]
            usage = resp.get("usage") or {}
            cost = float(usage.get("cost") or 0.0)
            BUDGET.add(model, cost, usage, tag)
            return {"message": choice["message"], "finish_reason": choice.get("finish_reason"), "cost": cost,
                    "usage": usage}
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:500]
            last_err = LLMError(f"HTTP {e.code}: {msg}")
            if e.code in (400, 401, 402, 403, 404):
                raise last_err
        except (urllib.error.URLError, http.client.HTTPException, OSError, json.JSONDecodeError, KeyError,
                IndexError, LLMError) as e:  # HTTPException covers IncompleteRead (connection cut mid-response)
            last_err = e
        time.sleep(min(60, 2 ** attempt + random.random()))
    raise LLMError(f"{model}: giving up after {retries} attempts: {last_err}")
