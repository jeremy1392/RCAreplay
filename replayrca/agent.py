"""Tool-calling worker agent. Supports starting fresh or continuing from a message prefix (replay)."""
from __future__ import annotations

import json

from . import env, llm, media

SYSTEM_PROMPT = """You are an autonomous back-office operations agent for Maison&Co, an online retailer. Today is 2026-09-01.
You act only through the provided tools; nobody will answer questions. Work carefully and completely: your work is \
checked automatically against company policy and the exact final state of the systems.
When the task is fully complete, call the finish tool with a short summary."""

NUDGE = "Continue working with the tools. Call finish when the task is fully complete."
MAX_STEPS = 80
MAX_STEPS_MULTIMODAL = 160  # level 6: viewing + scanning attachments roughly doubles the action count


def initial_messages(task: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task["prompt"]}]


def _clean_assistant(msg: dict) -> dict:
    out = {"role": "assistant", "content": msg.get("content") or ""}
    if msg.get("tool_calls"):
        out["tool_calls"] = [{"id": tc["id"], "type": "function",
                              "function": {"name": tc["function"]["name"],
                                           "arguments": tc["function"].get("arguments") or "{}"}}
                             for tc in msg["tool_calls"]]
    return out


def parse_args(raw: str):
    try:
        args = json.loads(raw or "{}")
        return args if isinstance(args, dict) else None
    except json.JSONDecodeError:
        return None


def image_of(tool: str, obs: str) -> str | None:
    """Attachment id shown to the agent by a successful view_attachment call."""
    if tool != "view_attachment":
        return None
    try:
        return json.loads(obs).get("attachment_id")
    except json.JSONDecodeError:
        return None


def run(worker: dict, task: dict, *, world: env.World | None = None, messages: list[dict] | None = None,
        steps: list[dict] | None = None, tag: str = "") -> dict:
    """Run an episode, or continue one from (world, messages, steps). Returns a trajectory dict."""
    world = world or env.new_world(task)
    messages = messages if messages is not None else initial_messages(task)
    steps = steps if steps is not None else []
    cost, nudged, end = 0.0, False, "max_steps"
    tools = env.tool_schemas(task)
    max_tokens = worker.get("max_tokens", 16000)
    max_steps = MAX_STEPS_MULTIMODAL if env.is_multimodal(task) else MAX_STEPS
    while len(steps) < max_steps and not world.s["finished"]:
        try:
            r = llm.chat(worker["model"], media.expand(messages, task), tools=tools, temperature=worker.get("temperature", 0.7),
                         max_tokens=max_tokens, extra=worker.get("extra"), tag=tag)
        except llm.BudgetExceeded:
            raise
        except llm.LLMError as e:
            end = f"api_error: {str(e)[:200]}"
            break
        cost += r["cost"]
        msg = _clean_assistant(r["message"])
        calls = msg.get("tool_calls") or []
        if not calls and r["finish_reason"] == "length":
            # Output budget exhausted (often by hidden reasoning): an infrastructure artefact, not an agent decision.
            if max_tokens < 32000:
                max_tokens *= 2
                continue
            end = "truncated"
            break
        if not calls:
            messages.append(msg)
            if nudged:
                end = "no_tool_call"
                break
            nudged = True
            messages.append({"role": "user", "content": NUDGE})
            continue
        messages.append(msg)
        msg_index = len(messages) - 1
        images = []
        for ci, tc in enumerate(calls):
            name, raw = tc["function"]["name"], tc["function"]["arguments"]
            args = parse_args(raw)
            obs = world.call(name, args) if args is not None else json.dumps({"error": "arguments are not valid JSON"})
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": obs})
            step = {"msg_index": msg_index, "call_index": ci, "tool": name, "args": args, "args_raw": raw,
                    "thought": msg["content"] if ci == 0 else "", "observation": obs}
            if image_of(name, obs):
                step["image"] = image_of(name, obs)
                images.append(step["image"])
            steps.append(step)
        if images:
            messages.append(media.image_message(images))
        if world.s["finished"]:
            end = "finish"
    passed, violations = env.check(task, world)
    return {"worker": worker, "task_id": task["task_id"], "messages": messages, "steps": steps, "end": end,
            "passed": passed, "violations": violations, "cost": cost}
