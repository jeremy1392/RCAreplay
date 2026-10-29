"""Counterfactual replay: restore the world just before step t, swap in one action, let the worker continue."""
from __future__ import annotations

import copy
import json
import re

from . import agent, env, media

FREE_TEXT_CAP = {"query": 40, "subject": 80, "body": 300}
SUSPICIOUS = re.compile(r"\b(O-\d{5}|refund|cancel|adjust|INC-\d+|must|should|remember|note)\b", re.I)


def restore(task: dict, traj: dict, t: int) -> tuple[env.World, list[dict]]:
    """World state and message prefix right before step t (0-based). Verifies the replay is faithful."""
    world = env.new_world(task)
    steps = traj["steps"]
    for i in range(t):
        s = steps[i]
        obs = world.call(s["tool"], s["args"]) if s["args"] is not None else \
            json.dumps({"error": "arguments are not valid JSON"})
        if obs != s["observation"]:
            raise RuntimeError(f"non-deterministic replay at step {i} of {traj['task_id']}")
    st = steps[t]
    mi, ci = st["msg_index"], st["call_index"]
    msgs = copy.deepcopy(traj["messages"])
    prefix = msgs[:mi]
    if ci > 0:
        asst = msgs[mi]
        prefix.append({**asst, "tool_calls": asst["tool_calls"][:ci]})
        prefix.extend(msgs[mi + 1: mi + 1 + ci])
        images = [s["image"] for s in steps[:t] if s["msg_index"] == mi and s.get("image")]
        if images:
            prefix.append(media.image_message(images))
    return world, prefix


def sanitize(tool: str, args: dict) -> tuple[dict, list[str]]:
    """Strip/cap free-text arguments of a replacement action so it cannot carry a plan to the agent."""
    flags, out = [], {}
    for k, v in args.items():
        if k in ("reason", "summary"):
            out[k] = "-"
            continue
        if k in FREE_TEXT_CAP and isinstance(v, str):
            if len(v) > FREE_TEXT_CAP[k]:
                flags.append(f"{k} truncated")
                v = v[:FREE_TEXT_CAP[k]]
            if k in ("query", "subject") and SUSPICIOUS.search(v):
                flags.append(f"suspicious {k}")
        out[k] = v
    return out, flags


def continue_with(task: dict, traj: dict, t: int, tool: str, args: dict, tag: str = "") -> dict:
    """Replace step t by (tool, args) and let the original worker continue. Returns the new trajectory."""
    world, prefix = restore(task, traj, t)
    steps = copy.deepcopy(traj["steps"][:t])
    call_id = "call_fix_0"
    prefix.append({"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}]})
    obs = world.call(tool, args)
    prefix.append({"role": "tool", "tool_call_id": call_id, "content": obs})
    step = {"msg_index": len(prefix) - 2, "call_index": 0, "tool": tool, "args": args,
            "args_raw": json.dumps(args), "thought": "", "observation": obs, "replaced": True}
    if agent.image_of(tool, obs):
        step["image"] = agent.image_of(tool, obs)
        prefix.append(media.image_message([step["image"]]))
    steps.append(step)
    return agent.run(traj["worker"], task, world=world, messages=prefix, steps=steps, tag=tag)
