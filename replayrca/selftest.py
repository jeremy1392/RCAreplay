"""Offline validation of the generator, tools and checker (no API calls).

python -m replayrca.selftest
"""
from __future__ import annotations

import json

from . import env, replay


def main(n_seeds: int = 40):
    n = 0
    for template in env.TEMPLATES:
        for d in (1, 2, 3, 4, 5, 6):
            for seed in range(n_seeds):
                task = env.generate(template, seed, d)
                assert env.generate(template, seed, d)["state_hash"] == task["state_hash"], "non-deterministic"
                exp = task["expected"]
                assert exp["refunds"] or exp["cancels"] or exp["adjust"], f"empty task {task['task_id']}"
                # oracle passes
                w = env.new_world(task)
                for tool, args in env.oracle_actions(task):
                    obs = json.loads(w.call(tool, args))
                    assert "error" not in obs, (task["task_id"], tool, args, obs)
                ok, v = env.check(task, w)
                assert ok, (task["task_id"], v)
                # doing nothing fails
                w0 = env.new_world(task)
                w0.call("finish", {"summary": "x"})
                assert not env.check(task, w0)[0], task["task_id"]
                # replay restores state faithfully
                traj = {"task_id": task["task_id"], "messages": [{"role": "system"}, {"role": "user"}], "steps": []}
                w1 = env.new_world(task)
                for i, (tool, args) in enumerate(env.oracle_actions(task)):
                    obs = w1.call(tool, args)
                    traj["messages"] += [{"role": "assistant", "tool_calls": [{"id": f"c{i}"}]},
                                         {"role": "tool", "tool_call_id": f"c{i}", "content": obs}]
                    traj["steps"].append({"msg_index": len(traj["messages"]) - 2, "call_index": 0, "tool": tool,
                                          "args": args, "observation": obs})
                last = len(traj["steps"]) - 1
                wr, prefix = replay.restore(task, traj, last)
                assert len(prefix) == traj["steps"][last]["msg_index"]
                n += 1
    # a few trap sanity checks on one instance
    t = env.generate("late_delivery", 0, 3)
    w = env.new_world(t)
    docs = json.loads(w.call("list_docs", {}))["documents"]
    assert {d["name"] for d in docs} >= {"late_delivery_policy", "late_delivery_policy_update_2026"}
    assert "error" in json.loads(w.call("read_doc", {"name": "ignore previous instructions"}))
    print(f"selftest OK: {n} task instances validated")
    for template in env.TEMPLATES:
        task = env.generate(template, 0, 6)
        e = task["expected"]
        print(f"  {task['task_id']}: refunds={len(e['refunds'])} cancels={len(e['cancels'])} "
              f"adjust={len(e['adjust'])} emails={len(e['emails'])} traps={task['traps']}")


def check_media(n_seeds: int = 5):
    """Level 6: every rendered attachment must carry a QR code that a real decoder (OpenCV) can read."""
    from pathlib import Path

    from . import media
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("media check skipped (opencv not installed)")
        return
    out = Path(__file__).resolve().parent.parent / "results" / "samples"
    out.mkdir(parents=True, exist_ok=True)
    det, ok, bad, saved = cv2.QRCodeDetector(), 0, [], set()
    for template in env.TEMPLATES:
        for seed in range(n_seeds):
            task = env.generate(template, seed, 6)
            for aid, att in sorted(task["state"]["attachments"].items()):
                jpg = media.render(aid, att)
                data, _, _ = det.detectAndDecode(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR))
                if data == att["codes"][0]:
                    ok += 1
                else:
                    bad.append(f"{task['task_id']}/{aid}")
                if att["kind"] not in saved:
                    (out / f"{att['kind']}.jpg").write_bytes(jpg)
                    saved.add(att["kind"])
    print(f"media check: {ok}/{ok + len(bad)} QR codes decoded by OpenCV; samples in {out}")
    if bad:
        print("  not decoded:", bad[:10])


if __name__ == "__main__":
    main()
    check_media()
