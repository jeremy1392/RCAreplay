"""Offline pre-flight check of every stored trajectory (no API calls).

python -m replayrca.verify data/items.jsonl data/items_d6.jsonl ...

For each run: the generator still reproduces the recorded initial state (hash), the world can be
restored faithfully at several steps and at every oversight checkpoint (each restore replays the
recorded actions and compares every observation), placebo actions exist, and every image
attachment renders.
"""
from __future__ import annotations

import sys
from pathlib import Path

from . import media, oversight, replay
from .evaluate import load_items, placebo_of, task_of

ROOT = Path(__file__).resolve().parent.parent


def main(paths: list[str]):
    n_runs = n_restores = n_images = 0
    problems = []
    for p in paths:
        for run in load_items(ROOT / p):
            rid = run["item_id"]
            try:
                task = task_of(run)
            except RuntimeError as e:
                problems.append(f"{rid}: {e}")
                continue
            steps = run["traj"]["steps"]
            n = len(steps)
            points = sorted({0, n // 2, n - 1} | set(oversight.checkpoints(run["traj"])))
            for t in points:
                try:
                    replay.restore(task, run["traj"], t)
                    n_restores += 1
                except Exception as e:  # noqa: BLE001
                    problems.append(f"{rid} restore@{t}: {type(e).__name__}: {e}")
                    break
            for t in oversight.checkpoints(run["traj"]):
                if placebo_of(run, t) is None:
                    problems.append(f"{rid}: no placebo at checkpoint {t} (invalid original args)")
            for aid, att in (task["state"].get("attachments") or {}).items():
                try:
                    media.render(aid, att)
                    n_images += 1
                except Exception as e:  # noqa: BLE001
                    problems.append(f"{rid} image {aid}: {e}")
            n_runs += 1
    print(f"verified {n_runs} runs, {n_restores} faithful restores, {n_images} images rendered")
    if problems:
        print(f"{len(problems)} PROBLEM(S):")
        for x in problems[:40]:
            print("  -", x)
        sys.exit(1)
    print("no problem found")


if __name__ == "__main__":
    main(sys.argv[1:] or ["data/items.jsonl"])
