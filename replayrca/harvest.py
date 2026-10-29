"""Harvest systematic failures of worker agents into benchmark items.

python -m replayrca.harvest --workers openai/gpt-6-luna,anthropic/claude-haiku-5.5 --seeds 0-9 --difficulties 2,3

A failure becomes an item only if it is systematic: the worker fails the task in at least
2 of 3 independent fresh runs. Re-running the harvest with newer, stronger workers is what
keeps the benchmark from saturating: their remaining failures are, by construction, subtler.
"""
from __future__ import annotations

import argparse
import collections
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import agent, env, llm

ROOT = Path(__file__).resolve().parent.parent


def parse_range(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def _artefact(tr) -> bool:
    """Episodes that ended for infrastructure reasons are not agent failures."""
    return tr["end"].startswith("api_error") or tr["end"] == "truncated"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", required=True)
    ap.add_argument("--templates", default=",".join(env.TEMPLATES))
    ap.add_argument("--seeds", default="0-5")
    ap.add_argument("--difficulties", default="2,3")
    ap.add_argument("--reruns", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--out", default="data/items.jsonl")
    a = ap.parse_args()

    (ROOT / "runs").mkdir(exist_ok=True)
    llm.BUDGET.limit, llm.BUDGET.log_path = a.budget, ROOT / "runs" / "calls.jsonl"
    workers = [{"model": m.strip(), "temperature": a.temperature, "max_tokens": 16000} for m in a.workers.split(",")]
    tasks = [env.generate(t, s, d) for t in a.templates.split(",") for d in parse_range(a.difficulties)
             for s in parse_range(a.seeds)]
    by_id = {t["task_id"]: t for t in tasks}
    log_path = ROOT / "runs" / "harvest_trajectories.jsonl"

    def run_key(task_id, worker, attempt):
        return task_id, worker.get("model"), worker.get("temperature"), worker.get("max_tokens"), attempt

    # Resume: episodes already in the log (same task, worker config and attempt) are reused, never paid twice.
    done = {}
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                tr = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not _artefact(tr):
                done[run_key(tr["task_id"], tr.get("worker") or {}, tr.get("attempt", 0))] = tr
    traj_log = open(log_path, "a", encoding="utf-8")
    log_lock = threading.Lock()

    def job(worker, task, attempt):
        cached = done.get(run_key(task["task_id"], worker, attempt))
        if cached is not None:
            return worker, task["task_id"], attempt, cached, True
        tr = agent.run(worker, task, tag=f"harvest:{task['task_id']}:{attempt}")
        with log_lock:
            traj_log.write(json.dumps({**tr, "attempt": attempt}) + "\n")
            traj_log.flush()
        return worker, task["task_id"], attempt, tr, False

    def collect(futs, label):
        for f in as_completed(futs):
            try:
                w, tid, att, tr, cached = f.result()
            except llm.BudgetExceeded as e:
                print(f"STOP: {e}", flush=True)
                continue
            except Exception as e:  # noqa: BLE001 - one broken episode must not lose the others
                print(f"episode failed ({type(e).__name__}): {str(e)[:200]}", flush=True)
                continue
            results[(w["model"], tid)].append((att, tr))
            print(f"[{llm.BUDGET.spent:6.3f}$] {label}{att} {w['model']:32s} {tid:24s} "
                  f"{'PASS' if tr['passed'] else 'FAIL'} steps={len(tr['steps'])} end={tr['end'][:24]}"
                  f"{' (cached)' if cached else ''}", flush=True)

    results = collections.defaultdict(list)  # (model, task_id) -> [(attempt, traj)]
    with ThreadPoolExecutor(a.threads) as pool:
        collect([pool.submit(job, w, t, 0) for t in tasks for w in workers], "try ")
        failed = [(w, by_id[tid]) for (m, tid), trs in results.items() for w in workers
                  if w["model"] == m and not trs[0][1]["passed"] and not _artefact(trs[0][1])]
        collect([pool.submit(job, w, t, i + 1) for w, t in failed for i in range(a.reruns)], "rerun ")
    results = {k: [tr for _, tr in sorted(v, key=lambda x: x[0])] for k, v in results.items()}

    items, summary = [], collections.defaultdict(lambda: [0, 0])
    for (model, tid), trs in sorted(results.items()):
        task = by_id[tid]
        first = trs[0]
        summary[(model, task["template"], task["difficulty"])][0] += first["passed"]
        summary[(model, task["template"], task["difficulty"])][1] += 1
        if first["passed"] or _artefact(first) or len(first["steps"]) < 3:
            continue
        if first["end"] == "max_steps":
            continue  # ran out of steps: inefficiency, not a mistake a single intervention can repair
        n_fail = sum(not t["passed"] and not _artefact(t) for t in trs)
        if n_fail < 2:
            continue  # not systematic
        items.append({
            "item_id": f"{tid}__{model.replace('/', '_')}",
            "task": {k: task[k] for k in ("task_id", "template", "seed", "difficulty", "state_hash", "traps")},
            "worker": first["worker"], "fresh_pass": f"{len(trs) - n_fail}/{len(trs)}",
            "traj": {k: first[k] for k in ("task_id", "worker", "messages", "steps", "end", "violations")},
        })
    out = ROOT / a.out
    out.parent.mkdir(exist_ok=True)
    existing = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            it = json.loads(line)
            existing[it["item_id"]] = it
    for it in items:
        existing.setdefault(it["item_id"], it)
    out.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in existing.values()), encoding="utf-8")

    print("\nFirst-attempt pass rate (worker x template x difficulty):")
    for (m, t, d), (p, n) in sorted(summary.items()):
        print(f"  {m:32s} {t:14s} d{d}: {p}/{n}")
    print(f"\n{len(items)} new systematic-failure items ({len(existing)} total in {out}); spent ${llm.BUDGET.spent:.3f}")


if __name__ == "__main__":
    main()
