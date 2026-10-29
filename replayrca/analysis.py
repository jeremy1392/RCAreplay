"""Final analyses from the caches (no API calls).

python -m replayrca.analysis
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from . import oversight, report
from .evaluate import BASELINE, JsonlCache, load_items, placebo_of, rollout_key

ROOT = Path(__file__).resolve().parent.parent
K = 5
AUTOPSY = ["x-ai/grok-4.7@xhigh", "google/gemini-4-argon@high", "moonshotai/kimi-k3@max", "openai/gpt-6.1-sol@max",
           "google/gemini-3.8-flash@high", "z-ai/glm-5.3@max", "anthropic/claude-opus-5.5@high",
           "deepseek/deepseek-v4-pro-0813@max", "anthropic/claude-sonnet-5.5@high", "openai/gpt-6-luna@max",
           "anthropic/claude-haiku-5.5@high", "openai/gpt-6-astra@max", "anthropic/claude-fable-5.1@high", BASELINE]
SUPERVISORS = ["deepseek/deepseek-v4.1-flash-1017@max", "google/gemini-3.8-flash@high", "anthropic/claude-haiku-5.5@high",
               "openai/gpt-6-luna@max", "openai/gpt-6.1-sol@max"]


def causal(item, d, rolls):
    """Autopsy causal effect of one diagnosis, or None if a rollout is missing (invalid answer = 0)."""
    if not d.get("valid"):
        return 0.0
    r = rolls.get(rollout_key(item["item_id"], d["t"], d["tool"], d["args"]))
    pl = placebo_of(item, d["t"])
    rp = rolls.get(rollout_key(item["item_id"], d["t"], *pl)) if pl else None
    if not r or not rp or len(r["outcomes"]) < K or len(rp["outcomes"]) < K:
        return None
    return sum(r["outcomes"][:K]) / K - sum(rp["outcomes"][:K]) / K


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    items = load_items(ROOT / "data" / "items_calibrated.jsonl")
    by_id = {i["item_id"]: i for i in items}
    runs = load_items(ROOT / "data" / "oversight_all.jsonl")
    D = JsonlCache(ROOT / "runs" / "diagnoses.jsonl", "id").data
    S = JsonlCache(ROOT / "runs" / "oversight.jsonl", "id").data
    R = JsonlCache(ROOT / "runs" / "rollouts.jsonl", "key").data
    out = []

    # 0. oversight report on the common set (runs that every supervisor saw, with complete rollouts), as in the paper
    common = [r for r in runs if all(S.get(f"{m}|{r['item_id']}") and oversight.effect(S[f"{m}|{r['item_id']}"], r, R, K) is not None
                                     for m in SUPERVISORS)]
    oversight.report(common, SUPERVISORS, K, ROOT / "results" / "oversight.md")

    # 1. full autopsy leaderboard (also written to results/leaderboard.md/json)
    report.main(items_path=ROOT / "data" / "items_calibrated.jsonl", models=AUTOPSY, k=K)
    out.append((ROOT / "results" / "leaderboard.md").read_text(encoding="utf-8"))

    # 2. autopsy vs oversight on the same failing runs
    fails = [r for r in runs if r["label"] == "fail" and r["item_id"] in by_id]
    out.append(f"\n## Autopsy vs live oversight — same {len(fails)} failing runs\n")
    out.append("| Model | Autopsy (told it failed) | Live oversight (outcome unknown) | Gap |\n|---|---|---|---|")
    for m in SUPERVISORS:
        a, o = [], []
        for r in fails:
            d, s = D.get(f"{m}|{r['item_id']}"), S.get(f"{m}|{r['item_id']}")
            ca = causal(by_id[r["item_id"]], d, R) if d else None
            co = oversight.effect(s, r, R, K) if s else None
            if co is not None:
                o.append(co)
            if ca is not None:
                a.append(ca)
        at = f"{100 * mean(a):+.1f} (n={len(a)})" if a else "not run"
        gap = f"{100 * (mean(o) - mean(a)):+.1f}" if a and o else "–"
        out.append(f"| {m} | {at} | {100 * mean(o):+.1f} (n={len(o)}) | {gap} |")

    # 3. level 6 (images): vision vs text-only diagnosers, and failures where the agent never opened an attachment
    out.append("\n## Level 6 (multimodal) — autopsy causal repair on d6 items\n")
    d6 = [i for i in items if i["task"]["difficulty"] == 6]
    blind = sum(not any(s.get("image") for s in i["traj"]["steps"]) for i in d6)
    out.append(f"{len(d6)} d6 items; in {blind} of them the failing agent never opened an attachment.\n")
    out.append("| Model | d6 causal repair | d4–d5 causal repair | Sees images |\n|---|---|---|---|")
    for m in AUTOPSY[:-1]:
        c6, c45, vis = [], [], None
        for i in items:
            d = D.get(f"{m}|{i['item_id']}")
            if not d:
                continue
            vis = d.get("vision", vis)
            c = causal(i, d, R)
            if c is not None:
                (c6 if i["task"]["difficulty"] == 6 else c45).append(c)
        out.append(f"| {m} | {100 * mean(c6):+.1f} | {100 * mean(c45):+.1f} | {'yes' if vis else 'no'} |")

    # 4. cost-effectiveness of the autopsy track
    lb = json.loads((ROOT / "results" / "leaderboard.json").read_text(encoding="utf-8"))
    out.append("\n## Cost-effectiveness (autopsy)\n")
    out.append("| Model | Causal repair | Diagnosis cost per item | Points per dollar |\n|---|---|---|---|")
    for r in sorted(lb["table"], key=lambda r: -(r["uplift"] or 0)):
        if r["model"] == BASELINE or not r["n"] or r["uplift"] is None:
            continue  # incomplete models are reported in the leaderboard with a warning
        per = r["diag_cost"] / r["n"]
        out.append(f"| {r['model']} | {r['uplift']:+.1f} | ${per:.3f} | {r['uplift'] / per if per else float('inf'):,.0f} |")

    text = "\n".join(out) + "\n"
    (ROOT / "results" / "analysis.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
