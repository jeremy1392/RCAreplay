"""Build the leaderboard from cached diagnoses and rollouts.

python -m replayrca.report --models a,b,c
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

from .evaluate import BASELINE, JsonlCache, load_items, placebo_of, rollout_key

ROOT = Path(__file__).resolve().parent.parent


def bootstrap_ci(xs, n=2000, seed=0):
    if not xs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def main(items_path: Path | None = None, models: list[str] | None = None, k: int = 3, stem: str = "leaderboard"):
    """Write results/<stem>.md and .json. The canonical 'leaderboard' is produced by analysis.py over the full model
    list; evaluation runs write 'leaderboard_last_run' so that a partial run never overwrites it."""
    items = load_items(items_path or ROOT / "data" / "items.jsonl")
    diags = JsonlCache(ROOT / "runs" / "diagnoses.jsonl", "id").data
    rolls = JsonlCache(ROOT / "runs" / "rollouts.jsonl", "key").data
    models = models or sorted({d["model"] for d in diags.values()})
    table, per_item = [], collections.defaultdict(dict)
    for m in models:
        ps, by_tpl, strict, valid, missing, cost, rel = [], collections.defaultdict(list), 0, 0, 0, 0.0, []
        by_lvl, blind, toks, rtoks = collections.defaultdict(list), False, [], []
        uplift, placebo = [], []
        for it in items:
            d = diags.get(f"{m}|{it['item_id']}")
            p = 0.0
            if d is None:
                missing += 1
            else:
                cost += d.get("cost", 0.0)
                blind |= d.get("vision") is False and it["task"]["difficulty"] >= 6
                u = d.get("usage") or {}
                if u:
                    toks.append(u.get("completion_tokens") or 0)
                    rtoks.append((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
                if d.get("valid"):
                    valid += 1
                    rel.append(d["t"] / max(1, len(it["traj"]["steps"]) - 1))
                    r = rolls.get(rollout_key(it["item_id"], d["t"], d["tool"], d["args"]))
                    if r is None or len(r["outcomes"]) < k:
                        missing += 1
                    else:
                        p = sum(r["outcomes"][:k]) / k
                        pl = placebo_of(it, d["t"])
                        rp = rolls.get(rollout_key(it["item_id"], d["t"], *pl)) if pl else None
                        if rp and len(rp["outcomes"]) >= k:
                            uplift.append(p - sum(rp["outcomes"][:k]) / k)
                            placebo.append(sum(rp["outcomes"][:k]) / k)
                if d.get("valid") is False:
                    uplift.append(0.0)  # no usable intervention: no causal effect
            ps.append(p)
            by_tpl[it["task"]["template"]].append(p)
            by_lvl[f"d{it['task']['difficulty']}"].append(p)
            strict += p == 1.0
            per_item[it["item_id"]][m] = p
        n = len(items)
        lo, hi = bootstrap_ci(ps)
        table.append({"model": m, "fix_rate": 100 * sum(ps) / n, "ci95": [100 * lo, 100 * hi],
                      "fully_fixed": strict, "valid": valid, "n": n, "missing": missing, "diag_cost": cost,
                      "mean_rel_step": sum(rel) / len(rel) if rel else None,
                      "by_template": {t: 100 * sum(v) / len(v) for t, v in sorted(by_tpl.items())},
                      "by_level": {t: 100 * sum(v) / len(v) for t, v in sorted(by_lvl.items())}, "blind": blind,
                      "mean_out_tokens": sum(toks) / len(toks) if toks else None,
                      "mean_reasoning_tokens": sum(rtoks) / len(rtoks) if rtoks else None,
                      # causal repair: success with the fix minus success of the null intervention at the same step
                      "uplift": 100 * sum(uplift) / len(uplift) if len(uplift) == n else None,
                      "uplift_ci95": [100 * x for x in bootstrap_ci(uplift)] if len(uplift) == n else None,
                      "placebo": 100 * sum(placebo) / len(placebo) if placebo else None})
    table.sort(key=lambda r: -(r["uplift"] if r["uplift"] is not None else r["fix_rate"]))
    real = [m for m in models if m != BASELINE]
    ceiling = [max(per_item[it["item_id"]].get(m, 0) for m in real) for it in items] if real else []
    tpls = sorted({it["task"]["template"] for it in items})
    lvls = sorted({f"d{it['task']['difficulty']}" for it in items})

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / f"{stem}.json").write_text(json.dumps({"k": k, "n_items": len(items), "table": table,
                                                      "pool_ceiling": 100 * sum(ceiling) / len(ceiling) if ceiling else None,
                                                      "per_item": per_item}, indent=1), encoding="utf-8")
    lines = ["# ReplayRCA — leaderboard\n",
             f"{len(items)} systematic-failure items · k = {k} continuations per intervention · "
             f"score = mean success rate of the worker after the model's single-action fix\n",
             "| # | Model | Causal repair (fix − placebo) | Fix rate | 95% CI | Fully fixed | Valid | "
             + " | ".join(tpls + lvls) + " | Out tokens (reasoning) | Diag. cost |",
             "|---|---|---|---|---|---|---|" + "---|" * (len(tpls) + len(lvls)) + "---|---|"]
    for i, r in enumerate(table, 1):
        flag = (" ⚠ incomplete" if r["missing"] else "") + (" (text-only: images hidden)" if r["blind"] else "")
        causal = (f"**{r['uplift']:+.1f}** ({r['uplift_ci95'][0]:+.0f}…{r['uplift_ci95'][1]:+.0f})"
                  if r["uplift"] is not None else "–")
        lines.append(f"| {i} | {r['model']}{flag} | {causal} | **{r['fix_rate']:.1f}%** | {r['ci95'][0]:.0f}–{r['ci95'][1]:.0f} | "
                     f"{r['fully_fixed']}/{r['n']} | {r['valid']}/{r['n']} | "
                     + " | ".join(f"{r['by_template'].get(t, 0):.0f}%" for t in tpls) + " | "
                     + " | ".join(f"{r['by_level'].get(t, 0):.0f}%" for t in lvls)
                     + (f" | {r['mean_out_tokens']:,.0f} ({r['mean_reasoning_tokens']:,.0f})" if r["mean_out_tokens"] is not None
                        else " | -") + f" | ${r['diag_cost']:.2f} |")
    if ceiling:
        lines.append(f"\nPool ceiling (best model per item): **{100 * sum(ceiling) / len(ceiling):.1f}%** · "
                     f"items no model fixed even once: {sum(c == 0 for c in ceiling)}/{len(ceiling)}")
    comp = collections.Counter((it["task"]["template"], it["task"]["difficulty"], it["worker"]["model"]) for it in items)
    lines.append("\n## Item pool\n\n| Template | Difficulty | Worker | Items |\n|---|---|---|---|")
    lines += [f"| {t} | d{d} | {w} | {c} |" for (t, d, w), c in sorted(comp.items())]
    (out / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=None, help="items file (default: data/items.jsonl)")
    ap.add_argument("--models", default="")
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    main(items_path=Path(a.items) if a.items else None, models=[m for m in a.models.split(",") if m] or None, k=a.k)
