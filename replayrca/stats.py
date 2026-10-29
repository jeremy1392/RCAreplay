"""Statistical analyses for the paper, computed from the caches (no API calls).

python -m replayrca.stats [--renewal-items data/items_qwen_cal.jsonl --renewal-models m1,m2]

Writes results/stats.json (every number the paper quotes) and results/stats.md (readable).
Paired comparisons use the item (or run) as the unit: bootstrap CI of the mean difference and a
sign-flip permutation test; families of pairwise tests are Holm-corrected.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

from . import oversight
from .analysis import AUTOPSY, K, SUPERVISORS, causal
from .evaluate import BASELINE, JsonlCache, load_items, placebo_of, rollout_key

ROOT = Path(__file__).resolve().parent.parent
N_BOOT = 10000
PENALTY = 0.05
HUMAN = "human/author"  # the author's own diagnoses (paper/human_baseline.py)
POSITION_BINS = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.0))  # relative step; the last bin is the final step
READ_TOOLS = {"read_doc", "list_docs", "read_ticket", "list_tickets", "search_customers", "get_customer",
              "list_orders", "search_orders", "get_order", "get_inventory", "view_attachment", "scan_code"}


def rng():
    return np.random.default_rng(0)


def boot_mean(x):
    x = np.asarray(x, float)
    idx = rng().integers(0, len(x), (N_BOOT, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(a, b):
    """Paired test of mean(a - b): bootstrap CI and sign-flip permutation p-value."""
    d = np.asarray(a, float) - np.asarray(b, float)
    mean, lo, hi = boot_mean(d)
    signs = rng().choice([-1.0, 1.0], (N_BOOT, len(d)))
    perm = (d * signs).mean(1)
    p = float(((np.abs(perm) >= abs(d.mean()) - 1e-12).sum() + 1) / (N_BOOT + 1))
    return {"n": int(len(d)), "mean_diff": 100 * mean, "ci95": [100 * lo, 100 * hi], "p": p}


def unpaired(a, b):
    """Difference of means between two independent samples: Welch t-test and bootstrap CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p = sps.ttest_ind(a, b, equal_var=False)
    g = rng()
    ia = g.integers(0, len(a), (N_BOOT, len(a)))
    ib = g.integers(0, len(b), (N_BOOT, len(b)))
    diff = a[ia].mean(1) - b[ib].mean(1)
    return {"n_a": int(len(a)), "n_b": int(len(b)), "mean_a": 100 * float(a.mean()), "mean_b": 100 * float(b.mean()),
            "mean_diff": 100 * float(a.mean() - b.mean()),
            "ci95": [100 * float(np.percentile(diff, 2.5)), 100 * float(np.percentile(diff, 97.5))],
            "p_welch": float(p)}


def holm(pvals: dict) -> dict:
    """Holm step-down adjusted p-values for a family of tests {name: p}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, adj, running = len(items), {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, (m - i) * p)
        adj[name] = min(1.0, running)
    return adj


def effects_matrix(items, D, R, models):
    """{model: {item_id: causal effect}} with None for missing rollouts (invalid answers count as 0)."""
    out = {}
    for m in models:
        out[m] = {}
        for it in items:
            d = D.get(f"{m}|{it['item_id']}")
            out[m][it["item_id"]] = causal(it, d, R) if d else None
    return out


def common(eff, a, b):
    ids = [i for i in eff[a] if eff[a][i] is not None and eff[b].get(i) is not None]
    return [eff[a][i] for i in ids], [eff[b][i] for i in ids], ids


def pairwise_family(eff, models, label):
    tests, pv = {}, {}
    for a, b in itertools.combinations(models, 2):
        xa, xb, _ = common(eff, a, b)
        if len(xa) < 5:
            continue
        tests[f"{a} vs {b}"] = paired(xa, xb)
        pv[f"{a} vs {b}"] = tests[f"{a} vs {b}"]["p"]
    adj = holm(pv)
    for k in tests:
        tests[k]["p_holm"] = adj[k]
    sig = sorted(k for k in tests if tests[k]["p_holm"] < 0.05)
    return {"label": label, "tests": tests, "n_pairs": len(tests), "significant_holm": sig}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renewal-items", default="data/items_qwen_cal.jsonl")
    ap.add_argument("--renewal-models", default="")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    items = load_items(ROOT / "data" / "items_calibrated.jsonl")
    by_id = {i["item_id"]: i for i in items}
    runs_all = load_items(ROOT / "data" / "oversight_all.jsonl")
    D = JsonlCache(ROOT / "runs" / "diagnoses.jsonl", "id").data
    S = JsonlCache(ROOT / "runs" / "oversight.jsonl", "id").data
    R = JsonlCache(ROOT / "runs" / "rollouts.jsonl", "key").data
    out, md = {}, []
    # the oversight track proper: every run that all supervisors have supervised, with complete continuations
    runs = [r for r in runs_all
            if all((S.get(f"{m}|{r['item_id']}") or {}) and oversight.effect(S[f"{m}|{r['item_id']}"], r, R, K) is not None
                   for m in SUPERVISORS)]
    out["oversight_runs"] = {"all": len(runs_all), "common": len(runs), "fail": sum(r["label"] == "fail" for r in runs),
                             "success": sum(r["label"] == "success" for r in runs)}

    # ---------------------------------------------------------------- autopsy
    core = [m for m in AUTOPSY if m != BASELINE]
    extras = [m for m in ("openai/gpt-6.1-sol@max#last", "openai/gpt-6.1-sol@medium", "anthropic/claude-haiku-5.5@max",
                          "openai/gpt-6-luna@max#rep2", "openai/gpt-6-luna@max#alt", HUMAN)
              if any(k.startswith(m + "|") for k in D)]
    eff = effects_matrix(items, D, R, core + extras + [BASELINE])
    coverage = {m: sum(v is not None for v in eff[m].values()) for m in eff}
    out["coverage"] = coverage
    summary = {}
    for m in eff:
        xs = [v for v in eff[m].values() if v is not None]
        if xs:
            mean, lo, hi = boot_mean(xs)
            summary[m] = {"n": len(xs), "causal_repair": 100 * mean, "ci95": [100 * lo, 100 * hi]}
    out["autopsy_summary"] = summary
    md.append("# Statistical analyses\n\n## Autopsy: causal repair per model (bootstrap over items)\n")
    md.append("| Model | n | Causal repair | 95% CI |\n|---|---|---|---|")
    for m, s in sorted(summary.items(), key=lambda kv: -kv[1]["causal_repair"]):
        md.append(f"| {m} | {s['n']} | {s['causal_repair']:+.1f} | {s['ci95'][0]:+.1f} … {s['ci95'][1]:+.1f} |")

    fam = pairwise_family(eff, core, "autopsy pairwise (13 models)")
    out["autopsy_pairwise"] = fam
    top = max(core, key=lambda m: summary[m]["causal_repair"])
    md.append(f"\n## Autopsy: pairwise paired tests ({fam['n_pairs']} pairs, Holm-corrected)\n")
    md.append(f"Pairs significant at 0.05 after Holm: {len(fam['significant_holm'])}")
    for k in fam["significant_holm"]:
        t = fam["tests"][k]
        md.append(f"- {k}: Δ = {t['mean_diff']:+.1f} [{t['ci95'][0]:+.1f}, {t['ci95'][1]:+.1f}], p_holm = {t['p_holm']:.3f}")
    md.append(f"\nTop model {top} vs each other model (raw p, sign-flip):")
    for m in core:
        if m == top:
            continue
        t = fam["tests"].get(f"{top} vs {m}") or fam["tests"].get(f"{m} vs {top}")
        if t:
            sign = 1 if f"{top} vs {m}" in fam["tests"] else -1
            md.append(f"- vs {m}: Δ = {sign * t['mean_diff']:+.1f}, p = {t['p']:.3f}, p_holm = {t['p_holm']:.3f}")

    # structural tendencies: relative step, last-step share, read-vs-act, agreement
    pos, last_share, read_share, kind = {}, {}, {}, collections.defaultdict(list)
    for m in core + extras:
        rel, last, reads = [], 0, 0
        for it in items:
            d = D.get(f"{m}|{it['item_id']}")
            if not d or not d.get("valid"):
                continue
            n = len(it["traj"]["steps"])
            rel.append(d["t"] / max(1, n - 1))
            last += d["t"] == n - 1
            reads += d["tool"] in READ_TOOLS
            e = eff[m][it["item_id"]]
            if e is not None and m in core:  # the read-vs-act pool covers the main-table models only
                kind["read" if d["tool"] in READ_TOOLS else "act"].append(e)
        if rel:
            pos[m] = float(np.mean(rel))
            last_share[m] = last / len(rel)
            read_share[m] = reads / len(rel)
    out["relative_step"] = pos
    out["last_step_share"] = last_share
    out["autopsy_read_share"] = read_share
    out["read_vs_act"] = {k: dict(zip(("mean", "lo", "hi"), [100 * x for x in boot_mean(v)]), n=len(v))
                          for k, v in kind.items()}
    md.append("\n## Where models intervene\n\n| Model | Mean relative step (0 = first, 1 = last) | Chose the last step |\n|---|---|---|")
    for m in core + extras:
        if m in pos:
            md.append(f"| {m} | {pos[m]:.2f} | {100 * last_share[m]:.0f}% |")
    md.append("\nCausal repair by kind of replacement action (all core models pooled; not causal, items differ):")
    for k, v in out["read_vs_act"].items():
        md.append(f"- {k}: {v['mean']:+.1f} [{v['lo']:+.1f}, {v['hi']:+.1f}] (n = {v['n']})")

    # agreement on the decisive step
    agree, modal_eff, other_eff = [], [], []
    for it in items:
        choices = {m: D[f"{m}|{it['item_id']}"]["t"] for m in core
                   if f"{m}|{it['item_id']}" in D and D[f"{m}|{it['item_id']}"].get("valid")}
        if len(choices) < 2:
            continue
        pairs = list(itertools.combinations(choices.values(), 2))
        agree.append(sum(x == y for x, y in pairs) / len(pairs))
        modal = collections.Counter(choices.values()).most_common(1)[0][0]
        for m, t in choices.items():
            e = eff[m][it["item_id"]]
            if e is not None:
                (modal_eff if t == modal else other_eff).append(e)
    out["step_agreement"] = {"mean_pairwise_agreement": float(np.mean(agree)),
                             "effect_modal_step": 100 * float(np.mean(modal_eff)),
                             "effect_other_step": 100 * float(np.mean(other_eff)),
                             "n_modal": len(modal_eff), "n_other": len(other_eff)}
    md.append(f"\nPairwise agreement on the chosen step: {100 * np.mean(agree):.0f}% of model pairs per item. "
              f"Causal repair when choosing the modal step: {100 * np.mean(modal_eff):+.1f} (n={len(modal_eff)}) "
              f"vs other steps {100 * np.mean(other_eff):+.1f} (n={len(other_eff)}).")

    # item difficulty distribution and failure categories
    per_item = {it["item_id"]: [eff[m][it["item_id"]] for m in core if eff[m][it["item_id"]] is not None] for it in items}
    item_mean = {k: float(np.mean(v)) for k, v in per_item.items() if v}
    out["item_means"] = {k: 100 * v for k, v in item_mean.items()}
    out["item_difficulty"] = {"n": len(item_mean), "mean_of_item_means": 100 * float(np.mean(list(item_mean.values()))),
                              "share_hard_lt20": float(np.mean([v < 0.2 for v in item_mean.values()])),
                              "share_easy_gt80": float(np.mean([v > 0.8 for v in item_mean.values()]))}
    cats = collections.defaultdict(list)
    for it in items:
        kinds = sorted({v.split(" on ")[0].split(" of ")[0].split(" for ")[0].split(" to ")[0] for v in it["traj"]["violations"]})
        key = " + ".join(kinds) if len(kinds) <= 2 else "3+ kinds"
        if it["item_id"] in item_mean:
            cats[key].append(item_mean[it["item_id"]])
    out["failure_categories"] = {k: {"n": len(v), "mean_repair": 100 * float(np.mean(v))} for k, v in cats.items()}
    md.append("\n## Item difficulty\n")
    md.append(f"Mean over items of the model-averaged causal repair: {out['item_difficulty']['mean_of_item_means']:.1f}; "
              f"items with model-averaged repair < 20%: {100 * out['item_difficulty']['share_hard_lt20']:.0f}%, "
              f"> 80%: {100 * out['item_difficulty']['share_easy_gt80']:.0f}%.\n")
    md.append("| Failure category (checker violations) | Items | Model-averaged causal repair |\n|---|---|---|")
    for k, v in sorted(out["failure_categories"].items(), key=lambda kv: -kv[1]["n"]):
        md.append(f"| {k} | {v['n']} | {v['mean_repair']:+.1f} |")
    traps = collections.defaultdict(list)
    for it in items:
        if it["item_id"] in item_mean:
            for tr in it["task"]["traps"]:
                traps[tr].append(item_mean[it["item_id"]])
    out["traps"] = {k: {"n": len(v), "mean_repair": 100 * float(np.mean(v))} for k, v in traps.items()}

    # level 6 (multimodal) vs levels 4-5, per model
    lvl = {}
    for m in core:
        d6 = [eff[m][it["item_id"]] for it in items if it["task"]["difficulty"] == 6 and eff[m][it["item_id"]] is not None]
        d45 = [eff[m][it["item_id"]] for it in items if it["task"]["difficulty"] < 6 and eff[m][it["item_id"]] is not None]
        if d6 and d45:
            lvl[m] = {"d6": dict(zip(("mean", "lo", "hi"), [100 * x for x in boot_mean(d6)]), n=len(d6)),
                      "d45": dict(zip(("mean", "lo", "hi"), [100 * x for x in boot_mean(d45)]), n=len(d45)),
                      "vision": bool((D.get(f"{m}|{items[0]['item_id']}") or {}).get("vision", True))}
    d6_items = [it for it in items if it["task"]["difficulty"] == 6]
    never_opened = [it for it in d6_items if not any(s.get("image") for s in it["traj"]["steps"])]
    pooled6 = [v for m in core for v in [eff[m][it["item_id"]] for it in d6_items] if v is not None]
    pooled45 = [v for m in core for v in [eff[m][it["item_id"]] for it in items if it["task"]["difficulty"] < 6] if v is not None]
    out["level6"] = {"per_model": lvl, "n_d6_items": len(d6_items), "n_never_opened": len(never_opened),
                     "pooled_d6": 100 * float(np.mean(pooled6)), "pooled_d45": 100 * float(np.mean(pooled45)),
                     "never_opened_repair": 100 * float(np.mean([item_mean[it["item_id"]] for it in never_opened if it["item_id"] in item_mean])),
                     "opened_repair": 100 * float(np.mean([item_mean[it["item_id"]] for it in d6_items
                                                           if it not in never_opened and it["item_id"] in item_mean]))}
    md.append(f"\n## Level 6 (images)\n\n{len(d6_items)} level-6 items; the failing agent never opened an attachment in "
              f"{len(never_opened)} of them (model-averaged repair {out['level6']['never_opened_repair']:+.1f} vs "
              f"{out['level6']['opened_repair']:+.1f} when it did). Pooled causal repair: d6 {out['level6']['pooled_d6']:+.1f} "
              f"vs d4–5 {out['level6']['pooled_d45']:+.1f}.\n")
    md.append("| Model | d4–5 | d6 | Sees images |\n|---|---|---|---|")
    for m, v in sorted(lvl.items(), key=lambda kv: -kv[1]["d6"]["mean"]):
        md.append(f"| {m} | {v['d45']['mean']:+.1f} | {v['d6']['mean']:+.1f} | {'yes' if v['vision'] else 'no'} |")

    # reasoning tokens vs repair (model level)
    lb = json.loads((ROOT / "results" / "leaderboard.json").read_text(encoding="utf-8"))
    rows = [r for r in lb["table"] if r["model"] in core and r.get("uplift") is not None and r.get("mean_reasoning_tokens")]
    if len(rows) >= 5:
        rho, p = sps.spearmanr([r["mean_reasoning_tokens"] for r in rows], [r["uplift"] for r in rows])
        out["reasoning_vs_repair"] = {"n_models": len(rows), "spearman_rho": float(rho), "p": float(p)}
        md.append(f"\nSpearman correlation between mean reasoning tokens and causal repair across {len(rows)} models: "
                  f"ρ = {rho:.2f} (p = {p:.2f}).")

    # ---------------------------------------------------------------- ablations
    md.append(f"\n## Ablations (paired on the {len(items)} items)\n")
    abl = {}
    for name, a_, b_ in (("last-step baseline vs free choice (GPT-6.1 Sol max)", "openai/gpt-6.1-sol@max#last", "openai/gpt-6.1-sol@max"),
                         ("GPT-6.1 Sol: medium vs max effort", "openai/gpt-6.1-sol@medium", "openai/gpt-6.1-sol@max"),
                         ("Claude Haiku 5.5: max vs high effort", "anthropic/claude-haiku-5.5@max", "anthropic/claude-haiku-5.5@high")):
        if a_ in eff and b_ in eff:
            xa, xb, _ = common(eff, a_, b_)
            if len(xa) >= 5:
                abl[name] = {**paired(xa, xb), "mean_a": 100 * float(np.mean(xa)), "mean_b": 100 * float(np.mean(xb)),
                             "a": a_, "b": b_}
                t = abl[name]
                md.append(f"- {name}: {t['mean_a']:+.1f} vs {t['mean_b']:+.1f}, Δ = {t['mean_diff']:+.1f} "
                          f"[{t['ci95'][0]:+.1f}, {t['ci95'][1]:+.1f}], p = {t['p']:.3f}, n = {t['n']}")
    # a second sample of the same diagnoser (test-retest) and a reworded prompt, both for GPT-6 Luna
    for name, a_, b_ in (("GPT-6 Luna: second sample vs first sample (test-retest)", "openai/gpt-6-luna@max#rep2", "openai/gpt-6-luna@max"),
                         ("GPT-6 Luna: reworded prompt vs original prompt", "openai/gpt-6-luna@max#alt", "openai/gpt-6-luna@max")):
        if a_ in eff and b_ in eff:
            xa, xb, ids = common(eff, a_, b_)
            if len(xa) >= 5:
                same = [D[f"{a_}|{i}"].get("t") == D[f"{b_}|{i}"].get("t") for i in ids
                        if D[f"{a_}|{i}"].get("valid") and D[f"{b_}|{i}"].get("valid")]
                abl[name] = {**paired(xa, xb), "mean_a": 100 * float(np.mean(xa)), "mean_b": 100 * float(np.mean(xb)),
                             "a": a_, "b": b_, "same_step_share": float(np.mean(same)) if same else None,
                             "item_corr": float(np.corrcoef(xa, xb)[0, 1]) if len(xa) > 2 else None,
                             "mean_abs_item_diff": 100 * float(np.mean(np.abs(np.asarray(xa) - np.asarray(xb))))}
                t = abl[name]
                md.append(f"- {name}: {t['mean_a']:+.1f} vs {t['mean_b']:+.1f}, Δ = {t['mean_diff']:+.1f} "
                          f"[{t['ci95'][0]:+.1f}, {t['ci95'][1]:+.1f}], p = {t['p']:.3f}, n = {t['n']}; same step chosen in "
                          f"{100 * (t['same_step_share'] or 0):.0f}% of items; item-level correlation {t['item_corr']:.2f}")
    for m in ("openai/gpt-6.1-sol@medium", "anthropic/claude-haiku-5.5@max", "openai/gpt-6.1-sol@max", "anthropic/claude-haiku-5.5@high",
              "openai/gpt-6-luna@max", "openai/gpt-6-luna@max#rep2", "openai/gpt-6-luna@max#alt"):
        toks = [((D[k].get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
                for k in D if k.startswith(m + "|") and D[k]["item_id"] in by_id]
        if toks:
            abl.setdefault("reasoning_tokens", {})[m] = float(np.mean(toks))
    out["ablations"] = abl

    # ---------------------------------------------------------------- reliability of the ranking
    ids = [it["item_id"] for it in items]
    M = np.array([[eff[m][i] if eff[m][i] is not None else np.nan for i in ids] for m in core])
    means = np.nanmean(M, 1)
    g = rng()
    rhos = []
    for _ in range(2000):
        perm = g.permutation(len(ids))
        a_half, b_half = perm[: len(ids) // 2], perm[len(ids) // 2:]
        rhos.append(sps.spearmanr(np.nanmean(M[:, a_half], 1), np.nanmean(M[:, b_half], 1)).correlation)
    rel = {"split_half_spearman": float(np.mean(rhos)), "split_half_p5": float(np.percentile(rhos, 5)),
           "split_half_p95": float(np.percentile(rhos, 95)),
           "spearman_brown": float(np.mean([2 * r / (1 + r) for r in rhos]))}

    def eff_k(m, it, idx_fix, idx_pl):
        d = D.get(f"{m}|{it['item_id']}")
        if not d:
            return np.nan
        if not d.get("valid"):
            return 0.0
        r = R.get(rollout_key(it["item_id"], d["t"], d["tool"], d["args"]))
        pl = placebo_of(it, d["t"])
        rp = R.get(rollout_key(it["item_id"], d["t"], *pl)) if pl else None
        if not r or not rp or len(r["outcomes"]) < K or len(rp["outcomes"]) < K:
            return np.nan
        return float(np.mean([r["outcomes"][i] for i in idx_fix]) - np.mean([rp["outcomes"][i] for i in idx_pl]))

    rel["k_subsampling"] = {}
    for kk in range(1, K):
        rs, dev = [], []
        for _ in range(200):
            Mk = np.array([[eff_k(m, it, g.choice(K, kk, replace=False), g.choice(K, kk, replace=False)) for it in items]
                           for m in core])
            mk = np.nanmean(Mk, 1)
            rs.append(sps.spearmanr(mk, means).correlation)
            dev.append(100 * float(np.mean(np.abs(mk - means))))
        rel["k_subsampling"][kk] = {"spearman_with_full": float(np.mean(rs)), "spearman_p5": float(np.percentile(rs, 5)),
                                    "mean_abs_score_change": float(np.mean(dev))}
    top6 = [core[j] for j in np.argsort(-means)[:6]]
    sds = []
    for a_, b_ in itertools.combinations(top6, 2):
        d = M[core.index(a_)] - M[core.index(b_)]
        sds.append(100 * float(np.std(d[~np.isnan(d)], ddof=1)))
    sd = float(np.mean(sds))
    rel["paired_sd_top6"] = sd
    rel["items_for_delta"] = {delta: float(((1.96 + 0.84) * sd / delta) ** 2) for delta in (5, 10, 15)}
    out["reliability"] = rel
    md.append("\n## Reliability of the ranking\n")
    md.append(f"Split-half Spearman of the {len(core)}-model ranking: {rel['split_half_spearman']:.2f} "
              f"(5–95%: {rel['split_half_p5']:.2f}..{rel['split_half_p95']:.2f}); Spearman–Brown {rel['spearman_brown']:.2f}.")
    for kk, v in rel["k_subsampling"].items():
        md.append(f"- k'={kk}: Spearman with the k={K} ranking {v['spearman_with_full']:.2f} (5%: {v['spearman_p5']:.2f}); "
                  f"mean |score change| {v['mean_abs_score_change']:.1f} points")
    md.append(f"Mean SD of paired item differences among the top six: {sd:.1f} points; items for 80% power at Δ=5/10/15: "
              + ", ".join(f"{v:.0f}" for v in rel["items_for_delta"].values()))

    # ---------------------------------------------------------------- placebo, fix and repair by position of the chosen step
    pl_rows, fix_rows = {}, []
    for m in core:
        for it in items:
            d = D.get(f"{m}|{it['item_id']}")
            if not d or not d.get("valid"):
                continue
            relpos = d["t"] / max(1, len(it["traj"]["steps"]) - 1)
            pl = placebo_of(it, d["t"])
            rp = R.get(rollout_key(it["item_id"], d["t"], *pl)) if pl else None
            r = R.get(rollout_key(it["item_id"], d["t"], d["tool"], d["args"]))
            if rp and len(rp["outcomes"]) >= K:
                pl_rows[(it["item_id"], d["t"])] = (relpos, float(np.mean(rp["outcomes"][:K])))
                if r and len(r["outcomes"]) >= K:
                    fix_rows.append((relpos, float(np.mean(r["outcomes"][:K])), float(np.mean(rp["outcomes"][:K]))))
    rel_p = np.array([v[0] for v in pl_rows.values()])
    suc_p = np.array([v[1] for v in pl_rows.values()])
    fr = np.array(fix_rows)
    rho_p, p_p = sps.spearmanr(rel_p, suc_p)
    pos_out = {"n_distinct_placebos": int(len(rel_p)), "spearman_relstep_placebo": float(rho_p), "p": float(p_p), "bins": []}
    for lo, hi in POSITION_BINS:
        if lo >= 1.0:  # the final step itself
            sel_p, sel_f = rel_p >= 1.0, fr[:, 0] >= 1.0
        else:
            sel_p, sel_f = (rel_p >= lo) & (rel_p < hi), (fr[:, 0] >= lo) & (fr[:, 0] < hi)
        pos_out["bins"].append({"lo": lo, "hi": hi, "n_placebos": int(sel_p.sum()), "placebo": 100 * float(suc_p[sel_p].mean()) if sel_p.sum() else None,
                                "n_diagnoses": int(sel_f.sum()), "fix": 100 * float(fr[sel_f, 1].mean()) if sel_f.sum() else None,
                                "causal": 100 * float((fr[sel_f, 1] - fr[sel_f, 2]).mean()) if sel_f.sum() else None})
    last_pl = []
    for it in items:
        d = D.get(f"openai/gpt-6.1-sol@max#last|{it['item_id']}")
        if d and d.get("valid"):
            pl = placebo_of(it, d["t"])
            rp = R.get(rollout_key(it["item_id"], d["t"], *pl)) if pl else None
            if rp and len(rp["outcomes"]) >= K:
                last_pl.append(float(np.mean(rp["outcomes"][:K])))
    pos_out["placebo_at_forced_last_step"] = 100 * float(np.mean(last_pl)) if last_pl else None
    pos_out["n_forced_last"] = len(last_pl)
    out["position"] = pos_out
    md.append("\n## Placebo, fix and causal repair by relative position of the chosen step (core models pooled)\n")
    md.append(f"{len(rel_p)} distinct (item, step) placebos; Spearman(relative step, placebo success) = {rho_p:.2f} (p = {p_p:.3f}); "
              f"placebo at the forced last step: {pos_out['placebo_at_forced_last_step']:.1f}% (n = {len(last_pl)}).\n")
    md.append("| Relative step | Placebos | Placebo success | Diagnoses | Fix | Causal |\n|---|---|---|---|---|---|")
    for b in pos_out["bins"]:
        md.append(f"| [{b['lo']:.2f}, {b['hi']:.2f}{')' if b['hi'] < 1 else ']'} | {b['n_placebos']} | {b['placebo']:.1f}% | "
                  f"{b['n_diagnoses']} | {b['fix']:.1f}% | {b['causal']:+.1f} |")

    # ---------------------------------------------------------------- scoring by a consensus label instead of by replay
    agree_m, last_modal = {}, 0
    for m in core:
        hits, n = 0, 0
        for it in items:
            d = D.get(f"{m}|{it['item_id']}")
            others = [D[f"{o}|{it['item_id']}"]["t"] for o in core if o != m and (D.get(f"{o}|{it['item_id']}") or {}).get("valid")]
            if not others:
                continue
            modal = collections.Counter(others).most_common(1)[0][0]
            hits += bool(d and d.get("valid") and d["t"] == modal)  # an invalid answer agrees with nothing
            n += 1
        agree_m[m] = hits / n
    for it in items:
        ts = [D[f"{o}|{it['item_id']}"]["t"] for o in core if (D.get(f"{o}|{it['item_id']}") or {}).get("valid")]
        last_modal += collections.Counter(ts).most_common(1)[0][0] == len(it["traj"]["steps"]) - 1
    lab = {"agreement": agree_m, "spearman_with_causal": float(sps.spearmanr([agree_m[m] for m in core], means).correlation),
           "label_rank": {m: 1 + int(sum(agree_m[o] > agree_m[m] for o in core)) for m in core},
           "causal_rank": {m: 1 + int(sum(means[core.index(o)] > means[core.index(m)] for o in core)) for m in core},
           "consensus_is_last_step": int(last_modal), "n_items": len(items)}
    out["label_vs_causal"] = lab
    md.append("\n## Scoring by agreement with a consensus label (leave-one-out modal step) vs causal repair\n")
    md.append("| Model | Causal repair | Rank | Consensus agreement | Rank |\n|---|---|---|---|---|")
    for m in sorted(core, key=lambda m: -means[core.index(m)]):
        md.append(f"| {m} | {100 * means[core.index(m)]:+.1f} | {lab['causal_rank'][m]} | {100 * agree_m[m]:.0f}% | {lab['label_rank'][m]} |")
    md.append(f"Spearman between the two rankings: {lab['spearman_with_causal']:.2f}; the consensus step is the last step in "
              f"{last_modal}/{len(items)} items.")

    # ---------------------------------------------------------------- oversight
    ov = {}
    for m in SUPERVISORS:
        ov[m] = {}
        for r in runs:
            rec = S.get(f"{m}|{r['item_id']}")
            e = oversight.effect(rec, r, R, K) if rec else None
            if e is not None:
                ov[m][r["item_id"]] = e - PENALTY * (rec["decision"] == "intervene")
    ov_summary = {m: dict(zip(("net", "lo", "hi"), [100 * x for x in boot_mean(list(v.values()))]), n=len(v))
                  for m, v in ov.items() if v}
    out["oversight_summary"] = ov_summary
    # sensitivity of the net score to the interruption cost
    lam_out = {}
    for lam in (0.0, 0.05, 0.10, 0.20):
        nets = {}
        for m in SUPERVISORS:
            xs = []
            for r in runs:
                rec = S.get(f"{m}|{r['item_id']}")
                e = oversight.effect(rec, r, R, K) if rec else None
                if e is not None:
                    xs.append(e - lam * (rec["decision"] == "intervene"))
            if xs:
                nets[m] = 100 * float(np.mean(xs))
        lam_out[str(int(100 * lam))] = {"net": nets, "order": sorted(nets, key=lambda m: -nets[m])}
    out["oversight_lambda"] = lam_out
    # does the worker's identity leak the outcome? false alarms by worker of the succeeding run, and scores without Qwen runs
    QWEN = "qwen/qwen3.8-flash"
    wk = {}
    for m in SUPERVISORS:
        fa_q = n_q = fa_o = n_o = n_fail = 0
        net_all, net_noq = [], []
        for r in runs_all:
            rec = S.get(f"{m}|{r['item_id']}")
            e = oversight.effect(rec, r, R, K) if rec else None
            if e is None:
                continue
            net = e - PENALTY * (rec["decision"] == "intervene")
            net_all.append(net)
            qwen = r["worker"]["model"] == QWEN
            if r["label"] == "success":
                if qwen:
                    n_q += 1
                    fa_q += rec["decision"] == "intervene"
                else:
                    n_o += 1
                    fa_o += rec["decision"] == "intervene"
            else:
                n_fail += 1
            if not qwen:
                net_noq.append(net)
        wk[m] = {"fa_qwen": fa_q, "n_qwen": n_q, "fa_other": fa_o, "n_other": n_o, "n_fail": n_fail,
                 "net_all": 100 * float(np.mean(net_all)), "n_all": len(net_all),
                 "net_no_qwen": 100 * float(np.mean(net_noq)), "n_no_qwen": len(net_noq)}
    out["oversight_workers"] = wk
    md.append("\n## Worker identity: false alarms by worker of the succeeding run; net score without Qwen runs\n")
    md.append("| Supervisor | FA on Qwen successes | FA on other successes | Net, all its runs | Net, non-Qwen runs |\n|---|---|---|---|---|")
    for m, v in sorted(wk.items(), key=lambda kv: -kv[1]["net_no_qwen"]):
        md.append(f"| {m} | {v['fa_qwen']}/{v['n_qwen']} | {v['fa_other']}/{v['n_other']} | {v['net_all']:+.1f} (n={v['n_all']}) | "
                  f"{v['net_no_qwen']:+.1f} (n={v['n_no_qwen']}) |")
    fam_o = pairwise_family(ov, [m for m in SUPERVISORS if m in ov_summary], "oversight pairwise (5 supervisors)")
    out["oversight_pairwise"] = fam_o
    md.append("\n## Oversight: net damage prevented (with interruption cost), bootstrap over runs\n")
    md.append("| Supervisor | n | Net | 95% CI |\n|---|---|---|---|")
    for m, s in sorted(ov_summary.items(), key=lambda kv: -kv[1]["net"]):
        md.append(f"| {m} | {s['n']} | {s['net']:+.1f} | {s['lo']:+.1f} … {s['hi']:+.1f} |")
    md.append(f"\nPairs significant after Holm ({fam_o['n_pairs']} pairs): {len(fam_o['significant_holm'])}")
    for k in fam_o["significant_holm"]:
        t = fam_o["tests"][k]
        md.append(f"- {k}: Δ = {t['mean_diff']:+.1f} [{t['ci95'][0]:+.1f}, {t['ci95'][1]:+.1f}], p_holm = {t['p_holm']:.3f}")
    # false alarms and timing
    timing = {}
    for m in SUPERVISORS:
        fa, nfa, cp, earlier, n_both, reads, n_int, first, fa_reads = 0, 0, [], 0, 0, 0, 0, 0, 0
        for r in runs:
            rec = S.get(f"{m}|{r['item_id']}")
            if not rec:
                continue
            if r["label"] == "success":
                nfa += 1
                fa += rec["decision"] == "intervene"
                fa_reads += rec["decision"] == "intervene" and rec["tool"] in READ_TOOLS
            elif rec["decision"] == "intervene":
                n_int += 1
                reads += rec["tool"] in READ_TOOLS
                first += rec["checkpoint"] == 0
                cp.append(rec["checkpoint"] / max(1, rec["n_checkpoints"] - 1) if rec["n_checkpoints"] > 1 else 0.0)
                d = D.get(f"{m}|{r['item_id']}")
                if d and d.get("valid"):
                    n_both += 1
                    earlier += rec["t"] < d["t"]
        timing[m] = {"false_alarm_rate": fa / nfa if nfa else None, "mean_relative_checkpoint": float(np.mean(cp)) if cp else None,
                     "intervened_earlier_than_autopsy": earlier / n_both if n_both else None, "n_both": n_both,
                     "read_share": reads / n_int if n_int else None, "first_checkpoint_share": first / n_int if n_int else None,
                     "false_alarm_read_share": fa_reads / fa if fa else None}
    out["oversight_timing"] = timing

    # autopsy vs oversight on the same failing runs
    md.append("\n## Autopsy vs live oversight on the same failing runs (paired by run)\n")
    md.append("| Model | n | Autopsy | Oversight (no penalty) | Δ (oversight − autopsy) | 95% CI | p |\n|---|---|---|---|---|---|---|")
    avo = {}
    for m in SUPERVISORS:
        xa, xo = [], []
        for r in runs:
            if r["label"] != "fail" or r["item_id"] not in by_id:
                continue
            rec, d = S.get(f"{m}|{r['item_id']}"), D.get(f"{m}|{r['item_id']}")
            if not rec or not d:
                continue
            eo = oversight.effect(rec, r, R, K)
            ea = causal(by_id[r["item_id"]], d, R)
            if eo is None or ea is None:
                continue
            xa.append(ea)
            xo.append(eo)
        if len(xa) >= 5:
            avo[m] = {**paired(xo, xa), "autopsy": 100 * float(np.mean(xa)), "oversight": 100 * float(np.mean(xo))}
            t = avo[m]
            md.append(f"| {m} | {t['n']} | {t['autopsy']:+.1f} | {t['oversight']:+.1f} | {t['mean_diff']:+.1f} | "
                      f"{t['ci95'][0]:+.1f} … {t['ci95'][1]:+.1f} | {t['p']:.3f} |")
    out["autopsy_vs_oversight"] = avo

    # ---------------------------------------------------------------- self-renewal pilot
    ren_path = ROOT / a.renewal_items
    if ren_path.exists() and a.renewal_models:
        ren_items = load_items(ren_path)
        models = [m for m in a.renewal_models.split(",") if m]
        eff_r = effects_matrix(ren_items, D, R, models)
        strong = [float(np.mean([eff_r[m][i["item_id"]] for m in models if eff_r[m][i["item_id"]] is not None]))
                  for i in ren_items if any(eff_r[m][i["item_id"]] is not None for m in models)]
        weak_items = [i for i in items if i["worker"]["model"] == "deepseek/deepseek-v4.1-flash-1017"
                      and i["task"]["template"] == "late_delivery"]
        weak = [float(np.mean([eff[m][i["item_id"]] for m in models if eff[m].get(i["item_id"]) is not None]))
                for i in weak_items if any(eff[m].get(i["item_id"]) is not None for m in models)]
        per_model = {}
        for m in models:
            s_ = [v for v in eff_r[m].values() if v is not None]
            w_ = [eff[m][i["item_id"]] for i in weak_items if eff[m].get(i["item_id"]) is not None]
            if s_ and w_:
                per_model[m] = unpaired(s_, w_)
        out["self_renewal"] = {"models": models, "n_strong_items": len(strong), "n_weak_items": len(weak),
                               "strong_worker": "qwen/qwen3.8-flash", "weak_worker": "deepseek/deepseek-v4.1-flash-1017",
                               "pooled": unpaired(strong, weak), "per_model": per_model,
                               "p0_strong": float(np.mean([i.get("p0", 0) for i in ren_items])),
                               "p0_weak": float(np.mean([i.get("p0", 0) for i in weak_items if "p0" in i])) if any("p0" in i for i in weak_items) else None}
        t = out["self_renewal"]["pooled"]
        md.append("\n## Self-renewal pilot: failures of a stronger worker vs a weaker worker (late_delivery)\n")
        md.append(f"Model-averaged causal repair: stronger worker items {t['mean_a']:+.1f} (n={t['n_a']}) vs weaker worker items "
                  f"{t['mean_b']:+.1f} (n={t['n_b']}); Δ = {t['mean_diff']:+.1f} [{t['ci95'][0]:+.1f}, {t['ci95'][1]:+.1f}], "
                  f"Welch p = {t['p_welch']:.3f}.")
        for m, v in per_model.items():
            md.append(f"- {m}: {v['mean_a']:+.1f} vs {v['mean_b']:+.1f}, Δ = {v['mean_diff']:+.1f} "
                      f"[{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}], p = {v['p_welch']:.3f}")

    # ---------------------------------------------------------------- human reference (the author's diagnoses on a subset)
    if HUMAN in eff:
        sub = [i for i in ids if eff[HUMAN][i] is not None]
        if len(sub) >= 5:
            hx = [eff[HUMAN][i] for i in sub]
            hmean, hlo, hhi = boot_mean(hx)
            per_model = {}
            for m in core:
                xm = [eff[m][i] for i in sub]
                if all(v is not None for v in xm):
                    per_model[m] = {**paired(xm, hx), "model_mean": 100 * float(np.mean(xm))}
            model_avg = [float(np.mean([eff[m][i] for m in core if eff[m][i] is not None])) for i in sub]
            out["human_reference"] = {"n": len(sub), "items": sub, "human": 100 * hmean, "ci95": [100 * hlo, 100 * hhi],
                                      "per_model": per_model, "model_average_on_subset": 100 * float(np.mean(model_avg)),
                                      "human_vs_model_average": paired(hx, model_avg),
                                      "n_valid": sum(1 for i in sub if (D.get(f"{HUMAN}|{i}") or {}).get("valid")),
                                      "minutes": [D[f"{HUMAN}|{i}"].get("minutes") for i in sub if D[f"{HUMAN}|{i}"].get("minutes")]}
            md.append(f"\n## Human reference (author) on {len(sub)} items\n")
            md.append(f"Human causal repair {100 * hmean:+.1f} [{100 * hlo:+.1f}, {100 * hhi:+.1f}]; model average on the same items "
                      f"{out['human_reference']['model_average_on_subset']:+.1f}.")
            for m, v in sorted(per_model.items(), key=lambda kv: -kv[1]["model_mean"]):
                md.append(f"- {m}: {v['model_mean']:+.1f} (Δ vs human {v['mean_diff']:+.1f}, p = {v['p']:.3f})")

    (ROOT / "results" / "stats.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (ROOT / "results" / "stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
