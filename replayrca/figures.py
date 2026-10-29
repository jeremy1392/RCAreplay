"""Paper figures from results/leaderboard.json and results/stats.json (no API calls).

python -m replayrca.figures   ->  paper/figures/*.pdf (+ .png previews)

Conventions: one hue for a single series, two validated categorical hues for paired
series, hollow markers as a second channel for text-only models, hairline grid, no
dual axes, values reported in a text column rather than on every mark.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .evaluate import BASELINE  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3, validated all-pairs on white
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
NAMES = {
    "x-ai/grok-4.7@xhigh": "Grok 4.7", "google/gemini-4-argon@high": "Gemini 4 Argon",
    "moonshotai/kimi-k3@max": "Kimi K3", "openai/gpt-6.1-sol@max": "GPT-6.1 Sol",
    "google/gemini-3.8-flash@high": "Gemini 3.8 Flash", "z-ai/glm-5.3@max": "GLM 5.3",
    "openai/gpt-6-astra@max": "GPT-6 Astra", "anthropic/claude-opus-5.5@high": "Claude Opus 5.5",
    "deepseek/deepseek-v4-pro-0813@max": "DeepSeek V4 Pro", "anthropic/claude-sonnet-5.5@high": "Claude Sonnet 5.5",
    "openai/gpt-6-luna@max": "GPT-6 Luna", "anthropic/claude-fable-5.1@high": "Claude Fable 5.1",
    "anthropic/claude-haiku-5.5@high": "Claude Haiku 5.5", "deepseek/deepseek-v4.1-flash-1017@max": "DeepSeek V4.1 Flash",
    "openai/gpt-6.1-sol@medium": "GPT-6.1 Sol (medium)", "openai/gpt-6.1-sol@max#last": "GPT-6.1 Sol (forced last step)",
    "anthropic/claude-haiku-5.5@max": "Claude Haiku 5.5 (max)", BASELINE: "Random-step replay (no change)",
}
TEXT_ONLY = {"z-ai/glm-5.3@max", "deepseek/deepseek-v4-pro-0813@max"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK, "axes.labelcolor": INK2, "text.color": INK, "axes.titlecolor": INK,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def name(m):
    return NAMES.get(m, m.split("/", 1)[-1])


def style(ax, xgrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3, color=AXIS)
    if xgrid:
        ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png")
    plt.close(fig)


def dot_ci(ax, y, x, lo, hi, color, hollow=False, ms=5.5):
    ax.plot([lo, hi], [y, y], color=color, linewidth=1.2, solid_capstyle="butt", zorder=2)
    ax.plot([x], [y], marker="o", ms=ms, color=color, markerfacecolor="white" if hollow else color,
            markeredgewidth=1.4, linestyle="none", zorder=3)


def fig_autopsy(lb, core):
    rows = sorted([r for r in lb["table"] if r["model"] in core and r["uplift"] is not None],
                  key=lambda r: r["uplift"], reverse=True)
    base = next((r for r in lb["table"] if r["model"] == BASELINE), None)
    if base:
        rows.append(base)
    fig, ax = plt.subplots(figsize=(6.5, 0.27 * len(rows) + 0.9))
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        is_base = r["model"] == BASELINE
        c = MUTED if is_base else BLUE
        dot_ci(ax, y, r["uplift"], r["uplift_ci95"][0], r["uplift_ci95"][1], c, hollow=r["model"] in TEXT_ONLY)
        ax.text(1.0, y, f"{r['uplift']:+.1f}  [{r['uplift_ci95'][0]:+.0f}, {r['uplift_ci95'][1]:+.0f}]",
                transform=ax.get_yaxis_transform(), ha="left", va="center", color=INK2, fontsize=7.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([name(r["model"]) for r in reversed(rows)])
    ax.axvline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax.set_xlim(-10, 70)
    ax.set_xlabel("Causal repair (success with the fix − success of the paired placebo), points")
    style(ax)
    ax.legend(handles=[Line2D([], [], marker="o", color=BLUE, linestyle="none", ms=5.5, label="sees images"),
                       Line2D([], [], marker="o", color=BLUE, markerfacecolor="white", markeredgewidth=1.4,
                              linestyle="none", ms=5.5, label="text-only (images hidden)")],
              loc="upper left", handletextpad=0.4)
    save(fig, "fig_autopsy")


# rank-number placement for the reasoning-token scatter: (dx, dy) in points, tuned so that numbers in the dense
# cluster do not touch
NUM_OFFSET = {"deepseek/deepseek-v4-pro-0813@max": (-11, -11)}


def fig_tokens(lb, core):
    """Causal repair against mean reasoning tokens per diagnosis. Points carry their rank (same order as the
    leaderboard); the names sit in a column on the right."""
    rows = sorted([r for r in lb["table"] if r["model"] in core and r["uplift"] is not None and r["n"]],
                  key=lambda r: -r["uplift"])
    fig = plt.figure(figsize=(6.5, 3.6))
    ax = fig.add_axes([0.085, 0.145, 0.60, 0.82])
    for i, r in enumerate(rows, 1):
        x, y = r["mean_reasoning_tokens"], r["uplift"]
        ax.plot([x, x], [r["uplift_ci95"][0], r["uplift_ci95"][1]], color=BLUE, linewidth=0.8, alpha=0.5, zorder=2)
        ax.plot([x], [y], marker="o", ms=5.5, color=BLUE, markerfacecolor="white" if r["model"] in TEXT_ONLY else BLUE,
                markeredgewidth=1.4, linestyle="none", zorder=3)
        dx, dy = NUM_OFFSET.get(r["model"], (4, 4))
        ax.annotate(str(i), (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color=INK, fontweight="bold")
        fig.text(0.735, 0.95 - 0.062 * i, str(i), ha="right", va="center", fontsize=7.5, color=INK, fontweight="bold")
        fig.text(0.745, 0.95 - 0.062 * i, name(r["model"]) + (" (text-only)" if r["model"] in TEXT_ONLY else ""),
                 ha="left", va="center", fontsize=7.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("Mean reasoning tokens per diagnosis (log scale)")
    ax.set_ylabel("Causal repair, points")
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    save(fig, "fig_tokens")


def fig_autopsy_vs_oversight(st):
    avo = st["autopsy_vs_oversight"]
    rows = sorted(avo.items(), key=lambda kv: kv[1]["oversight"], reverse=True)
    fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(rows) + 1.0))
    for i, (m, v) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([v["autopsy"], v["oversight"]], [y, y], color=AXIS, linewidth=1.2, zorder=1)
        ax.plot([v["autopsy"]], [y], "o", ms=6, color=BLUE, zorder=3)
        ax.plot([v["oversight"]], [y], "o", ms=6, color=ORANGE, zorder=3)
        ax.text(1.0, y, f"Δ = {v['mean_diff']:+.1f}  [{v['ci95'][0]:+.0f}, {v['ci95'][1]:+.0f}],  p = {v['p']:.3f}",
                transform=ax.get_yaxis_transform(), ha="left", va="center", color=INK2, fontsize=7.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([name(m) for m, _ in reversed(rows)])
    ax.axvline(0, color=AXIS, linewidth=0.8, zorder=0)
    ax.set_xlabel("Causal effect on the same failing runs, points")
    style(ax)
    ax.legend(handles=[Line2D([], [], marker="o", color=BLUE, linestyle="none", ms=6, label="autopsy (told the run failed)"),
                       Line2D([], [], marker="o", color=ORANGE, linestyle="none", ms=6, label="live oversight (outcome unknown)")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handletextpad=0.4)
    save(fig, "fig_autopsy_vs_oversight")


def fig_oversight(st):
    s, t = st["oversight_summary"], st["oversight_timing"]
    rows = sorted(s.items(), key=lambda kv: kv[1]["net"], reverse=True)
    fig, ax = plt.subplots(figsize=(6.5, 0.4 * len(rows) + 1.0))
    for i, (m, v) in enumerate(rows):
        y = len(rows) - 1 - i
        dot_ci(ax, y, v["net"], v["lo"], v["hi"], BLUE)
        fa = t.get(m, {}).get("false_alarm_rate")
        ax.text(1.0, y, f"{v['net']:+.1f}  [{v['lo']:+.0f}, {v['hi']:+.0f}]" + (f"   false alarms {100 * fa:.0f}%" if fa is not None else ""),
                transform=ax.get_yaxis_transform(), ha="left", va="center", color=INK2, fontsize=7.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([name(m) for m, _ in reversed(rows)])
    ax.axvline(0, color=AXIS, linewidth=0.8, zorder=1)
    n_runs = next(iter(s.values()))["n"]
    ax.set_xlabel(f"Net damage prevented with interruption cost (points over {n_runs} runs)")
    style(ax)
    save(fig, "fig_oversight")


def fig_level6(st):
    lv = st["level6"]["per_model"]
    rows = sorted(lv.items(), key=lambda kv: kv[1]["d6"]["mean"], reverse=True)
    fig, ax = plt.subplots(figsize=(6.5, 0.27 * len(rows) + 1.0))
    for i, (m, v) in enumerate(rows):
        y = len(rows) - 1 - i
        a, b = v["d45"]["mean"], v["d6"]["mean"]
        ax.plot([a, b], [y, y], color=AXIS, linewidth=1.2, zorder=1)
        hollow = not v["vision"]
        ax.plot([a], [y], "o", ms=5.5, color=BLUE, markerfacecolor="white" if hollow else BLUE, markeredgewidth=1.4, zorder=3)
        ax.plot([b], [y], "o", ms=5.5, color=ORANGE, markerfacecolor="white" if hollow else ORANGE, markeredgewidth=1.4, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([name(m) for m, _ in reversed(rows)])
    ax.axvline(0, color=AXIS, linewidth=0.8, zorder=0)
    ax.set_xlabel("Causal repair, points")
    style(ax)
    ax.legend(handles=[Line2D([], [], marker="o", color=BLUE, linestyle="none", ms=5.5, label="levels 4–5 (text only)"),
                       Line2D([], [], marker="o", color=ORANGE, linestyle="none", ms=5.5, label="level 6 (QR codes, photos, handwriting)"),
                       Line2D([], [], marker="o", color=MUTED, markerfacecolor="white", markeredgewidth=1.4, linestyle="none",
                              ms=5.5, label="hollow: text-only diagnoser")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, handletextpad=0.4)
    save(fig, "fig_level6")


def fig_ablation(st):
    s = st["autopsy_summary"]
    groups = [("GPT-6.1 Sol", [("openai/gpt-6.1-sol@medium", "medium effort"), ("openai/gpt-6.1-sol@max", "max effort"),
                               ("openai/gpt-6.1-sol@max#last", "max effort, forced last step")]),
              ("Claude Haiku 5.5", [("anthropic/claude-haiku-5.5@high", "high effort"), ("anthropic/claude-haiku-5.5@max", "max effort")]),
              ("GPT-6 Luna", [("openai/gpt-6-luna@max", "max effort, first sample"), ("openai/gpt-6-luna@max#rep2", "max effort, second sample"),
                             ("openai/gpt-6-luna@max#alt", "max effort, reworded prompt")])]
    rows = [(g, m, lab) for g, ms in groups for m, lab in ms if m in s]
    n_items = s[rows[0][1]]["n"] if rows else 0
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.5, 0.42 * len(rows) + 1.0))
    for i, (g, m, lab) in enumerate(rows):
        y = len(rows) - 1 - i
        v = s[m]
        dot_ci(ax, y, v["causal_repair"], v["ci95"][0], v["ci95"][1], BLUE)
        ax.text(1.0, y, f"{v['causal_repair']:+.1f}  [{v['ci95'][0]:+.0f}, {v['ci95'][1]:+.0f}]",
                transform=ax.get_yaxis_transform(), ha="left", va="center", color=INK2, fontsize=7.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{g} — {lab}" for g, _, lab in reversed(rows)])
    ax.axvline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax.set_xlim(-10, 70)
    ax.set_xlabel(f"Causal repair, points ({n_items} items)")
    style(ax)
    save(fig, "fig_ablation")


BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # one hue, light -> dark


def fig_pairwise(lb, st, core):
    """Lower-triangular matrix of raw sign-flip p-values between models (rows/columns in leaderboard order);
    cells significant after Holm correction carry a thick border."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    order = [r["model"] for r in sorted((r for r in lb["table"] if r["model"] in core and r["uplift"] is not None),
                                         key=lambda r: -r["uplift"])]
    tests = st["autopsy_pairwise"]["tests"]
    n = len(order)
    P = np.full((n, n), np.nan)
    H = np.zeros((n, n), bool)
    for i in range(n):
        for j in range(i):
            t = tests.get(f"{order[i]} vs {order[j]}") or tests.get(f"{order[j]} vs {order[i]}")
            if t:
                P[i, j] = t["p"]
                H[i, j] = t["p_holm"] < 0.05
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    sig = -np.log10(np.where(np.isnan(P), 1.0, P))
    masked = np.ma.masked_where(np.isnan(P), sig)
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=3, aspect="equal")
    for i in range(n):
        for j in range(i):
            if not np.isnan(P[i, j]):
                dark = sig[i, j] > 1.5
                ax.text(j, i, f"{P[i, j]:.2f}" if P[i, j] >= 0.005 else "<.01", ha="center", va="center", fontsize=6.5,
                        color="white" if dark else INK)
                if H[i, j]:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.6))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([name(m) for m in order], rotation=45, ha="right", fontsize=7.5)
    ax.set_yticklabels([name(m) for m in order], fontsize=7.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Raw sign-flip p-value of the paired difference (darker = smaller); bold border = significant after Holm",
                 fontsize=8, color=INK2, loc="left")
    save(fig, "fig_pairwise")


def fig_items(st):
    """Distribution over items of the model-averaged causal repair (one hue, bars)."""
    import numpy as np
    vals = list(st["item_means"].values())
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    lo = min(-20, 10 * int(np.floor(min(vals) / 10)))
    bins = list(range(lo, 101, 10))
    counts, _ = np.histogram(vals, bins=bins)
    ax.bar([b + 5 for b in bins[:-1]], counts, width=8.4, color=BLUE, zorder=3)
    ax.set_xticks(bins)
    ax.axvline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax.set_xlabel("Model-averaged causal repair per item, points")
    ax.set_ylabel("Items")
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    save(fig, "fig_items")


def main():
    lb = json.loads((ROOT / "results" / "leaderboard.json").read_text(encoding="utf-8"))
    st = json.loads((ROOT / "results" / "stats.json").read_text(encoding="utf-8"))
    core = [m for m in NAMES if m != BASELINE and "#" not in m and "@medium" not in m
            and m != "anthropic/claude-haiku-5.5@max" and m != "deepseek/deepseek-v4.1-flash-1017@max"]
    fig_autopsy(lb, core)
    fig_tokens(lb, core)
    fig_autopsy_vs_oversight(st)
    fig_oversight(st)
    fig_level6(st)
    fig_ablation(st)
    fig_pairwise(lb, st, core)
    fig_items(st)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
