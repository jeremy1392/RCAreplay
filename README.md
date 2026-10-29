# ReplayRCA

**Can a model find the decisive mistake in another agent's failed run and fix it with a single action, after the fact or live?**

ReplayRCA scores such diagnoses by **counterfactual replay** instead of by agreement with annotations or by an LLM judge. The environment is restored to the exact state preceding the chosen step, the proposed action replaces the agent's own, and the same agent continues on its own. The result is compared with a **paired placebo**: the same replay with the agent's original action, so that success the agent would have had anyway is removed from the score.

Two tracks share this protocol:

| | **Autopsy** | **Live oversight** |
|---|---|---|
| What the evaluated model sees | The complete transcript of a run that failed, and the fact that it failed | The run as it unfolds, up to the irreversible action the agent is about to take; the outcome is unknown |
| What it must do | Name the decisive step and propose **one** replacement action | Before every irreversible action: let it pass, or intervene **once** |
| Score | Causal repair = success rate with the correction − success rate of the paired placebo | Net damage prevented, with an interruption cost per intervention, over a mix of runs that will fail and runs that will succeed |

This repository contains the environment, the harvesting and calibration pipeline, the two evaluation tracks, the OpenRouter client and the reporting scripts. The harvested items, the calibrated pools and the caches of diagnoses and continuations used for the companion paper (in preparation) are not part of this repository.

**Contents:** [Results](#results-october-2026) · [How the score is computed](#how-the-score-is-computed) · [The environment](#the-environment-opsworld) · [Pipeline](#pipeline) · [Protocol and guardrails](#protocol-and-guardrails) · [Installation and usage](#installation-and-usage) · [Repository layout](#repository-layout) · [Limitations](#limitations-of-the-design) · [Licence and citation](#licence-and-citation) · [Project page](https://jeremy1392.github.io/RCAreplay/)

## Results (October 2026)

Setup: 62 calibrated systematic failures (29 at levels 4–5, 33 at level 6) harvested from three small workers (DeepSeek V4.1 Flash 41, Claude Haiku 5.5 11, GPT-6 Luna 10); 13 diagnosers, each at its maximal reasoning effort within a 64 000-token output budget (Anthropic models at `high`, see below); `k = 5` continuations per intervention; live oversight on the 87 runs seen by all five supervisors (the 62 failures plus 25 calibrated successes as controls). Intervals are 95 % bootstrap intervals over items or runs; pairwise comparisons are paired sign-flip permutation tests with Holm correction. Scores are in points of continuation success.

### Autopsy track

| # | Model | Effort | Causal repair | 95 % CI | Reasoning tokens / diagnosis |
|---|---|---|---|---|---|
| 1 | Grok 4.7 | xhigh | **+51.3** | +40 … +63 | 38,820 |
| 2 | Kimi K3 | max | +48.4 | +38 … +59 | 12,099 |
| 3 | Gemini 4 Argon | high | +48.4 | +37 … +60 | 19,125 |
| 4 | Gemini 3.8 Flash | high | +47.1 | +35 … +59 | 41,704 |
| 5 | GPT-6.1 Sol | max | +45.2 | +35 … +55 | 3,385 |
| 6 | GLM 5.3 (text-only) | max | +45.2 | +36 … +55 | 26,772 |
| 7 | Claude Opus 5.5 | high | +42.9 | +33 … +53 | 1,428 |
| 8 | GPT-6 Astra | max | +42.9 | +32 … +54 | 4,163 |
| 9 | DeepSeek V4 Pro (text-only) | max | +42.3 | +31 … +53 | 25,370 |
| 10 | Claude Sonnet 5.5 | high | +38.4 | +28 … +48 | 1,116 |
| 11 | GPT-6 Luna | max | +37.1 | +26 … +47 | 13,214 |
| 12 | Claude Fable 5.1 | high | +36.1 | +26 … +47 | 2,591 |
| 13 | Claude Haiku 5.5 | high | +32.3 | +22 … +43 | 3,921 |
| — | Placebo baseline: the original action replayed at a random step | — | −0.6 | −2 … +0 | — |

Causal repair = success rate of the worker's continuation after the correction minus that of the paired placebo (same step, original action). Text-only models receive a placeholder in place of level-6 attachments.

- **The top of the ranking is not resolved.** None of the 78 pairwise comparisons is significant after Holm correction, which neither establishes a hierarchy nor shows equivalence (the intervals are about ±10 points wide); against Grok 4.7, the first model that is significantly worse even without correction is DeepSeek V4 Pro (rank 9). The split-half reliability of the ranking is ρ = 0.47 (0.62 after Spearman–Brown), and separating two models 10 points apart with 80 % power needs about 84 items.
- **Flagship models do not lead.** GPT-6 Astra ranks 8th and Claude Fable 5.1 12th. The Claude models ran at `high`, and an ablation shows that `max` is worth +15.2 points to Claude Haiku 5.5 (p = 0.011), so the four Claude ranks are lower bounds. Reasoning volume is only loosely related to rank (ρ = 0.54, p = 0.056 across the 13 models).
- **Hard items exist.** 5 items are repaired by no model in any continuation; 35 % of items have a model-averaged repair below 20 points and 19 % above 80.
- **The last-step shortcut does not transfer.** GPT-6.1 Sol forced to choose the final step scores +28.1 against +45.2 with free choice (−17.1 points, p = 0.004). Scoring by agreement with a consensus step instead of by replay would rank GPT-6 Astra first and Grok 4.7 third (ρ = 0.43 between the two scores); the consensus step is the last step in only 21 % of items.
- **Scores are reproducible, individual diagnoses less so.** A second sample of GPT-6 Luna scores +33.5 against +37.1 (p = 0.53) and chooses the same step in 74 % of items (item-level correlation 0.58); a reworded prompt gives +30.6 (p = 0.16), the same step in 71 %. GPT-6.1 Sol at `medium` effort scores +40.6 against +45.2 at `max` (p = 0.28).
- **Level 6 (images).** In 17 of the 33 level-6 failures the worker never opened the attachment. Most vision models lose points at level 6 (Grok 4.7 +57.9 → +45.5, GPT-6 Astra +51.0 → +35.8, Claude Opus 5.5 +49.7 → +37.0), while two text-only models stay competitive (DeepSeek V4 Pro +42.4, GLM 5.3 +39.4): making the worker open the image does not require seeing it.

### Live oversight track

87 runs: 62 heading for failure, 25 for success.

| # | Supervisor | Net damage prevented | 95 % CI | False alarms | Output tokens / call |
|---|---|---|---|---|---|
| 1 | Gemini 3.8 Flash | **+29.1** | +21 … +38 | 2/25 | 5,906 |
| 2 | DeepSeek V4.1 Flash (open weights) | **+27.5** | +19 … +36 | 3/25 | 8,660 |
| 3 | Claude Haiku 5.5 | +21.7 | +14 … +30 | 1/25 | 1,732 |
| 4 | GPT-6 Luna | +19.7 | +12 … +28 | 8/25 | 1,766 |
| 5 | GPT-6.1 Sol | +18.1 | +11 … +26 | 3/25 | 1,008 |

Net = mean over runs of (success with the intervention − success of the paired placebo) minus 5 points per intervention. The two leaders are not statistically separable (which does not establish equivalence) and stay in front for every interruption cost from 0 to 20 points; 3 of the 10 pairs are significant after Holm correction (Gemini 3.8 Flash above GPT-6.1 Sol and GPT-6 Luna, DeepSeek V4.1 Flash above GPT-6.1 Sol).

**The worker's identity does not leak the outcome.** Every failing run comes from the three weaker workers and 18 of the 25 control successes from Qwen 3.8 Flash, so a supervisor that recognised a worker's style could score well without reading the run. Over the 59 (supervisor, run) pairs that involve a success of a weaker worker (6 extra successes were added for this check, seen by every supervisor but Gemini 3.8 Flash), the supervisors raise 5 false alarms (8 %), against 16 over the 90 pairs that involve a Qwen success (18 %); removing every Qwen run leaves the ranking unchanged.

### Explaining versus preventing

Same 62 failing runs, same models; the live effect is reported without the interruption cost.

| Model | Autopsy | Live oversight | Δ | 95 % CI | p |
|---|---|---|---|---|---|
| GPT-6.1 Sol | +45.2 | +30.6 | **−14.5** | −23 … −5 | 0.002 |
| GPT-6 Luna | +37.1 | +32.9 | −4.2 | −14 … +6 | 0.45 |
| Gemini 3.8 Flash | +47.1 | +45.5 | −1.6 | −10 … +7 | 0.77 |
| Claude Haiku 5.5 | +32.3 | +35.5 | +3.2 | −8 … +14 | 0.61 |

GPT-6.1 Sol is among the best post-hoc diagnosers and the weakest live supervisor. Its interventions come early (16 % at the first irreversible action, mean relative position 0.31 on a 0–1 scale) and 81 % of them make the worker read something rather than redo the pending action. The gap is the only significant one of the four (p = 0.010 after Holm correction); its size and whether it is specific to this model remain to be established on more runs and models.

### Compute

6,258 worker continuations in total. A transcript shown to a diagnoser averages 14,347 tokens and 66 steps; a supervised run has 17 checkpoints on average, each a separate call to the supervisor.

## How the score is computed

- A **diagnosis** is a step index and a replacement tool call. The replay (`replayrca/replay.py`) restores the world at that step, executes the replacement call, and lets the original worker agent continue from there. Each continuation is run `k` times (default 5), and the success rate is measured by the task checker.
- The **placebo** replays the worker's own original action at the same step through the same mechanism. Causal repair is the difference between the two success rates, so a diagnosis that merely restarts the agent earns nothing.
- In the **oversight** track, the supervisor is asked before every irreversible action whether to intervene. The first intervention ends the supervision; its effect is measured by the same replay, against the same placebo, and an interruption cost (default 5 points) is charged per intervention. Interventions on runs that were going to succeed count as false alarms.
- An invalid or unparseable answer counts as a causal effect of zero.

## The environment: OpsWorld

`replayrca/env.py` simulates the back office of an e-commerce company. Every world is generated from a seed and is fully deterministic, so any state can be restored and every restoration is verified. The agent uses 15 tools (documents, tickets, customers, orders, refunds, cancellations, inventory, e-mail, finish), plus two at level 6 (`view_attachment`, `scan_code`).

Three task families, each with a set of traps:

| Family | Traps |
|---|---|
| `late_delivery` | Outdated policy; a neutrally titled memo that changes the promised dates; 5 pages of orders; a namesake customer; an excluded promo code; edge cases; refunds already paid that must be deducted; an annual cap with an order of imputation; a customer who lies; level 6: delivery slips with a handwritten DD/MM date and the order number only in the QR code |
| `recall` | An old SKU code; a product bundle; a batch range with an exception; a quality memo adding a batch; partial refunds; level 6: batch readable only in the label's QR code, relabelled parcels |
| `damage` | Corrections, including chained ones; duplicates; cartons; warehouse aliases; damage in transit; full or partial repairs; withdrawn reports; level 6: photographed forms with QR codes and handwritten corrections |

Levels 1 to 3 enable a fixed number of traps, level 4 draws them at random, level 5 enables all of them, and level 6 adds images (`replayrca/media.py`: photographed forms, delivery slips and packing labels rendered with Pillow and segno; the self-test checks that the QR codes are decodable with OpenCV when it is installed).

## Pipeline

1. **Harvest** (`replayrca/harvest.py`). Worker agents attempt generated tasks. A failure becomes a candidate item only if it is systematic: the worker fails at least two of three independent fresh attempts. Episodes that ended for infrastructure reasons (API errors, truncated outputs) or by exhausting the step budget are excluded.
2. **Calibration** (`python -m replayrca.oversight calibrate`). The worker's own action at its first checkpoint is replayed `k` times without any change. Failures whose spontaneous success is at most 20 % and successes whose spontaneous success is at least 80 % are kept, so that the outcome of every retained run is stable under restart.
3. **Autopsy** (`replayrca/evaluate.py`). The diagnoser receives the transcript (images included for vision models), its answer is parsed and sanitised, then `k` continuations of the correction and `k` of the placebo are run. Continuations are cached and shared between models.
4. **Oversight** (`python -m replayrca.oversight run`). The supervisor is queried before every irreversible action of a recorded run. Every checkpoint is cached, so a run is resumable.
5. **Reports** (`replayrca/report.py`, `python -m replayrca.oversight report`): leaderboards from the caches, with bootstrap confidence intervals.

## Protocol and guardrails

- **Reasoning effort.** `--effort top` sets each model to the highest reasoning effort that fits in 64 000 output tokens; a model can also be pinned with `provider/model@effort`. The same output budget applies to every call, and an answer that exhausts it is invalid.
- **Paired placebo.** The original action is replayed by the same mechanism at the same step; only the difference counts.
- **Anti-injection.** Tools never return the free text passed to them. In replacement actions, `reason` and `summary` are erased and `query`, `subject` and `body` are truncated, so a diagnosis cannot smuggle instructions to the worker.
- **Budget.** Every command takes `--budget` (US dollars, enforced from the usage reported by OpenRouter) and logs each call to `runs/calls.jsonl`. Caches in `runs/` make every command resumable and idempotent.
- **Offline checks.** `python -m replayrca.selftest` validates the generator, the tools and the checker; `python -m replayrca.verify <items.jsonl>` checks, for every stored run, that the generator still reproduces the recorded initial state, that the world can be restored faithfully at several steps and at every checkpoint (each restoration replays the recorded actions and compares every observation), that placebo actions exist and that every attachment renders.

## Installation and usage

Python 3.10 or later.

```bash
pip install -r requirements.txt            # segno, Pillow, numpy (+ optional opencv-python, scipy, matplotlib)
cp .env.example .env                        # OPENROUTER_API_KEY=...
python -m replayrca.selftest
python tests/test_scoring.py
```

A full cycle, from harvesting to the two leaderboards:

```bash
# 1. harvest systematic failures of small worker agents
python -m replayrca.harvest --workers deepseek/deepseek-v4.1-flash-1017,qwen/qwen3.8-flash \
       --templates late_delivery,recall --seeds 10-29 --difficulties 4,5,6 --out data/items_all.jsonl --budget 10

# 2. build a pool of failing and succeeding runs, calibrate it, export the calibrated failures as autopsy items
python -m replayrca.oversight build --items data/items_all.jsonl --n-fail 81 --n-success 65 --max-difficulty 6 --out data/pool_full.jsonl
python -m replayrca.oversight calibrate --runs data/pool_full.jsonl --out data/pool_full_cal.jsonl --k 5
python -m replayrca.oversight export-fails --runs data/pool_full_cal.jsonl --out data/items_calibrated.jsonl
python -m replayrca.oversight sample --runs data/pool_full_cal.jsonl --out data/oversight_subset.jsonl --n-fail 35 --n-success 25

# 3. autopsy track
python -m replayrca.evaluate --items data/items_calibrated.jsonl --effort top --k 5 --budget 100 \
       --models x-ai/grok-4.7,openai/gpt-6.1-sol,anthropic/claude-opus-5.5,baseline/replay-original-random
python -m replayrca.report --items data/items_calibrated.jsonl --models x-ai/grok-4.7,openai/gpt-6.1-sol,anthropic/claude-opus-5.5 --k 5

# 4. oversight track
python -m replayrca.oversight run --runs data/oversight_subset.jsonl --effort top --k 5 --budget 60 \
       --models deepseek/deepseek-v4.1-flash-1017,google/gemini-3.8-flash,openai/gpt-6.1-sol
python -m replayrca.oversight report --runs data/oversight_subset.jsonl --k 5 \
       --models deepseek/deepseek-v4.1-flash-1017,google/gemini-3.8-flash,openai/gpt-6.1-sol
```

Model identifiers are OpenRouter ids, optionally with an effort and a variant: `provider/model@effort#variant`. The variants are `#last` (a structural baseline forced to choose the final step), `#alt` (a reworded diagnosis prompt) and `#rep<N>` (an independent second sample of the same diagnoser, for test–retest). `baseline/replay-original-random` replays the original action at a random step.

Concurrency is set with `--threads` (calls to the evaluated model) and the environment variable `REPLAYRCA_PARALLEL` (concurrent worker continuations, default 32). With many simultaneous requests to an expensive model, OpenRouter reserves the maximum cost of each request in flight and may refuse requests (HTTP 402) when the credit is tight; lower both settings and rerun, the caches avoid paying twice.

## Repository layout

| Path | Contents |
|---|---|
| `replayrca/env.py`, `media.py` | World, tools, generators, checker, oracle; level-6 image rendering |
| `replayrca/agent.py`, `replay.py`, `llm.py` | Worker agent, counterfactual replay, OpenRouter client (budget, retries, effort) |
| `replayrca/harvest.py` | Resumable harvest of systematic failures |
| `replayrca/evaluate.py`, `report.py` | Autopsy track and its leaderboard |
| `replayrca/oversight.py` | Oversight track: pool construction, calibration, sampling, run, report |
| `replayrca/selftest.py`, `verify.py` | Offline checks of the generator and of stored trajectories |
| `replayrca/analysis.py`, `stats.py`, `figures.py` | Analyses written for the companion paper: paired bootstrap and permutation tests, reliability of the ranking, figures. They expect the data files and caches of a complete run, with the model lists set at the top of `analysis.py` and `stats.py` |
| `tests/test_scoring.py` | Scoring tests on a synthetic cache (no API calls) |

Generated at run time (git-ignored): `data/` (items and pools), `runs/` (caches of diagnoses, supervisions, continuations and trajectories, plus the cost log), `results/` (leaderboards and analyses).

## Limitations of the design

- One simulated environment. The protocol only requires an environment whose state can be saved and restored and an executable checker, and transfers to containers, databases or virtual machines, but this is not demonstrated here.
- The pool is a function of the workers it is harvested from. Harvesting from stronger workers needs more episodes, and continuations with large workers cost one to two orders of magnitude more per intervention.
- Anthropic models at their `max` effort fill the whole 64 000-token output budget; run them at `high`.

## Licence and citation

Code under the MIT licence (see `LICENSE`). To cite this repository, see `CITATION.cff`.
