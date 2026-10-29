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
