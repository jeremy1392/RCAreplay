"""Scoring tests on synthetic caches with hand-computed expected values (no API calls).

python tests/test_scoring.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from replayrca import evaluate, oversight, report  # noqa: E402


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def make_item(iid, label=None, template="late_delivery", difficulty=4):
    steps = [{"tool": "read_ticket", "args": {"ticket_id": "T-1"}},
             {"tool": "issue_refund", "args": {"order_id": "O-1", "amount": 5.0, "reason": "x"}},
             {"tool": "finish", "args": {"summary": "done"}}]
    it = {"item_id": iid, "task": {"template": template, "difficulty": difficulty}, "worker": {"model": "w"},
          "traj": {"steps": steps}}
    if label:
        it["label"] = label
    return it


def outcomes(p, k=5):
    n = round(p * k)
    return [True] * n + [False] * (k - n)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "runs").mkdir()
        (root / "results").mkdir()
        report.ROOT = oversight.ROOT = root
        rk = evaluate.rollout_key

        # ---------------- autopsy ----------------
        items = [make_item("A"), make_item("B")]
        write_jsonl(root / "items.jsonl", items)
        M = "model/x@max"
        write_jsonl(root / "runs" / "diagnoses.jsonl", [
            {"id": f"{M}|A", "model": M, "item_id": "A", "valid": True, "t": 1, "tool": "read_doc",
             "args": {"name": "p"}, "cost": 0.5, "usage": {"completion_tokens": 100,
                                                          "completion_tokens_details": {"reasoning_tokens": 80}}},
            {"id": f"{M}|B", "model": M, "item_id": "B", "valid": False, "error": "no JSON", "cost": 0.25},
        ])
        placebo_A = evaluate.placebo_of(items[0], 1)
        write_jsonl(root / "runs" / "rollouts.jsonl", [
            {"key": rk("A", 1, "read_doc", {"name": "p"}), "item_id": "A", "t": 1, "outcomes": outcomes(0.8)},
            {"key": rk("A", 1, *placebo_A), "item_id": "A", "t": 1, "outcomes": outcomes(0.2)},
        ])
        report.main(items_path=root / "items.jsonl", models=[M], k=5)
        row = json.loads((root / "results" / "leaderboard.json").read_text())["table"][0]
        assert abs(row["fix_rate"] - 40.0) < 1e-9, row["fix_rate"]          # (0.8 + 0) / 2
        assert abs(row["uplift"] - 30.0) < 1e-9, row["uplift"]              # ((0.8 - 0.2) + 0) / 2
        assert row["valid"] == 1 and row["missing"] == 0, row
        assert abs(row["diag_cost"] - 0.75) < 1e-9
        assert row["fully_fixed"] == 0

        # missing rollout -> flagged incomplete, never silently counted as success
        write_jsonl(root / "runs" / "rollouts.jsonl", [])
        report.main(items_path=root / "items.jsonl", models=[M], k=5)
        row = json.loads((root / "results" / "leaderboard.json").read_text())["table"][0]
        assert row["missing"] == 1 and row["fix_rate"] == 0.0 and row["uplift"] is None, row

        # ---------------- oversight ----------------
        runs = [make_item("F", "fail"), make_item("S", "success")]
        write_jsonl(root / "runs.jsonl", runs)
        S = "sup/y@high"
        write_jsonl(root / "runs" / "oversight.jsonl", [
            {"id": f"{S}|F", "model": S, "run_id": "F", "label": "fail", "decision": "intervene", "t": 1,
             "tool": "get_order", "args": {"order_id": "O-1"}, "calls": 2, "cost": 0.1, "out_tokens": 10},
            {"id": f"{S}|S", "model": S, "run_id": "S", "label": "success", "decision": "proceed",
             "calls": 3, "cost": 0.2, "out_tokens": 30},
        ])
        placebo_F = evaluate.placebo_of(runs[0], 1)
        write_jsonl(root / "runs" / "rollouts.jsonl", [
            {"key": rk("F", 1, "get_order", {"order_id": "O-1"}), "item_id": "F", "t": 1, "outcomes": outcomes(0.6)},
            {"key": rk("F", 1, *placebo_F), "item_id": "F", "t": 1, "outcomes": outcomes(0.0)},
        ])
        sup = evaluate.JsonlCache(root / "runs" / "oversight.jsonl", "id").data
        rolls = evaluate.JsonlCache(root / "runs" / "rollouts.jsonl", "key").data
        assert abs(oversight.effect(sup[f"{S}|F"], runs[0], rolls, 5) - 0.6) < 1e-9
        assert oversight.effect(sup[f"{S}|S"], runs[1], rolls, 5) == 0.0
        out = root / "results" / "oversight.md"
        oversight.report(runs, [S], 5, out, penalty=0.05)
        line = [l for l in out.read_text(encoding="utf-8").splitlines() if l.startswith("| 1 |")][0]
        cells = [c.strip() for c in line.split("|")]
        assert cells[3] == "**+27.5 pts**", cells           # (0.6 - 0.05 * 1 intervention) / 2 runs
        assert cells[4] == "+30.0", cells                   # (0.6 + 0) / 2
        assert cells[7] == "1/1" and cells[9] == "0/1", cells  # intervened on failing / false alarms
        assert cells[10] == "5", cells                       # calls

        # ---------------- answer parsing / validation ----------------
        assert evaluate.parse_answer('```json\n{"step": 2}\n```') == {"step": 2}
        assert evaluate.parse_answer("{ unterminated") is None
        assert evaluate.parse_answer("no json at all") is None
    print("scoring tests OK")


if __name__ == "__main__":
    main()
