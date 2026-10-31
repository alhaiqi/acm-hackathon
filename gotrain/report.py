"""Collect test scores, leaderboards and journals into results/REPORT.md (the submission summary)."""
from __future__ import annotations

import json

from . import config, registry


def write_report() -> str:
    lines = ["# GoTrain results", "", "| task | metric | test | val | chosen run | runs tried |", "|---|---|---|---|---|---|"]
    tasks = sorted(p.stem for p in config.RESULTS_DIR.glob("*.json") if not p.name.endswith(".lock.json"))
    for t in tasks:
        r = json.loads((config.RESULTS_DIR / f"{t}.json").read_text())
        lines.append(f"| {t} | {r['primary_metric']} | {r['test_score']:.4f} | {r['val_score']:.4f} | "
                     f"{r['run_id']} | {r['n_runs_tried']} |")
    for t in tasks:
        lines += ["", f"## {t}: every run (VALID)", "", "| run | parent | val | hypothesis |", "|---|---|---|---|"]
        for row in registry.leaderboard(t):
            v = "-" if row["val_score"] is None else f"{row['val_score']:.4f}"
            lines.append(f"| {row['run_id']} | {row['parent'] or '-'} | {v} | {row['hypothesis'].replace('|', '/')} |")
        lines += ["", "### Journal", registry.read_journal(t)]
    out = config.RESULTS_DIR / "REPORT.md"
    out.write_text("\n".join(lines))
    return str(out)


if __name__ == "__main__":
    print(write_report())
