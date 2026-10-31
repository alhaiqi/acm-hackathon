"""Experiment registry: every run is a folder, every folder is a row on the leaderboard.

runs/<task>/<run_id>/
    config.json          what was tried, why (hypothesis), and which run it builds on (parent)
    status.json          running / done / failed, current epoch, best val so far
    train.log            full training log
    history.jsonl        one line per epoch
    metrics.json         final VALID metrics (written when the run ends)
    model.pt             best-on-VALID checkpoint
runs/<task>/journal.md   the agent's reasoning: hypotheses, observations, decisions
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import config


def task_runs_dir(task: str) -> Path:
    d = config.RUNS_DIR / task
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_run(task: str, cfg: dict, hypothesis: str, parent: str | None) -> Path:
    d = task_runs_dir(task)
    k = len([p for p in d.iterdir() if p.is_dir()]) + 1
    run_id = f"{task}__r{k:03d}"
    rd = d / run_id
    rd.mkdir()
    (rd / "config.json").write_text(json.dumps({
        "run_id": run_id, "task": task, "config": cfg, "hypothesis": hypothesis,
        "parent": parent, "created": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    return rd


def run_dir(run_id: str) -> Path:
    task = run_id.split("__")[0]
    rd = config.RUNS_DIR / task / run_id
    if not rd.exists():
        raise FileNotFoundError(f"no run {run_id}")
    return rd


def read(run_id: str) -> dict:
    rd = run_dir(run_id)
    info = json.loads((rd / "config.json").read_text())
    for name in ("status", "metrics"):
        p = rd / f"{name}.json"
        info[name] = json.loads(p.read_text()) if p.exists() else None
    return info


def all_runs(task: str) -> list[dict]:
    out = []
    for rd in sorted(task_runs_dir(task).iterdir()):
        if rd.is_dir() and (rd / "config.json").exists():
            out.append(read(rd.name))
    return out


def leaderboard(task: str) -> list[dict]:
    rows = []
    for r in all_runs(task):
        st, m = r["status"] or {}, r["metrics"] or {}
        rows.append({
            "run_id": r["run_id"], "state": st.get("state", "created"),
            "val_score": m.get("val_score", st.get("best_val")),
            "metric": m.get("primary_metric"), "epochs": m.get("epochs_run", st.get("epoch")),
            "minutes": m.get("minutes"), "parent": r["parent"], "hypothesis": r["hypothesis"][:140],
            "config_changes": r["config"],
        })
    rows.sort(key=lambda x: -(x["val_score"] if x["val_score"] is not None else -1))
    return rows


def best_run(task: str) -> dict | None:
    done = [r for r in leaderboard(task) if r["state"] == "done" and r["val_score"] is not None]
    return done[0] if done else None


def journal(task: str, text: str) -> None:
    with open(task_runs_dir(task) / "journal.md", "a") as f:
        f.write(f"\n### {time.strftime('%Y-%m-%d %H:%M:%S')}\n{text.strip()}\n")


def read_journal(task: str) -> str:
    p = task_runs_dir(task) / "journal.md"
    return p.read_text() if p.exists() else ""
