"""Reproduce receipts: a fingerprint per registry version, and a one-command check that it reproduces.

A receipt pins what produced a model version:
  data     sha256 of the TRAIN/VALID cache files the run trained and selected on (never test)
  config   the full training config (defaults filled in) and the seed
  code     the git commit checked out when the run was created (from git history by timestamp: the run
           itself did not record it, so this is a best-effort reconstruction and is labelled as such)
  result   the original VALID score and its bootstrap 95% CI (from the run's val_preds.npz)

reproduce() retrains the same config + seed in an isolated home (links to TRAIN/VALID only), and passes if
the new VALID score lands inside the original CI and the data hashes still match. GPU kernels are not
bit-deterministic, so "inside the CI" is the criterion, not equality.

Receipts live in GOTRAIN_HOME/receipts/<task>/v<N>.json (outside the store, so the S3 layout is unchanged).
Never reads TEST; never writes runs/ or results/.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import config, registry, stats
from .train import DEFAULTS

REPO = Path(__file__).resolve().parent.parent
DATA_FILES = ("train_X.npy", "train_y.npy", "valid_X.npy", "valid_y.npy")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def data_hashes(task: str) -> dict:
    d = config.CACHE_DIR / task
    return {f: _sha256(d / f) for f in DATA_FILES}


def _git(*args) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def _entry(task: str, version: int) -> dict:
    from .store import Store
    v = next((e for e in Store().versions(task) if e["version"] == int(version)), None)
    if v is None:
        raise KeyError(f"{task} has no registry version {version}")
    return v


def _path(task: str, version: int) -> Path:
    d = config.HOME / "receipts" / task
    d.mkdir(parents=True, exist_ok=True)
    return d / f"v{int(version)}.json"


def make_receipt(task: str, version: int) -> dict:
    v = _entry(task, version)
    info = registry.read(v["run_id"])
    rd = registry.run_dir(v["run_id"])
    cfg = {**DEFAULTS, **info["config"]}
    vp = np.load(rd / "val_preds.npz")
    ci = stats.bootstrap_ci(stats.scorecard(vp["probs"], vp["y"]))
    created = info.get("created")
    commit = _git("rev-list", "-1", f"--before={created}", "HEAD") if created else ""
    rec = {"task": task, "version": int(version), "run_id": v["run_id"], "stage": v.get("stage"),
           "data_sha256": data_hashes(task), "config": cfg, "seed": cfg.get("seed", 0),
           "code_commit": commit or None,
           "code_commit_how": "git rev-list -1 --before=<run created> HEAD (best effort: not recorded by the run)",
           "run_created": created, "original_val": (info["metrics"] or {}).get("val_score"),
           "original_val_ci95": [ci["low"], ci["high"]], "metric": ci["metric"],
           "receipt_created": time.strftime("%Y-%m-%d %H:%M:%S")}
    rec["fingerprint"] = hashlib.sha256(json.dumps(
        {k: rec[k] for k in ("data_sha256", "config", "seed", "code_commit")}, sort_keys=True).encode()).hexdigest()[:16]
    _path(task, version).write_text(json.dumps(rec, indent=1))
    return rec


def reproduce(task: str, version: int, log=print) -> dict:
    p = _path(task, version)
    rec = json.loads(p.read_text()) if p.exists() else make_receipt(task, version)
    now = data_hashes(task)
    data_ok = now == rec["data_sha256"]
    home = config.HOME / "experiments" / f"reproduce_{task}_v{int(version)}_{time.strftime('%H%M%S')}"
    cache = home / "cache" / task
    cache.mkdir(parents=True)
    for f in DATA_FILES + ("meta.json", "train_stats.json"):
        (cache / f).symlink_to(config.CACHE_DIR / task / f)
    over = {k: v for k, v in rec["config"].items() if k != "time_limit_min"}
    over["time_limit_min"] = 120
    env = {**os.environ, "GOTRAIN_HOME": str(home), "PYTHONPATH": str(REPO)}
    head = _git("rev-parse", "HEAD")
    log(f"[receipt] retraining {rec['run_id']} config, seed {rec['seed']}, code now {head[:10]} "
        f"(receipt {str(rec['code_commit'])[:10]}) in {home}")
    t0 = time.time()
    out = subprocess.run([sys.executable, "-m", "gotrain.train", "--task", task, "--config", json.dumps(over)],
                         env=env, capture_output=True, text=True)
    rd = sorted((home / "runs" / task).glob(f"{task}__r*"))[-1]
    m = json.loads((rd / "metrics.json").read_text()) if (rd / "metrics.json").exists() else {}
    new = m.get("val_score")
    lo, hi = rec["original_val_ci95"]
    ok = new is not None and lo <= new <= hi and data_ok
    res = {"task": task, "version": int(version), "fingerprint": rec["fingerprint"], "original_val": rec["original_val"],
           "original_ci95": [lo, hi], "reproduced_val": new, "inside_ci": new is not None and lo <= new <= hi,
           "data_hashes_match": data_ok, "code_now": head or None, "code_at_receipt": rec["code_commit"],
           "same_code": bool(head) and head == rec["code_commit"], "minutes": round((time.time() - t0) / 60, 1),
           "verdict": "REPRODUCED" if ok else "NOT REPRODUCED", "run_dir": str(rd),
           "train_tail": out.stdout.splitlines()[-2:]}
    (p.with_name(p.stem + "_reproduce.json")).write_text(json.dumps(res, indent=1))
    log(f"[receipt] {res['verdict']}: val {new} vs original {rec['original_val']} CI [{lo:.4f}, {hi:.4f}], "
        f"data match {data_ok}")
    return res


# --------------------------------------------------------------------------- MCP tools (no training)
def model_receipt(task: str, version: int) -> str:
    """Fingerprint of a registry model version: data hashes (TRAIN/VALID), full config, seed, code commit,
    and the original VALID score with its 95% CI. Also shows the last reproduce check, if one was run
    (scripts/reproduce_version.py retrains and checks the new VALID score lands inside that CI).

    Args:
        task: task name.
        version: registry version number (see model_versions).
    """
    p = _path(task, version)
    rec = json.loads(p.read_text()) if p.exists() else make_receipt(task, version)
    chk = p.with_name(p.stem + "_reproduce.json")
    rec["last_reproduce_check"] = json.loads(chk.read_text()) if chk.exists() else None
    return json.dumps(rec, indent=1, default=str)


MCP_TOOLS = [model_receipt]
