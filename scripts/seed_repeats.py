"""Retrain an existing run's config with other seeds, VALID only, in an isolated GOTRAIN_HOME.

    python scripts/seed_repeats.py m-eurosat__r003 --seeds 2 3

The real run is only read (config.json, metrics.json). New runs go to
~/gotrain_experiments/seeds_<run>/ whose cache holds links to TRAIN/VALID only (no test file), so a
finalized task's runs/ and results/ are never written. Prints val per seed and mean +- SD (incl. original).
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))  # run from anywhere
import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

from gotrain import config

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2, 3])
    a = ap.parse_args()
    task = a.run_id.split("__")[0]
    src = config.RUNS_DIR / task / a.run_id
    spec = json.loads((src / "config.json").read_text())
    orig = json.loads((src / "metrics.json").read_text())
    home = Path.home() / "gotrain_experiments" / f"seeds_{a.run_id}"
    cache = home / "cache" / task
    cache.mkdir(parents=True, exist_ok=True)
    for f in ("train_X.npy", "train_y.npy", "valid_X.npy", "valid_y.npy", "meta.json", "train_stats.json"):
        if not (cache / f).exists():
            (cache / f).symlink_to(config.CACHE_DIR / task / f)
    env = {**os.environ, "GOTRAIN_HOME": str(home), "PYTHONPATH": str(REPO)}
    vals = {f"seed {spec['config'].get('seed', 0)} (original {a.run_id})": orig["val_score"]}
    for s in a.seeds:
        cfg = {**spec["config"], "seed": s, "time_limit_min": 60}
        out = subprocess.run([sys.executable, "-m", "gotrain.train", "--task", task, "--config", json.dumps(cfg)],
                             env=env, capture_output=True, text=True)
        line = [l for l in out.stdout.splitlines() if l.startswith("DONE")] or out.stdout.splitlines()[-3:]
        rd = sorted((home / "runs" / task).glob(f"{task}__r*"))[-1]
        m = json.loads((rd / "metrics.json").read_text()) if (rd / "metrics.json").exists() else {}
        vals[f"seed {s} ({rd.name})"] = m.get("val_score")
        print(f"seed {s}: {line}", flush=True)
    v = [x for x in vals.values() if x is not None]
    res = {"run": a.run_id, "config": spec["config"], "val_by_seed": vals, "n": len(v),
           "mean": statistics.mean(v), "sd": statistics.stdev(v) if len(v) > 1 else None,
           "range": [min(v), max(v)], "home": str(home)}
    (home / "seed_repeats.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print("SEEDS DONE")


if __name__ == "__main__":
    main()
