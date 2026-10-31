"""Baseline for the agent: random search over the same search space, same wall-clock budget, VALID only.

    GOTRAIN_HOME=<isolated home> python scripts/random_search.py m-eurosat --budget-min 60 [--seed 0]

Each run samples a curated candidate (CANDIDATE_POOL, the agent's menu) and the knobs the agent most
often changes: input size, learning rate, epochs, augmentation, label smoothing, mixup, TTA. Runs go one at
a time on the GPU through the same trainer the agent uses (train.py subprocess), each capped so it ends
inside the budget. It never finalizes; it cannot read test. Every sample is logged to random_search.jsonl.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))  # run from anywhere
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from gotrain import config, data, registry
from gotrain.train import CANDIDATE_POOL

AUGMENTS = [["flip", "rot90"], ["flip", "rot90", "scale_jitter"], ["flip", "rot90", "noise"]]


def sample(rng, kind: str, native: int, k: int) -> dict:
    pool = [e for e in CANDIDATE_POOL[kind] if e.get("source") != "hf"]        # curated menu only
    cfg = dict(pool[rng.integers(len(pool))]["config"])
    if kind != "segmentation" and native <= 64:
        cfg["img_size"] = [None, 128, 224][rng.integers(3)]
    cfg.update(lr=float(10 ** rng.uniform(-4, -2.5)), epochs=int(rng.choice([10, 20, 30])),
               augment=AUGMENTS[rng.integers(len(AUGMENTS))], label_smoothing=float(rng.choice([0.0, 0.1])),
               tta=bool(rng.integers(2)), seed=k)
    if kind == "classification":
        cfg["mixup"] = float(rng.choice([0.0, 0.2]))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--budget-min", type=float, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    meta = data.task_meta(a.task)
    native = min(meta["patch_size"] or [64])
    rng = np.random.default_rng(a.seed)
    deadline = time.time() + 60 * a.budget_min
    log = config.HOME / "random_search.jsonl"
    env = {**os.environ, "PYTHONPATH": str(_pl.Path(__file__).resolve().parent.parent)}
    k = 0
    while (left := (deadline - time.time()) / 60) > 2:
        k += 1
        cfg = sample(rng, meta["kind"], native, k)
        cfg["time_limit_min"] = round(min(20.0, left - 1), 1)
        rd = registry.new_run(a.task, cfg, hypothesis=f"random search sample {k} (seed {a.seed})", parent=None)
        t0 = time.time()
        with open(rd / "stdout.log", "w") as out:
            subprocess.run([sys.executable, "-m", "gotrain.train", "--run-dir", str(rd)], stdout=out,
                           stderr=subprocess.STDOUT, env=env)
        m = json.loads((rd / "metrics.json").read_text()) if (rd / "metrics.json").exists() else {}
        rec = {"k": k, "run_id": rd.name, "config": cfg, "val_score": m.get("val_score"),
               "minutes": round((time.time() - t0) / 60, 2), "minutes_left": round((deadline - time.time()) / 60, 1)}
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({k_: rec[k_] for k_ in ("k", "run_id", "val_score", "minutes", "minutes_left")}), flush=True)
    print("RANDOM SEARCH DONE", k, "runs")


if __name__ == "__main__":
    main()
