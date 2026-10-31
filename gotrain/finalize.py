"""The only code path that reads the TEST split. It runs once per task.

A lock file (results/<task>.lock.json) is written BEFORE the test data is opened, so a
second call is refused even if the first one crashed halfway. If that ever happens,
a human decides what to do and the lock file records it; the agent cannot retry.
"""
from __future__ import annotations

import json
import time

import torch

from . import config, data, registry
from .train import DEFAULTS, Normalizer, build_model, evaluate


def finalize(task: str, run_id: str | None = None, reason: str = "") -> dict:
    lock = config.RESULTS_DIR / f"{task}.lock.json"
    if lock.exists():
        prev = json.loads(lock.read_text())
        return {"refused": True, "why": "test split already used for this task", "previous": prev}
    best = registry.best_run(task)
    if run_id is None:
        if best is None:
            return {"refused": True, "why": "no finished run to finalize"}
        run_id = best["run_id"]
    info = registry.read(run_id)
    if (info["status"] or {}).get("state") != "done":
        return {"refused": True, "why": f"{run_id} is not a finished run"}
    lock.write_text(json.dumps({"task": task, "run_id": run_id, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "reason": reason, "best_val_run_at_that_time": best and best["run_id"]}, indent=1))

    rd = registry.run_dir(run_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(rd / "model.pt", map_location=device, weights_only=False)
    cfg = {**DEFAULTS, **ck["config"], "pretrained": False}   # weights come from the checkpoint
    model, _, _ = build_model(ck["meta"], cfg, ck["in_ch"])
    model.load_state_dict(ck["state_dict"])
    model.to(device)
    Xte, yte = data.load_split(task, "test", allow_test=True)
    if ck["meta"]["kind"] == "segmentation":
        import numpy as np
        n = ck["meta"]["n_classes"]
        yte = np.where((yte >= 0) & (yte < n), yte, -100).astype(np.int64)
    norm = Normalizer(data.channel_stats(task), cfg, device)
    m, _ = evaluate(model, Xte, yte, ck["meta"], cfg, norm, device)
    primary = {"classification": "accuracy", "multilabel": "f1_micro", "segmentation": "miou"}[ck["meta"]["kind"]]
    result = {"task": task, "run_id": run_id, "primary_metric": primary, "test_score": m[primary],
              "val_score": (info["metrics"] or {}).get("val_score"), "test_metrics": m,
              "n_runs_tried": len(registry.all_runs(task)), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    (config.RESULTS_DIR / f"{task}.json").write_text(json.dumps(result, indent=1))
    lock.write_text(json.dumps({**json.loads(lock.read_text()), "done": True, "test_score": m[primary]}, indent=1))
    try:
        from .store import Store
        result["published"] = Store().publish_run(run_id, stage="production", test_metrics=m)["version"]
    except Exception as e:  # storage problems must not hide the score
        result["publish_error"] = str(e)
    return result
