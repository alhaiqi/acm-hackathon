"""The only code path that reads the TEST split. It runs once per task.

A lock file (results/<task>.lock.json) is written BEFORE the test data is opened, so a
second call is refused even if the first one crashed halfway. If that ever happens,
a human decides what to do and the lock file records it; the agent cannot retry.

finalize() takes one run or several (an ensemble chosen on VALID): each member predicts,
the probabilities are averaged, and the average is scored. All test-touching code lives here.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from . import config, data, registry
from .train import DEFAULTS, PRIMARY, Normalizer, build_model, evaluate, metrics_from_probs


def _predict(run_id: str, split: str, allow_test: bool = False):
    """(metrics, probs, y, meta) of one run's best checkpoint on a split. probs as in val_preds.npz."""
    rd = registry.run_dir(run_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(rd / "model.pt", map_location=device, weights_only=False)
    cfg = {**DEFAULTS, **ck["config"], "pretrained": False}   # weights come from the checkpoint
    model, _, _ = build_model(ck["meta"], cfg, ck["in_ch"])
    model.load_state_dict(ck["state_dict"])
    model.to(device)
    task, meta = ck["task"], ck["meta"]
    X, y = data.load_split(task, split, allow_test=allow_test)
    if meta["kind"] == "segmentation":
        n = meta["n_classes"]
        y = np.where((y >= 0) & (y < n), y, -100).astype(np.int64)
    norm = Normalizer(data.channel_stats(task), cfg, device)
    m, probs = evaluate(model, X, y, meta, cfg, norm, device, return_probs=True)
    return m, probs, y, meta


def predict(run_id: str, split: str = "valid"):
    """Probabilities of one run on TRAIN or VALID: (probs, y, metrics). TEST is refused here;
    it is only read inside finalize()."""
    if split == "test":
        raise PermissionError("The test split is only read by finalize(), once, with the final model.")
    m, probs, y, _ = _predict(run_id, split)
    return probs, y, m


def _ensemble(run_ids: list[str], split: str, allow_test: bool = False):
    """Average the members' probabilities. Returns (metrics, per-member primary scores)."""
    acc, members, meta, y = None, {}, None, None
    for r in run_ids:
        m, probs, y, meta = _predict(r, split, allow_test=allow_test)
        members[r] = m[PRIMARY[meta["kind"]]]
        p = np.asarray(probs, dtype=np.float32)
        acc = p if acc is None else acc + p
        del probs, p
    return metrics_from_probs(meta, y, acc / len(run_ids)), members


def _parse_ids(run_id) -> list[str]:
    if run_id is None:
        return []
    if isinstance(run_id, str):
        return [r.strip() for r in run_id.replace("+", ",").split(",") if r.strip()]
    return list(run_id)


def finalize(task: str, run_id: str | list[str] | None = None, reason: str = "") -> dict:
    """run_id: one run, a list, or "r1,r2,r3" (ensemble: probabilities averaged). None = best VALID run."""
    lock = config.RESULTS_DIR / f"{task}.lock.json"
    if lock.exists():
        prev = json.loads(lock.read_text())
        return {"refused": True, "why": "test split already used for this task", "previous": prev}
    best = registry.best_run(task)
    ids = _parse_ids(run_id)
    if not ids:
        if best is None:
            return {"refused": True, "why": "no finished run to finalize"}
        ids = [best["run_id"]]
    for r in ids:
        if not r.startswith(task + "__"):
            return {"refused": True, "why": f"{r} is not a run of {task}"}
        try:
            info = registry.read(r)
        except FileNotFoundError:
            return {"refused": True, "why": f"no run {r}"}
        if (info["status"] or {}).get("state") != "done":
            return {"refused": True, "why": f"{r} is not a finished run"}
    ids = list(dict.fromkeys(ids))   # no duplicate members
    ens = len(ids) > 1
    primary = PRIMARY[data.task_meta(task)["kind"]]

    # VALID score of exactly what will be tested (for an ensemble: the averaged probabilities). No test yet.
    if ens:
        val_m, val_members = _ensemble(ids, "valid")
        val_score = val_m[primary]
    else:
        val_members = None
        val_score = (registry.read(ids[0])["metrics"] or {}).get("val_score")

    lock.write_text(json.dumps({"task": task, "run_ids": ids, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "reason": reason, "best_val_run_at_that_time": best and best["run_id"]}, indent=1))

    if ens:
        m, test_members = _ensemble(ids, "test", allow_test=True)
    else:
        m, _, _, _ = _predict(ids[0], "test", allow_test=True)
        test_members = None
    result = {"task": task, "run_id": "+".join(ids), "run_ids": ids, "ensemble": ens, "primary_metric": primary,
              "test_score": m[primary], "val_score": val_score, "test_metrics": m,
              "n_runs_tried": len(registry.all_runs(task)), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    if ens:
        result["members_val"], result["members_test"] = val_members, test_members
    (config.RESULTS_DIR / f"{task}.json").write_text(json.dumps(result, indent=1))
    lock.write_text(json.dumps({**json.loads(lock.read_text()), "done": True, "test_score": m[primary]}, indent=1))
    try:
        from .store import Store
        st = Store()
        result["published"] = [st.publish_run(r, stage="production", test_metrics=m,
                                              ensemble=ids if ens else None)["version"] for r in ids]
    except Exception as e:  # storage problems must not hide the score
        result["publish_error"] = str(e)
    return result
