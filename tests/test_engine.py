"""Engine checks (Person A side), offline on the fake GEO-Bench tasks, CPU, no Bedrock:

- every run writes the frozen contract files, val_preds.npz has the agreed shapes and dtypes,
  and re-scoring val_preds reproduces the logged best VALID score;
- 8-way TTA views invert exactly;
- predict() works on VALID and refuses TEST;
- finalize() accepts an ensemble, scores it once, and refuses a second call;
- the store publishes, promotes and pulls model versions.

    GEO_BENCH_DIR=... GOTRAIN_HOME=... python tests/test_engine.py
"""
import json

import numpy as np
import torch

from gotrain import config, registry
from gotrain.finalize import finalize, predict
from gotrain.registry import new_run
from gotrain.store import Store
from gotrain.train import PRIMARY, adapt_first_conv, metrics_from_probs, run, tta_views

FAST = {"pretrained": False, "epochs": 2, "num_workers": 0, "batch_size": 16}


def train(task, extra, hyp):
    rd = new_run(task, {**FAST, **extra}, hypothesis=hyp, parent=None)
    run(rd)
    return rd


def check_contract(rd, kind, n):
    cfg = json.loads((rd / "config.json").read_text())
    assert {"task", "config", "hypothesis", "parent", "seed"} <= set(cfg), cfg.keys()
    st = json.loads((rd / "status.json").read_text())
    assert st["state"] == "done" and st["epoch"] >= 1 and st["best_val"] is not None, st
    bvm = json.loads((rd / "best_val_metrics.json").read_text())
    assert isinstance(bvm["primary"], float), bvm
    assert len((rd / "history.jsonl").read_text().splitlines()) == st["epoch"]
    vp = np.load(rd / "val_preds.npz")
    probs, y = vp["probs"], vp["y"]
    assert probs.dtype == np.float16 and probs.shape[1] == n and len(probs) == len(y), (probs.dtype, probs.shape)
    assert probs.ndim == (4 if kind == "segmentation" else 2)
    meta = {"kind": kind, "n_classes": n, "class_names": None}
    again = metrics_from_probs(meta, y, probs)[PRIMARY[kind]]
    assert abs(again - bvm["primary"]) < 0.02, (again, bvm["primary"])   # float16 rounding only
    return bvm["primary"]


def main():
    # 1. TTA views are exact inverses (segmentation relies on this)
    t = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).view(2, 3, 4, 4)
    views = tta_views(True, True)
    assert len(views) == 8 and all(torch.equal(b(f(t)), t) for f, b in views)
    assert len({tuple(f(t).flatten().tolist()) for f, _ in views}) == 8          # 8 distinct symmetries
    assert len(tta_views(True, False)) == 3 and len(tta_views(False, True)) == 1

    # 2. first-conv adaptation keeps pretrained filters for a band subset
    conv = torch.nn.Conv2d(13, 8, 3)
    assert torch.equal(adapt_first_conv(conv, 3, [3, 2, 1]).weight, conv.weight[:, [3, 2, 1]])
    assert adapt_first_conv(conv, 5).weight.shape == (8, 5, 3, 3)

    # 3. classification runs (one with TTA) + contract
    c1 = train("m-eurosat", {}, "baseline")
    c2 = train("m-eurosat", {"tta": True, "seed": 1}, "tta + other seed")
    for rd in (c1, c2):
        print("cls", rd.name, check_contract(rd, "classification", 3))

    # 4. segmentation runs: unet, and segformer if this smp has it (4 bands -> adapter or native)
    s1 = train("m-pv4ger-seg", {}, "unet baseline")
    print("seg", s1.name, check_contract(s1, "segmentation", 3))
    try:
        s2 = train("m-pv4ger-seg", {"seg_arch": "segformer", "backbone": "mit_b0", "tta": True}, "segformer")
        print("seg", s2.name, check_contract(s2, "segmentation", 3), "|", (s2 / "train.log").read_text().count("adapter"), "adapter note(s)")
    except SystemExit:
        print("segformer unavailable in this smp version:", (registry.task_runs_dir("m-pv4ger-seg")).name)

    # 5. predict: VALID ok and equal to the training-time val_preds; TEST refused
    probs, y, m = predict(c1.name, "valid")
    saved = np.load(c1 / "val_preds.npz")["probs"]
    assert np.allclose(probs, saved, atol=2e-3), np.abs(probs - saved).max()
    try:
        predict(c1.name, "test")
        raise AssertionError("predict on test must be refused")
    except PermissionError:
        pass

    # 6. finalize an ensemble once; second call refused; bad ids refused before the lock
    assert finalize("m-eurosat", "m-eurosat__r999")["refused"]
    assert not (config.RESULTS_DIR / "m-eurosat.lock.json").exists(), "a refused call must not burn the lock"
    res = finalize("m-eurosat", f"{c1.name},{c2.name}", "ensemble of the two runs (VALID)")
    assert res["ensemble"] and 0 <= res["test_score"] <= 1 and set(res["members_test"]) == {c1.name, c2.name}, res
    print("ensemble val", res["val_score"], "members", res["members_val"], "| test", res["test_score"])
    assert finalize("m-eurosat", c1.name)["refused"]
    seg = finalize("m-pv4ger-seg", s1.name, "single run")
    assert not seg.get("ensemble") and 0 <= seg["test_score"] <= 1, seg

    # 7. store: versions, ensemble lineage, promote, pull
    st = Store()
    vers = st.versions("m-eurosat")
    assert [v["stage"] for v in vers] == ["production", "production"] and vers[0]["ensemble_members"], vers
    assert st.promote("m-eurosat", 1, "archived")["stage"] == "archived"
    pulled = st.pull_model("m-eurosat")                      # newest production = v2
    assert pulled["version"] == 2 and all(__import__("os").path.exists(f) for f in pulled["files"]), pulled

    print("ENGINE OK")


if __name__ == "__main__":   # guard: Windows spawns cache workers that re-import this file
    main()
