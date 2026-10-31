"""Checks that the fake runs follow the same contract as real ones (tests/test_engine.py checks the
real side), so strategy code built on them works unchanged on SageMaker:

- every finished run has the contract files, val_preds.npz has the agreed shapes and dtypes;
- re-scoring val_preds with train.metrics_from_probs (if torch is installed) reproduces the logged score;
- crashed runs are listed as failed and have no val_preds;
- the registry reads them: leaderboard, parent chains back to the screen runs;
- the planted effects are visible: the real gain beats its parent, the real loss falls below its parent.

    python tests/test_fake_runs.py            # uses a fresh temporary GOTRAIN_HOME
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_fake_runs  # noqa: E402


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
    assert y.dtype == (np.int16 if kind == "segmentation" else np.int64), y.dtype
    return bvm["primary"], y, probs


def main():
    home = Path(tempfile.mkdtemp(prefix="gotrain_fake_"))
    truths = make_fake_runs.main(home)
    from gotrain import registry
    try:
        from gotrain.train import PRIMARY, metrics_from_probs
    except ImportError as e:            # laptop without torch: the numpy re-score in make_fake_runs is used
        print(f"train.metrics_from_probs unavailable ({e}); skipping the cross-check")
        metrics_from_probs = None

    for task, truth in truths.items():
        kind, n = truth["kind"], truth["n_classes"]
        scores = {}
        for rid, info in truth["runs"].items():
            rd = registry.run_dir(rid)
            if info["skill"] is None:
                st = json.loads((rd / "status.json").read_text())
                assert st["state"] == "failed" and not (rd / "val_preds.npz").exists(), rid
                continue
            primary, y, probs = check_contract(rd, kind, n)
            if metrics_from_probs:
                again = metrics_from_probs({"kind": kind, "n_classes": n, "class_names": None}, y, probs)[PRIMARY[kind]]
                assert abs(again - primary) < 1e-6, (rid, again, primary)
            scores[rid] = primary

        # the registry sees every run, failed ones included, and every chain ends at a screen run
        board = registry.leaderboard(task)
        assert len(board) == len(truth["runs"]), (len(board), len(truth["runs"]))
        assert sum(r["state"] == "failed" for r in board) == 1
        for rid in truth["runs"]:
            r = rid
            while registry.read(r)["parent"]:
                r = registry.read(r)["parent"]
            assert truth["runs"][r]["role"] == "screen", (rid, r)

        # the planted effects show in the scores
        for rid, info in truth["runs"].items():
            if info["role"] in ("real_gain", "real_loss"):
                parent = scores[info["parent"]]
                assert (scores[rid] > parent) == (info["role"] == "real_gain"), (rid, scores[rid], parent)

        best = max(scores, key=scores.get)
        print(f"{task}: {len(truth['runs'])} runs, best {best} {scores[best]:.4f} "
              f"({truth['runs'][best]['role']}, {truth['runs'][best]['candidate']})")
    print(f"home: {home}")
    print("FAKE RUNS OK")


if __name__ == "__main__":
    main()
