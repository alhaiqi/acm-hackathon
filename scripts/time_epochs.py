"""A3: measure minutes per epoch for each task on this GPU, so the tournament can size k and run length.

    python scripts/time_epochs.py                          # the 4 tasks of WORK_SPLIT.md, default model
    python scripts/time_epochs.py m-eurosat --config '{"backbone": "resnet50", "img_size": 128}'

Runs 3 epochs per task (a normal run: it shows up on the leaderboard as "timing run") and writes
timing.json (in GOTRAIN_HOME, not results/: the report reads every results/*.json as a final score): {task: {sec_per_epoch, min_for_<E>_epochs, ...}}. Share that file with the Brain side.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))  # run from anywhere
import argparse
import json
import statistics

from gotrain import config
from gotrain.registry import new_run
from gotrain.train import DEFAULTS, run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*", default=["m-eurosat", "m-pv4ger-seg", "m-brick-kiln", "m-nz-cattle"])
    ap.add_argument("--config", default="{}", help="JSON overrides, same as start_training")
    ap.add_argument("--epochs", type=int, default=3)
    a = ap.parse_args()

    out_path = config.HOME / "timing.json"
    timing = json.loads(out_path.read_text()) if out_path.exists() else {}
    full_epochs = json.loads(a.config).get("epochs", DEFAULTS["epochs"])
    for t in a.tasks:
        cfg = {**json.loads(a.config), "epochs": a.epochs, "patience": 99}
        rd = new_run(t, cfg, hypothesis="timing run (A3), not a search step", parent=None)
        try:
            run(rd)
        except SystemExit:
            print(f"FAILED {t}: see {rd / 'train.log'}")
            continue
        secs = [json.loads(l)["sec"] for l in (rd / "history.jsonl").read_text().splitlines()]
        per = statistics.median(secs[1:] or secs)          # epoch 1 includes cuDNN warm-up
        timing[t] = {"run_id": rd.name, "config": cfg, "sec_per_epoch": per, "epoch_secs": secs,
                     f"min_for_{full_epochs}_epochs": round(per * full_epochs / 60, 1),
                     "min_for_30pct_subset_10_epochs": round(per * 0.3 * 10 / 60, 1)}
        print(t, json.dumps(timing[t]), flush=True)
    out_path.write_text(json.dumps(timing, indent=1))
    print("wrote", out_path)


if __name__ == "__main__":   # cache workers re-import this file under spawn (Windows/macOS)
    main()
