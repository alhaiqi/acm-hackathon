"""Run the Data Advisor on a classification task (TRAIN/VALID only; output in GOTRAIN_HOME/advisor/<task>/).

    python scripts/run_advisor.py m-eurosat                 # best existing config, full compute (GPU)
    python scripts/run_advisor.py m-eurosat --tiny          # plumbing check on CPU: tiny model, few steps
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))  # run from anywhere
import argparse
import json

from gotrain import advisor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--tiny", action="store_true", help="CPU plumbing check: untrained resnet18, 6 steps per model")
    ap.add_argument("--steps", type=int, default=None, help="optimizer steps per model (default: full run's steps)")
    a = ap.parse_args()
    over = ({"backbone": "resnet18", "pretrained": False, "img_size": None, "num_workers": 0, "batch_size": 16,
             "tta": False} if a.tiny else {})
    rep = advisor.run(a.task, cfg_override=over, steps=6 if a.tiny else a.steps)
    print(json.dumps({k: rep[k] for k in ("learning_curve", "minutes")}, indent=1, default=str))
    print("ADVISOR DONE ->", advisor.out_dir(a.task))


if __name__ == "__main__":
    main()
