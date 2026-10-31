"""A2: download the picked tasks and build their train/valid caches, then print a one-line summary each.

    python scripts/prepare_data.py                         # the 4 tasks of WORK_SPLIT.md
    python scripts/prepare_data.py m-chesapeake m-so2sat   # any others

The test split is NOT cached here (finalize.py does that, once). Safe to re-run: every step is idempotent.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))  # run from anywhere
import sys
import time

from gotrain import data

TASKS = sys.argv[1:] or ["m-eurosat", "m-pv4ger-seg", "m-brick-kiln", "m-nz-cattle"]

for t in TASKS:
    t0 = time.time()
    try:
        print(data.download_task(t), flush=True)
        for split in ("train", "valid"):
            print(" ", data.build_cache(t, split), flush=True)
        d = data.describe(t)
        print(f"OK {t}: {d['kind']}, {d['n_classes']} classes, input {d['input_shape_CHW']} {d['dtype']}, "
              f"train {d['train_n']} valid {d['valid_n']} ({time.time() - t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"FAILED {t}: {type(e).__name__}: {e}", flush=True)
