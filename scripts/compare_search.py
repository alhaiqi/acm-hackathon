"""Compare two search methods that ran in separate GOTRAIN_HOMEs on the same task (VALID only).

    python scripts/compare_search.py m-eurosat agent=<home_a> random=<home_b>

For each: runs started / finished, best VALID score with a bootstrap 95% CI, which run it was, and
after how many runs and minutes the best was first reached. Then a paired test best-vs-best on the
same val images (McNemar for classification, paired bootstrap for segmentation).
Reads run folders directly, so it needs no GOTRAIN_HOME of its own.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))  # run from anywhere
import json
import sys
from pathlib import Path

import numpy as np

from gotrain import stats


def summarize(home: Path, task: str) -> dict:
    runs = []
    for rd in sorted((home / "runs" / task).glob(f"{task}__r*")):
        st = json.loads((rd / "status.json").read_text()) if (rd / "status.json").exists() else {}
        m = json.loads((rd / "metrics.json").read_text()) if (rd / "metrics.json").exists() else {}
        runs.append({"run_id": rd.name, "dir": rd, "state": st.get("state"), "started": st.get("started"),
                     "val": m.get("val_score"), "has_preds": (rd / "val_preds.npz").exists()})
    done = [r for r in runs if r["val"] is not None and r["has_preds"]]
    if not done:
        return {"runs_started": len(runs), "runs_finished": 0}
    best = max(done, key=lambda r: r["val"])
    t0 = min(r["started"] for r in runs if r["started"])
    order = sorted(done, key=lambda r: r["started"])
    first = next(i for i, r in enumerate(order, 1) if r["val"] >= best["val"])
    vp = np.load(best["dir"] / "val_preds.npz")
    card = stats.scorecard(vp["probs"], vp["y"])
    return {"runs_started": len(runs), "runs_finished": len(done), "best_run": best["run_id"],
            "best_val": best["val"], "ci95": stats.bootstrap_ci(card), "best_reached_after_runs": first,
            "best_reached_after_min": round((order[first - 1]["started"] - t0) / 60, 1), "_card": card}


if __name__ == "__main__":
    task, specs = sys.argv[1], dict(s.split("=", 1) for s in sys.argv[2:])
    out = {name: summarize(Path(h), task) for name, h in specs.items()}
    for name, s in out.items():
        ci = s.get("ci95", {})
        print(f"{name:>7}: {s['runs_finished']}/{s['runs_started']} runs finished | best {s.get('best_run')} "
              f"val {s.get('best_val', float('nan')):.4f} [95% CI {ci.get('low', float('nan')):.4f}, "
              f"{ci.get('high', float('nan')):.4f}] | best reached after run {s.get('best_reached_after_runs')} "
              f"(~{s.get('best_reached_after_min')} min)")
    names = [n for n in out if "_card" in out[n]]
    if len(names) == 2:
        a, b = names
        print(f"paired {a} best vs {b} best:", json.dumps(stats.paired_test(out[a]["_card"], out[b]["_card"])))
    print(json.dumps({n: {k: v for k, v in s.items() if k != "_card"} for n, s in out.items()}, indent=1, default=str))
