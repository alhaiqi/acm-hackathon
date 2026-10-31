"""stats.py graded against the fake runs, whose true effects are known (tests/make_fake_runs.py):

- scores rebuilt from per-image counts equal the logged scores (and train.metrics_from_probs, if torch is here);
- the confidence interval contains the score; a run against itself is a perfect tie;
- a real gain is accepted and a real loss is detected; the "gain" that is pure luck is NOT accepted,
  even on m-nz-cattle where it tops the leaderboard;
- runs on different val sets are refused;
- seed repeats rank the race correctly; the noise floor is small but positive;
- the val halves are fixed, disjoint and cover every image.

    python tests/test_stats.py
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_fake_runs  # noqa: E402


def accept(t, floor):
    """The planned noise rule: a gain counts if it is significant AND larger than training noise."""
    return t["diff"] > 0 and t["p"] < 0.05 and t["diff"] > floor["diff_sd"]


def main():
    truths = make_fake_runs.main(Path(tempfile.mkdtemp(prefix="gotrain_stats_")))
    from gotrain import registry, stats
    try:
        from gotrain.train import metrics_from_probs
    except ImportError:
        metrics_from_probs = None

    for task, truth in truths.items():
        runs = {r: i for r, i in truth["runs"].items() if i["skill"] is not None}

        # 1. same numbers as training logged
        for rid in runs:
            logged = json.loads((registry.run_dir(rid) / "best_val_metrics.json").read_text())["primary"]
            assert abs(stats.score(rid) - logged) < 1e-9, (rid, stats.score(rid), logged)
            if metrics_from_probs:
                probs, y = stats.load_preds(rid)
                m = metrics_from_probs({"kind": truth["kind"], "n_classes": truth["n_classes"], "class_names": None}, y, probs)
                assert abs(stats.score(rid) - m[truth["primary"]]) < 1e-9, rid

        # 2. confidence interval
        best = max(runs, key=stats.score)
        t0 = time.time()
        ci = stats.bootstrap_ci(best)
        assert ci["low"] < ci["score"] < ci["high"] and ci["se"] > 0, ci
        print(f"{task}: best {best} {ci['metric']} {ci['score']:.4f}  95% [{ci['low']:.4f}, {ci['high']:.4f}]  "
              f"se {ci['se']:.4f}  ({time.time() - t0:.2f}s, {ci['n_images']} images)")
        same = stats.paired_test(best, best)
        assert same["diff"] == 0 and same["p"] == 1.0, same

        # 3. noise floor from the race seeds; groups rank like the true skills
        race = {}
        for rid, i in runs.items():
            if i["role"] == "race":
                race.setdefault(i["candidate"], []).append(rid)
        floor = stats.noise_floor(list(race.values()))
        assert 0 < floor["seed_sd"] < 0.02 and floor["dof"] == 6, floor
        groups = {c: stats.group_score(g) for c, g in race.items()}
        by_mean = sorted(groups, key=lambda c: -groups[c]["mean"])
        by_skill = sorted(groups, key=lambda c: -runs[race[c][0]]["base_skill"])
        assert by_mean[0] == by_skill[0] and by_mean[-1] == by_skill[-1], (by_mean, by_skill)
        print(f"  noise floor: seed sd {floor['seed_sd']:.4f}, a gain must clear {floor['diff_sd']:.4f}")
        for c in by_mean:
            g = groups[c]
            print(f"  race {c:16s} {g['mean']:.4f} +- {g['se']:.4f}  ({g['n_runs']} seeds)")

        # 4. the planted effects, judged by the noise rule
        for rid, i in runs.items():
            if i["role"] not in ("real_gain", "noise", "real_loss", "small_gain"):
                continue
            t = stats.paired_test(rid, i["parent"])
            verdict = "accept" if accept(t, floor) else ("worse" if t["diff"] < 0 and t["p"] < 0.05 else "tie")
            print(f"  {i['role']:10s} {rid} vs parent: diff {t['diff']:+.4f} [{t['low']:+.4f}, {t['high']:+.4f}] "
                  f"p {t['p']:.3g} -> {verdict}")
            if i["role"] == "real_gain":
                assert verdict == "accept", (rid, t)
            if i["role"] == "noise":
                assert verdict != "accept", (rid, t)
            if i["role"] == "real_loss":
                assert verdict == "worse", (rid, t)

    # 5. different val sets are refused
    a = next(r for r, i in truths["m-eurosat"]["runs"].items() if i["skill"] is not None)
    b = next(r for r, i in truths["m-nz-cattle"]["runs"].items() if i["skill"] is not None)
    try:
        stats.paired_test(a, b)
        raise AssertionError("runs on different val sets must be refused")
    except ValueError:
        pass

    # 6. halves: fixed, disjoint, complete; a score on half A differs from the full score
    ha, hb = stats.halves(1000)
    assert len(ha) == len(hb) == 500 and not set(ha) & set(hb) and len(set(ha) | set(hb)) == 1000
    assert np.array_equal(ha, stats.halves(1000)[0])
    assert stats.score(a, ha) != stats.score(a, hb)
    print("STATS OK")


if __name__ == "__main__":
    main()
