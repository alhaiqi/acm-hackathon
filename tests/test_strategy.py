"""strategy.py graded on the fake runs (tests/make_fake_runs.py), whose true effects are known:

- propose_candidates: the menu carries data-specific notes (SSL4EO matches 13 Sentinel-2 bands);
- submit_shortlist: refuses unknown models, missing reasons and too few families; adds the safe
  default; writes the reasoning to the journal; returns screen configs on 30% of the data;
- race_decision: drops the clearly weak screen model; after the race picks branch A and a branch B
  from a DIFFERENT family (skipping the same-family runner-up);
- compare_runs: KEEP the real gain, TIE the lucky one, REVERT the real loss, FAILED for a crash;
- branch_status: finds both branches and prunes the one that is clearly behind;
- branch_status remembers every compare_runs verdict and flags finished runs nobody judged;
- fit_race: a race that would spend more than half the time left is cut (seeds, models, time limit);
- the tools are registered in tools.ALL_TOOLS.

    python tests/test_strategy.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_fake_runs  # noqa: E402

FACTS = {
    "m-eurosat": {"kind": "classification", "n_classes": 10, "bands": 13, "image_size": [64, 64],
                  "train_n": 2000, "valid_n": 1000, "known_issue": None},
    "m-nz-cattle": {"kind": "segmentation", "n_classes": 2, "bands": 3, "image_size": [500, 500],
                    "train_n": 524, "valid_n": 66, "known_issue": None},
}


def main():
    truths = make_fake_runs.main(Path(tempfile.mkdtemp(prefix="gotrain_strategy_")))
    from gotrain import registry, strategy, tools
    strategy.task_facts = lambda task: FACTS[task]          # fake runs have no dataset behind them
    j = json.loads

    # 1. the menu
    menu = {m["name"]: m for m in j(strategy.propose_candidates("m-eurosat"))["menu"]}
    assert "matches this data" in " ".join(menu["ssl4eo_resnet18"]["notes"]), menu["ssl4eo_resnet18"]
    assert "upsamples to 128px" in " ".join(menu["convnext_tiny"]["notes"])
    seg = {m["name"]: m for m in j(strategy.propose_candidates("m-nz-cattle"))["menu"]}
    assert "overfitting risk" in " ".join(seg["unet-tu-convnext_tiny"]["notes"]), seg["unet-tu-convnext_tiny"]
    print("menu:", sorted(menu), "|", sorted(seg))

    # 2. the shortlist
    why = "13 Sentinel-2 bands: weights pretrained on the same sensor"
    sub = lambda picks: j(strategy.submit_shortlist("m-eurosat", json.dumps(picks), json.dumps(
        ["spectral: band values carry the signal", "spatial: shapes and texture carry it"])))
    assert sub([{"name": "vit_huge", "why": why}])["refused"]
    assert sub([{"name": "ssl4eo_resnet18", "why": ""}])["refused"]
    assert sub([{"name": "ssl4eo_resnet18", "why": why}, {"name": "ssl4eo_resnet50", "why": why}])["refused"]
    ok = sub([{"name": "ssl4eo_resnet18", "why": why, "risk": "slower to converge"},
              {"name": "convnext_tiny", "why": "modern CNN for shapes at 128px"},
              {"name": "efficientnet_b0", "why": "cheap, suits a small train set"}])
    assert ok["added_safe_default"] == ["resnet18"] and len(ok["families"]) == 4, ok
    assert all(r["config"]["train_fraction"] == 0.3 for r in ok["screen_runs"])
    assert "MODEL SHORTLIST" in registry.read_journal("m-eurosat") and why in registry.read_journal("m-eurosat")
    print("shortlist:", ok["accepted"])

    for task, truth in truths.items():
        runs = strategy._runs(task)

        # 3a. screen: the top 3 race (>= 2 families), the weakest does not
        sc = strategy.screen_result(task, [r for r in runs if r["screen"]])
        dropped = [r["model"] for r in sc["results"] if r["verdict"] == "not raced"]
        assert len(set(sc["families"].values())) >= 2 and sc["seeds"] == 3, sc
        assert all(r["config"].get("train_fraction", 1) == 1 and "seed" in r["config"] for r in sc["race_runs"])
        print(f"{task}: screen -> race {sc['race_models']}, dropped {dropped}")
        skill = {i["candidate"]: i["skill"] for i in truth["runs"].values() if i["role"] == "screen"}
        top3 = sorted(skill, key=lambda c: -skill[c])[:3]
        name = {i["candidate"]: r["model"] for r in runs for i in [truth["runs"][r["id"]]] if r["screen"]}
        raced = {name[c] for c in top3} & set(sc["race_models"])
        assert len(raced) >= 2 and len(sc["race_models"]) == 3, (top3, sc["race_models"])   # mostly the truly strong
        if task == "m-eurosat":
            assert "efficientnet_b0" in dropped, sc["results"]

        # 3b. race: winners from different families
        rd = j(strategy.race_decision(task))
        assert rd["stage"] == "race", rd
        a, b = rd["branches"]["A"]["model"], rd["branches"]["B"]["model"]
        skipped = [x["model"] for x in rd["ranking"] if x["verdict"].startswith("skipped")]
        print(f"  race: A {a}, B {b}, skipped {skipped}, noise sd {rd['noise_floor']['seed_sd']:.4f}")
        if task == "m-eurosat":
            assert (a, b, skipped) == ("ssl4eo_resnet18", "convnext_tiny", ["ssl4eo_resnet50"]), rd["ranking"]
        else:
            assert (a, b) == ("segformer-mit_b2", "unet-efficientnet-b4"), rd["ranking"]

        # 4. keep / tie / revert / failed
        want = {"real_gain": "KEEP", "noise": "TIE", "real_loss": "REVERT", "failed": "FAILED"}
        for rid, info in truth["runs"].items():
            if info["role"] in want:
                c = j(strategy.compare_runs(rid))
                print(f"  {info['role']:9s} {rid}: {c['verdict']}" + (f" (gain {c['gain']:+.4f}, p {c['p']:.3g})" if "gain" in c else ""))
                assert c["verdict"] == want[info["role"]], c

        # 5. branches: both found, the one clearly behind is pruned
        bs = j(strategy.branch_status(task))
        assert set(bs["branches"]) == {"A", "B"} and bs["branches"]["A"]["model"] == a, bs
        assert bs["branches"]["A"]["failed"] == 1 and bs["prune"] == "B" and bs["next"] == "A", bs
        assert bs["ensemble_candidates"] == [bs["branches"]["A"]["best_run"]]

        # 5b. memory: judged runs carry their verdict; a finished run nobody judged is flagged
        tried = {t["run"]: t["verdict"] for b in bs["branches"].values() for t in b["tried"]}
        for rid, info in truth["runs"].items():
            if info["role"] in want:
                assert tried[rid] == want[info["role"]], (rid, tried)
        small = [r for r, i in truth["runs"].items() if i["role"] == "small_gain"]
        assert small and set(small) <= set(bs["unjudged"]) and "compare_runs" in bs["first"], bs
        strategy.compare_runs(small[0])
        assert small[0] not in j(strategy.branch_status(task))["unjudged"]
        print(f"  branches: A best {bs['branches']['A']['best_score']:.4f} ({bs['branches']['A']['tries']} tries), "
              f"B best {bs['branches']['B']['best_score']:.4f} ({bs['branches']['B']['tries']} tries) -> prune {bs['prune']}")

    # 6. the race fits the budget: m-chesapeake-like (~16 min per full run) is cut, m-eurosat-like is not
    row = lambda model, fam, minutes: {"model": model, "family": fam, "minutes": minutes,
                                       "cfg": {"backbone": model, "train_fraction": 0.3, "epochs": 10}}
    slow = [row("unet-resnet34", "unet-resnet", 1.6), row("unet-efficientnet-b4", "unet-efficientnet", 2.4),
            row("segformer-mit_b2", "segformer", 2.0)]
    f = strategy.fit_race(slow, 3, 140)
    assert [r["model"] for r in f["race"]] == ["unet-resnet34", "unet-efficientnet-b4"] and f["seeds"] == 2, f
    assert f["time_limit_min"] and f["minutes_est"] <= 0.5 * 140 + 1e-9, f
    fast = [row("ssl4eo_resnet18", "ssl4eo-resnet", 0.05), row("convnext_tiny", "convnext", 0.08),
            row("resnet18", "resnet", 0.04)]
    f = strategy.fit_race(fast, 3, 90)
    assert len(f["race"]) == 3 and f["seeds"] == 3 and f["time_limit_min"] is None and not f["notes"], f
    assert strategy.fit_race(slow, 3, None)["race"] == slow                 # no deadline (tests, MCP): unchanged
    print("race budget: chesapeake-like ->", strategy.fit_race(slow, 3, 140)["notes"])

    # 7. registered for the agent (and therefore for MCP)
    names = {f.__name__ for f in tools.ALL_TOOLS}
    assert {"propose_candidates", "submit_shortlist", "race_decision", "compare_runs", "branch_status"} <= names
    print("STRATEGY OK")


if __name__ == "__main__":
    main()
