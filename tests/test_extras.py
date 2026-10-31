"""Extras (MCP-only features) on the fake GEO-Bench tasks, CPU, no Bedrock, no test split.

    GEO_BENCH_DIR=... GOTRAIN_HOME=... python tests/test_extras.py
"""
import json

import numpy as np


def main():
    from gotrain import advisor, config, registry
    from gotrain.registry import new_run
    from gotrain.train import run

    # a finished run to take the "best config" from (the advisor reads the leaderboard)
    rd = new_run("m-eurosat", {"pretrained": False, "epochs": 1, "num_workers": 0, "batch_size": 16}, "base", None)
    run(rd)
    runs_before = sorted(p.name for p in (config.RUNS_DIR / "m-eurosat").iterdir())

    # P1 data advisor
    t = np.linspace(1, 8, 4)
    a, b, r2 = advisor.fit_power_law(t * 100, 0.3 * (t * 100) ** -0.5)
    assert abs(b - 0.5) < 1e-6 and r2 > 0.999, (a, b, r2)
    rep = advisor.run("m-eurosat", cfg_override={"backbone": "resnet18", "pretrained": False, "num_workers": 0,
                                                 "batch_size": 16, "img_size": None, "tta": False}, steps=6)
    assert [c["fraction"] for c in rep["curve"]] == [0.1, 0.25, 0.5, 1.0]
    assert all(c["steps"] >= 6 for c in rep["curve"]), rep["curve"]
    lc = rep["learning_curve"]
    assert lc["gain_2x_ci95"][0] <= lc["gain_2x_ci95"][1]
    ld = rep["label_doctor"]
    assert ld["injected_flips"] >= 1 and 0 <= ld["recall_injected_at_K"] <= 1
    out = advisor.out_dir("m-eurosat")
    for f in ("SUMMARY.md", "advisor.json", "label_doctor.html", "labeling_queue.html", "labeling_queue.json"):
        assert (out / f).exists(), f
    assert "data:image/png;base64," in (out / "label_doctor.html").read_text() or ld["natural_suspects"] == 0
    assert "Verdict" in advisor.data_advisor_report("m-eurosat")
    # rules: no new runs, no results, no test cache
    assert sorted(p.name for p in (config.RUNS_DIR / "m-eurosat").iterdir()) == runs_before
    assert not list(config.RESULTS_DIR.glob("m-eurosat*"))
    assert not (config.CACHE_DIR / "m-eurosat" / "test_X.npy").exists()
    print(f"advisor OK: curve {[round(c['val_acc'], 3) for c in rep['curve']]}, "
          f"doctor recall {ld['recall_injected_at_K']:.2f}")

    # P2 reproduce receipt: publish the base run, fingerprint it, retrain it, check
    from gotrain import receipt
    from gotrain.store import Store
    v = Store().publish_run(rd.name, stage="candidate")["version"]
    rec = receipt.make_receipt("m-eurosat", v)
    assert set(rec["data_sha256"]) == {"train_X.npy", "train_y.npy", "valid_X.npy", "valid_y.npy"}
    assert rec["fingerprint"] and rec["original_val_ci95"][0] <= rec["original_val"] <= rec["original_val_ci95"][1]
    res = receipt.reproduce("m-eurosat", v, log=lambda *_: None)
    assert res["data_hashes_match"] and res["reproduced_val"] is not None, res
    assert res["verdict"] == "REPRODUCED", res          # same seed on CPU: deterministic, must land in the CI
    assert json.loads(receipt.model_receipt("m-eurosat", v))["last_reproduce_check"]["verdict"] == "REPRODUCED"
    assert sorted(p.name for p in (config.RUNS_DIR / "m-eurosat").iterdir()) == runs_before   # nothing written
    print(f"receipt OK: {rec['fingerprint']} val {res['original_val']} -> {res['reproduced_val']}")

    # MCP: extras exposed to MCP only, not to the training agent
    from gotrain import mcp_server, tools
    names = {f.__name__ for f in mcp_server.EXTRA_TOOLS}
    assert names and not names & {f.__name__ for f in tools.ALL_TOOLS}, names
    print("EXTRAS OK")


if __name__ == "__main__":   # guard: cache workers re-import this file under spawn
    main()
