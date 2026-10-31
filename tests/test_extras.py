"""Extras (MCP-only features) on the fake GEO-Bench tasks, CPU, no Bedrock, no test split.

    GEO_BENCH_DIR=... GOTRAIN_HOME=... python tests/test_extras.py
"""
import json

import numpy as np


def _exists(p):
    import os
    return os.path.exists(p)


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

    # P3 predict_geotiff: band-name matching rules (no rasterio needed)
    from gotrain import inference
    assert [inference.canon(b) for b in ("B8", "b08a", "nir08", "08A - Vegetation Red Edge", "B1", "swir22")] == \
        ["B08", "B8A", "B8A", "B8A", "B01", "B12"]
    mb = inference.match_bands(["B01", "B10", "B8A"], ["coastal", "nir08"])
    assert mb["missing"] == ["B10"] and inference.match_bands(["B01", "B10"], ["B1"], ["B10"])["plan"] == [0, "fill"]
    try:
        import rasterio
    except ImportError:
        rasterio = None
    if rasterio is None:
        assert "rasterio is not installed" in inference.predict_geotiff("x.tif")
        print("inference OK (rasterio missing: clear message)")
    else:
        from rasterio.transform import from_origin
        Store().promote("m-eurosat", v, "production")
        meta = advisor.data.task_meta("m-eurosat")
        Xva, _ = advisor.data.load_split("m-eurosat", "valid")
        img = np.concatenate([np.concatenate(list(np.asarray(Xva[i * 3:i * 3 + 3])), -1) for i in range(3)], -2)
        tif = config.HOME / "fake_scene.tif"
        prof = {"driver": "GTiff", "height": img.shape[1], "width": img.shape[2], "count": img.shape[0],
                "dtype": "float32", "crs": "EPSG:32636", "transform": from_origin(700000, 3750000, 10, 10)}
        with rasterio.open(tif, "w", **prof) as d:
            d.write(img.astype("float32"))
            for i, b in enumerate(meta["bands"], 1):
                d.set_band_description(i, b)
        r = json.loads(inference.predict_geotiff(str(tif), task="m-eurosat"))
        with rasterio.open(r["output_geotiff"]) as o:
            assert o.count == 2 and str(o.crs) == "EPSG:32636" and o.transform == prof["transform"]
            assert o.shape == img.shape[1:] and o.read(2).max() <= 100
        assert (config.HOME / "inference").exists() and _exists(r["preview_png"])
        # drop a band -> refused; allow filling it -> runs with a warning
        tif2 = config.HOME / "fake_scene_3b.tif"
        with rasterio.open(tif2, "w", **{**prof, "count": img.shape[0] - 1}) as d:
            d.write(img[1:].astype("float32"))
            for i, b in enumerate(meta["bands"][1:], 1):
                d.set_band_description(i, b)
        ref = json.loads(inference.predict_geotiff(str(tif2), task="m-eurosat"))
        assert ref.get("refused") and "lacks bands" in ref["why"], ref
        ok = json.loads(inference.predict_geotiff(str(tif2), task="m-eurosat",
                                                  fill_missing=inference.canon(meta["bands"][0])))
        assert ok["filled_bands"] and any("filled" in w for w in ok["warnings"]), ok
        print(f"inference OK: {r['shape']} -> {r['class_area']}, refusal + opt-in fill checked")

    # MCP: extras exposed to MCP only, not to the training agent
    from gotrain import mcp_server, tools
    names = {f.__name__ for f in mcp_server.EXTRA_TOOLS}
    assert names and not names & {f.__name__ for f in tools.ALL_TOOLS}, names
    print("EXTRAS OK")


if __name__ == "__main__":   # guard: cache workers re-import this file under spawn
    main()
