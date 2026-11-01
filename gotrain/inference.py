"""predict_geotiff: run a task's production model (from the registry) on any GeoTIFF.

    GeoTIFF in -> GeoTIFF out with the same CRS/transform: band 1 = class index, band 2 = confidence (%),
    plus a PNG preview (true colour | class map with legend) and a JSON summary.

Safety checks, in this order:
  bands       input bands are matched BY NAME (band descriptions, Sentinel-2 names or STAC asset names) to the
              bands the model was trained on. Missing bands -> clear refusal listing them, unless the caller
              explicitly allows filling them (fill_missing): then they are set to the training mean, with a warning.
  value range each band's median is compared with the TRAIN statistics; a mismatch (e.g. L2A reflectance with
              the +1000 offset vs L1C top-of-atmosphere, or a different scale) is reported as a WARNING.
  tiling      the image is cut into training-size windows with overlap; window probabilities are averaged per
              pixel (classification models label whole windows, so their map is as coarse as one window).

m-eurosat band order: GEO-Bench 1.0 stores wrong band NAMES for this task (geo-bench issue #28): the true
channel order is B01..B12 then B8A (as in torchgeo / the EuroSAT authors). TRAINED_BANDS uses that order.

rasterio is optional: without it predict_geotiff returns a clear message and nothing else in GoTrain is affected.
Never reads any dataset split; only the model checkpoint, its config and the TRAIN statistics.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from . import config

S2 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
TRAINED_BANDS = {   # channel order the models saw; None = use the task's stored band names
    "m-eurosat": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"],
}
STAC_NAMES = {"coastal": "B01", "blue": "B02", "green": "B03", "red": "B04", "rededge1": "B05", "rededge2": "B06",
              "rededge3": "B07", "nir": "B08", "nir08": "B8A", "nir09": "B09", "cirrus": "B10", "swir16": "B11",
              "swir22": "B12"}


def canon(name) -> str:
    """'B8' / 'B08' / 'b08a' / 'nir08' / '08A - Vegetation Red Edge' -> canonical Sentinel-2 name when possible."""
    s = str(name or "").strip()
    if s.lower() in STAC_NAMES:
        return STAC_NAMES[s.lower()]
    m = re.match(r"^(?:B)?0?(\d{1,2})(A?)\b", s.upper().replace(" ", ""))
    if m:
        n, a = int(m.group(1)), m.group(2)
        return "B8A" if (n == 8 and a) else f"B{n:02d}"
    return s.upper()


def match_bands(trained: list[str], given: list[str], fill_missing=()) -> dict:
    """Index of each trained band in the input, or 'fill' / missing."""
    pos = {canon(g): i for i, g in enumerate(given)}
    plan, missing = [], []
    allow = {canon(f) for f in (fill_missing or ())}
    for b in trained:
        c = canon(b)
        if c in pos:
            plan.append(pos[c])
        elif c in allow:
            plan.append("fill")
        else:
            missing.append(c)
    return {"plan": plan, "missing": missing}


def _production(task: str):
    """The production version(s) to run: the finalized ensemble if production carries one, else the newest."""
    from .store import Store
    st = Store()
    prod = [v for v in st.versions(task) if v.get("stage") == "production"]
    if not prod:
        raise LookupError(f"{task} has no production model in the registry")
    ens = prod[-1].get("ensemble_members")
    chosen = [v for v in prod if v["run_id"] in ens] if ens else [prod[-1]]
    return [st.pull_model(task, v["version"]) for v in chosen]


def _load(files: list[str], device):
    import torch

    from .train import DEFAULTS, build_model
    ck = torch.load(next(f for f in files if f.endswith("model.pt")), map_location=device, weights_only=False)
    cfg = {**DEFAULTS, **ck["config"], "pretrained": False}
    model, _, _ = build_model(ck["meta"], cfg, ck["in_ch"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), cfg, ck["meta"]


def _train_stats(task: str) -> dict:
    p = config.CACHE_DIR / task / "train_stats.json"
    if not p.exists():
        p = config.STORE_DIR / "datasets" / task / "train_stats.json"
    return json.loads(p.read_text())


def range_warnings(x: np.ndarray, names: list[str], st: dict, idx: list) -> list[str]:
    """x: (C, H, W) in trained-band order (filled bands skipped). Compare medians with TRAIN p01/p99 and mean."""
    out = []
    for k, (b, i) in enumerate(zip(names, idx)):
        if i == "fill":
            continue
        med = float(np.median(x[k]))
        lo, hi, mean = st["p01"][k], st["p99"][k], st["mean"][k]
        if not (lo <= med <= hi) or (mean > 0 and not 0.5 <= med / mean <= 2.0):
            out.append(f"{b}: input median {med:.0f} vs TRAIN mean {mean:.0f} (p01-p99 {lo:.0f}-{hi:.0f})")
    if out:
        out.insert(0, "WARNING value range differs from the training data. Possible causes: L2A surface reflectance "
                      "(with the +1000 offset of processing baseline >= 04.00) vs L1C top-of-atmosphere, or a different "
                      "scale. Predictions may be unreliable:")
    return out


def predict_array(members, meta, st, x: np.ndarray, overlap: float = 0.5, batch: int = 64):
    """members: [(model, cfg)] (an ensemble averages them, each with its own img size / TTA / normalisation).
    x: (C, H, W) float32 in trained-band order -> per-pixel class probabilities (n_classes, H, W)."""
    import torch
    import torch.nn.functional as F

    from .train import Normalizer, resize, tta_views
    device = next(members[0][0].parameters()).device
    size = int(min(meta.get("patch_size") or [64]))
    step = max(1, int(round(size * (1 - overlap))))
    C, H, W = x.shape
    ph, pw = max(0, size - H), max(0, size - W)
    xp = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect") if (ph or pw) else x
    Hp, Wp = xp.shape[1:]
    ys = list(range(0, Hp - size + 1, step)); xs = list(range(0, Wp - size + 1, step))
    if ys[-1] != Hp - size: ys.append(Hp - size)
    if xs[-1] != Wp - size: xs.append(Wp - size)
    n = meta["n_classes"]
    acc = np.zeros((n, Hp, Wp), np.float32); cnt = np.zeros((Hp, Wp), np.float32)
    norms = [Normalizer(st, {**c, "channels": None}, device) for _, c in members]   # x is already in model order
    seg = meta["kind"] == "segmentation"
    wins = [(r, c) for r in ys for c in xs]
    with torch.no_grad():
        for i in range(0, len(wins), batch):
            part = wins[i:i + batch]
            raw = torch.from_numpy(np.stack([xp[:, r:r + size, c:c + size] for r, c in part])).float().to(device)
            prob = 0
            for (m, mc), norm in zip(members, norms):
                xb, _ = resize(norm(raw), None, mc["img_size"], seg)
                views = tta_views(mc["tta"], xb.shape[-1] == xb.shape[-2])
                pm = 0
                for fwd, back in views:
                    lo = m(fwd(xb)).float()
                    pm = pm + (back(lo) if seg else lo).softmax(1)
                pm = pm / len(views)
                if seg and pm.shape[-1] != size:
                    pm = F.interpolate(pm, size=(size, size), mode="bilinear", align_corners=False)
                prob = prob + pm
            prob = (prob / len(members)).cpu().numpy()
            for (r, c), p in zip(part, prob):
                acc[:, r:r + size, c:c + size] += p if seg else p[:, None, None]
                cnt[r:r + size, c:c + size] += 1
    return (acc / np.maximum(cnt, 1))[:, :H, :W], {"window": size, "stride": step, "windows": len(wins)}


def _preview(png: Path, rgb: np.ndarray | None, cls: np.ndarray, conf: np.ndarray, names: list[str], title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    cmap = plt.get_cmap("tab10" if len(names) <= 10 else "tab20")
    fig, ax = plt.subplots(1, 3 if rgb is not None else 2, figsize=(15 if rgb is not None else 10, 5.5))
    k = 0
    if rgb is not None:
        lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
        ax[0].imshow(np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)); ax[0].set_title("input (true colour)"); k = 1
    ax[k].imshow(cls, cmap=cmap, vmin=0, vmax=cmap.N - 1, interpolation="nearest"); ax[k].set_title("predicted class")
    present = sorted(set(np.unique(cls).tolist()))
    ax[k].legend(handles=[Patch(color=cmap(i), label=f"{names[i]} ({(cls == i).mean():.0%})") for i in present],
                 loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    im = ax[k + 1].imshow(conf, vmin=0, vmax=100, cmap="viridis"); ax[k + 1].set_title("confidence (%)")
    fig.colorbar(im, ax=ax[k + 1], fraction=0.046)
    for a in ax:
        a.axis("off")
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(png, dpi=110); plt.close(fig)


def predict_geotiff(path: str, task: str = "m-eurosat", out_dir: str = "", fill_missing: str = "",
                    overlap: float = 0.5) -> str:
    """Run the task's PRODUCTION model (from the model registry) on a GeoTIFF and write a georeferenced
    result: band 1 = class index, band 2 = confidence in %, same CRS and transform as the input, plus a PNG
    preview with a legend. Input bands are matched by name to the model's training bands; if any is missing
    the call is refused unless fill_missing lists it (then it is filled with the training mean, with a
    warning). A value-range mismatch with the training data (e.g. L2A vs L1C) is reported as a warning.

    Args:
        path: input GeoTIFF; band descriptions should name the bands (B02, B8A, red, nir08, ...).
        task: task whose production model to use, e.g. "m-eurosat".
        out_dir: output folder (default GOTRAIN_HOME/inference/<input name>).
        fill_missing: comma-separated band names the caller accepts to fill with the training mean, e.g. "B10".
        overlap: tile overlap fraction in [0, 0.9].
    """
    try:
        import rasterio
    except ImportError:
        return json.dumps({"error": "rasterio is not installed: pip install rasterio (the rest of GoTrain works without it)"})
    import torch
    t0 = time.time()
    src = Path(path)
    with rasterio.open(src) as ds:
        x = ds.read().astype(np.float32)
        given = [d or f"band{i + 1}" for i, d in enumerate(ds.descriptions)]
        profile, crs, transform = ds.profile, ds.crs, ds.transform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pulled = _production(task)
    loaded = [_load(p["files"], device) for p in pulled]
    members, cfg, meta = [(m, c) for m, c, _ in loaded], loaded[0][1], loaded[0][2]
    fixed = TRAINED_BANDS.get(task)
    trained = fixed if fixed and len(fixed) == len(meta["bands"]) else meta["bands"]
    if cfg["channels"] is not None:
        trained = [trained[i] for i in cfg["channels"]]
    fill = [f for f in fill_missing.split(",") if f.strip()]
    mb = match_bands(trained, given, fill)
    if mb["missing"]:
        return json.dumps({"refused": True, "why": f"input lacks bands the model was trained on: {mb['missing']}",
                           "model_bands": [canon(b) for b in trained], "input_bands": [canon(g) for g in given],
                           "hint": "provide them, or pass fill_missing to fill them with the training mean (less reliable)"},
                          indent=1)
    st_all = _train_stats(task)
    sel = cfg["channels"] if cfg["channels"] is not None else list(range(len(trained)))
    st = {k: [st_all[k][i] for i in sel] for k in ("mean", "std", "p01", "p99")}
    xin = np.stack([x[i] if i != "fill" else np.full(x.shape[1:], st["mean"][k], np.float32)
                    for k, i in enumerate(mb["plan"])])
    warns = range_warnings(xin, [canon(b) for b in trained], st, mb["plan"])
    filled = [canon(b) for b, i in zip(trained, mb["plan"]) if i == "fill"]
    if filled:
        warns.insert(0, f"WARNING bands filled with the training mean (not observed): {filled}")
    probs, tiling = predict_array(members, meta, st, xin, overlap=overlap)
    cls = probs.argmax(0).astype(np.uint8)
    conf = np.round(probs.max(0) * 100).astype(np.uint8)
    names = meta["class_names"] or [str(i) for i in range(meta["n_classes"])]
    out = Path(out_dir) if out_dir else config.HOME / "inference" / src.stem
    out.mkdir(parents=True, exist_ok=True)
    tif = out / f"{src.stem}_{task}_pred.tif"
    prof = {**profile, "count": 2, "dtype": "uint8", "nodata": None, "compress": "deflate"}
    with rasterio.open(tif, "w", **prof) as dst:
        dst.write(cls, 1); dst.write(conf, 2)
        dst.set_band_description(1, "class"); dst.set_band_description(2, "confidence_pct")
        dst.update_tags(task=task, classes=json.dumps(names), models=json.dumps([p["run_id"] for p in pulled]))
    rgb_idx = [given.index(g) for g in given if canon(g) in ("B04", "B03", "B02")]
    order = {canon(given[i]): i for i in rgb_idx}
    rgb = np.stack([x[order[b]] for b in ("B04", "B03", "B02")], -1) if len(order) == 3 else None
    png = out / f"{src.stem}_{task}_preview.png"
    _preview(png, rgb, cls, conf, names, f"{task} production model(s) {[p['run_id'] for p in pulled]} on {src.name}")
    area = {names[i]: round(float((cls == i).mean()), 4) for i in np.unique(cls)}
    res = {"input": str(src), "crs": str(crs), "transform": list(transform)[:6], "shape": list(x.shape),
           "models": [{"version": p["version"], "run_id": p["run_id"], "stage": p["stage"]} for p in pulled],
           "warnings": warns, "filled_bands": filled, "tiling": tiling, "class_area": dict(sorted(area.items(), key=lambda kv: -kv[1])),
           "mean_confidence_pct": float(conf.mean()), "output_geotiff": str(tif), "preview_png": str(png),
           "seconds": round(time.time() - t0, 1)}
    (out / f"{src.stem}_{task}_summary.json").write_text(json.dumps(res, indent=1))
    return json.dumps(res, indent=1)


MCP_TOOLS = [predict_geotiff]
