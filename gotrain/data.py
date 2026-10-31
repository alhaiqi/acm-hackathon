"""Download GEO-Bench tasks, turn them into fast numpy caches, and describe them.

Why a cache: GEO-Bench stores one HDF5 file per sample. Opening thousands of small
files every epoch makes the GPU wait on the disk. We read each split once, stack it
into one array (C, H, W per sample) saved as .npy, and memory-map it during training.

The TEST split is never cached by these functions unless `allow_test=True`, which only
`finalize.py` passes. That is how the "touch test once" rule is enforced in code, not
just in the prompt.
"""
from __future__ import annotations

import json
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import config

SPLITS = ("train", "valid", "test")
# geobench warns on every dataset open for m-eurosat / m-brick-kiln; we report it once in task_meta instead
warnings.filterwarnings("ignore", message=".*stores incorrect band names.*")
KNOWN_BAND_ORDER_ISSUES = {
    "m-eurosat": "stored band NAMES are wrong (pixel data is fine): never select bands by name, use channel indices",
    "m-brick-kiln": "stored band NAMES are wrong (pixel data is fine): never select bands by name, use channel indices",
}


# --------------------------------------------------------------------------- download
def download_task(task: str) -> str:
    """Download only one task's files from the Hugging Face repo and unzip them."""
    import zipfile

    from huggingface_hub import HfApi, hf_hub_download

    bench = config.benchmark_of(task)
    target = config.GEO_BENCH_DIR / bench / task
    if (target / "task_specs.pkl").exists():
        return f"{task} already present at {target}"
    files = HfApi().list_repo_files(config.HF_REPO, repo_type="dataset")
    # files of this task: its own zip/folder; plus small per-benchmark files (normalizers, readme)
    mine = [f for f in files if f.startswith(bench + "/") and (f"/{task}" in f)]
    # avoid "m-pv4ger" also grabbing "m-pv4ger-seg" (they are in different benchmarks, but be safe)
    mine = [f for f in mine if Path(f).name.split(".")[0] == task or f"/{task}/" in f]
    if not mine:
        raise FileNotFoundError(f"No files for {task} under {bench}/ in {config.HF_REPO}")
    for f in mine:
        hf_hub_download(config.HF_REPO, f, repo_type="dataset", local_dir=config.GEO_BENCH_DIR)
        p = config.GEO_BENCH_DIR / f
        if p.suffix == ".zip":
            with zipfile.ZipFile(p) as z:
                z.extractall(p.parent)
            p.unlink()
    if not (target / "task_specs.pkl").exists():
        raise FileNotFoundError(f"Downloaded {mine} but {target}/task_specs.pkl is missing")
    return f"downloaded {task} ({len(mine)} file(s)) to {target}"


# --------------------------------------------------------------------------- task info
def load_task(task: str):
    import geobench

    return geobench.load_task_specs(config.task_dir(task))


def task_kind(task_specs) -> str:
    name = type(task_specs.label_type).__name__
    if name == "SegmentationClasses":
        return "segmentation"
    if name == "MultiLabelClassification":
        return "multilabel"
    if name == "Classification":
        return "classification"
    raise ValueError(f"Unsupported label type {name}")


def task_meta(task: str) -> dict:
    ts = load_task(task)
    lt = ts.label_type
    names = getattr(lt, "class_names", None)
    return {
        "task": task,
        "kind": task_kind(ts),
        "n_classes": int(lt.n_classes),
        "class_names": list(names) if names else None,
        "bands": [str(b.name) for b in ts.bands_info],
        "patch_size": list(ts.patch_size) if ts.patch_size else None,
        "resolution_m": ts.spatial_resolution,
        "known_issue": KNOWN_BAND_ORDER_ISSUES.get(task),
    }


# --------------------------------------------------------------------------- caching
def _cache_dir(task: str) -> Path:
    d = config.CACHE_DIR / task
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sample_to_arrays(sample, kind):
    try:
        x, _ = sample.pack_to_3d(band_names=None, resample=True)          # (H, W, C)
    except AssertionError:                                               # time series: stack dates as channels
        x4, _, _ = sample.pack_to_4d(band_names=None, resample=True)      # (T, H, W, C)
        x = np.concatenate(list(x4), axis=-1)
    x = np.ascontiguousarray(np.moveaxis(x, -1, 0))                       # (C, H, W)
    lab = sample.label
    if kind == "segmentation":
        y = np.asarray(lab.data)
        if y.ndim == 3:
            y = y[..., 0]
        y = y.astype(np.int16)
    elif kind == "multilabel":
        y = np.asarray(lab, dtype=np.float32).reshape(-1)
    else:
        y = np.int64(lab)
    return x, y


def _fit(x, shape):
    """Crop or zero-pad (C, H, W) to `shape` (rare: only if samples differ in size)."""
    out = np.zeros(shape, dtype=x.dtype)
    c, h, w = (min(a, b) for a, b in zip(x.shape, shape))
    out[:c, :h, :w] = x[:c, :h, :w]
    return out


def _load_chunk(args):
    task, split, idx = args
    ts = load_task(task)
    ds = ts.get_dataset(split=split, band_names=None)
    kind = task_kind(ts)
    return [_sample_to_arrays(ds[i], kind) for i in idx]


def build_cache(task: str, split: str, allow_test: bool = False, workers: int = 8) -> dict:
    """Read one split of a task into X.npy (N, C, H, W) and y.npy. Idempotent."""
    if split == "test" and not allow_test:
        raise PermissionError("The test split is only read by finalize.py, once, with the final model.")
    d = _cache_dir(task)
    xp, yp = d / f"{split}_X.npy", d / f"{split}_y.npy"
    if xp.exists() and yp.exists():
        return {"split": split, "X": str(xp), "y": str(yp), "cached": True}

    ts = load_task(task)
    kind = task_kind(ts)
    n = len(ts.get_dataset(split=split, band_names=None))
    chunks = [(task, split, list(range(i, min(i + 64, n)))) for i in range(0, n, 64)]
    first_x, first_y = _load_chunk((task, split, [0]))[0]
    X = np.lib.format.open_memmap(str(xp) + ".tmp.npy", mode="w+", dtype=first_x.dtype, shape=(n, *first_x.shape))
    ys, misfit, k = [], 0, 0
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for res in pool.map(_load_chunk, chunks):
            for x, y in res:
                if x.shape != first_x.shape:
                    misfit += 1
                    x = _fit(x, first_x.shape)
                    if kind == "segmentation":
                        y = _fit(y[None], (1, *first_x.shape[1:]))[0]
                X[k] = x
                ys.append(y)
                k += 1
    X.flush()
    del X
    Path(str(xp) + ".tmp.npy").rename(xp)
    np.save(yp, np.stack(ys))
    if misfit:
        warnings.warn(f"{task}/{split}: {misfit} samples had a different shape and were cropped/padded")
    meta = task_meta(task)
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    return {"split": split, "X": str(xp), "y": str(yp), "cached": False, "n": n, "reshaped": misfit}


def load_split(task: str, split: str, allow_test: bool = False):
    if split == "test" and not allow_test:
        raise PermissionError("The test split is only read by finalize.py, once, with the final model.")
    build_cache(task, split, allow_test=allow_test)
    d = _cache_dir(task)
    return np.load(d / f"{split}_X.npy", mmap_mode="r"), np.load(d / f"{split}_y.npy")


def channel_stats(task: str, max_samples: int = 2000) -> dict:
    """Per-channel statistics on TRAIN only (used for normalisation). Robust: 1st/99th percentiles too."""
    d = _cache_dir(task)
    p = d / "train_stats.json"
    if p.exists():
        return json.loads(p.read_text())
    X, _ = load_split(task, "train")
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(X), size=min(max_samples, len(X)), replace=False))
    sub = np.asarray(X[idx], dtype=np.float64)                              # (n, C, H, W)
    flat = np.moveaxis(sub, 1, 0).reshape(sub.shape[1], -1)
    st = {
        "mean": flat.mean(1).tolist(),
        "std": (flat.std(1) + 1e-6).tolist(),
        "min": flat.min(1).tolist(),
        "max": flat.max(1).tolist(),
        "p01": np.percentile(flat, 1, axis=1).tolist(),
        "p99": np.percentile(flat, 99, axis=1).tolist(),
    }
    p.write_text(json.dumps(st))
    return st


# --------------------------------------------------------------------------- inspection
def describe(task: str) -> dict:
    """Everything the agent needs to choose an approach, from TRAIN and VALID only."""
    meta = task_meta(task)
    out = {**meta}
    for split in ("train", "valid"):
        X, y = load_split(task, split)
        out[f"{split}_n"] = int(len(X))
        out["input_shape_CHW"] = list(X.shape[1:])
        out["dtype"] = str(X.dtype)
        k = meta["kind"]
        names = meta["class_names"] or [str(i) for i in range(meta["n_classes"])]
        if k == "classification":
            cnt = Counter(y.tolist())
            dist = {names[c] if c < len(names) else str(c): int(v) for c, v in sorted(cnt.items())}
        elif k == "multilabel":
            dist = {names[i]: int(v) for i, v in enumerate(y.sum(0))}
            out[f"{split}_labels_per_image_mean"] = float(y.sum(1).mean())
        else:
            vals, cnts = np.unique(y, return_counts=True)
            tot = cnts.sum()
            dist = {(names[v] if 0 <= v < len(names) else f"value {v}"): round(float(c / tot), 4) for v, c in zip(vals, cnts)}
            out["label_values_present"] = [int(v) for v in vals]
        out[f"{split}_label_distribution"] = dist
    st = channel_stats(task)
    out["channel_stats_train"] = {
        (meta["bands"][i] if i < len(meta["bands"]) else f"ch{i}"): {k: round(st[k][i], 2) for k in ("mean", "std", "p01", "p99")}
        for i in range(len(st["mean"]))
    }
    tr = out["train_label_distribution"]
    if meta["kind"] in ("classification", "multilabel") and tr:
        v = list(tr.values())
        out["imbalance_ratio_max_over_min"] = round(max(v) / max(1, min(v)), 2)
    return out
