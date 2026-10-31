"""Quick probe: a random forest on hand-made spectral features, in seconds, before any GPU run.

Why: it answers three questions the deep model cannot answer cheaply.
1. Floor: what score does a model with NO spatial understanding get? The CNN must beat it clearly.
2. Spectral vs spatial: if the forest is close to the CNN, the signal is mostly in band values
   (keep all bands, care about normalisation). If the CNN is far ahead, the signal is in shapes and
   texture (spend effort on resolution, backbone, augmentation).
3. Which bands matter: feature importances, summed per channel, guide band-subset ablations.

Classification: one feature row per image (per-band mean, std, 10th/90th percentile, and the
normalised difference of every band pair's means, the family NDVI/NDWI belong to).
Segmentation: one row per sampled pixel (band values + pairwise normalised differences); the
score is an estimate from sampled pixels. Uses TRAIN to fit and VALID to score, never TEST.
"""
from __future__ import annotations

import json
import time
from itertools import combinations

import numpy as np

from . import config, data
from .train import PRIMARY, metrics_from


def _nd_pairs(m, pairs):
    """Normalised differences (a-b)/(a+b) for band pairs; m has channels on the last axis."""
    a, b = m[..., [p[0] for p in pairs]], m[..., [p[1] for p in pairs]]
    return (a - b) / (np.abs(a) + np.abs(b) + 1e-6)


def _image_features(X, idx, pairs):
    out = []
    for s in range(0, len(idx), 256):
        x = np.asarray(X[idx[s:s + 256]], dtype=np.float32)          # (n, C, H, W)
        flat = x.reshape(x.shape[0], x.shape[1], -1)
        mean = flat.mean(-1)
        q10, q90 = np.percentile(flat, [10, 90], axis=-1)
        out.append(np.concatenate([mean, flat.std(-1), q10, q90, _nd_pairs(mean, pairs)], axis=1))
    return np.concatenate(out)


def _pixel_features(X, y, idx, pairs, per_image, rng, n_classes):
    feats, labels = [], []
    for i in idx:
        x = np.asarray(X[i], dtype=np.float32)                        # (C, H, W)
        t = np.asarray(y[i]).reshape(-1)
        px = x.reshape(x.shape[0], -1).T                               # (H*W, C)
        ok = np.flatnonzero((t >= 0) & (t < n_classes))
        if len(ok) == 0:
            continue
        pick = rng.choice(ok, size=min(per_image, len(ok)), replace=False)
        feats.append(np.concatenate([px[pick], _nd_pairs(px[pick], pairs)], axis=1))
        labels.append(t[pick])
    return np.concatenate(feats), np.concatenate(labels)


def quick_probe(task: str, max_images: int = 3000, pixels_per_image: int = 200, seed: int = 0) -> dict:
    from sklearn.ensemble import RandomForestClassifier

    t0 = time.time()
    meta = data.task_meta(task)
    kind, n = meta["kind"], meta["n_classes"]
    Xtr, ytr = data.load_split(task, "train")
    Xva, yva = data.load_split(task, "valid")
    C = Xtr.shape[1]
    pairs = list(combinations(range(C), 2)) if C <= 20 else list(combinations(range(min(C, 12)), 2))
    rng = np.random.default_rng(seed)
    tr_idx = np.sort(rng.choice(len(Xtr), size=min(max_images, len(Xtr)), replace=False))
    va_idx = np.arange(len(Xva)) if kind != "segmentation" else np.sort(
        rng.choice(len(Xva), size=min(max_images // 3, len(Xva)), replace=False))
    names = meta["bands"] if len(meta["bands"]) == C else [f"ch{i}" for i in range(C)]

    if kind == "segmentation":
        Ftr, Ltr = _pixel_features(Xtr, ytr, tr_idx, pairs, pixels_per_image, rng, n)
        Fva, Lva = _pixel_features(Xva, yva, va_idx, pairs, pixels_per_image, rng, n)
        groups = [[i] for i in range(C)] + [[a, b] for a, b in pairs]
    else:
        Ftr, Fva = _image_features(Xtr, tr_idx, pairs), _image_features(Xva, va_idx, pairs)
        Ltr, Lva = np.asarray(ytr)[tr_idx], np.asarray(yva)
        groups = [[i] for i in range(C)] * 4 + [[a, b] for a, b in pairs]

    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed,
                                class_weight="balanced_subsample" if kind != "multilabel" else None)
    rf.fit(Ftr, Ltr)
    if kind == "segmentation":
        pred = rf.predict(Fva).astype(np.int64)
        cm = np.zeros((n, n), dtype=np.int64)
        np.add.at(cm, (Lva.astype(np.int64), pred), 1)
        m = metrics_from(kind, n, Lva, cm, meta["class_names"])
        m["note"] = f"estimated from {len(Lva)} sampled valid pixels"
    elif kind == "multilabel":
        prob = np.stack([p[:, 1] if p.shape[1] > 1 else np.zeros(len(p)) for p in rf.predict_proba(Fva)], axis=1)
        m = metrics_from(kind, n, Lva, prob, meta["class_names"])
    else:
        prob = np.zeros((len(Fva), n))
        prob[:, rf.classes_] = rf.predict_proba(Fva)
        m = metrics_from(kind, n, Lva, prob, meta["class_names"])

    # importance per band: each feature's importance is split evenly over the bands it uses
    band_imp = np.zeros(C)
    for imp, g in zip(rf.feature_importances_, groups):
        band_imp[g] += imp / len(g)
    band_imp /= band_imp.sum()
    order = np.argsort(band_imp)[::-1]
    top_pairs = []
    if pairs:
        off = len(groups) - len(pairs)
        pi = np.argsort(rf.feature_importances_[off:])[::-1][:3]
        top_pairs = [f"ND({names[pairs[i][0]]}, {names[pairs[i][1]]})" for i in pi]

    result = {
        "task": task, "method": "random forest on spectral features (no spatial context)",
        "primary_metric": PRIMARY[kind], "val_score": m[PRIMARY[kind]], "val_metrics": m,
        "band_importance": {f"ch{i} {names[i]}": round(float(band_imp[i]), 3) for i in order},
        "top_normalized_differences": top_pairs,
        "train_rows": int(len(Ftr)), "seconds": round(time.time() - t0, 1),
        "how_to_use": ("This is the floor. A CNN far above it means spatial patterns matter (work on resolution, "
                       "backbone, augmentation). A CNN close to it means the signal is spectral (keep all bands, try "
                       "percentile normalisation, try channel subsets from band_importance)."),
    }
    (config.RUNS_DIR / task).mkdir(parents=True, exist_ok=True)
    (config.RUNS_DIR / task / "probe.json").write_text(json.dumps(result, indent=1))
    return result
