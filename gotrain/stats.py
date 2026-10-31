"""Is a difference between runs real, or luck? Statistics on saved VALID predictions (val_preds.npz).

Val sets are small (m-nz-cattle: 66 images), so a score can move by a point from luck alone. Two
sources of luck, two tools:

- which images happen to be in val: the bootstrap redraws the val images (bootstrap_ci, paired_test);
- training randomness: the same config with another seed scores differently (group_score, noise_floor).

Every metric here is computed from per-image counts that add up (right/total, or tp/fp/fn per class),
so one bootstrap redraw is a weighted sum of those counts: fast even for segmentation, where the raw
probabilities can be GBs. The numbers match train.metrics_from_probs exactly.

This module only measures; strategy.py makes the keep / drop decisions.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import registry

PRIMARY = {"classification": "accuracy", "multilabel": "f1_micro", "segmentation": "miou"}
N_BOOT = 2000


@dataclass(frozen=True)
class Scorecard:
    """One run's VALID predictions reduced to additive per-image counts: units (N, k)."""
    kind: str
    n_classes: int
    units: np.ndarray
    y_hash: str       # two runs can only be compared on the same val images

    @property
    def n(self) -> int:
        return len(self.units)


# --------------------------------------------------------------------------- scorecards
def scorecard(probs, y) -> Scorecard:
    """From probabilities (N, C) or (N, C, H, W) and labels, as stored in val_preds.npz.
    Also works for an ensemble: pass the averaged probabilities."""
    y = np.asarray(y)
    n = probs.shape[1]
    y_hash = hashlib.sha1(np.ascontiguousarray(y).tobytes()).hexdigest()
    if probs.ndim == 4:
        units = np.zeros((len(probs), 3 * n))       # tp, fp, fn per class, per image
        for i in range(0, len(probs), 32):          # chunked: segmentation probabilities can be GBs
            pred = np.asarray(probs[i:i + 32]).argmax(1).reshape(len(probs[i:i + 32]), -1)
            t = y[i:i + 32].reshape(len(pred), -1).astype(np.int64)
            for j in range(len(pred)):
                ok = (t[j] >= 0) & (t[j] < n)       # same ignore rule as train.metrics_from_probs
                cm = np.bincount(t[j][ok] * n + pred[j][ok], minlength=n * n).reshape(n, n)
                tp = np.diag(cm)
                units[i + j] = np.concatenate([tp, cm.sum(0) - tp, cm.sum(1) - tp])
        return Scorecard("segmentation", n, units, y_hash)
    p = np.asarray(probs, dtype=np.float32)
    if y.ndim == 2:                                 # multilabel: 0/1 per class
        pred, t = p > 0.5, y > 0.5
        units = np.stack([(pred & t).sum(1), (pred & ~t).sum(1), (~pred & t).sum(1)], 1).astype(float)
        return Scorecard("multilabel", n, units, y_hash)
    right = (p.argmax(1) == y).astype(float)
    return Scorecard("classification", n, np.stack([right, np.ones_like(right)], 1), y_hash)


def load_preds(run_id: str):
    """(probs, y) of a finished run."""
    vp = np.load(registry.run_dir(run_id) / "val_preds.npz")
    return vp["probs"], vp["y"]


def load(run) -> Scorecard:
    """A run id, or a Scorecard (returned as is)."""
    if isinstance(run, Scorecard):
        return run
    p = registry.run_dir(run) / "val_preds.npz"
    if not p.exists():
        raise FileNotFoundError(f"{run} has no val_preds.npz (failed, still running, or trained before the contract)")
    return _load_cached(run, p.stat().st_mtime_ns)


@lru_cache(maxsize=128)
def _load_cached(run_id: str, mtime: int) -> Scorecard:   # mtime: a re-written file is re-read
    return scorecard(*load_preds(run_id))


# --------------------------------------------------------------------------- metrics from counts
def metric(kind: str, sums: np.ndarray):
    """Primary metric from summed units (..., k); vectorised over leading axes (bootstrap draws)."""
    if kind == "classification":
        return sums[..., 0] / np.maximum(sums[..., 1], 1)
    if kind == "multilabel":
        tp, fp, fn = sums[..., 0], sums[..., 1], sums[..., 2]
        return 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    n = sums.shape[-1] // 3
    tp, fp, fn = sums[..., :n], sums[..., n:2 * n], sums[..., 2 * n:]
    iou = tp / np.maximum(tp + fp + fn, 1)
    present = (tp + fn) > 0                         # classes absent from the labels are not averaged
    return (iou * present).sum(-1) / np.maximum(present.sum(-1), 1)


def score(run, idx=None) -> float:
    """Primary VALID score, optionally on a subset of val images (e.g. one half from halves())."""
    c = load(run)
    u = c.units if idx is None else c.units[idx]
    return float(metric(c.kind, u.sum(0)))


def _weights(n: int, n_boot: int, seed: int) -> np.ndarray:
    """Bootstrap draws as counts: row b says how often each image appears in redraw b."""
    return np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=n_boot).astype(float)


def _units(c: Scorecard, idx):
    return c.units if idx is None else c.units[idx]


# --------------------------------------------------------------------------- one run
def bootstrap_ci(run, idx=None, n_boot: int = N_BOOT, conf: float = 0.95, seed: int = 0) -> dict:
    """How sure is this score? Redraw the val images n_boot times and recompute it."""
    c = load(run)
    u = _units(c, idx)
    draws = metric(c.kind, _weights(len(u), n_boot, seed) @ u)
    a = (1 - conf) / 2
    return {"metric": PRIMARY[c.kind], "score": float(metric(c.kind, u.sum(0))),
            "low": float(np.quantile(draws, a)), "high": float(np.quantile(draws, 1 - a)),
            "se": float(draws.std(ddof=1)), "n_images": len(u), "conf": conf}


# --------------------------------------------------------------------------- two runs
def mcnemar_p(a_only: int, b_only: int) -> float:
    """Exact two-sided McNemar test: of the images exactly one run got right, is the split 50/50?"""
    k, n = min(a_only, b_only), a_only + b_only
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired_test(a, b, idx=None, n_boot: int = N_BOOT, conf: float = 0.95, seed: int = 0) -> dict:
    """Is run a really better than run b? Both answered the same val images, so compare them image
    by image: the same bootstrap redraws for both, and McNemar for classification.
    diff > 0 means a scores higher. p is two-sided: small p = the gap is unlikely to be luck."""
    ca, cb = load(a), load(b)
    if ca.y_hash != cb.y_hash or ca.kind != cb.kind:
        raise ValueError("the two runs were not scored on the same val images")
    ua, ub = _units(ca, idx), _units(cb, idx)
    w = _weights(len(ua), n_boot, seed)
    d = metric(ca.kind, w @ ua) - metric(ca.kind, w @ ub)
    diff = float(metric(ca.kind, ua.sum(0)) - metric(ca.kind, ub.sum(0)))
    al = (1 - conf) / 2
    out = {"metric": PRIMARY[ca.kind], "diff": diff,
           "low": float(np.quantile(d, al)), "high": float(np.quantile(d, 1 - al)), "se": float(d.std(ddof=1)),
           "n_images": len(ua), "conf": conf}
    if ca.kind == "classification":
        a_only = int(((ua[:, 0] == 1) & (ub[:, 0] == 0)).sum())
        b_only = int(((ua[:, 0] == 0) & (ub[:, 0] == 1)).sum())
        out.update(method="mcnemar", p=mcnemar_p(a_only, b_only), a_only_right=a_only, b_only_right=b_only)
    else:
        p = 2 * min((d <= 0).sum() + 1, (d >= 0).sum() + 1) / (n_boot + 1)
        out.update(method="paired bootstrap over images", p=float(min(1.0, p)))
    if diff == 0 and np.array_equal(ua, ub):
        out["p"] = 1.0
    return out


# --------------------------------------------------------------------------- repeated runs (seeds)
def group_score(runs) -> dict:
    """Mean primary score of the same config trained with different seeds, with its standard error."""
    s = np.array([score(r) for r in runs])
    sd = float(s.std(ddof=1)) if len(s) > 1 else None
    return {"mean": float(s.mean()), "sd": sd, "se": None if sd is None else sd / math.sqrt(len(s)),
            "n_runs": len(s), "scores": [float(v) for v in s]}


def noise_floor(groups) -> dict:
    """Training noise, pooled over groups of seed repeats ([[r1, r2, r3], [r4, r5], ...]).
    seed_sd: how much one run moves from the seed alone. diff_sd: the same for the difference of two
    single runs (sqrt 2 larger), the ruler a gain must clear before it counts."""
    ss, dof = 0.0, 0
    for g in groups:
        if len(g) > 1:
            s = np.array([score(r) for r in g])
            ss += ((s - s.mean()) ** 2).sum()
            dof += len(s) - 1
    if dof == 0:
        return {"seed_sd": None, "diff_sd": None, "dof": 0, "note": "no seed repeats yet: race with >= 2 seeds"}
    sd = math.sqrt(ss / dof)
    return {"seed_sd": sd, "diff_sd": sd * math.sqrt(2), "dof": dof}


# --------------------------------------------------------------------------- val halves
def halves(n: int, seed: int = 0):
    """A fixed split of the n val images into half A (choose on it) and half B (confirm on it)."""
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
