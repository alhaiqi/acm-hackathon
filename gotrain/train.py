"""One training run: config in, checkpoint + validation metrics out.

Runs as its own process (`python -m gotrain.train --run-dir RUN`) so the agent can launch
it in the background, poll it, and kill it, and so a crash never takes the agent down.

Rules baked in:
- trains on TRAIN, selects the epoch by VALID score, never opens TEST;
- hard wall-clock limit (`time_limit_min`) so the agent can plan a budget;
- every epoch is appended to history.jsonl, so every score has a trail.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config, data

DEFAULTS = {
    "backbone": "resnet18",          # timm name, ssl4eo_resnet18/50 (classification) or smp encoder name (segmentation)
    "pretrained": True,              # ImageNet (or SSL4EO) weights; first conv is adapted to any number of bands
    "seg_arch": "unet",              # unet | unetplusplus | fpn | deeplabv3plus | linknet | pspnet | segformer
    "channel_adapter": False,        # segmentation: learned 1x1 conv bands -> 3 in front of an RGB-only encoder (set automatically if needed)
    "custom_model": None,            # name of a file written with the write_custom_model tool
    "channels": None,                # list of channel indices to keep, None = all
    "img_size": None,                # resize inputs to this square size (None = native)
    "normalization": "zscore",       # zscore | percentile (clip to 1st-99th pct of train, then scale)
    "epochs": 30,
    "batch_size": 64,
    "lr": 1e-3,                      # head / decoder learning rate
    "backbone_lr_mult": 0.1,         # pretrained encoder learns slower
    "weight_decay": 0.05,
    "warmup_epochs": 1,
    "freeze_backbone_epochs": 0,
    "loss": "ce",                    # ce | focal | ce+dice (segmentation)
    "label_smoothing": 0.0,
    "class_weights": "none",         # none | balanced
    "augment": ["flip", "rot90"],    # flip, rot90, scale_jitter, noise
    "mixup": 0.0,                    # classification only: Beta(a, a) mixing, 0 = off
    "drop_rate": 0.0,
    "tta": False,                    # average predictions over the 8 flips/rotations at evaluation (3 flips if not square)
    "patience": 8,                   # epochs without a better VALID score before stopping
    "time_limit_min": 20.0,          # hard wall-clock limit for the run
    "train_fraction": 1.0,           # < 1 to try ideas quickly on a subset
    "seed": 0,
    "num_workers": 4,
    "amp": True,
}

PRIMARY = {"classification": "accuracy", "multilabel": "f1_micro", "segmentation": "miou"}

# The curated candidate pool for the model tournament (strategy.propose_candidates reads it).
# Each entry is a config fragment plus facts the strategy uses to filter (params, what data it suits).
CANDIDATE_POOL = {
    "classification": [
        {"config": {"backbone": "resnet18"}, "family": "resnet", "weights": "imagenet", "params_M": 11.7,
         "suits": "any; fast baseline"},
        {"config": {"backbone": "resnet50"}, "family": "resnet", "weights": "imagenet", "params_M": 25.6,
         "suits": "any; more capacity"},
        {"config": {"backbone": "convnext_tiny", "img_size": 128}, "family": "convnext", "weights": "imagenet",
         "params_M": 28.6, "suits": "any; strong modern CNN, wants img_size >= 128"},
        {"config": {"backbone": "efficientnet_b0"}, "family": "efficientnet", "weights": "imagenet", "params_M": 5.3,
         "suits": "small datasets; cheap"},
        {"config": {"backbone": "ssl4eo_resnet18"}, "family": "resnet-ssl4eo", "weights": "SSL4EO-S12 S2 MoCo",
         "params_M": 11.7, "suits": "Sentinel-2 13 bands; falls back to ImageNet if torchgeo/weights missing"},
        {"config": {"backbone": "ssl4eo_resnet50"}, "family": "resnet-ssl4eo", "weights": "SSL4EO-S12 S2 MoCo",
         "params_M": 25.6, "suits": "Sentinel-2 13 bands; falls back to ImageNet if torchgeo/weights missing"},
    ],
    "segmentation": [
        {"config": {"seg_arch": "unet", "backbone": "resnet34"}, "family": "unet-resnet", "weights": "imagenet",
         "params_M": 24.4, "suits": "any; reliable baseline"},
        {"config": {"seg_arch": "unet", "backbone": "efficientnet-b4"}, "family": "unet-efficientnet",
         "weights": "imagenet", "params_M": 20.2, "suits": "small objects, high resolution"},
        {"config": {"seg_arch": "unet", "backbone": "tu-convnext_tiny"}, "family": "unet-convnext",
         "weights": "imagenet", "params_M": 32.0, "suits": "any; strong encoder"},
        {"config": {"seg_arch": "segformer", "backbone": "mit_b0"}, "family": "segformer", "weights": "imagenet",
         "params_M": 3.7, "suits": "fast transformer; RGB encoder (adapter added for >3 bands)"},
        {"config": {"seg_arch": "segformer", "backbone": "mit_b2"}, "family": "segformer", "weights": "imagenet",
         "params_M": 24.7, "suits": "transformer, global context; RGB encoder (adapter added for >3 bands)"},
    ],
}
CANDIDATE_POOL["multilabel"] = CANDIDATE_POOL["classification"]

SSL4EO = {"ssl4eo_resnet18": ("resnet18", "ResNet18_Weights"), "ssl4eo_resnet50": ("resnet50", "ResNet50_Weights")}


# --------------------------------------------------------------------------- model
class PadTo32(nn.Module):
    """Segmentation encoders downsample by 32: pad input up, crop logits back."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        h, w = x.shape[-2:]
        H, W = math.ceil(h / 32) * 32, math.ceil(w / 32) * 32
        if (H, W) != (h, w):
            x = F.pad(x, (0, W - w, 0, H - h), mode="reflect" if min(h, w) > 1 else "constant")
        return self.net(x)[..., :h, :w]


def adapt_first_conv(conv: nn.Conv2d, in_ch: int, keep=None) -> nn.Conv2d:
    """Same conv for `in_ch` input bands. keep = indices of the pretrained bands to reuse (e.g. a band
    subset of the same sensor); otherwise every new band starts from the mean of the pretrained filters."""
    new = nn.Conv2d(in_ch, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
    w = conv.weight.data
    with torch.no_grad():
        if keep is not None and len(keep) == in_ch and max(keep) < w.shape[1]:
            new.weight.copy_(w[:, list(keep)])
        else:
            new.weight.copy_(w.mean(1, keepdim=True).repeat(1, in_ch, 1, 1) * w.shape[1] / in_ch)
        if conv.bias is not None:
            new.bias.copy_(conv.bias.data)
    return new


def _ssl4eo_model(cfg: dict, in_ch: int, n: int, notes: list):
    """timm ResNet with torchgeo SSL4EO-S12 Sentinel-2 (13 bands, MoCo) weights; ImageNet if unavailable.
    The checkpoint stores the adapted weights, so pretrained=False (finalize) rebuilds the same shapes."""
    import timm

    arch, wname = SSL4EO[cfg["backbone"]]
    if not cfg["pretrained"]:
        return timm.create_model(arch, pretrained=False, in_chans=in_ch, num_classes=n, drop_rate=cfg["drop_rate"])
    try:
        import torchgeo.models as tgm

        w = getattr(tgm, wname).SENTINEL2_ALL_MOCO
        model = getattr(tgm, arch)(weights=w)
        model.reset_classifier(n)
        model.drop_rate = cfg["drop_rate"]
        src = int(w.meta.get("in_chans", 13))
        if in_ch != src:
            model.conv1 = adapt_first_conv(model.conv1, in_ch, cfg["channels"])
            notes.append(f"SSL4EO first conv adapted {src} -> {in_ch} bands")
        notes.append("SSL4EO-S12 Sentinel-2 MoCo weights (pretrained on S2 L1C; our z-score normalisation differs, fine-tuning absorbs it)")
        return model
    except Exception as e:
        notes.append(f"SSL4EO weights unavailable ({type(e).__name__}: {str(e)[:120]}); using ImageNet {arch}")
        try:
            return timm.create_model(arch, pretrained=True, in_chans=in_ch, num_classes=n, drop_rate=cfg["drop_rate"])
        except Exception as e2:
            notes.append(f"ImageNet weights unavailable too ({type(e2).__name__}); trained from scratch")
            return timm.create_model(arch, pretrained=False, in_chans=in_ch, num_classes=n, drop_rate=cfg["drop_rate"])


def build_model(meta: dict, cfg: dict, in_ch: int):
    """Returns (model, list of encoder parameters, notes).

    May set cfg["channel_adapter"] = True (segmentation encoder that only accepts 3 bands); cfg is saved
    in the checkpoint, so finalize rebuilds the same architecture."""
    kind, n = meta["kind"], meta["n_classes"]
    notes = []
    if cfg.get("custom_model"):
        import importlib.util

        path = config.CUSTOM_DIR / f"{cfg['custom_model']}.py"
        spec = importlib.util.spec_from_file_location(cfg["custom_model"], path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = mod.build_model(in_ch, n, kind, cfg)
        enc = list(model.encoder.parameters()) if hasattr(model, "encoder") else []
        return model, enc, [f"custom model {path.name}"]

    if kind == "segmentation":
        import segmentation_models_pytorch as smp

        def make(weights, ch):
            return smp.create_model(cfg["seg_arch"], encoder_name=cfg["backbone"], encoder_weights=weights,
                                    in_channels=ch, classes=n)

        def with_adapter(net):
            # learned bands -> 3 mixing in front of an encoder that only takes RGB (e.g. mit_* in some smp versions)
            return nn.Sequential(nn.Conv2d(in_ch, 3, 1), net)

        weights = "imagenet" if cfg["pretrained"] else None
        if cfg["channel_adapter"]:
            net = make(weights, 3)
            return PadTo32(with_adapter(net)), list(net.encoder.parameters()), [f"channel adapter {in_ch} -> 3"]
        try:
            net = make(weights, in_ch)
        except Exception as e:
            if in_ch != 3:
                try:   # encoder may refuse in_channels != 3: keep its pretrained weights behind an adapter
                    net = make(weights, 3)
                    cfg["channel_adapter"] = True
                    notes.append(f"{cfg['backbone']} refused {in_ch} input bands ({type(e).__name__}); "
                                 f"added a learned 1x1 adapter {in_ch} -> 3")
                    return PadTo32(with_adapter(net)), list(net.encoder.parameters()), notes
                except Exception:
                    pass
            notes.append(f"pretrained weights unavailable ({type(e).__name__}: {str(e)[:120]}); trained from scratch")
            net = make(None, in_ch)
        return PadTo32(net), list(net.encoder.parameters()), notes

    if cfg["backbone"] in SSL4EO:
        model = _ssl4eo_model(cfg, in_ch, n, notes)
        head = {id(p) for p in model.get_classifier().parameters()}
        return model, [p for p in model.parameters() if id(p) not in head], notes

    import timm

    def make(pre):
        return timm.create_model(cfg["backbone"], pretrained=pre, in_chans=in_ch, num_classes=n,
                                 drop_rate=cfg["drop_rate"])
    try:
        model = make(cfg["pretrained"])
    except Exception as e:
        notes.append(f"pretrained weights unavailable ({type(e).__name__}: {str(e)[:120]}); trained from scratch")
        model = make(False)
    head = {id(p) for p in model.get_classifier().parameters()}
    enc = [p for p in model.parameters() if id(p) not in head]
    return model, enc, notes


# --------------------------------------------------------------------------- data on GPU
class ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, idx, channels):
        self.X, self.y, self.idx, self.ch = X, y, idx, channels

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x = np.asarray(self.X[j], dtype=np.float32)
        if self.ch is not None:
            x = x[self.ch]
        return torch.from_numpy(x), torch.as_tensor(self.y[j])


class Normalizer:
    def __init__(self, stats, cfg, device):
        ch = cfg["channels"]
        pick = (lambda v: np.asarray(v)[ch]) if ch is not None else np.asarray
        t = lambda v: torch.tensor(pick(v), dtype=torch.float32, device=device).view(1, -1, 1, 1)
        self.mode = cfg["normalization"]
        self.mean, self.std, self.lo, self.hi = t(stats["mean"]), t(stats["std"]), t(stats["p01"]), t(stats["p99"])

    def __call__(self, x):
        if self.mode == "percentile":
            x = torch.minimum(torch.maximum(x, self.lo), self.hi)
            return (x - self.lo) / (self.hi - self.lo + 1e-6) * 2 - 1
        return (x - self.mean) / self.std


def augment(x, y, cfg, seg):
    aug = cfg["augment"] or []
    if "flip" in aug:
        if torch.rand(1).item() < 0.5:
            x = x.flip(-1); y = y.flip(-1) if seg else y
        if torch.rand(1).item() < 0.5:
            x = x.flip(-2); y = y.flip(-2) if seg else y
    if "rot90" in aug and x.shape[-1] == x.shape[-2]:
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            x = torch.rot90(x, k, (-2, -1)); y = torch.rot90(y, k, (-2, -1)) if seg else y
    if "scale_jitter" in aug:   # per-sample, per-channel brightness change (sensor/atmosphere variation)
        x = x * (1 + 0.1 * torch.randn(x.shape[0], x.shape[1], 1, 1, device=x.device))
    if "noise" in aug:
        x = x + 0.05 * torch.randn_like(x)
    return x, y


def resize(x, y, size, seg):
    if not size or x.shape[-1] == size and x.shape[-2] == size:
        return x, y
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    if seg and y is not None:
        y = F.interpolate(y[:, None].float(), size=(size, size), mode="nearest")[:, 0].long()
    return x, y


# --------------------------------------------------------------------------- losses & metrics
def make_loss(meta, cfg, y_train, device):
    kind, n = meta["kind"], meta["n_classes"]
    w = None
    if cfg["class_weights"] == "balanced":
        if kind == "classification":
            cnt = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=n).astype(np.float64)
        elif kind == "multilabel":
            cnt = np.asarray(y_train).sum(0).astype(np.float64)
        else:
            yy = np.asarray(y_train[: min(len(y_train), 500)]).reshape(-1)
            yy = yy[(yy >= 0) & (yy < n)]
            cnt = np.bincount(yy, minlength=n).astype(np.float64)
        w = cnt.sum() / (n * np.maximum(cnt, 1))
        w = torch.tensor(np.clip(w, 0.1, 10), dtype=torch.float32, device=device)

    if kind == "multilabel":
        pos_w = w if w is not None else None
        return lambda logits, y: F.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=pos_w)

    ls = cfg["label_smoothing"]

    def ce(logits, y):
        return F.cross_entropy(logits, y, weight=w, label_smoothing=ls, ignore_index=-100)

    if cfg["loss"] == "focal":
        def focal(logits, y):
            lp = F.log_softmax(logits, 1)
            nll = F.nll_loss(lp, y, weight=w, reduction="none", ignore_index=-100)
            p = torch.exp(-F.nll_loss(lp, y, reduction="none", ignore_index=-100))
            m = (y != -100).float()
            return ((1 - p) ** 2 * nll * m).sum() / m.sum().clamp(min=1)
        return focal
    if cfg["loss"] == "ce+dice" and kind == "segmentation":
        def ce_dice(logits, y):
            valid = (y != -100)
            prob = logits.softmax(1)
            oh = F.one_hot(y.clamp(min=0), n).permute(0, 3, 1, 2).float() * valid[:, None]
            inter = (prob * oh).sum((0, 2, 3)); den = (prob * valid[:, None]).sum((0, 2, 3)) + oh.sum((0, 2, 3))
            dice = 1 - ((2 * inter + 1) / (den + 1)).mean()
            return ce(logits, y) + dice
        return ce_dice
    return ce


def metrics_from(kind, n, y_true, out, names):
    """y_true numpy, out numpy: probabilities (cls/ml) or confusion matrix (seg)."""
    names = names or [str(i) for i in range(n)]
    if kind == "classification":
        pred = out.argmax(1)
        cm = np.zeros((n, n), dtype=np.int64)
        np.add.at(cm, (y_true, pred), 1)
        tp = np.diag(cm); prec = tp / np.maximum(cm.sum(0), 1); rec = tp / np.maximum(cm.sum(1), 1)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        off = cm.copy(); np.fill_diagonal(off, 0)
        top = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1][:3], off.shape))[0]
        return {"accuracy": float(tp.sum() / cm.sum()), "f1_macro": float(f1.mean()),
                "per_class_recall": {names[i]: round(float(rec[i]), 3) for i in range(n)},
                "top_confusions": [f"{names[a]} -> {names[b]}: {int(off[a, b])}" for a, b in top if off[a, b] > 0]}
    if kind == "multilabel":
        pred = out > 0.5
        t = y_true > 0.5
        tp = (pred & t).sum(); fp = (pred & ~t).sum(); fn = (~pred & t).sum()
        f1c = 2 * (pred & t).sum(0) / np.maximum(2 * (pred & t).sum(0) + (pred & ~t).sum(0) + (~pred & t).sum(0), 1)
        try:
            from sklearn.metrics import average_precision_score
            mAP = float(average_precision_score(t, out, average="macro"))
        except Exception:
            mAP = float("nan")
        return {"f1_micro": float(2 * tp / max(2 * tp + fp + fn, 1)), "f1_macro": float(f1c.mean()), "mAP_macro": mAP,
                "per_class_f1": {names[i]: round(float(f1c[i]), 3) for i in range(n)}}
    cm = out
    tp = np.diag(cm); denom = cm.sum(0) + cm.sum(1) - tp
    present = cm.sum(1) > 0
    iou = tp / np.maximum(denom, 1)
    return {"miou": float(iou[present].mean()) if present.any() else 0.0,
            "pixel_accuracy": float(tp.sum() / max(cm.sum(), 1)),
            "per_class_iou": {names[i]: round(float(iou[i]), 3) for i in range(n) if present[i]}}


def metrics_from_probs(meta: dict, y_true, probs) -> dict:
    """Metrics from saved probabilities (val_preds.npz, or an ensemble average of several runs).
    probs: (N, C) for classification/multilabel, (N, C, H, W) for segmentation; y_true as stored."""
    kind, n = meta["kind"], meta["n_classes"]
    if kind != "segmentation":
        return metrics_from(kind, n, np.asarray(y_true), np.asarray(probs, dtype=np.float32), meta["class_names"])
    cm = np.zeros((n, n), dtype=np.int64)
    for i in range(0, len(probs), 32):      # chunked: segmentation probabilities can be GBs
        pred = np.asarray(probs[i:i + 32]).argmax(1).reshape(-1)
        t = np.asarray(y_true[i:i + 32]).reshape(-1).astype(np.int64)
        ok = (t >= 0) & (t < n)
        np.add.at(cm, (t[ok], pred[ok]), 1)
    return metrics_from(kind, n, None, cm, meta["class_names"])


def tta_views(on: bool, square: bool):
    """(forward, inverse) pairs. Satellite images have no 'up', so all 8 symmetries of the square (D4) are valid."""
    ident = (lambda t: t, lambda t: t)
    if not on:
        return [ident]
    if not square:   # rot90 would change the shape: flips only
        return [ident, (lambda t: t.flip(-1), lambda t: t.flip(-1)), (lambda t: t.flip(-2), lambda t: t.flip(-2))]
    views = []
    for f in (False, True):
        for k in range(4):
            views.append((lambda t, k=k, f=f: torch.rot90(t.flip(-1) if f else t, k, (-2, -1)),
                          lambda t, k=k, f=f: (lambda u: u.flip(-1) if f else u)(torch.rot90(t, -k, (-2, -1)))))
    return views


@torch.no_grad()
def evaluate(model, X, y, meta, cfg, norm, device, batch=None, return_probs=False):
    """Returns (metrics, probs). probs is (N, C) for classification/multilabel always, and
    (N, C, H, W) float16 for segmentation only if return_probs (otherwise None: it can be GBs)."""
    kind, n = meta["kind"], meta["n_classes"]
    seg = kind == "segmentation"
    model.eval()
    batch = batch or cfg["batch_size"] * 2
    probs, cm = [], np.zeros((n, n), dtype=np.int64)
    ch = cfg["channels"]
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(np.asarray(X[i:i + batch], dtype=np.float32))
        if ch is not None:
            xb = xb[:, ch]
        xb = norm(xb.to(device))
        H, W = xb.shape[-2:]
        xr, _ = resize(xb, None, cfg["img_size"], seg)
        views = tta_views(cfg["tta"], xr.shape[-1] == xr.shape[-2])
        acc = 0
        for fwd, back in views:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cfg["amp"] and device.type == "cuda"):
                lo = model(fwd(xr)).float()
            if seg:
                lo = back(lo)
                acc = acc + lo.softmax(1)
            else:
                acc = acc + (lo.sigmoid() if kind == "multilabel" else lo.softmax(1))
        acc = acc / len(views)
        if seg:
            if acc.shape[-2:] != (H, W):
                acc = F.interpolate(acc, size=(H, W), mode="bilinear", align_corners=False)
            pred = acc.argmax(1).cpu().numpy().reshape(-1)
            t = np.asarray(y[i:i + batch]).reshape(-1).astype(np.int64)
            ok = (t >= 0) & (t < n)
            np.add.at(cm, (t[ok], pred[ok]), 1)
            if return_probs:
                probs.append(acc.half().cpu().numpy())
        else:
            probs.append(acc.cpu().numpy())
    out = cm if seg else np.concatenate(probs)
    p = (np.concatenate(probs) if return_probs else None) if seg else out
    return metrics_from(kind, n, np.asarray(y), out, meta["class_names"]), p


def save_val_preds(run_dir: Path, probs, y, kind: str) -> None:
    """Contract file for the strategy side: probs (N,C) or (N,C,H,W) float16 + labels. Atomic (a killed
    run never leaves a half-written file)."""
    y = np.asarray(y)
    y = y.astype(np.int16) if kind == "segmentation" else y
    tmp = run_dir / "val_preds.tmp.npz"
    np.savez(tmp, probs=np.asarray(probs).astype(np.float16), y=y)
    os.replace(tmp, run_dir / "val_preds.npz")


# --------------------------------------------------------------------------- run
def run(run_dir: Path):
    run_dir = Path(run_dir)
    spec = json.loads((run_dir / "config.json").read_text())
    task, cfg = spec["task"], {**DEFAULTS, **spec["config"]}
    status = {"state": "running", "pid": os.getpid(), "started": time.time(), "epoch": 0, "best_val": None}

    def put_status(**kw):
        status.update(kw)
        (run_dir / "status.json").write_text(json.dumps(status))

    put_status()
    log = open(run_dir / "train.log", "a", buffering=1)
    say = lambda s: (log.write(s + "\n"), print(s, flush=True))
    try:
        torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        meta = data.task_meta(task)
        kind, seg = meta["kind"], meta["kind"] == "segmentation"
        Xtr, ytr = data.load_split(task, "train")
        Xva, yva = data.load_split(task, "valid")
        stats = data.channel_stats(task)
        if seg:  # labels outside [0, n) are "no data": ignored by the loss and the metric
            fix = lambda y: np.where((y >= 0) & (y < meta["n_classes"]), y, -100).astype(np.int64)
            ytr, yva = fix(ytr), fix(yva)
        in_ch = Xtr.shape[1] if cfg["channels"] is None else len(cfg["channels"])
        model, enc, notes = build_model(meta, cfg, in_ch)
        model = model.to(device)
        for n_ in notes:
            say("NOTE " + n_)
        n_params = sum(p.numel() for p in model.parameters())
        say(f"task {task} ({kind}, {meta['n_classes']} classes) | train {len(Xtr)} valid {len(Xva)} | "
            f"input {tuple(Xtr.shape[1:])} -> {in_ch} ch | model {cfg['custom_model'] or cfg['backbone']} "
            f"{n_params / 1e6:.1f}M params | device {device}")
        say("config " + json.dumps(cfg))

        rng = np.random.default_rng(cfg["seed"])
        idx = np.arange(len(Xtr))
        if cfg["train_fraction"] < 1:
            idx = np.sort(rng.choice(idx, size=max(8, int(len(idx) * cfg["train_fraction"])), replace=False))
        dl = torch.utils.data.DataLoader(ArrayDataset(Xtr, ytr, idx, cfg["channels"]), batch_size=cfg["batch_size"],
                                         shuffle=True, drop_last=len(idx) > cfg["batch_size"],
                                         num_workers=cfg["num_workers"], pin_memory=device.type == "cuda",
                                         persistent_workers=cfg["num_workers"] > 0)
        norm = Normalizer(stats, cfg, device)
        loss_fn = make_loss(meta, cfg, ytr[idx], device)
        enc_ids = {id(p) for p in enc}
        groups = [{"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": cfg["lr"]}]
        if enc:
            groups.append({"params": enc, "lr": cfg["lr"] * cfg["backbone_lr_mult"]})
        opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
        base = [g["lr"] for g in opt.param_groups]
        steps = cfg["epochs"] * max(1, len(dl)); warm = cfg["warmup_epochs"] * max(1, len(dl))
        scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and device.type == "cuda")
        primary = PRIMARY[kind]
        best, best_ep, bad, step = -1.0, 0, 0, 0
        t0 = time.time()
        for ep in range(1, cfg["epochs"] + 1):
            te = time.time()
            frozen = ep <= cfg["freeze_backbone_epochs"]
            for p in enc:
                p.requires_grad_(not frozen)
            model.train(); tot, cnt = 0.0, 0
            for xb, yb in dl:
                xb, yb = norm(xb.to(device, non_blocking=True)), yb.to(device, non_blocking=True)
                xb, yb = augment(xb, yb, cfg, seg)
                xb, yb = resize(xb, yb, cfg["img_size"], seg)
                lr_f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
                for g, b in zip(opt.param_groups, base):
                    g["lr"] = b * lr_f
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cfg["amp"] and device.type == "cuda"):
                    if cfg["mixup"] > 0 and kind == "classification":
                        lam = float(np.random.beta(cfg["mixup"], cfg["mixup"]))
                        perm = torch.randperm(xb.shape[0], device=device)
                        out = model(lam * xb + (1 - lam) * xb[perm])
                        loss = lam * loss_fn(out.float(), yb) + (1 - lam) * loss_fn(out.float(), yb[perm])
                    else:
                        loss = loss_fn(model(xb).float(), yb)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update()
                tot += loss.item() * xb.shape[0]; cnt += xb.shape[0]; step += 1
                if not math.isfinite(loss.item()):
                    raise FloatingPointError("loss became NaN/inf: lower lr or check normalization")
            m, probs = evaluate(model, Xva, yva, meta, cfg, norm, device, return_probs=True)
            score = m[primary]
            mark = ""
            if score > best:
                best, best_ep, bad, mark = score, ep, 0, " *best"
                torch.save({"state_dict": model.state_dict(), "config": cfg, "task": task, "meta": meta,
                            "in_ch": in_ch, "epoch": ep, "val_metrics": m}, run_dir / "model.pt")
                save_val_preds(run_dir, probs, yva, kind)
                (run_dir / "best_val_metrics.json").write_text(json.dumps(
                    {"epoch": ep, "primary_metric": primary, "primary": score, **m}, indent=1))
            else:
                bad += 1
            del probs
            rec = {"epoch": ep, "train_loss": tot / max(cnt, 1), f"val_{primary}": score,
                   "lr": opt.param_groups[0]["lr"], "sec": round(time.time() - te, 1)}
            with open(run_dir / "history.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            say(f"epoch {ep:>3}  train loss {rec['train_loss']:.4f}  val {primary} {score:.4f}  ({rec['sec']}s){mark}")
            put_status(epoch=ep, best_val=best, best_epoch=best_ep)
            if bad >= cfg["patience"]:
                say(f"early stop: {cfg['patience']} epochs without a better val {primary}")
                break
            if (time.time() - t0) / 60 > cfg["time_limit_min"]:
                say(f"time limit {cfg['time_limit_min']} min reached")
                break
        result = {"task": task, "primary_metric": primary, "val_score": best, "best_epoch": best_ep,
                  "epochs_run": ep, "minutes": round((time.time() - t0) / 60, 2), "params_M": round(n_params / 1e6, 2),
                  "notes": notes,
                  "val_metrics": json.loads((run_dir / "best_val_metrics.json").read_text())}
        (run_dir / "metrics.json").write_text(json.dumps(result, indent=1))
        put_status(state="done", finished=time.time())
        say(f"DONE best val {primary} {best:.4f} at epoch {best_ep}")
    except Exception as e:
        tb = traceback.format_exc()
        say("FAILED\n" + tb)
        put_status(state="failed", error=f"{type(e).__name__}: {e}", traceback=tb[-3000:], finished=time.time())
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train one GoTrain run (VALID only).")
    ap.add_argument("--run-dir", help="existing run folder with config.json (what the agent uses)")
    ap.add_argument("--task", help="for humans: create a new run for this task")
    ap.add_argument("--config", default="{}", help="JSON overrides of DEFAULTS")
    a = ap.parse_args()
    if a.run_dir:
        run(Path(a.run_dir))
    else:
        from .registry import new_run
        rd = new_run(a.task, json.loads(a.config), hypothesis="manual run", parent=None)
        print("run dir:", rd)
        run(rd)
