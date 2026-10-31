"""Write fake but contract-exact run folders, so the strategy side (stats, tournament, branches,
ensembles) can be built and tested on a laptop: no GPU, no data, no Bedrock.

Each fake model has a known "skill". Every validation image has a difficulty shared by all models,
so two models are right and wrong on mostly the same images, like real models. A model gets an image
(or a pixel) right with probability sigmoid(skill - difficulty), and its probabilities are built to
match. The answers are known in advance, so the stats can be checked against them:

- screen: every pool candidate once on train_fraction 0.3 (lower skill than on full data)
- race:   the top 3 families x seeds 0, 1, 2 on full data (seed jitter = run-to-run noise)
- branch A (spectral) from the best race run: a real gain, a gain inside the noise, a crashed run
- branch B (spatial) from another family: a real loss, a small gain

Tasks (sizes match the real val splits, so the noise is realistic):
    m-eurosat    classification, 10 classes, 1000 val images
    m-nz-cattle  segmentation, 2 classes, 66 val images (64 x 64 crops, real ones are larger)

runs/<task>/fake_truth.json lists every run's true skill and its role, so tests can check that
e.g. compare_runs accepts the real gain and calls the noisy one a tie.

    python tests/make_fake_runs.py DIR      # writes DIR/runs/...; use DIR as GOTRAIN_HOME
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

TASKS = {
    "m-eurosat": {"kind": "classification", "n_classes": 10, "n_val": 1000, "primary": "accuracy"},
    "m-nz-cattle": {"kind": "segmentation", "n_classes": 2, "n_val": 66, "size": 64, "primary": "miou"},
}

# Configs mirror train.CANDIDATE_POOL; skill is on a logit scale (2.0 is about 85% accuracy on eurosat).
PLANS = {
    "m-eurosat": {
        "screen": [("resnet18", "resnet", {"backbone": "resnet18"}, 1.4),
                   ("resnet50", "resnet", {"backbone": "resnet50"}, 1.5),
                   ("convnext_tiny", "convnext", {"backbone": "convnext_tiny", "img_size": 128}, 1.7),
                   ("efficientnet_b0", "efficientnet", {"backbone": "efficientnet_b0"}, 1.1),
                   ("ssl4eo_resnet18", "resnet-ssl4eo", {"backbone": "ssl4eo_resnet18"}, 1.9),
                   ("ssl4eo_resnet50", "resnet-ssl4eo", {"backbone": "ssl4eo_resnet50"}, 1.85)],
        "race": {"ssl4eo_resnet18": 2.3, "ssl4eo_resnet50": 2.25, "convnext_tiny": 2.0},
        "branch_a": ("ssl4eo_resnet18", "spectral: all 13 bands carry the signal, SSL4EO Sentinel-2 weights"),
        "branch_b": ("convnext_tiny", "spatial: shapes and texture carry the signal, upsampled RGB-like input"),
        "a_steps": [("real_gain", {"img_size": 128}, "upsample 64 -> 128 so the pretrained filters see larger shapes", +0.35),
                    ("noise", {"img_size": 128, "mixup": 0.2}, "mixup 0.2 against overfitting", +0.02),
                    ("failed", {"img_size": 128, "mixup": 0.2, "lr": 0.1}, "much larger lr to converge faster", None)],
        "b_steps": [("real_loss", {"channels": [3, 2, 1]}, "RGB only: test whether the extra bands matter", -0.45),
                    ("small_gain", {"tta": True}, "8-way test-time augmentation", +0.08)],
    },
    "m-nz-cattle": {
        "screen": [("unet_resnet34", "unet-resnet", {"seg_arch": "unet", "backbone": "resnet34"}, 1.0),
                   ("unet_effb4", "unet-efficientnet", {"seg_arch": "unet", "backbone": "efficientnet-b4"}, 1.15),
                   ("unet_convnext", "unet-convnext", {"seg_arch": "unet", "backbone": "tu-convnext_tiny"}, 1.05),
                   ("segformer_b0", "segformer", {"seg_arch": "segformer", "backbone": "mit_b0"}, 0.8),
                   ("segformer_b2", "segformer", {"seg_arch": "segformer", "backbone": "mit_b2"}, 1.2)],
        "race": {"segformer_b2": 1.6, "unet_effb4": 1.55, "unet_convnext": 1.4},
        "branch_a": ("segformer_b2", "spatial context: global attention finds cattle groups"),
        "branch_b": ("unet_effb4", "fine detail: cattle are a few pixels, keep full resolution"),
        "a_steps": [("real_gain", {"loss": "ce+dice"}, "ce+dice because cattle are a tiny share of pixels", +0.5),
                    ("noise", {"loss": "ce+dice", "augment": ["flip", "rot90", "noise"]}, "add noise augmentation", +0.03),
                    ("failed", {"loss": "ce+dice", "batch_size": 64}, "batch 64 for steadier gradients", None)],
        "b_steps": [("real_loss", {"img_size": 32}, "half resolution to train faster", -0.6),
                    ("small_gain", {"tta": True}, "8-way test-time augmentation", +0.1)],
    },
}

SEED_JITTER = 0.05      # skill change from the training seed alone


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fake_labels(task, rng):
    t = TASKS[task]
    if t["kind"] == "classification":
        return rng.integers(0, t["n_classes"], t["n_val"]).astype(np.int64)
    s = t["size"]
    y = np.zeros((t["n_val"], s, s), dtype=np.int16)
    for i in range(t["n_val"]):              # a few square "cattle" per image, a small share of pixels
        for _ in range(rng.integers(0, 6)):
            r, c = rng.integers(0, s - 4, 2)
            y[i, r:r + 4, c:c + 4] = 1
    return y


def fake_probs(y, n, skill, difficulty, rng):
    """Probabilities whose argmax is right with probability sigmoid(skill - difficulty)."""
    if y.ndim == 1:
        p_right = sigmoid(skill - difficulty)
    else:   # segmentation: background pixels are easier, so a rare class drives mIoU like in real data
        d = difficulty[:, None, None]
        p_right = np.where(y == 0, sigmoid(skill + 2.5 - d), sigmoid(skill - d))
    right = rng.random(y.shape) < p_right
    wrong = (y + rng.integers(1, n, y.shape)) % n
    pred = np.where(right, y, wrong)
    logits = rng.normal(0, 0.5, (n,) + y.shape)
    np.put_along_axis(logits, pred[None], np.take_along_axis(logits, pred[None], 0) + rng.uniform(1.5, 3.5, y.shape), 0)
    e = np.exp(logits - logits.max(0))
    probs = np.moveaxis(e / e.sum(0), 0, 1)          # (N, C) or (N, C, H, W)
    return probs.astype(np.float16)


def score(kind, n, y, probs):
    """Same numbers as train.metrics_from_probs, in numpy only."""
    pred = probs.astype(np.float32).argmax(1)
    if kind == "classification":
        acc = float((pred == y).mean())
        return {"accuracy": acc}
    t, p = y.reshape(-1).astype(np.int64), pred.reshape(-1)
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (t, p), 1)
    tp = np.diag(cm)
    iou = tp / np.maximum(cm.sum(0) + cm.sum(1) - tp, 1)
    present = cm.sum(1) > 0
    return {"miou": float(iou[present].mean()), "pixel_accuracy": float(tp.sum() / cm.sum())}


def write_run(registry, task, cfg, hypothesis, parent, skill, y, difficulty, rng, epochs):
    t = TASKS[task]
    rd = registry.new_run(task, cfg, hypothesis, parent)
    t0 = time.time()
    sec = 6.0 * cfg.get("train_fraction", 1.0)
    if skill is None:                               # a crashed run: no metrics, no val_preds
        (rd / "status.json").write_text(json.dumps({
            "state": "failed", "pid": None, "started": t0, "epoch": 2, "best_val": None,
            "error": "FloatingPointError: loss became NaN/inf: lower lr or check normalization", "finished": t0 + 2 * sec}))
        (rd / "train.log").write_text("epoch   1  train loss nan\nfailed: loss became NaN/inf\n")
        return rd
    probs = fake_probs(y, t["n_classes"], skill, difficulty, rng)
    m = score(t["kind"], t["n_classes"], y, probs)
    best = m[t["primary"]]
    best_ep = int(rng.integers(max(1, epochs - 4), epochs + 1))
    with open(rd / "history.jsonl", "w") as f:      # val rises to the best epoch, then wobbles below it
        for ep in range(1, epochs + 1):
            v = best * (1 - 0.5 * (1 - ep / best_ep) ** 2) if ep <= best_ep else best - abs(rng.normal(0, 0.005))
            f.write(json.dumps({"epoch": ep, "train_loss": round(1.5 / ep, 4), f"val_{t['primary']}": v,
                                "lr": 1e-3, "sec": sec}) + "\n")
    np.savez(rd / "val_preds.npz", probs=probs, y=y)
    bvm = {"epoch": best_ep, "primary_metric": t["primary"], "primary": best, **m}
    (rd / "best_val_metrics.json").write_text(json.dumps(bvm, indent=1))
    (rd / "metrics.json").write_text(json.dumps({
        "task": task, "primary_metric": t["primary"], "val_score": best, "best_epoch": best_ep, "epochs_run": epochs,
        "minutes": round(epochs * sec / 60, 2), "params_M": None, "notes": ["fake run (tests/make_fake_runs.py)"],
        "val_metrics": bvm}, indent=1))
    (rd / "status.json").write_text(json.dumps({
        "state": "done", "pid": None, "started": t0, "epoch": epochs, "best_val": best, "best_epoch": best_ep,
        "finished": t0 + epochs * sec}))
    (rd / "train.log").write_text(f"fake run, best val {t['primary']} {best:.4f} at epoch {best_ep}\n")
    return rd


def make_task(registry, task, seed=0):
    t, plan = TASKS[task], PLANS[task]
    rng = np.random.default_rng(seed)
    y = fake_labels(task, rng)
    difficulty = rng.normal(0, 1, len(y))           # shared by all models: their errors overlap
    truth = {"task": task, "kind": t["kind"], "n_classes": t["n_classes"], "primary": t["primary"], "runs": {}}

    def add(role, name, family, cfg, hyp, parent, skill, epochs, **extra):
        rd = write_run(registry, task, cfg, hyp, parent, skill, y, difficulty, rng, epochs)
        truth["runs"][rd.name] = {"role": role, "candidate": name, "family": family, "skill": skill,
                                  "parent": parent, **extra}
        return rd.name

    screen = {}
    for name, family, cfg, skill in plan["screen"]:
        screen[name] = (add("screen", name, family, {**cfg, "train_fraction": 0.3, "epochs": 5},
                            f"screen {name} on 30% of train", None, skill, 5), family, cfg)

    race = {}
    for name, skill in plan["race"].items():
        sid, family, cfg = screen[name]
        race[name] = [add("race", name, family, {**cfg, "seed": s}, f"race {name}, seed {s}", sid,
                          skill + rng.normal(0, SEED_JITTER), 15, seed=s, base_skill=skill) for s in range(3)]

    for branch, steps in (("a", plan["a_steps"]), ("b", plan["b_steps"])):
        name, assumption = plan[f"branch_{branch}"]
        family, cfg = screen[name][1], screen[name][2]
        parent = race[name][0]                      # the branch starts from the seed-0 race run
        skill = plan["race"][name]
        truth[f"branch_{branch}"] = {"root": parent, "assumption": assumption, "runs": []}
        for role, change, hyp, gain in steps:
            new_skill = None if gain is None else skill + gain
            rid = add(role, name, family, {**cfg, **change}, f"branch {branch.upper()}: {hyp}", parent, new_skill, 15,
                      branch=branch, gain=gain)
            truth[f"branch_{branch}"]["runs"].append(rid)
            if gain is not None and gain > 0.05:    # a branch builds on what it keeps
                parent, skill = rid, new_skill
    (registry.task_runs_dir(task) / "fake_truth.json").write_text(json.dumps(truth, indent=1))
    return truth


def main(home: Path, seed: int = 0):
    os.environ["GOTRAIN_HOME"] = str(home)          # before importing gotrain: config reads it at import
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from gotrain import registry
    out = {}
    for task in TASKS:
        if any(p.is_dir() for p in registry.task_runs_dir(task).iterdir()):
            raise SystemExit(f"{registry.task_runs_dir(task)} already has runs: use an empty GOTRAIN_HOME")
        out[task] = make_task(registry, task, seed)
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for task, truth in main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 0).items():
        print(f"{task}: {len(truth['runs'])} runs")
