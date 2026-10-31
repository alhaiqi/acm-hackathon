"""Data Advisor: should the next unit of effort go into the MODEL or into the DATA? (classification tasks)

Three measurements with the task's best existing config (read-only from the leaderboard):
  learning curve   train on nested 10/25/50/100% of TRAIN with the same number of optimizer steps, score on
                   VALID with bootstrap CIs, fit error = a * n^-b, and extrapolate the gain from 2x labels with a
                   band from paired bootstrap over val images (refit per draw).
  label doctor     2-fold out-of-fold predictions on TRAIN: images whose given label gets a low probability while
                   another class is confident. Checked by injecting 2% random label flips into the training copy and
                   measuring how many the doctor recovers (recall@K). Gallery: thumbnail, given vs predicted, confidence.
  labeling queue   a model trained on half of TRAIN ranks the other half by margin (top-1 minus top-2 probability):
                   the most uncertain images to label next. Checked by retraining on half + top-N vs half + random-N.

Rules: never reads TEST; never writes runs/ or results/ (output goes to GOTRAIN_HOME/advisor/<task>/);
reuses train.py's building blocks without changing it. Run with scripts/run_advisor.py.
"""
from __future__ import annotations

import base64
import io
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn

from . import config, data, registry, stats
from .train import (DEFAULTS, ArrayDataset, Normalizer, augment, build_model, evaluate, make_loss,
                    resize)

FRACTIONS = (0.10, 0.25, 0.50, 1.00)
INJECT = 0.02          # fraction of TRAIN labels flipped to test the label doctor
QUEUE_N = 100          # images in the labeling queue (and in its check)


def out_dir(task: str):
    d = config.HOME / "advisor" / task
    d.mkdir(parents=True, exist_ok=True)
    return d


def best_config(task: str) -> tuple[str, dict]:
    """The best finished run's config overrides (read-only)."""
    best = registry.best_run(task)
    if best is None:
        raise RuntimeError(f"{task} has no finished run to take a config from")
    cfg = {k: v for k, v in best["config_changes"].items() if k not in ("time_limit_min", "train_fraction")}
    return best["run_id"], cfg


# --------------------------------------------------------------------------- a small trainer (train.py parts)
def fit(meta, cfg, X, y, idx, stats_, device, steps=None, seed=0):
    """Train on X[idx] with labels y (may be a modified copy). steps: total optimizer steps (equal-compute curves).
    No validation-based epoch selection: the last epoch is used, so VALID stays a clean measurement."""
    cfg = {**DEFAULTS, **cfg, "seed": seed}
    torch.manual_seed(seed); np.random.seed(seed)
    in_ch = X.shape[1] if cfg["channels"] is None else len(cfg["channels"])
    model, enc, _ = build_model(meta, cfg, in_ch)
    model.to(device)
    bs = min(cfg["batch_size"], len(idx))
    dl = torch.utils.data.DataLoader(ArrayDataset(X, y, np.sort(idx), cfg["channels"]), batch_size=bs, shuffle=True,
                                     drop_last=len(idx) > bs, num_workers=0)
    norm = Normalizer(stats_, cfg, device)
    loss_fn = make_loss(meta, cfg, y[np.sort(idx)], device)
    enc_ids = {id(p) for p in enc}
    groups = [{"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": cfg["lr"]}]
    if enc:
        groups.append({"params": enc, "lr": cfg["lr"] * cfg["backbone_lr_mult"]})
    opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    base = [g["lr"] for g in opt.param_groups]
    per_epoch = max(1, len(dl))
    epochs = max(1, math.ceil(steps / per_epoch)) if steps else cfg["epochs"]
    total, warm = epochs * per_epoch, cfg["warmup_epochs"] * per_epoch
    amp = cfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    step = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = norm(xb.to(device)), yb.to(device)
            xb, yb = augment(xb, yb, cfg, False)
            xb, yb = resize(xb, yb, cfg["img_size"], False)
            f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * f
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = loss_fn(model(xb).float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            step += 1
    return model, norm, cfg, step


def predict(model, norm, cfg, meta, X, y, idx, device):
    idx = np.sort(np.asarray(idx))
    m, probs = evaluate(model, np.asarray(X[idx]), np.asarray(y)[idx], meta, cfg, norm, device)
    return m, probs, idx


# --------------------------------------------------------------------------- (a) learning curve
def fit_power_law(n, err):
    """err = a * n^-b in log-log space. Returns (a, b, r2)."""
    lx, ly = np.log(n), np.log(np.maximum(err, 1e-6))
    b, la = np.polyfit(lx, ly, 1)
    pred = la + b * lx
    r2 = 1 - ((ly - pred) ** 2).sum() / max(((ly - ly.mean()) ** 2).sum(), 1e-12)
    return float(np.exp(la)), float(-b), float(r2)


def learning_curve(points, n_boot=2000, seed=0):
    """points: [(n_train, val_right (N,) 0/1 array)], same val images. Paired bootstrap of the 2x-label gain."""
    n = np.array([p[0] for p in points], dtype=float)
    right = np.stack([p[1] for p in points])                       # (fractions, N val)
    acc = right.mean(1)
    a, b, r2 = fit_power_law(n, 1 - acc)
    N2 = 2 * n[-1]
    gain = (1 - acc[-1]) - a * N2 ** -b                           # observed full-data error minus predicted at 2N
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(n_boot):
        w = rng.integers(0, right.shape[1], right.shape[1])
        acc_b = right[:, w].mean(1)
        ab, bb, _ = fit_power_law(n, 1 - acc_b)
        gains.append((1 - acc_b[-1]) - ab * N2 ** -bb)
    lo, hi = np.quantile(gains, [0.025, 0.975])
    return {"n_train": n.tolist(), "val_acc": acc.tolist(), "power_law": {"a": a, "b": b, "r2": r2},
            "n_2x": int(N2), "pred_acc_2x": float(1 - a * N2 ** -b), "gain_2x": float(gain),
            "gain_2x_ci95": [float(lo), float(hi)], "monotonic": bool(np.all(np.diff(acc) >= -0.005))}


# --------------------------------------------------------------------------- (b) label doctor, (c) queue
def suspects(probs, given, idx):
    """Rows where the given label looks wrong: another class is predicted; ranked by p(given) ascending."""
    p_given = probs[np.arange(len(idx)), given]
    pred = probs.argmax(1)
    order = np.argsort(p_given)
    return [{"train_index": int(idx[i]), "given": int(given[i]), "predicted": int(pred[i]),
             "p_given": float(p_given[i]), "p_predicted": float(probs[i, pred[i]])}
            for i in order if pred[i] != given[i]]


def margin_queue(probs, idx, n):
    top2 = np.sort(probs, 1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    order = np.argsort(margin)[:n]
    return [{"train_index": int(idx[i]), "margin": float(margin[i]), "predicted": int(probs[i].argmax())} for i in order]


# --------------------------------------------------------------------------- thumbnails / gallery
def thumbnail(x, stats_, rgb=(3, 2, 1), size=96) -> str:
    """Base64 PNG from channels rgb (by INDEX: standard Sentinel-2 order B04,B03,B02), 2-98% stretch per image."""
    from PIL import Image
    ch = [c for c in rgb if c < x.shape[0]] or [0]
    img = np.stack([x[c] for c in (ch * 3)[:3]], -1).astype(np.float32)
    lo, hi = np.percentile(img, 2), np.percentile(img, 98)
    img = (np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).resize((size, size), Image.NEAREST).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def gallery(title, rows, X, names, stats_, note="") -> str:
    cards = []
    for r in rows:
        img = thumbnail(np.asarray(X[r["train_index"]]), stats_)
        lab = (f"given <b>{names[r['given']]}</b> (p {r['p_given']:.2f})<br>model: <b>{names[r['predicted']]}</b> "
               f"(p {r['p_predicted']:.2f})" if "given" in r else
               f"model: <b>{names[r['predicted']]}</b><br>margin {r['margin']:.3f}")
        cards.append(f'<figure><img src="data:image/png;base64,{img}"><figcaption>#{r["train_index"]}<br>{lab}'
                     f'</figcaption></figure>')
    return (f"<!doctype html><meta charset=utf-8><title>{title}</title><style>body{{font:14px system-ui;margin:16px}}"
            f"figure{{display:inline-block;margin:6px;width:150px;vertical-align:top}}img{{width:96px;height:96px;"
            f"image-rendering:pixelated}}figcaption{{font-size:12px}}</style><h2>{title}</h2><p>{note}</p>"
            + "".join(cards))


# --------------------------------------------------------------------------- the whole advisor
def run(task: str, cfg_override: dict | None = None, steps: int | None = None, seed: int = 0, log=print) -> dict:
    meta = data.task_meta(task)
    if meta["kind"] != "classification":
        raise ValueError(f"the data advisor handles single-label classification; {task} is {meta['kind']}")
    src_run, cfg = best_config(task)
    cfg.update(cfg_override or {})
    ignored = {k: cfg.pop(k) for k in ("mixup",) if cfg.get(k)}     # not implemented in fit(): say so
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr, ytr = data.load_split(task, "train")
    Xva, yva = data.load_split(task, "valid")
    ytr, yva = np.asarray(ytr), np.asarray(yva)
    st = data.channel_stats(task)
    names = meta["class_names"] or [str(i) for i in range(meta["n_classes"])]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xtr))
    if steps is None:   # same compute at every fraction: the full-data run's optimizer steps
        steps = max(1, math.ceil(len(Xtr) / {**DEFAULTS, **cfg}["batch_size"])) * {**DEFAULTS, **cfg}["epochs"]
    out, t0 = out_dir(task), time.time()
    rep = {"task": task, "config_from": src_run, "config": cfg, "steps_per_model": steps, "device": device.type,
           "n_train": len(Xtr), "n_valid": len(Xva), "ignored_config": ignored}

    # (a) learning curve, nested subsets
    points, half_model = [], None
    for f in FRACTIONS:
        idx = perm[: max(8, int(round(f * len(Xtr))))]
        model, norm, c, done = fit(meta, cfg, Xtr, ytr, idx, st, device, steps=steps, seed=seed)
        m, probs, _ = predict(model, norm, c, meta, Xva, yva, np.arange(len(Xva)), device)
        right = (probs.argmax(1) == yva).astype(float)
        ci = stats.bootstrap_ci(stats.scorecard(probs, yva))
        points.append((len(idx), right))
        rep.setdefault("curve", []).append({"fraction": f, "n_train": len(idx), "val_acc": m["accuracy"],
                                            "ci95": [ci["low"], ci["high"]], "steps": done})
        log(f"[advisor] {int(f * 100)}% ({len(idx)} imgs): val acc {m['accuracy']:.4f} "
            f"[{ci['low']:.4f}, {ci['high']:.4f}]  ({(time.time() - t0) / 60:.1f} min)")
        if f == 0.5:
            half_model = (model, norm, c)
        else:
            del model
    rep["learning_curve"] = learning_curve(points, seed=seed)
    lc = rep["learning_curve"]
    log(f"[advisor] power law r2 {lc['power_law']['r2']:.3f}; 2x labels -> {lc['gain_2x'] * 100:+.2f} pts "
        f"[{lc['gain_2x_ci95'][0] * 100:.2f}, {lc['gain_2x_ci95'][1] * 100:.2f}]")

    # (b) label doctor: 2-fold OOF on TRAIN with INJECT random flips (a modified COPY of the labels)
    noisy = ytr.copy()
    n_inj = max(1, int(round(INJECT * len(ytr))))
    inj = rng.choice(len(ytr), n_inj, replace=False)
    for i in inj:
        noisy[i] = rng.choice([c for c in range(meta["n_classes"]) if c != ytr[i]])
    A, B = perm[: len(perm) // 2], perm[len(perm) // 2:]
    oof_p, oof_i = [], []
    for tr, te in ((A, B), (B, A)):
        model, norm, c, _ = fit(meta, cfg, Xtr, noisy, tr, st, device, steps=steps, seed=seed)
        _, probs, idx = predict(model, norm, c, meta, Xtr, noisy, te, device)
        oof_p.append(probs); oof_i.append(idx)
        del model
    probs, idx = np.concatenate(oof_p), np.concatenate(oof_i)
    sus = suspects(probs, noisy[idx], idx)
    K = 2 * n_inj
    top = {s["train_index"] for s in sus[:K]}
    recall = len(top & set(inj.tolist())) / n_inj
    natural = [s for s in sus if s["train_index"] not in set(inj.tolist())]
    rep["label_doctor"] = {"injected_flips": n_inj, "flagged_total": len(sus), "check_K": K,
                           "recall_injected_at_K": recall, "natural_suspects": len(natural),
                           "natural_suspect_rate": len(natural) / len(ytr), "top_natural": natural[:24]}
    (out / "label_doctor.html").write_text(gallery(
        f"{task}: suspicious TRAIN labels (out-of-fold)", natural[:24], Xtr, names, st,
        f"{len(natural)} natural suspects ({len(natural) / len(ytr):.1%} of TRAIN). The method recovered "
        f"{recall:.0%} of {n_inj} injected label flips within its top {K}. Thumbnails: channels 3,2,1 by index."))
    log(f"[advisor] label doctor: recall {recall:.0%} of {n_inj} injected flips in top {K}; "
        f"{len(natural)} natural suspects ({len(natural) / len(ytr):.1%})")

    # (c) labeling queue from the clean 50% model, checked: half + top-N vs half + random-N
    model, norm, c = half_model
    half = perm[: int(round(0.5 * len(Xtr)))]
    pool = perm[int(round(0.5 * len(Xtr))):]
    _, pprobs, pidx = predict(model, norm, c, meta, Xtr, ytr, pool, device)
    qn = min(QUEUE_N, len(pool) // 2)
    queue = margin_queue(pprobs, pidx, qn)
    del model
    chk = {}
    for name, extra in (("uncertain", np.array([q["train_index"] for q in queue])),
                        ("random", rng.choice(pool, qn, replace=False))):
        model, norm, c, _ = fit(meta, cfg, Xtr, ytr, np.concatenate([half, extra]), st, device, steps=steps, seed=seed)
        _, probs, _ = predict(model, norm, c, meta, Xva, yva, np.arange(len(Xva)), device)
        chk[name] = stats.scorecard(probs, yva)
        del model
    pt = stats.paired_test(chk["uncertain"], chk["random"])
    rep["labeling_queue"] = {"pool": "the half of TRAIN not used by the 50% model", "n": qn, "top": queue[:24],
                             "check": {"val_acc_half_plus_uncertain": float(chk["uncertain"].units[:, 0].mean()),
                                       "val_acc_half_plus_random": float(chk["random"].units[:, 0].mean()),
                                       "paired_test": pt}}
    (out / "labeling_queue.html").write_text(gallery(
        f"{task}: next {qn} images to label (most uncertain)", queue[:24], Xtr, names, st,
        f"Ranked by margin from a model trained on 50% of TRAIN. Check: +{qn} uncertain vs +{qn} random "
        f"gives val {pt['diff'] * 100:+.2f} pts [{pt['low'] * 100:+.2f}, {pt['high'] * 100:+.2f}], p {pt['p']:.3g}."))
    (out / "labeling_queue.json").write_text(json.dumps(queue, indent=1))
    log(f"[advisor] queue check: uncertain vs random {pt['diff'] * 100:+.2f} pts, p {pt['p']:.3g}")

    rep["minutes"] = round((time.time() - t0) / 60, 1)
    (out / "advisor.json").write_text(json.dumps(rep, indent=1, default=str))
    (out / "SUMMARY.md").write_text(summary(rep))
    return rep


def summary(rep) -> str:
    lc, ld, q = rep["learning_curve"], rep["label_doctor"], rep["labeling_queue"]["check"]
    pt = q["paired_test"]
    g, (glo, ghi) = lc["gain_2x"] * 100, (v * 100 for v in lc["gain_2x_ci95"])
    rows = "\n".join(f"| {int(c['fraction'] * 100)}% | {c['n_train']} | {c['val_acc']:.4f} | "
                     f"[{c['ci95'][0]:.4f}, {c['ci95'][1]:.4f}] |" for c in rep["curve"])
    fit_ok = lc["power_law"]["r2"] >= 0.9 and lc["monotonic"]
    if not fit_ok:
        verdict = ("**Inconclusive.** The learning curve does not follow a power law, so the 2x-label estimate is not "
                   "trustworthy; decide from the label doctor and the model comparisons instead.")
    elif glo > 0.5:
        verdict = "**Data.** Doubling the labels is expected to add a clearly positive amount (lower bound > 0.5 pts)."
    else:
        verdict = ("**Model, not more of the same labels.** The 2x-label gain is small or not distinguishable from "
                   "zero; label quality and model choices are the better levers.")
    weak = []
    if lc["power_law"]["r2"] < 0.9 or not lc["monotonic"]:
        weak.append(f"the power-law fit is weak (r2 {lc['power_law']['r2']:.2f}, monotonic {lc['monotonic']}): "
                    "treat the 2x estimate as rough")
    if pt["p"] > 0.05:
        weak.append(f"uncertainty sampling did not beat random sampling significantly (p {pt['p']:.2f})")
    return f"""# Data Advisor: {rep['task']} - model or data?

Config from `{rep['config_from']}` (best VALID run), {rep['steps_per_model']} optimizer steps per model, {rep['device']},
{rep['minutes']} min. TRAIN {rep['n_train']}, VALID {rep['n_valid']}. Test never read.

## Verdict
{verdict}

## Learning curve (same compute per point, VALID accuracy, bootstrap 95% CI)
| TRAIN used | images | val acc | 95% CI |
|---|---|---|---|
{rows}

Power law on error: b = {lc['power_law']['b']:.3f}, r2 = {lc['power_law']['r2']:.3f}.
2x labels ({lc['n_2x']} images): predicted val acc {lc['pred_acc_2x']:.4f}, gain **{g:+.2f} pts** [95% {glo:+.2f}, {ghi:+.2f}].

## Label doctor (2-fold out-of-fold on TRAIN)
Sanity check: {ld['injected_flips']} labels flipped on purpose; the doctor found **{ld['recall_injected_at_K']:.0%}** of them
in its top {ld['check_K']}. Natural suspects: {ld['natural_suspects']} ({ld['natural_suspect_rate']:.1%} of TRAIN):
see `label_doctor.html` (they are candidates to re-check, not proven errors).

## Labeling queue
Next {rep['labeling_queue']['n']} images by margin: `labeling_queue.html` / `labeling_queue.json`.
Check: 50% + {rep['labeling_queue']['n']} uncertain -> val {q['val_acc_half_plus_uncertain']:.4f}; 50% + {rep['labeling_queue']['n']} random -> {q['val_acc_half_plus_random']:.4f};
paired diff {pt['diff'] * 100:+.2f} pts [{pt['low'] * 100:+.2f}, {pt['high'] * 100:+.2f}], p {pt['p']:.3g}.

## Caveats
{chr(10).join('- ' + w for w in weak) or '- none flagged by the checks'}
- config keys ignored by the advisor's trainer: {rep['ignored_config'] or 'none'}
- one seed per point: training noise is not in the CIs (they cover val-image sampling only)
- thumbnails use channels 3,2,1 by index (m-eurosat stores wrong band NAMES in GEO-Bench 1.0)
"""


# --------------------------------------------------------------------------- MCP tool (read-only)
def data_advisor_report(task: str) -> str:
    """Read the Data Advisor's last report for a task: learning curve with the expected gain from 2x labels,
    suspicious TRAIN labels, and the next images to label (produced by scripts/run_advisor.py; VALID/TRAIN only).

    Args:
        task: task name.
    """
    p = out_dir(task) / "SUMMARY.md"
    return p.read_text() if p.exists() else f"no advisor report for {task}: run python scripts/run_advisor.py {task}"


MCP_TOOLS = [data_advisor_report]
