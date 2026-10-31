"""Model discovery on the Hugging Face Hub, as agent tools.

Why HF and not web search: the SageMaker instance reaches huggingface.co (no key needed), but there is
no search API key and the Bedrock model has no search tool. The Hub API is the one searchable model
catalogue we can rely on, and every timm model on it loads directly:
    classification  backbone "hf-hub:timm/convnext_tiny.in12k_ft_in1k"        (timm adapts the first conv to any band count)
    segmentation    seg_arch "unet", backbone "tu-hf-hub:timm/..."             (smp builds the encoder through timm)

Leakage guard: GEO-Bench tasks are subsets of public datasets. A Hub model fine-tuned on, e.g., the full
EuroSAT has very likely seen our m-eurosat TEST images, so any model whose id, tags or training datasets
mention a GEO-Bench source dataset is refused. That keeps the "test touched once" rule meaningful.

Trust rules: the Hub has no reviews, so usage is the signal. Models from TRUSTED_AUTHORS (timm: the
maintained weights) pass; any other uploader needs >= MIN_DOWNLOADS downloads (last 30 days) or
>= MIN_LIKES likes. Likes, downloads, license and last update are shown to the agent; a non-commercial
license is flagged (fine for the benchmark, not for GoPilot production).

Accepted models join the task's candidate menu (CANDIDATE_POOL), so propose_candidates / submit_shortlist
treat them like the curated ones. They are persisted in GOTRAIN_HOME/hub_pool.json.
"""
from __future__ import annotations

import json
import re

from . import config, data
from .train import CANDIDATE_POOL

# substrings of GEO-Bench source datasets (lower case, '-' and '_' removed before matching)
BENCHMARK_SOURCES = ("eurosat", "bigearthnet", "bigearth", "so2sat", "lcz42", "brickkiln", "forestnet", "pv4ger",
                     "chesapeake", "cashew", "sacrop", "croptype", "nzcattle", "neontree", "geobench")
POOL_FILE = config.HOME / "hub_pool.json"
TRUSTED_AUTHORS = ("timm",)
MIN_DOWNLOADS = 1000        # last 30 days
MIN_LIKES = 10              # likes are rare even on popular timm weights (convnext_tiny.in12k: 6), so either is enough
EXPAND = ["safetensors", "downloads", "likes", "lastModified", "tags", "cardData"]


def _j(obj) -> str:
    return json.dumps(obj, indent=1, default=str)


def contamination(model_id: str, tags=(), datasets=()) -> str | None:
    """Why this model may have seen GEO-Bench data, or None."""
    for field, text in (("id", model_id), ("tags", " ".join(tags or ())), ("datasets", " ".join(datasets or ()))):
        flat = re.sub(r"[-_ ]", "", str(text).lower())
        hit = next((s for s in BENCHMARK_SOURCES if s in flat), None)
        if hit:
            return f"{field} mentions '{hit}': likely trained on a GEO-Bench source dataset (test leakage)"
    return None


def little_used(model_id: str, downloads, likes) -> str | None:
    """Why this model is too little used to trust, or None."""
    if model_id.split("/")[0] in TRUSTED_AUTHORS or (downloads or 0) >= MIN_DOWNLOADS or (likes or 0) >= MIN_LIKES:
        return None
    return (f"little used: {downloads or 0} downloads in 30 days, {likes or 0} likes; uploaders other than "
            f"{'/'.join(TRUSTED_AUTHORS)} need >= {MIN_DOWNLOADS} downloads or >= {MIN_LIKES} likes")


def license_note(lic) -> str | None:
    if lic and "nc" in re.split(r"[-_. ]", str(lic).lower()):
        return f"non-commercial license ({lic}): fine for the benchmark, not for GoPilot production"
    return None


def _usage(m) -> dict:
    """Likes, downloads, license, last update of a Hub ModelInfo."""
    card = m.card_data or {}
    lic = card.get("license") if card else None
    return {"downloads": m.downloads, "likes": m.likes, "license": lic,
            "last_modified": str(m.last_modified)[:10] if m.last_modified else None,
            **({"license_note": license_note(lic)} if license_note(lic) else {})}


def _family(model_id: str) -> str:
    arch = model_id.split("/")[-1].split(".")[0]                 # timm/convnext_tiny.in12k -> convnext_tiny
    return "hf-" + re.sub(r"(_|\d).*$", "", arch) or "hf"        # -> hf-convnext


def search_hf_models(query: str, max_params_M: float = 60.0, author: str = "timm", limit: int = 8) -> str:
    """Search the Hugging Face Hub for pretrained timm backbones (sorted by downloads) to extend the model
    menu beyond the curated pool. Returns id, parameters, downloads and pretraining datasets. Models that
    may have been trained on a GEO-Bench source dataset (EuroSAT, BigEarthNet, ...) are listed as EXCLUDED:
    using them would leak test images. Uploaders other than timm need real usage (downloads or likes);
    each model shows likes, downloads, license and last update. Follow with add_hf_candidate for the ones you want to screen.

    Args:
        query: architecture or keyword, e.g. "convnext", "efficientnetv2", "regnety", "swin".
        max_params_M: skip models larger than this (millions of parameters); the GPU is a 16 GB T4.
        author: Hub organisation; "timm" holds the maintained weights. "" searches all authors.
        limit: how many usable models to return.
    """
    from huggingface_hub import HfApi

    found, excluded, little = [], [], []
    for m in HfApi().list_models(search=query, author=author or None, filter="timm", sort="downloads",
                                 limit=limit * 4, expand=EXPAND):
        datasets = (m.card_data or {}).get("datasets") if m.card_data else None
        datasets = [datasets] if isinstance(datasets, str) else list(datasets or [])
        why = contamination(m.id, m.tags, datasets)
        if why:
            excluded.append({"id": m.id, "why": why})
            continue
        low = little_used(m.id, m.downloads, m.likes)
        if low:
            little.append({"id": m.id, "why": low})
            continue
        params = round(m.safetensors.total / 1e6, 1) if m.safetensors and m.safetensors.total else None
        if params is not None and params > max_params_M:
            continue
        found.append({"id": m.id, "params_M": params, "pretrained_on": datasets, "family": _family(m.id), **_usage(m)})
        if len(found) >= limit:
            break
    return _j({"query": query, "models": found, "excluded_for_leakage": excluded, "excluded_little_used": little,
               "next": "add_hf_candidate(task, model_id, why) for each model you want on the menu"})


def _entry(kind: str, model_id: str, params_M, why: str) -> dict:
    cfg = ({"seg_arch": "unet", "backbone": f"tu-hf-hub:{model_id}"} if kind == "segmentation"
           else {"backbone": f"hf-hub:{model_id}"})
    return {"config": cfg, "family": _family(model_id), "weights": f"HF Hub {model_id}", "params_M": params_M,
            "suits": f"agent-added from the Hugging Face Hub: {why}", "source": "hf"}


def _load_persisted() -> dict:
    return json.loads(POOL_FILE.read_text()) if POOL_FILE.exists() else {}


def _register(kind: str, entry: dict) -> None:
    pool = CANDIDATE_POOL[kind]
    if all(e["config"] != entry["config"] for e in pool):
        pool.append(entry)            # same list object strategy.py reads: visible to propose_candidates now


def load_hub_pool() -> None:
    """Re-add models accepted in earlier sessions (called at import)."""
    for kind, entries in _load_persisted().items():
        for e in entries:
            if kind in CANDIDATE_POOL:
                _register(kind, e)


def add_hf_candidate(task: str, model_id: str, why: str) -> str:
    """Check a Hugging Face model and add it to this task's candidate menu (then it can be picked in
    submit_shortlist like the curated models). Checks: no GEO-Bench leakage, enough usage (timm, or
    downloads / likes), it builds for this task's band count and kind (classification head, or U-Net
    encoder for segmentation), and its real size.
    Downloads only the model config here; weights download when it first trains.

    Args:
        task: task name.
        model_id: Hub id from search_hf_models, e.g. "timm/convnext_tiny.in12k_ft_in1k".
        why: how this model fits THIS data (recorded in the journal with the decision).
    """
    import torch
    from huggingface_hub import HfApi

    from . import registry

    meta = data.task_meta(task)
    kind = meta["kind"]
    try:
        info = HfApi().model_info(model_id, expand=EXPAND)
    except Exception as e:
        return _j({"refused": True, "why": f"not found on the Hub: {type(e).__name__}: {str(e)[:150]}"})
    datasets = (info.card_data or {}).get("datasets") if info.card_data else None
    datasets = [datasets] if isinstance(datasets, str) else list(datasets or [])
    leak = contamination(model_id, info.tags, datasets)
    if leak:
        return _j({"refused": True, "why": leak})
    low = little_used(model_id, info.downloads, info.likes)
    if low:
        return _j({"refused": True, "why": low})
    usage = _usage(info)
    if "timm" not in (info.tags or []):
        return _j({"refused": True, "why": "not a timm model: the trainer loads Hub models through timm"})
    bands = len(meta["bands"]) or 3
    entry = _entry("segmentation" if kind == "segmentation" else "classification", model_id, None, why)
    native = min(meta["patch_size"] or [64])
    try:   # build without weights: proves it runs for this band count / kind, gives the true size
        if kind == "segmentation":
            import segmentation_models_pytorch as smp
            net = smp.create_model("unet", encoder_name=entry["config"]["backbone"], encoder_weights=None,
                                   in_channels=bands, classes=meta["n_classes"])
            out = net(torch.zeros(1, bands, 64, 64))
        else:
            import timm
            net = timm.create_model(entry["config"]["backbone"], pretrained=False, in_chans=bands,
                                    num_classes=meta["n_classes"])
            net.eval()
            try:
                out = net(torch.zeros(1, bands, native, native))
            except Exception:   # fixed-size models (ViT, Swin): train at the size they were built for
                size = int(net.pretrained_cfg.get("input_size", (3, 224, 224))[-1])
                out = net(torch.zeros(1, bands, size, size))
                entry["config"]["img_size"] = size
    except Exception as e:
        return _j({"refused": True, "why": f"does not build as a {kind} model with {bands} bands: "
                                          f"{type(e).__name__}: {str(e)[:200]}"})
    entry["params_M"] = round(sum(p.numel() for p in net.parameters()) / 1e6, 1)
    entry["usage"] = usage
    pool_kind = "segmentation" if kind == "segmentation" else "classification"
    _register(pool_kind, entry)
    saved = _load_persisted()
    saved.setdefault(pool_kind, [])
    if all(e["config"] != entry["config"] for e in saved[pool_kind]):
        saved[pool_kind].append(entry)
        POOL_FILE.write_text(json.dumps(saved, indent=1))
    registry.journal(task, f"HF HUB CANDIDATE added: {model_id} ({entry['params_M']}M params, "
                           f"pretrained on {datasets or 'unknown'}; {usage['downloads']} downloads, {usage['likes']} likes, "
                           f"license {usage['license']}): {why}")
    return _j({"added": model_id, "config": entry["config"], "family": entry["family"],
               "params_M": entry["params_M"], "output_shape": list(out.shape), **usage,
               "next": "propose_candidates(task) now lists it; pick it in submit_shortlist"})


ALL_HUB_TOOLS = [search_hf_models, add_hf_candidate]

load_hub_pool()
