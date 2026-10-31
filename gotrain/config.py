"""All paths and settings in one place, overridable with environment variables.

GEO_BENCH_DIR      where GEO-Bench tasks live (same variable the geobench package reads)
GOTRAIN_HOME       where caches, runs, logs and the local model store go
GOTRAIN_S3_BUCKET  if set, datasets and model versions are also pushed to this bucket
GOTRAIN_MODEL_ID   Bedrock model id for the agent
GOTRAIN_REGION     Bedrock region
"""
from __future__ import annotations

import os
from pathlib import Path

GEO_BENCH_DIR = Path(os.environ.get("GEO_BENCH_DIR", Path.home() / "dataset" / "geobench"))
os.environ.setdefault("GEO_BENCH_DIR", str(GEO_BENCH_DIR))  # geobench reads it at import

HOME = Path(os.environ.get("GOTRAIN_HOME", Path.home() / "gotrain_home"))
CACHE_DIR = HOME / "cache"      # numpy arrays per task and split
RUNS_DIR = HOME / "runs"        # one folder per training run
LOGS_DIR = HOME / "logs"        # agent session logs (every tool call)
STORE_DIR = HOME / "store"      # local model/dataset store (mirror of S3 layout)
CUSTOM_DIR = HOME / "custom_models"
RESULTS_DIR = HOME / "results"  # final test scores, one file per task

S3_BUCKET = os.environ.get("GOTRAIN_S3_BUCKET") or None
S3_PREFIX = os.environ.get("GOTRAIN_S3_PREFIX", "gotrain")

MODEL_ID = os.environ.get("GOTRAIN_MODEL_ID", "eu.anthropic.claude-sonnet-5-5")
REGION = os.environ.get("GOTRAIN_REGION", "eu-central-1")

HF_REPO = "recursix/geo-bench-1.0"
BENCHMARKS = {"classification": "classification_v1.0", "segmentation": "segmentation_v1.0"}

CLASSIFICATION_TASKS = ["m-bigearthnet", "m-so2sat", "m-brick-kiln", "m-forestnet", "m-eurosat", "m-pv4ger"]
SEGMENTATION_TASKS = ["m-pv4ger-seg", "m-chesapeake", "m-cashew-plant", "m-SA-crop-type", "m-nz-cattle", "m-NeonTree"]

for d in (CACHE_DIR, RUNS_DIR, LOGS_DIR, STORE_DIR, CUSTOM_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def benchmark_of(task: str) -> str:
    if task in CLASSIFICATION_TASKS:
        return BENCHMARKS["classification"]
    if task in SEGMENTATION_TASKS:
        return BENCHMARKS["segmentation"]
    # unknown name: look on disk
    for b in BENCHMARKS.values():
        if (GEO_BENCH_DIR / b / task).exists():
            return b
    raise ValueError(f"Unknown task {task!r}. Known: {CLASSIFICATION_TASKS + SEGMENTATION_TASKS}")


def task_dir(task: str) -> Path:
    return GEO_BENCH_DIR / benchmark_of(task) / task
