"""Versioned storage for datasets and models: S3 when a bucket is configured, local folder otherwise.

Layout (identical in S3 and locally, so switching is one environment variable):
    datasets/<task>/<split>_X.npy, <split>_y.npy, meta.json, train_stats.json
    models/<task>/v<N>/model.pt, config.json, metrics.json, history.jsonl, train.log
    models/<task>/registry.json   list of versions: run, val score, test score, stage, time, lineage
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from . import config, registry


class Store:
    def __init__(self):
        self.s3 = None
        if config.S3_BUCKET:
            import boto3
            self.s3 = boto3.client("s3")
        self.root = config.STORE_DIR

    # -- primitives
    def _key(self, rel: str) -> str:
        return f"{config.S3_PREFIX}/{rel}"

    def put_file(self, src: Path, rel: str) -> str:
        dst = self.root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if self.s3:
            self.s3.upload_file(str(src), config.S3_BUCKET, self._key(rel))
            return f"s3://{config.S3_BUCKET}/{self._key(rel)}"
        return str(dst)

    def get_file(self, rel: str, dst: Path) -> Path:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if self.s3:
            self.s3.download_file(config.S3_BUCKET, self._key(rel), str(dst))
        else:
            shutil.copy2(self.root / rel, dst)
        return dst

    def put_json(self, obj, rel: str) -> str:
        tmp = self.root / (rel + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(obj, indent=1))
        uri = self.put_file(tmp, rel)
        tmp.unlink()
        return uri

    def get_json(self, rel: str, default=None):
        try:
            if self.s3:
                body = self.s3.get_object(Bucket=config.S3_BUCKET, Key=self._key(rel))["Body"].read()
                return json.loads(body)
            return json.loads((self.root / rel).read_text())
        except Exception:
            return default

    # -- datasets
    def push_dataset(self, task: str) -> list[str]:
        d = config.CACHE_DIR / task
        return [self.put_file(p, f"datasets/{task}/{p.name}") for p in sorted(d.glob("*")) if p.is_file()]

    def pull_dataset(self, task: str, files=("train_X.npy", "train_y.npy", "valid_X.npy", "valid_y.npy",
                                              "meta.json", "train_stats.json")) -> list[str]:
        d = config.CACHE_DIR / task
        return [str(self.get_file(f"datasets/{task}/{f}", d / f)) for f in files]

    # -- models
    def versions(self, task: str) -> list[dict]:
        return self.get_json(f"models/{task}/registry.json", default=[])

    def publish_run(self, run_id: str, stage: str = "candidate", test_metrics: dict | None = None) -> dict:
        info = registry.read(run_id)
        task = info["task"]
        rd = registry.run_dir(run_id)
        vers = self.versions(task)
        existing = [v for v in vers if v["run_id"] == run_id]
        if existing:
            entry = existing[0]
        else:
            n = len(vers) + 1
            files = {}
            for name in ("model.pt", "config.json", "metrics.json", "history.jsonl", "train.log"):
                if (rd / name).exists():
                    files[name] = self.put_file(rd / name, f"models/{task}/v{n}/{name}")
            entry = {"version": n, "run_id": run_id, "parent_run": info["parent"],
                     "val_score": (info["metrics"] or {}).get("val_score"),
                     "metric": (info["metrics"] or {}).get("primary_metric"),
                     "hypothesis": info["hypothesis"], "files": files,
                     "published": time.strftime("%Y-%m-%d %H:%M:%S")}
            vers.append(entry)
        entry["stage"] = stage
        if test_metrics is not None:
            entry["test_metrics"] = test_metrics
        self.put_json(vers, f"models/{task}/registry.json")
        return entry
