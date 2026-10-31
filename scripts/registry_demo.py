"""A7 demo: datasets and model versions in S3 (or the local store), with lineage and stages.

    export GOTRAIN_S3_BUCKET=my-team-bucket GOTRAIN_S3_PREFIX=group4/gotrain   # optional; else written locally
    python scripts/registry_demo.py m-eurosat

Uses the best VALID run that is not yet published (finalized production versions are left alone): publishes it as a candidate, promotes it to staging, pulls it back
as another machine would, and prints the registry. Never touches TEST.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))  # run from anywhere
import json
import sys

from gotrain import config, registry
from gotrain.store import Store

task = sys.argv[1] if len(sys.argv) > 1 else "m-eurosat"
st = Store()
print("store:", f"s3://{config.S3_BUCKET}/{config.S3_PREFIX}" if config.S3_BUCKET else config.STORE_DIR)

published = {v["run_id"] for v in st.versions(task)}
done = [r for r in registry.leaderboard(task) if r["state"] == "done" and r["val_score"] is not None]
best = next((r for r in done if r["run_id"] not in published), None)   # never re-stage a finalized model
if best is None:
    sys.exit(f"no unpublished finished run for {task}: train one first (python -m gotrain.train --task {task})")

print("\n1. push the prepared dataset cache")
for uri in st.push_dataset(task):
    print("  ", uri)

print(f"\n2. publish best unpublished VALID run {best['run_id']} (val {best['val_score']:.4f}) as a candidate")
entry = st.publish_run(best["run_id"], stage="candidate")
print("   version", entry["version"], "parent run", entry["parent_run"])

print("\n3. promote it to staging")
print("  ", st.promote(task, entry["version"], "staging")["stage"])

print("\n4. pull it back (what GoPilot / another instance would do)")
print("  ", json.dumps(st.pull_model(task, entry["version"]), indent=1))

print("\n5. registry")
for v in st.versions(task):
    print(f"   v{v['version']}  {v['stage']:<10}  run {v['run_id']}  val {v.get('val_score')}  "
          f"test {(v.get('test_metrics') or {}).get(v.get('metric'), '-')}  parent {v.get('parent_run')}")
