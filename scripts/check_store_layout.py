"""Check that the local store maps 1:1 to S3: every file under STORE_DIR is an allowed key, and
registry.json lists files by key. Filesystem only (no AWS calls).

    python scripts/check_store_layout.py          # report; exit 1 on any violation
    python scripts/check_store_layout.py --fix    # repair stores written by older code, then re-check:
                                                  #   absolute paths in registry.json -> keys (backup kept)
                                                  #   datasets/*/test_* removed (copies; the cache is untouched)
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))  # run from anywhere
import json
import re
import shutil
import sys
import time

from gotrain import config

ALLOWED = [
    r"datasets/[^/]+/(train|valid)_(X|y)\.npy",
    r"datasets/[^/]+/(meta|train_stats)\.json",
    r"models/[^/]+/registry\.json",
    r"models/[^/]+/v\d+/(model\.pt|config\.json|metrics\.json|history\.jsonl|train\.log)",
]


def check(root, fix=False):
    problems, fixed = [], []
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    for rel in files:
        if rel.startswith("datasets/") and rel.split("/")[-1].startswith("test_"):
            if fix:
                (root / rel).unlink(); fixed.append(f"removed {rel} (test data never goes to the store)")
            else:
                problems.append(f"{rel}: test data in the store")
        elif not any(re.fullmatch(a, rel) for a in ALLOWED):
            problems.append(f"{rel}: not an allowed key")
    for reg in sorted(root.glob("models/*/registry.json")):
        task, vers, changed = reg.parent.name, json.loads(reg.read_text()), False
        for v in vers:
            for name, val in v.get("files", {}).items():
                key = f"models/{task}/v{v['version']}/{name}"
                if val != key:
                    if fix and str(val).replace("\\", "/").endswith(key):
                        v["files"][name], changed = key, True
                    else:
                        problems.append(f"{reg.relative_to(root)} v{v['version']} {name}: '{val}' is not the key '{key}'")
                if not (root / key).exists():
                    problems.append(f"{key}: listed in registry.json but missing")
        on_disk = {int(p.name[1:]) for p in reg.parent.glob("v*") if p.is_dir() and p.name[1:].isdigit()}
        orphans = on_disk - {v["version"] for v in vers}
        problems += [f"models/{task}/v{n}/: version folder not in registry.json" for n in sorted(orphans)]
        if changed:
            backup = config.HOME / "tmp" / f"registry_{task}_{time.strftime('%Y%m%d_%H%M%S')}.json"
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(reg, backup)
            reg.write_text(json.dumps(vers, indent=1))
            fixed.append(f"{reg.relative_to(root)}: file paths rewritten as keys (backup {backup})")
    return files, problems, fixed


if __name__ == "__main__":
    root = config.STORE_DIR
    files, problems, fixed = check(root, fix="--fix" in sys.argv)
    for f in fixed:
        print("FIXED", f)
    if fixed:
        files, problems, _ = check(root)
    dest = f"s3://{config.S3_BUCKET or '<bucket>'}/{config.S3_PREFIX}"
    print(f"STORE_DIR {root}  ->  {dest}\n")
    for rel in files:
        if (root / rel).exists():
            print(f"  {rel:<48} -> {dest}/{rel}")
    print()
    for p in problems:
        print("VIOLATION", p)
    print("LAYOUT OK: local path under STORE_DIR == S3 key" if not problems else f"{len(problems)} violation(s)")
    sys.exit(1 if problems else 0)
