"""Write the receipt of a registry version and retrain it to check it reproduces (VALID only).

    python scripts/reproduce_version.py m-eurosat 1
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))  # run from anywhere
import json
import sys

from gotrain import receipt

if __name__ == "__main__":
    task, version = sys.argv[1], int(sys.argv[2])
    rec = receipt.make_receipt(task, version)
    print("receipt", json.dumps({k: rec[k] for k in ("run_id", "fingerprint", "code_commit", "original_val",
                                                      "original_val_ci95")}))
    res = receipt.reproduce(task, version)
    print(json.dumps(res, indent=1))
    print("REPRODUCE DONE", res["verdict"])
