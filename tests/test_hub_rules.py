"""hub.py's trust rules, offline (a fake Hugging Face API stands in for the network):

- timm models pass whatever their likes (popular timm weights often have < 10 likes);
- other uploaders need >= MIN_DOWNLOADS downloads or >= MIN_LIKES likes, else "excluded_little_used";
- leakage still wins: a popular EuroSAT fine-tune is excluded for leakage;
- likes, downloads, license and last update are shown; a non-commercial license is flagged;
- add_hf_candidate refuses a little-used model before building anything.

    python tests/test_hub_rules.py
"""
import json
import sys
import types
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def model(mid, downloads, likes, lic="apache-2.0", datasets=("imagenet-1k",), params=28.6e6):
    return types.SimpleNamespace(id=mid, downloads=downloads, likes=likes, tags=["timm", "image-classification"],
                                 card_data={"license": lic, "datasets": list(datasets)},
                                 last_modified=datetime(2025, 1, 21), safetensors=types.SimpleNamespace(total=params))


HUB = [model("timm/convnext_tiny.in12k_ft_in1k", 742601, 6),
       model("timm/convnextv2_tiny.fcmae_ft_in1k", 50000, 0, lic="cc-by-nc-4.0"),
       model("someone/convnext_tiny_eurosat", 90000, 40),                    # popular, but leaks
       model("someone/convnext_tiny_myrun", 12, 0),                          # little used
       model("lab/convnext_tiny_rs", 5000, 2),                               # enough downloads
       model("lab2/convnext_tiny_rs2", 300, 15)]                             # enough likes


class FakeApi:
    def list_models(self, **kw):
        return iter(HUB)

    def model_info(self, mid, **kw):
        return next(m for m in HUB if m.id == mid)


def main():
    sys.modules["huggingface_hub"] = types.SimpleNamespace(HfApi=FakeApi)
    from gotrain import hub
    out = json.loads(hub.search_hf_models("convnext", author=""))
    ok = {m["id"]: m for m in out["models"]}
    leak = {m["id"] for m in out["excluded_for_leakage"]}
    low = {m["id"] for m in out["excluded_little_used"]}
    assert set(ok) == {"timm/convnext_tiny.in12k_ft_in1k", "timm/convnextv2_tiny.fcmae_ft_in1k",
                       "lab/convnext_tiny_rs", "lab2/convnext_tiny_rs2"}, sorted(ok)
    assert leak == {"someone/convnext_tiny_eurosat"} and low == {"someone/convnext_tiny_myrun"}, (leak, low)
    t = ok["timm/convnext_tiny.in12k_ft_in1k"]
    assert t["likes"] == 6 and t["downloads"] == 742601 and t["license"] == "apache-2.0"
    assert t["last_modified"] == "2025-01-21" and "license_note" not in t
    assert "non-commercial" in ok["timm/convnextv2_tiny.fcmae_ft_in1k"]["license_note"]
    print("search: kept", sorted(ok), "| leakage", sorted(leak), "| little used", sorted(low))

    hub.data.task_meta = lambda task: {"kind": "classification", "bands": list(range(13)), "n_classes": 10,
                                       "patch_size": [64, 64]}
    r = json.loads(hub.add_hf_candidate("m-eurosat", "someone/convnext_tiny_myrun", "test"))
    assert r["refused"] and "little used" in r["why"], r
    assert hub.little_used("timm/anything", 0, 0) is None and hub.license_note("cc-by-nc-sa-4.0")
    assert hub.license_note("apache-2.0") is None and hub.license_note(None) is None
    print("HUB RULES OK")


if __name__ == "__main__":
    main()
