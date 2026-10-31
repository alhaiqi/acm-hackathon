"""P3 demo: a real Sentinel-2 crop over Lebanon -> GeoTIFF -> the m-eurosat production model.

    python scripts/demo_predict_lebanon.py [--bbox 35.86 33.82 35.92 33.87]

1. Finds a cloud-free Sentinel-2 L2A scene in the public Element 84 Earth Search STAC catalogue (no account, no
   credentials; the COGs are read over plain HTTPS) and writes a 10 m multi-band GeoTIFF of the bbox with band
   descriptions B01..B12 (L2A has no B10).
2. Runs predict_geotiff with the default rules: it must REFUSE (B10 missing) - the safety check working.
3. Runs it again with fill_missing="B10" (explicit opt-in): output GeoTIFF + PNG, with the warnings it raises.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))  # run from anywhere
import argparse
import json

import numpy as np
import requests

from gotrain import config
from gotrain.inference import STAC_NAMES, predict_geotiff

STAC = "https://earth-search.aws.element84.com/v1/search"


def find_scene(bbox):
    body = {"collections": ["sentinel-2-l2a"], "bbox": bbox, "datetime": "2024-05-01T00:00:00Z/2024-09-30T00:00:00Z",
            "query": {"eo:cloud_cover": {"lt": 3}}, "limit": 20}
    feats = requests.post(STAC, json=body, timeout=60).json()["features"]
    return min(feats, key=lambda f: f["properties"]["eo:cloud_cover"])


def build_geotiff(item, bbox, out):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    assets = {STAC_NAMES[k]: v["href"] for k, v in item["assets"].items() if k in STAC_NAMES}
    order = [b for b in ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")
             if b in assets]
    with rasterio.open(assets["B04"]) as ref:                       # 10 m grid
        b = transform_bounds("EPSG:4326", ref.crs, *bbox)
        win = from_bounds(*b, transform=ref.transform).round_offsets().round_lengths()
        H, W = int(win.height), int(win.width)
        transform, crs = ref.window_transform(win), ref.crs
    bands = []
    for name in order:
        with rasterio.open(assets[name]) as ds:
            w = from_bounds(*b, transform=ds.transform)
            bands.append(ds.read(1, window=w, out_shape=(H, W), resampling=Resampling.bilinear, boundless=True))
    prof = {"driver": "GTiff", "height": H, "width": W, "count": len(bands), "dtype": "uint16", "crs": crs,
            "transform": transform, "compress": "deflate"}
    with rasterio.open(out, "w", **prof) as dst:
        for i, (name, arr) in enumerate(zip(order, bands), 1):
            dst.write(arr.astype(np.uint16), i)
            dst.set_band_description(i, name)
        dst.update_tags(source=item["id"], stac=STAC, cloud_cover=str(item["properties"]["eo:cloud_cover"]),
                        bbox_wgs84=json.dumps(bbox))
    return order, (H, W), str(crs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", type=float, nargs=4, default=[35.86, 33.82, 35.92, 33.87],
                    help="lon_min lat_min lon_max lat_max (default: around Zahle, Bekaa valley, Lebanon)")
    ap.add_argument("--task", default="m-eurosat")
    a = ap.parse_args()
    try:
        import rasterio  # noqa: F401
    except ImportError:
        print(predict_geotiff("none.tif", task=a.task)); return
    item = find_scene(a.bbox)
    out_dir = config.HOME / "inference"; out_dir.mkdir(parents=True, exist_ok=True)
    tif = out_dir / f"lebanon_{item['id']}.tif"
    order, shape, crs = build_geotiff(item, a.bbox, tif)
    print(f"scene {item['id']} cloud {item['properties']['eo:cloud_cover']}% -> {tif} {shape} {crs} bands {order}")
    print("\n--- 1) default rules ---")
    print(predict_geotiff(str(tif), task=a.task))
    print("\n--- 2) fill_missing='B10' (explicit opt-in) ---")
    print(predict_geotiff(str(tif), task=a.task, fill_missing="B10"))
    print("DEMO DONE")


if __name__ == "__main__":
    main()
