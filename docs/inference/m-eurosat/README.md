# P3 real demo: m-eurosat production ensemble on a Sentinel-2 crop over Zahle, Lebanon

Run on SageMaker (T4), 2026-11-01, `python scripts/demo_predict_lebanon.py` (feature/extras, 88ca3fd).
Full output: `lebanon_zahle_demo_output.txt`. Preview: `lebanon_zahle_preview.png`.

- **Input**: Element 84 Earth Search, collection `sentinel-2-c1-l2a`, scene `S2B_T37SBT_20240718T082953_L2A`
  (cloud 0.0008%), bbox 35.86-35.92 E / 33.82-33.87 N, 571 x 572 px at 10 m, EPSG:32637, 12 bands (L2A has no B10).
  Read anonymously over HTTPS (public COGs, no credentials). Two earlier scenes were skipped: no data over the bbox.
- **Run 1, default rules: REFUSED** - "input lacks bands the model was trained on: ['B10']".
- **Run 2, `fill_missing="B10"` (explicit opt-in)**: production ensemble v1-v3 (r003, r007, r004), 289 tiles of
  64 px, 8 s. Warnings raised: B10 filled with the training mean; value range differs on 6 bands (input medians
  2.4-5.6x the training means: L2A with the +1000 offset vs the L1C training data).
- **Output**: 2-band GeoTIFF (class, confidence %) with the input's CRS/transform. Area: Annual Crop 95.1%,
  Industrial Buildings 2.5%, Highway 2.4%; mean confidence 74%.

## Honest reading

The safety layer worked: it refused the missing band, and after the opt-in it warned that the value ranges
do not match the training data. The prediction itself is **not reliable**: Zahle town (right half of the crop)
is labelled Annual Crop, not Residential. The confidence map shows it - confidence drops from ~85-95% on the
fields to ~40-60% over the town. Causes, in order of likely size: L2A surface reflectance vs L1C top-of-atmosphere
(the range warning), a filled band, and a domain shift (EuroSAT is European land cover at 64 px).
Fix for real use: feed L1C (or retrain on L2A), which is what the warning tells the user.
