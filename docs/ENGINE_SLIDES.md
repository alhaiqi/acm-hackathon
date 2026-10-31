# GoTrain - engine slides (draft, Person A)

Every number below comes from a real run on the team's SageMaker instance (Tesla T4 16 GB, real GEO-Bench
data, Claude Sonnet on Bedrock), 31 Oct 2026. Sources are given so each number can be traced. GEO-Bench paper
baselines are NOT included yet: look them up in the paper before adding them (never estimate them).

---

## 1. What the engine is

One sentence: **the agent decides, the engine executes - and the engine is the only thing that can touch test.**

```
Strands Agent (Claude Sonnet, Bedrock)
  │  hooks: guards (task / test lock / budget) + tool-call log      (gotrain/guards.py)
  │  26 tools                                                         (gotrain/tools.py, strategy.py, hub.py)
  ▼
train.py   one run = one subprocess: TRAIN -> best epoch by VALID -> model.pt + val_preds.npz + history
finalize.py   the ONLY code path that reads TEST; lock file written BEFORE the data is opened
store.py   versioned model registry (local store, S3 layout); MCP server exposes the same tools to GoPilot
```

Run folder contract shared with the strategy side: `config.json, status.json, best_val_metrics.json,
val_preds.npz, history.jsonl` (WORK_SPLIT.md section 1).

---

## 2. Real results so far (test split used once per task)

| Task | Metric | Test | Val | Chosen model | Runs | Source |
|---|---|---|---|---|---|---|
| m-eurosat | accuracy | **0.979** | 0.987 | ensemble of 3 SSL4EO ResNet50 @128 px | 7 | `results/m-eurosat.json` |
| m-pv4ger-seg | mIoU | **0.9556** | 0.9616 | U-Net + ConvNeXt-tiny, 8-way TTA | 5 | `results/m-pv4ger-seg.json` |
| m-nz-cattle (extra) | mIoU | **0.857** ⚠ noisy | 0.860 | U-Net + ConvNeXt-tiny, 640 px, TTA | 5 | `results/m-nz-cattle.json` |
| m-brick-kiln | accuracy | *day 2* | | | | test untouched |
| m-chesapeake | mIoU | *day 2* | | | | test untouched |

pv4ger-seg per-class test IoU: no solar 0.984, **solar panels 0.927**.

⚠ m-nz-cattle is noisy: only 66 val / 65 test images. Val 95% CI [0.852, 0.867] (bootstrap over images);
cattle IoU on test 0.722 (2% of pixels). The paired tests vs the other runs are significant (p 0.001), but
they only measure image sampling on these 66 images, not seed-to-seed training noise (one seed per config).

Honest note for the eurosat ensemble: its val (0.987) was below the best single run (0.988) and its test
(0.979) below two of its members alone (0.981, 0.982). The agent's own reason was "near-tie within noise;
averaging reduces seed variance" - stats confirm the tie (next slide), so it was a defensible but not a
winning choice.

---

## 3. Evidence, not leaderboard noise (stats.py on the real VALID predictions)

**m-eurosat** (1000 val images): best run 0.988, 95% CI [0.981, 0.994]. Paired McNemar vs the next runs:
p = 1.0, 0.77, 0.34, 0.21 -> **the top 5 runs are statistically tied**.

**m-pv4ger-seg** (403 val images, paired bootstrap over images):

| best (r005) vs | diff mIoU | 95% CI | p |
|---|---|---|---|
| same model, other seed (r003) | +0.0022 | [-0.0007, +0.0049] | 0.12 -> tie |
| EfficientNet-B4 U-Net (r004) | +0.0032 | [+0.0013, +0.0056] | 0.001 |
| ResNet34 U-Net baseline (r002) | +0.0067 | [+0.0043, +0.0095] | 0.001 |

Message: "+0.2 points" that other teams would report as progress is noise here; we can say which gains are real.

---

## 4. Pretraining that matches the sensor wins

m-eurosat (13-band Sentinel-2), VALID accuracy:

| Model | Val |
|---|---|
| ImageNet ResNet18, 128 px | 0.944 |
| SSL4EO-S12 ResNet18 (Sentinel-2 MoCo) | 0.982 |
| **SSL4EO-S12 ResNet50 (Sentinel-2 MoCo)** | **0.988** |

The SSL4EO weights were added to the engine's candidate pool (torchgeo, first conv adapted to any band
count, ImageNet fallback); the agent found and chose them on its own.

---

## 5. Spectral or spatial? The 6-second random-forest probe

| Task | RF on band statistics (val) | Best CNN (val) | Reading |
|---|---|---|---|
| m-eurosat | 0.848 accuracy (6 s) | 0.988 | band values carry most of it; CNN adds +14 pts |
| m-pv4ger-seg | 0.654 mIoU (134 s) | 0.962 | almost all signal is spatial (panel shapes) |

Caveat on band importance: GEO-Bench 1.0 stores wrong band NAMES for m-eurosat (geo-bench #28/#29);
importances are reliable by channel index only. The engine always selects channels by index.

---

## 6. The test split is protected in code, three times

1. `data.load_split(..., "test")` raises unless called from `finalize.py`.
2. `finalize.py` writes `results/<task>.lock.json` **before** opening test; a second call is refused,
   even after a crash. Invalid run ids are rejected before the lock, so a typo cannot burn the test look.
3. A Strands `BeforeToolCallEvent` hook cancels `finalize_task` once the lock exists, and cancels any tool
   call naming another task (verified in the offline loop test: the second finalize is cancelled before
   it runs).

Plus a **leakage guard on Hugging Face models**: models whose id/tags/training datasets mention a
GEO-Bench source dataset are refused. A search for "eurosat" returned 9 models, all blocked - e.g.
`cm93/resnet50-eurosat` was fine-tuned on full EuroSAT and has very likely seen our test images.

---

## 7. Beyond the curated pool: Hugging Face Hub (no web search needed)

Web search is not available on the instance (no search API key, no Bedrock search tool); the HF Hub API is.
`search_hf_models` -> `add_hf_candidate` builds the model for the task's bands, records its true size and
fixed input size, and adds it to the tournament menu.

Real GPU runs with Hub models (VALID only, on tasks already finalized):
- `timm/convnext_tiny.in12k_ft_in1k` on m-eurosat, 13 bands: val 0.966 after 4 epochs
- U-Net + `timm/efficientnet_b2.ra_in1k` on m-pv4ger-seg: val mIoU 0.915 after 2 epochs

---

## 8. Real data found bugs that synthetic tests never did

| Symptom on real data | Cause | Fix |
|---|---|---|
| m-eurosat could not be cached | GEO-Bench specs store resolution as numpy int64 -> JSON error | cast to float |
| process killed on the 15 GB instance | train statistics loaded 2000 images as float64 (~10 GB) | streamed in chunks; exact mean/std (checked: diff 0.0 / 2e-7) |
| m-brick-kiln cache 6.4 GB | dataset ships float64 | cache as float32 (3.2 GB) |
| report crashed after first real run | timing file in results/ read as a score | moved to GOTRAIN_HOME |
| registry demo could demote production | re-publish overwrote the stage | production can only change via promote() |

---

## 9. Speed facts (T4, ResNet18 baseline, full data)

| Task | s / epoch | 30-epoch run |
|---|---|---|
| m-eurosat | 1.3 | 0.7 min |
| m-brick-kiln | 4.5 | 2.2 min |
| m-chesapeake | 31.75 | ~16 min |
| m-pv4ger-seg | 33.3 | ~17 min |

Data is read once into numpy memmaps (`data.build_cache`), so the GPU is not waiting on 1000s of HDF5 files
(GPU utilisation 100% during runs).

---

## 10. Production / GoPilot

- **MCP**: `python -m gotrain.mcp_server` exposes all 26 tools. A separate Strands agent using only MCPClient
  (no GoTrain imports) answered "which tasks used their test split and what eurosat versions exist" correctly
  from the live state (`scripts/strands_mcp_agent.py`).
- **Model registry**: versions with stage (candidate -> staging -> production), lineage (parent run),
  val/test scores, ensemble membership; `pull_model` fetches a version for inference elsewhere.
- **S3**: the local store is laid out as the bucket (next slide); switching is one env var.

---

## 11. Storage: the local store IS the S3 layout

**Rule: path under `$GOTRAIN_HOME/store/` = S3 key under `s3://<bucket>/<prefix>/`.**
Switch to S3 with one variable: `GOTRAIN_S3_BUCKET` (prefix optional). No code change.

```
store/                                   -> s3://<bucket>/<prefix>/
├── datasets/m-eurosat/                  train/valid arrays + meta + train stats (test never exported)
└── models/
    ├── m-eurosat/registry.json          v1-v3 production (ensemble), v4 staging; files listed by KEY
    │   └── v1..v4/ model.pt config.json metrics.json history.jsonl train.log
    └── m-pv4ger-seg/registry.json       v1 production
        └── v1/ ...
```

Real store on SageMaker: 33 files, `scripts/check_store_layout.py` -> **LAYOUT OK** (every file printed
with its S3 key). The check caught two real problems first, both fixed:
test arrays had been exported into the store, and `registry.json` listed absolute local paths
instead of keys. When a bucket exists: `scripts/migrate_store_to_s3.sh` (copy + read-back check).

---

## Open before the presentation

- [ ] Day-2 results: m-brick-kiln, m-chesapeake (overnight, `scripts/launch_overnight.sh`, after B6)
- [x] m-nz-cattle result: test 0.857 mIoU, labelled noisy
- [ ] GEO-Bench paper baselines per task (look up, cite)
- [x] S3: organisers want the local layout only (slide 11); no bucket needed
