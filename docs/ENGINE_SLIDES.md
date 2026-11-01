# GoTrain - engine slides (draft, Person A)

Every number below comes from a real run on the team's SageMaker instance (Tesla T4 16 GB, real GEO-Bench
data, Claude Sonnet on Bedrock), 31 Oct 2026. Sources are given so each number can be traced. GEO-Bench
baselines come from the authors' published per-seed results (slide 2b), never estimated.

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
| m-brick-kiln | accuracy | **0.983** | 0.992 | ConvNeXt-tiny @128 px, mixup 0.2, 12 epochs | 16 | `results/m-brick-kiln.json` |
| m-chesapeake | mIoU | **0.751** | 0.770 | U-Net + ConvNeXt-tiny, CE+Dice, TTA, 12 epochs | 15 | `results/m-chesapeake.json` |

m-brick-kiln and m-chesapeake ran overnight (A8, `scripts/launch_overnight.sh`, 100 / 150 min, agent with B6 tools),
each finalized by the agent itself (no auto-finalize). m-chesapeake per-class test IoU: 0.938 / 0.968 / 0.839 /
0.785 / **0.533 / 0.603 / 0.593** (the three rare classes are the weak point).

pv4ger-seg per-class test IoU: no solar 0.984, **solar panels 0.927**.

⚠ m-nz-cattle is noisy: only 66 val / 65 test images. Val 95% CI [0.852, 0.867] (bootstrap over images);
cattle IoU on test 0.722 (2% of pixels). The paired tests vs the other runs are significant (p 0.001), but
they only measure image sampling on these 66 images, not seed-to-seed training noise (one seed per config).

Honest note for the eurosat ensemble: its val (0.987) was below the best single run (0.988) and its test
(0.979) below two of its members alone (0.981, 0.982). The agent's own reason was "near-tie within noise;
averaging reduces seed variance" - stats confirm the tie (next slide), so it was a defensible but not a
winning choice.

---

## 2b. Against the GEO-Bench baselines

Source: the GEO-Bench authors' raw per-seed results, `baseline_classification_results.csv` and
`baseline_segmentation_results.csv` in [ServiceNow/geo-bench](https://github.com/ServiceNow/geo-bench)
(commit `5a5a763c5b`, 2024-01-15), for the paper Lacoste et al., *GEO-Bench: Toward Foundation Models for
Earth Monitoring*, NeurIPS 2023 Datasets & Benchmarks ([arXiv 2306.03831](https://arxiv.org/abs/2306.03831)).
The paper itself shows these only as plots (Fig. 9, Fig. 11). Rows: full training set ("1.00x train"),
12 seeds per model, test metric mean ± SD; best baseline model per task.

| Task | Best GEO-Bench baseline (12 seeds) | GoTrain test (1 look) | Difference |
|---|---|---|---|
| m-eurosat (acc.) | ResNet18-MoCo-S2, RGB: **0.9800 ± 0.0028** (max 0.985) | **0.979** | -0.001: on par, inside the baseline's seed spread |
| m-pv4ger-seg (IoU) | ResNet18-U-Net: **0.9408 ± 0.0010** (max 0.9419) | **0.9556** | +0.015 (~15 baseline SDs) |
| m-nz-cattle (IoU) | ResNet50-U-Net: **0.8107 ± 0.0020** (max 0.8138) | **0.857** ⚠ | +0.046 (65 test images: noisy) |
| m-brick-kiln (acc.) | ConvNeXt-B, RGB: **0.9887 ± 0.0013** (max 0.991) | **0.983** | **-0.006: below** the best baseline (~ResNet18 level, 0.9823 ± 0.0042) |
| m-chesapeake (IoU) | ResNet50-U-Net: **0.7112 ± 0.0034** (max 0.7162) | **0.751** | +0.040 (~12 baseline SDs) |

Caveats, all of them material:
- **Benchmark version**: the baselines ran on classification v0.8.5 / segmentation v0.4; we use v1.0.
- **Inputs differ**: the eurosat baselines in the CSV at 1.00x are RGB-only (3 channels); ours uses all 13 bands.
- **Metric**: the paper says "IoU" without stating the class averaging. Ours is the mean over both classes;
  their pv4ger values (~0.94) fit a two-class mean better than panel-only IoU, but that is our inference.
- **Seeds**: their numbers average 12 seeds; ours is one model (one test look), so a single draw.

Honest summary: on eurosat we match the best baseline, we do not beat it; on m-brick-kiln we are **below** the
best baseline (a ConvNeXt-**B**; ours is a ConvNeXt-tiny at 128 px chosen from four VALID-tied runs); on all three
segmentation tasks the agent's U-Net + ConvNeXt-tiny clearly exceeds the published ResNet U-Nets.

## 2c. Val -> test gap (does our selection overfit VALID?)

| Task | GoTrain val | GoTrain test | gap | Baseline gap, same task (best model, 12 seeds) |
|---|---|---|---|---|
| m-eurosat | 0.987 (ensemble) | 0.979 | **-0.008** | -0.0016 ± 0.0030 |
| m-pv4ger-seg | 0.9616 | 0.9556 | **-0.006** | -0.0058 ± 0.0009 |
| m-nz-cattle | 0.860 | 0.857 | **-0.003** | -0.0044 ± 0.0009 |
| m-brick-kiln | 0.992 | 0.983 | **-0.009** | -0.0046 ± 0.0018 |
| m-chesapeake | 0.770 | 0.751 | **-0.019** | -0.0154 ± 0.0045 |

Reading: on all three segmentation tasks our gap is inside or near the baselines' (chesapeake -0.019 vs
-0.015 ± 0.005: within one SD; no sign of extra overfitting to VALID). On brick-kiln our gap (-0.009) is about
twice the baseline's - the same winner's curse as eurosat: the agent picked r016 from four runs tied at 0.992.
On eurosat our gap is larger than the baselines' typical gap (-0.008 vs -0.002 ± 0.003): picking the top of
7 runs that were statistically tied on VALID (slide 3) inflates the val score - the winner's curse.

---

## 2d. Is the agent better than random search? (m-eurosat, VALID only)

Same task, same trainer, same 60-min wall clock each, one run at a time on the T4, run one after the other
(16:57-18:54 UTC). Random search samples the agent's own menu (curated backbones, img size, lr, epochs,
augmentation, label smoothing, mixup, TTA). Isolated homes with TRAIN/VALID links only; the agent ran with
`--valid-only` (guard cancels finalize). **No lock file was created; test was never read.**

| | Agent (Claude) | Random search |
|---|---|---|
| runs finished in 60 min | 15 | 29 |
| best VALID accuracy | **0.991** [95% CI 0.985, 0.996] | **0.986** [0.979, 0.993] |
| best config | SSL4EO ResNet50, 128 px, TTA, 40 ep | SSL4EO ResNet18, 128 px, TTA, 30 ep |
| best reached | run 14, ~48 min | run 2, ~0.7 min (lucky early draw) |
| median run | 0.986 | 0.978 |
| runs >= 0.984 | 11 / 15 | 8 / 29 |

Paired test, agent best vs random best (same 1000 val images, McNemar): **+0.005 [-0.001, +0.011], p = 0.18**
-> **not significant**. Honest reading: with one hour each, the agent's best is not distinguishable from
random search's best. What the agent does better is *consistency*: its typical run is ~0.8 pts higher
(median 0.986 vs 0.978) and 73% of its runs reach 0.984 vs 28% for random, with half as many runs
(it spends time thinking and trains larger models). Caveat: one repetition of each method on one task.

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

**Seed noise, measured** (m-eurosat, chosen config r003 = SSL4EO ResNet50 @128 px, retrained with 2 more seeds,
VALID only, isolated home): 0.988 / 0.985 / 0.988 -> **mean 0.987 +- 0.0017 SD**. The 0.003 spread between
seeds is the same size as the gaps between the top runs: another reason those runs are a tie.

---

## 3b. Data Advisor: model or data? (m-eurosat, real GPU run, 12 min, VALID only)

Config of the best run (r003), equal compute per point (960 optimizer steps), nested subsets of TRAIN:

| TRAIN used | 200 (10%) | 500 (25%) | 1000 (50%) | 2000 (100%) |
|---|---|---|---|---|
| VALID acc | 0.941 | 0.964 | 0.978 | 0.984 |

- Power law on the error fits (r2 0.995). **2x labels -> +0.56 pts [+0.30, +0.84]**: real, and ~35% of the remaining
  1.6% error, but it costs 2000 new labels.
- **Label doctor** (2-fold out-of-fold on TRAIN): 40 labels flipped on purpose -> it found **all 40** in its top 80.
  57 natural suspects (2.9% of TRAIN) in a thumbnail gallery: the cheapest thing to fix first.
- **Labeling queue** (most uncertain first): +100 uncertain vs +100 random -> +0.60 pts [-0.10, +1.30], **p 0.15:
  not significant** - honest result: uncertainty sampling did not clearly beat random here.
- Verdict: both, cheapest first: re-check the 57 suspects, then labels/model. Exposed to GoPilot via MCP
  (`data_advisor_report`), not to the training agent. Outputs: `docs/advisor/m-eurosat/` on branch feature/extras.

---

## 3c. Reproduce receipts: can we rebuild a production model? (real, m-eurosat v1)

Each registry version gets a receipt: sha256 of the TRAIN/VALID cache files, full config + seed, the code commit at
training time (reconstructed from git history by the run's timestamp: best effort, labelled so), and the original
VALID score with its 95% CI. One command retrains it in an isolated home and checks it.

| | m-eurosat v1 (r003, production) |
|---|---|
| fingerprint | `a969712199346694` |
| original VALID | 0.988, CI [0.981, 0.994] |
| reproduced VALID | **0.988** (1.7 min, T4) |
| data hashes | match |
| code | newer (bffec55 vs f930064 at training) - result unchanged |
| verdict | **REPRODUCED** |

GPU kernels are not bit-deterministic, so the criterion is "inside the original CI", not equality.
MCP tool `model_receipt`; files in `docs/receipts/m-eurosat/` on branch feature/extras.

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

- [x] Day-2 results: m-brick-kiln 0.983 acc., m-chesapeake 0.751 mIoU (overnight A8, finalized by the agent)
- [x] m-nz-cattle result: test 0.857 mIoU, labelled noisy
- [x] GEO-Bench baselines (slide 2b, cited from the authors' per-seed CSVs) and val->test gaps (2c)
- [x] S3: organisers want the local layout only (slide 11); no bucket needed
