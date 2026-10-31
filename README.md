# GoTrain

An AI agent that trains geospatial models on its own. Give it a GEO-Bench task and a time budget:
it explores the data, picks an approach, trains on the GPU, reads the **validation** score, keeps or
discards each idea, and finally evaluates its best model on the **test** split exactly once.

Built with Strands Agents + Claude Sonnet on Amazon Bedrock, for the AUB ACM × RASID GoTrain Challenge.

## Quick start (SageMaker JupyterLab terminal)

```bash
git clone https://github.com/<you>/gotrain && cd gotrain
pip install -r requirements.txt
bash scripts/smoke_test.sh                 # offline end-to-end test, ~1 min, no Bedrock needed
python scripts/bedrock_check.py            # agent reaches Claude on Bedrock and calls a tool

# download only the tasks you picked (not the 17.5 GB benchmark)
python -c "from gotrain.data import download_task as d; [print(d(t)) for t in ['m-eurosat','m-pv4ger-seg']]"

# by hand (step 2 of the brief): one training run, validation only
python -m gotrain.train --task m-eurosat --config '{"backbone":"resnet18","img_size":128,"epochs":15}'

# the agent: 90 minutes per task, runs in the background, survives closing the browser tab
nohup python -m gotrain.agent --tasks m-eurosat m-brick-kiln m-pv4ger-seg m-nz-cattle --budget-min 90 > agent.out 2>&1 &
tail -f agent.out
```

Results land in `~/gotrain_home/results/REPORT.md`.

## How it works

```
             ┌──────────────── Strands ReAct loop (Claude Sonnet, Bedrock) ────────────────┐
  task ───►  │ inspect_dataset → baseline → screen ideas (subset) → exploit best branches   │ ──► REPORT.md
             │      ▲                 │ start_training / check_training / stop_training      │     test score
             │      └── journal ◄─────┘ leaderboard / read_run_log / write_custom_model       │     logs
             └───────────────────────────────┬────────────────────────────────────────────┘
                                             │ subprocess per run (crash-safe, time-limited)
                     train.py: TRAIN → best epoch by VALID → model.pt + history.jsonl + metrics.json
                                             │
       finalize.py (only code path that can read TEST; lock file ⇒ once per task) → publish to store
                                             │
                  store.py: S3 (GOTRAIN_S3_BUCKET) or local, versioned model registry per task
```

**The agent's method** (in `gotrain/prompts.py`): understand the data, including a seconds-long random-forest
probe (`quick_probe`) that gives a no-spatial-context score floor and per-band importance → strong pretrained baseline →
screen single changes cheaply on a 30% subset → combine winners on full data while keeping the two best
branches alive (a light version of AIDE's tree search) → confirm near-ties → `tta` → finalize once.
Every run carries a written hypothesis and a parent run; after each result the agent writes to a journal.

**What the agent can change:** backbone (any timm model / smp encoder), segmentation architecture,
input resolution, band subset, normalization, learning rates (separate for the pretrained encoder),
schedule, loss (CE, focal, CE+Dice), class weights, augmentation, mixup, test-time augmentation,
training subset, and — through `write_custom_model` — its own PyTorch architecture.

## The test-split rule is enforced in code, not only in the prompt

* `data.load_split(..., "test")` raises unless called from `finalize.py`.
* `finalize.py` writes `results/<task>.lock.json` **before** opening the test data; any second call is refused.
* If the agent ends without finalizing, the best **validation** run is finalized automatically.

## Logs: how every score was produced

| file | content |
|---|---|
| `logs/<session>/tool_calls.jsonl` | every tool call: time, arguments, result, duration |
| `logs/<session>/<task>_messages.json` | the full agent conversation |
| `runs/<task>/<run>/config.json` | config, hypothesis, parent run |
| `runs/<task>/<run>/history.jsonl`, `train.log` | per-epoch training loss and validation score |
| `runs/<task>/journal.md` | the agent's reasoning |
| `results/<task>.json`, `REPORT.md` | test score of the chosen run, all runs, journals |

## Production pieces (extra points)

* **Model registry**: `publish_run` uploads checkpoint + config + metrics + logs as `models/<task>/v<N>/`
  and records version, lineage (parent run), val/test scores and stage (`candidate` → `production`).
* **Datasets**: `push_dataset` uploads the prepared train/valid arrays to `datasets/<task>/` so another machine trains without re-processing (the test cache is never exported).
* **S3**: see *Storage layout* below; without a bucket the same layout is written locally.
* **GoPilot integration**: `python -m gotrain.mcp_server` exposes the same 26 tools over MCP.

## Storage layout (S3-ready, no code changes)

**Rule: a file's path relative to `STORE_DIR` (`$GOTRAIN_HOME/store`) is its S3 key relative to
`s3://<bucket>/<prefix>/`.** One environment variable switches to S3: `GOTRAIN_S3_BUCKET`
(`GOTRAIN_S3_PREFIX` is optional, default `gotrain`). The store writes nothing else under `STORE_DIR`,
and `registry.json` lists files by key, never by local path.

```
$GOTRAIN_HOME/store/                         s3://<bucket>/<prefix>/
├── datasets/<task>/                         train/valid only: test never leaves the machine
│   ├── train_X.npy  train_y.npy
│   ├── valid_X.npy  valid_y.npy
│   └── meta.json    train_stats.json
└── models/<task>/
    ├── registry.json                        versions: run, val/test score, stage, lineage, files (keys)
    └── v<N>/
        ├── model.pt     config.json     metrics.json
        └── history.jsonl    train.log
```

Check it (filesystem only): `python scripts/check_store_layout.py` prints every file with its S3 key and
fails on anything that would not map 1:1. When a bucket exists, `scripts/migrate_store_to_s3.sh` is the
one-command path: copy the store, verify the registry reads back from S3, then keep `GOTRAIN_S3_BUCKET` set.

## Configuration

| variable | default |
|---|---|
| `GEO_BENCH_DIR` | `~/dataset/geobench` |
| `GOTRAIN_HOME` | `~/gotrain_home` |
| `GOTRAIN_MODEL_ID` | `eu.anthropic.claude-sonnet-5-5` |
| `GOTRAIN_REGION` | `eu-central-1` |
| `GOTRAIN_S3_BUCKET` | unset (local store) |

## Repository

```
gotrain/agent.py        Strands agent, budget loop, session resume (FileSessionManager), structured final report
gotrain/guards.py       Strands hooks: one task per session, test lock, budget; tool-call log
gotrain/prompts.py      system prompt: rules + search method
gotrain/tools.py        all 26 tools (shared by Strands and MCP): 19 core + 5 strategy + 2 hub
gotrain/strategy.py     model tournament, noise rule, branches (5 tools)
gotrain/stats.py        bootstrap CIs, paired tests, noise floor on VALID predictions
gotrain/hub.py          Hugging Face Hub model search with a GEO-Bench leakage guard (2 tools)
gotrain/probe.py        random-forest probe: score floor, band importance, spectral vs spatial
gotrain/train.py        one training run (classification, multi-label, segmentation)
gotrain/finalize.py     test evaluation, once per task
gotrain/data.py         per-task download, numpy cache, dataset description
gotrain/registry.py     runs, leaderboard, journal
gotrain/store.py        S3/local versioned store + model registry
gotrain/mcp_server.py   MCP server for GoPilot
gotrain/report.py       REPORT.md
tests/                  fake GEO-Bench tasks + scripted-model test of the full agent loop
```

## Known data issue

`m-eurosat` and `m-brick-kiln` store wrong band *names* in GEO-Bench 1.0 (pixel data is fine,
geo-bench issues #28/#29). GoTrain always selects channels by index, never by name.
