# GoTrain — Work Split (2 people)

AUB ACM × RASID GoTrain Challenge · 31 Oct – 1 Nov 2026

Goal: two people working in parallel with minimal dependency. The halves meet at **one frozen contract** (the run-folder files). Everything else is owned by one person only.

---

## 0. Adapting the design doc to GoTrain

The ML Engineering Agent design doc targets MLE-bench (12 h runs, many competitions). Here is what carries over to GoTrain and what changes.

| Doc idea | Verdict | Why |
|---|---|---|
| Model tournament (screen → race → rank by mean ± SE) | **Keep — headline feature** | It is the "creative workflow" judges score, and it stops the LLM from defaulting to ResNet every time |
| Noise-aware acceptance (gain must beat 1–2 SE) | **Keep** | Same idea as `compare_runs`; almost free |
| Decision trace for RL | **Keep, trimmed** | Cheap (extends the existing logger) and the strongest "production / future work" slide |
| Ensemble of top picks | **Keep** | Usually the cheapest real score gain |
| k folds as the k runs | **Change → seeds + bootstrap on the fixed val split** | GEO-Bench has a fixed val split and the rule says tune on val; k folds multiply GPU cost by k on one GPU with ~90 min/task |
| Lockbox, up to 3 looks | **Change → lockbox = half of val, never test** | The brief allows exactly one test look. Select on val half A, confirm on val half B |
| Web search for models | **Toggle; default = curated candidate pool** | SageMaker may not have open internet; search returns models too big for the compute. This is the doc's config B (defaults-only tournament) |
| 3 parallel branches | **Sequential "top-2 families"** | One GPU, so parallel branches would just take turns |

---

## 1. The contract (freeze in the first 30 min)

Every training run writes exactly these files:

```
runs/<task>/<run_id>/
  config.json            {task, config, hypothesis, parent, seed}
  status.json            {state: running|done|failed, epoch, best_val}
  best_val_metrics.json  {primary: float, ...}
  val_preds.npz          probs: (N,C) or (N,C,H,W) float16,  y: labels
  history.jsonl          one line per epoch
```

- **Person A produces** these files on the real GPU.
- **Person B consumes** them. B builds against a fake-run generator from minute one, so B needs no GPU, no data and no Bedrock.

Changing the contract after the freeze requires both people to agree.

---

## 2. Person A — Engine (owns the GPU)

**Files owned:** `data.py`, `train.py`, `finalize.py`, `store.py`, `mcp_server.py`, `scripts/`

- [ ] **A1. Environment.** Run `python scripts/bedrock_check.py` and `bash scripts/smoke_test.sh` on SageMaker. *Done when:* both pass.
- [ ] **A2. Data.** Download m-eurosat and m-pv4ger-seg, fix `download_task` if the HF layout differs, and build the caches. Then do the same for m-brick-kiln and m-nz-cattle. *Done when:* `inspect_dataset` works on all 4.
- [ ] **A3. Timing.** Run one manual `python -m gotrain.train` per task and record minutes per epoch. *Done when:* `t_full` per task is shared with B (it sets k in the tournament).
- [ ] **A4. Contract output.** Save `val_preds.npz` in `train.py`. **Do this first in `train.py`.**
- [ ] **A5. Candidate pool.** Add these as backbone options:
  - Classification: ImageNet resnet18/50, convnext or efficientnet, and torchgeo SSL4EO Sentinel-2 weights (with fallback to ImageNet).
  - Segmentation: U-Net and Segformer variants.
- [ ] **A6. Test-side ensemble.** Add `predict(run_id, split)`. `finalize` should accept a list of run_ids, run each on test and average them. All test-touching code stays in A's files.
- [ ] **A7. Extra points.** S3 upload/download (`GOTRAIN_S3_BUCKET`), a model-registry demo, and an MCP server demo (`python -m gotrain.mcp_server`).
- [ ] **A8. Launch overnight runs** after sync point 3.

---

## 3. Person B — Brain (laptop only)

**Files owned:** `agent.py`, `prompts.py`, `strategy.py` (new), `stats.py` (new), `trace.py` (new), `report.py`, `tests/`

- [ ] **B1. Fake runs.** Write `tests/make_fake_runs.py`, which generates run folders that follow the contract, with random `val_preds`.
- [ ] **B2. Statistics (`stats.py`).**
  - Bootstrap confidence interval on the primary metric.
  - Paired test: McNemar for classification, per-image bootstrap for segmentation.
  - Noise-floor estimate (SE of the leader).
- [ ] **B3. Tournament tools (`strategy.py`).**
  - `propose_candidates(task)` picks from the curated pool using the `inspect_dataset` output (band count, image size, dataset size). Web search sits behind a toggle.
  - `race_decision(run_ids)` returns keep or drop, dropping anything more than 2 SE below the leader.
  - `compare_runs(a, b)` returns the CI and p-value and says whether to accept.
- [ ] **B4. Ensemble selection.** `select_ensemble(run_ids)` does greedy selection on val probabilities and confirms on val half B.
- [ ] **B5. Decision trace (`trace.py`).** One JSONL record per decision with `decision_id, parent_id, state, options, choice, outcome, reward`, attached to the existing `ToolLogger`.
- [ ] **B6. Prompt rewrite.** Phases: intake → tournament → refine with the noise rule → ensemble → finalize once.
- [ ] **B7. Report and presentation.** `REPORT.md`, slides, and the real GEO-Bench paper baselines (looked up, never invented).

**Avoiding merge conflicts:** `tools.py` is the only shared file. B adds a single line to register the new tools:

```python
from gotrain.strategy import ALL_STRATEGY_TOOLS
ALL_TOOLS += ALL_STRATEGY_TOOLS
```

---

## 4. Sync points (only three)

| # | When | What | Time |
|---|---|---|---|
| 1 | Start | Agree on and freeze the contract (section 1) | 30 min |
| 2 | A has real runs on 1 task | B points the strategy tools at the real run folders and fixes the mismatches | ~1 h |
| 3 | Before the overnight launch | Merge, run `smoke_test.sh`, launch. **Leave 2 tasks unfinalized**, because finalize locks test permanently; day 2's improved agent runs on those | ~30 min |

---

## 5. Non-negotiable rules (both people)

- The agent tunes on **validation only**. Test is touched **once**, through `finalize_task` (the lock file is written before the read, and a second call is refused). Never weaken this.
- Keep every log: `logs/<session>/tool_calls.jsonl`, the journals, and the decision traces.
- Never invent baseline numbers.
- Commit small and often; each person commits only to the files they own.

---

## 6. Suggested owners

- **Person A (Engine): Mohammed.** Matches the AWS / infrastructure track and the S3/MCP extra points.
- **Person B (Brain): Amoor.** Wrote the tournament, noise-rule and decision-trace design.
