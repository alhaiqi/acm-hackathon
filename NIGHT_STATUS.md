# Night status (autonomous run, 31 Oct -> 1 Nov 2026)

Times are UTC (Beirut = UTC+2). Newest entry at the bottom. Questions for Mohammed go here, at the top.

## Needs you
- A teammate (~/amoor/trial_home) ran an agent on the shared GPU at ~20:15 UTC (m-brick-kiln, --valid-only,
  20 min). A8 waits for any running agent before it starts, so if teammates keep starting runs A8 keeps
  waiting. Please ask them to keep the GPU free until A8 finishes (~01:00 UTC).

## Queue (priority)
1. A8 when Omar's B6 lands (watching origin/main for gotrain/prompts.py, base 9cf2434)
2. agent-vs-random report (experiment running since 16:57)
3. 2 extra seeds of the chosen eurosat config (VALID only)
4. P1 Data Advisor -> P2 Reproduce receipt -> P3 predict_geotiff (branch feature/extras)

## Log
- 17:30 SageMaker session renewed until 05:25 UTC. Plan confirmed. Waiter armed (B6 / experiment / WSL).
- 17:30 Earlier today (already on main): GEO-Bench baselines + val->test gaps in ENGINE_SLIDES (9cf2434);
  P1 code written + CPU test passes on Windows (not committed: waits for the Linux smoke test before/after).
- Decision: Linux smoke test runs in WSL on this laptop (venv on D:), not on SageMaker, so the running
  experiment gets the GPU/CPU to itself. Reason: your rule "don't disturb it".
- 17:50 WSL Linux test env ready on D: (fixed a torchvision/torch CPU mismatch). Smoke test BEFORE P1
  (clean checkout 9cbfb8d): AGENT LOOP OK, store layout OK, ENGINE OK. AFTER P1: same + EXTRAS OK.
- 17:50 P1 code committed (CPU-tested on fake data; numbers there are meaningless by design). The real
  GPU run on m-eurosat waits for the experiment to free the GPU.
- 18:54 AGENT VS RANDOM (m-eurosat, 60 min each, VALID only) DONE. Agent best 0.991 [0.985, 0.996], 15 runs,
  best at run 14 (~48 min). Random best 0.986 [0.979, 0.993], 29 runs, best at run 2 (~0.7 min).
  Paired McNemar +0.005 [-0.001, +0.011], p 0.18: NOT significant. Agent more consistent (median 0.986 vs
  0.978; 11/15 vs 8/29 runs >= 0.984). No lock files. -> ENGINE_SLIDES 2d (main a313f88).
- 18:57 Seeds started on GPU: m-eurosat__r003 config (SSL4EO R50, 128 px, 30 ep) with seeds 2, 3, isolated
  home. Note: r003 stopped by early stopping (27 ep, 1.6 min), so its 8-min time limit is irrelevant.
- Decision: P2 code drafted + CPU-tested meanwhile (receipt OK), NOT committed until P1's GPU run is done,
  to keep "each feature fully finished before the next" in commit order.
- 18:58 SEEDS: m-eurosat r003 config (SSL4EO R50, 128 px) over 3 seeds = 0.988 / 0.985 / 0.988 ->
  mean 0.987 +- 0.0017 SD (VALID). Isolated home; finalized runs untouched.
- 19:12 P1 DATA ADVISOR real GPU run on m-eurosat (12.2 min): learning curve 0.941/0.964/0.978/0.984 at
  10/25/50/100% (power law r2 0.995); 2x labels -> +0.56 pts [+0.30, +0.84] (~35% of remaining error).
  Label doctor recovered 40/40 injected flips in its top 80; 57 natural suspects (2.9% of TRAIN).
  Labeling queue: +100 uncertain vs +100 random = +0.60 pts [-0.10, +1.30], p 0.15 (NOT significant).
- Decision: the first verdict text ("not distinguishable from zero") was wrong for a CI of [+0.30, +0.84];
  rewrote the verdict relative to the remaining error ("Both, cheapest first") and regenerated SUMMARY.md
  from the saved results (no retraining). Outputs copied to docs/advisor/m-eurosat/.
- 19:20 Smoke test after P1: AGENT LOOP OK, store layout OK, ENGINE OK, EXTRAS OK. P1 DONE.
- 19:26 P2 REPRODUCE RECEIPT demo on m-eurosat v1 (r003, production): fingerprint a969712199346694;
  retrained in an isolated home (1.7 min) -> VALID 0.988 = original 0.988, inside CI [0.981, 0.994],
  data hashes match -> REPRODUCED. Code differs (now bffec55, receipt f930064: reconstructed from git
  history by the run's timestamp, labelled best effort). Files: docs/receipts/m-eurosat/.
- Decision: the laptop ran out of memory (a waiter was killed, 1.1 GB free of 7.7). Cause: WSL VM holding
  RAM. I now start WSL only for each smoke test and shut it down after; restarted the waiter because your
  standing instruction requires one.
- 19:40 Smoke test after P2: AGENT LOOP OK, store layout OK, ENGINE OK, EXTRAS OK (advisor + receipt). P2 DONE.
- 19:46 B6 LANDED (Omar, B6a-B6g, 030384e: prompt follows tournament + branches, race fits budget,
  branch memory, Hub trust rules, arXiv search tool). Features paused (no feature GPU job was running).
- 19:48 SageMaker main checkout pulled to 030384e. smoke_test.sh: AGENT LOOP OK, store layout OK,
  ENGINE OK. Omar's new tests: PROMPTS OK, STRATEGY OK, HUB RULES OK, LITERATURE OK.
- 19:49 A8 LAUNCHED from a notebook kernel (busy kernel = activity for idle shutdown):
  launch_overnight.sh -> its own smoke test, then m-brick-kiln (100 min) and m-chesapeake (150 min).
  Expected end ~00:15 UTC. Checked every 30 min.
- 19:50 P3 predict_geotiff code written and CPU-tested (fake GeoTIFF: output keeps CRS/transform,
  2 bands, refusal on a missing band, opt-in fill with warning; rasterio-missing path returns a clear
  message). Real Lebanon demo waits for the GPU (after A8).
- 20:05 Merged main (B6) INTO feature/extras (main untouched). Merged branch: AGENT LOOP OK, store layout OK, ENGINE OK, EXTRAS OK (advisor, receipt, inference).
- 20:20 A8 attempt 1 did NOT run: launcher said "SMOKE TEST FAILED" although the smoke output had ENGINE OK.
  Cause: my launch_overnight.sh used `smoke | tee | grep -q` under pipefail (grep -q closes the pipe early,
  tee gets SIGPIPE -> false failure). No test split touched. Fixed on main (e500751, my engine file).
- 20:21 A8 relaunched from a new notebook kernel; it is waiting for the teammate's agent run to finish,
  then smoke test, then m-brick-kiln (100 min) + m-chesapeake (150 min).
- 20:32 A8 RUNNING: smoke passed (20:31), m-brick-kiln agent started 20:32 (100 min) using B6 tools (search_papers, propose_candidates, submit_shortlist). An 'A8_FAILED' alert at 20:32 was a false alarm from my own waiter matching the fix's commit message in the log; waiter pattern fixed.
