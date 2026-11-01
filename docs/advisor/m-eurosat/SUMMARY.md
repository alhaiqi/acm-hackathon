# Data Advisor: m-eurosat - model or data?

Config from `m-eurosat__r003` (best VALID run), 960 optimizer steps per model, cuda,
12.2 min. TRAIN 2000, VALID 1000. Test never read.

## Verdict
**Both, cheapest first.** 2x labels gives a real but small gain: +0.56 pts [+0.30, +0.84], i.e. it removes ~35% of the remaining 1.6% error at the cost of 2000 new labels. Cheaper first: re-check the 57 suspicious labels (2.9% of TRAIN, `label_doctor.html`), then the model side (seed noise and the agent's model search, slides 2d/3).

## Learning curve (same compute per point, VALID accuracy, bootstrap 95% CI)
| TRAIN used | images | val acc | 95% CI |
|---|---|---|---|
| 10% | 200 | 0.9410 | [0.9250, 0.9560] |
| 25% | 500 | 0.9640 | [0.9520, 0.9750] |
| 50% | 1000 | 0.9780 | [0.9680, 0.9860] |
| 100% | 2000 | 0.9840 | [0.9760, 0.9910] |

Power law on error: b = 0.580, r2 = 0.995.
2x labels (4000 images): predicted val acc 0.9896, gain **+0.56 pts** [95% +0.30, +0.84].

## Label doctor (2-fold out-of-fold on TRAIN)
Sanity check: 40 labels flipped on purpose; the doctor found **100%** of them
in its top 80. Natural suspects: 57 (2.9% of TRAIN):
see `label_doctor.html` (they are candidates to re-check, not proven errors).

## Labeling queue
Next 100 images by margin: `labeling_queue.html` / `labeling_queue.json`.
Check: 50% + 100 uncertain -> val 0.9840; 50% + 100 random -> 0.9780;
paired diff +0.60 pts [-0.10, +1.30], p 0.146.

## Caveats
- uncertainty sampling did not beat random sampling significantly (p 0.15)
- config keys ignored by the advisor's trainer: none
- one seed per point: training noise is not in the CIs (they cover val-image sampling only)
- thumbnails use channels 3,2,1 by index (m-eurosat stores wrong band NAMES in GEO-Bench 1.0)
- 100% point (0.984) is below r003's 0.988: the advisor trains to the LAST step with no val-based epoch choice, so VALID stays a clean measurement (train.py picks the best epoch on VALID)
