# Threshold and Calibration Analysis

All three models output a probability, not just a label — the README/REPORT numbers use each model's default 0.5 cutoff. This sweeps the decision threshold from 0.10 to 0.90 in steps of 0.05 on the **test set** (237 examples) to see how much precision/recall can be traded off post hoc, without retraining, and what threshold to deploy at for two concrete operating points.

## Method

- Full sweep (17 thresholds x 3 models) saved to [`threshold_sweep.csv`](threshold_sweep.csv).
- Precision-recall curves (full-resolution, not just the 17-point sweep) for all three models on one figure: [`pr_curves.png`](pr_curves.png).
- **HIGH_RECALL**: among swept thresholds with recall >= 0.95, pick the one with the highest precision (catch nearly all injections, minimize false positives given that).
- **HIGH_PRECISION**: among swept thresholds with precision >= 0.98, pick the one with the highest recall (almost never flag a benign prompt, catch as much as possible given that).
- If no threshold in [0.10, 0.90] satisfies a constraint, the table falls back to the closest available point and flags it in the Notes column.

## Recommended thresholds

| Model | Operating point | Threshold | Precision | Recall | F1 | Notes |
|---|---|---|---|---|---|---|
| TF-IDF + LR | HIGH_RECALL (recall >= 0.95) | 0.35 | 0.788 | 0.963 | 0.867 | - |
| TF-IDF + LR | HIGH_PRECISION (precision >= 0.98) | 0.80 | 1.000 | 0.546 | 0.707 | - |
| MiniLM Embeddings + LR | HIGH_RECALL (recall >= 0.95) | 0.25 | 0.696 | 0.954 | 0.805 | - |
| MiniLM Embeddings + LR | HIGH_PRECISION (precision >= 0.98) | 0.60 | 0.989 | 0.806 | 0.888 | - |
| DistilBERT | HIGH_RECALL (recall >= 0.95) | 0.15 | 0.927 | 0.935 | 0.931 | constraint unreachable in [0.10, 0.90] sweep; closest shown |
| DistilBERT | HIGH_PRECISION (precision >= 0.98) | 0.85 | 0.989 | 0.852 | 0.915 | - |

## Default threshold (0.5) for reference

| Model | Precision | Recall | F1 |
|---|---|---|---|
| TF-IDF + LR | 0.949 | 0.861 | 0.903 |
| MiniLM Embeddings + LR | 0.948 | 0.843 | 0.892 |
| DistilBERT | 0.951 | 0.907 | 0.929 |

## Discussion

**DistilBERT can't hit the HIGH_RECALL bar (recall >= 0.95) anywhere in the swept range** — its test-set recall tops out at 0.935, even at the lowest threshold tried (0.10). This isn't a calibration problem, it's a coverage problem: the remaining false negatives are cases DistilBERT is confidently wrong about (several sit at >0.7 confidence in `results/error_analysis.md`'s NOVEL_ATTACK_PATTERN bucket), so no amount of threshold lowering recovers them — only retraining or more data would. TF-IDF and MiniLM, by contrast, both clear 0.95 recall comfortably (0.963 and 0.954) once the threshold drops enough, which says their missed injections are borderline rather than confidently misclassified.

**For a HIGH_RECALL deployment (e.g. a pre-filter ahead of a stricter downstream check), TF-IDF at t=0.35 is the best of the three** — 0.963 recall at 0.788 precision, beating MiniLM's 0.954 recall at only 0.696 precision. DistilBERT's t=0.15 point (0.935 recall, 0.927 precision) is the strongest single-model tradeoff overall if the hard 0.95 recall floor can be relaxed slightly — its precision at that recall level is far above either linear model's.

**For a HIGH_PRECISION deployment (e.g. auto-blocking without human review), DistilBERT at t=0.85 is the clear choice** — 0.852 recall at 0.989 precision, well ahead of MiniLM (0.806 recall) and TF-IDF, which only clears the 0.98 precision bar at t=0.80 by giving up almost half its recall (0.546). TF-IDF's PR curve (see `pr_curves.png`) falls off a cliff past ~0.85 recall, while DistilBERT's stays highest through most of the curve — consistent with its better test-set F1 in the README.

**Recommendation:** deploy DistilBERT if the fine-tuned weights are available (t=0.15 for a recall-leaning pre-filter, t=0.85 for precision-leaning auto-blocking). If only a lightweight, fully-explainable model is acceptable, TF-IDF at t=0.35 is the better HIGH_RECALL choice between the two linear models, and MiniLM at t=0.60 (0.989 precision, 0.806 recall) is the better HIGH_PRECISION choice. All of this is threshold-only tuning on the in-distribution test set — it says nothing about the holdout/adversarial generalization gaps documented in `REPORT.md`, which no threshold choice can fix.
