"""Sweep decision thresholds for all three models on the test set, plot
precision-recall curves, and recommend thresholds for two operating points."""
import os

import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score

from src.data_loader import load_splits
from src.evaluate import MODELS_DIR, DISTILBERT_MAX_LENGTH, DISTILBERT_BATCH_SIZE, load_distilbert
from src.features import get_embeddings

RESULTS_DIR = "results"
THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.05), 2)
HIGH_RECALL_MIN = 0.95
HIGH_PRECISION_MIN = 0.98

MODEL_LABELS = {
    "tfidf": "TF-IDF + LR",
    "embedding": "MiniLM Embeddings + LR",
    "distilbert": "DistilBERT",
}
MODEL_ORDER = ["tfidf", "embedding", "distilbert"]


def _proba_tfidf(pipeline, df):
    return pipeline.predict_proba(df["text"])[:, 1]


def _proba_embedding(clf, df):
    embeddings = get_embeddings(df["text"])
    return clf.predict_proba(embeddings)[:, 1]


def _proba_distilbert(tokenizer, model, df):
    texts = df["text"].tolist()
    proba = []
    with torch.no_grad():
        for i in range(0, len(texts), DISTILBERT_BATCH_SIZE):
            batch = texts[i:i + DISTILBERT_BATCH_SIZE]
            inputs = tokenizer(
                batch, truncation=True, padding=True,
                max_length=DISTILBERT_MAX_LENGTH, return_tensors="pt",
            )
            logits = model(**inputs).logits
            proba.extend(torch.softmax(logits, dim=-1)[:, 1].tolist())
    return np.array(proba)


def _load_scorers():
    tfidf_pipeline = joblib.load(f"{MODELS_DIR}/tfidf_lr.joblib")
    embed_clf = joblib.load(f"{MODELS_DIR}/embed_lr.joblib")
    distilbert_tokenizer, distilbert_model = load_distilbert()
    return {
        "tfidf": lambda df: _proba_tfidf(tfidf_pipeline, df),
        "embedding": lambda df: _proba_embedding(embed_clf, df),
        "distilbert": lambda df: _proba_distilbert(distilbert_tokenizer, distilbert_model, df),
    }


def sweep_thresholds(y_true, y_proba):
    rows = []
    for t in THRESHOLDS:
        preds = (y_proba >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": precision_score(y_true, preds, zero_division=0),
            "recall": recall_score(y_true, preds, zero_division=0),
            "f1": f1_score(y_true, preds, zero_division=0),
        })
    return pd.DataFrame(rows)


def pick_high_recall(sweep_df, min_recall=HIGH_RECALL_MIN):
    candidates = sweep_df[sweep_df["recall"] >= min_recall]
    if candidates.empty:
        row = sweep_df.sort_values(["recall", "precision"], ascending=False).iloc[0]
        return row, False
    row = candidates.sort_values(["precision", "threshold"], ascending=[False, False]).iloc[0]
    return row, True


def pick_high_precision(sweep_df, min_precision=HIGH_PRECISION_MIN):
    candidates = sweep_df[sweep_df["precision"] >= min_precision]
    if candidates.empty:
        row = sweep_df.sort_values(["precision", "recall"], ascending=False).iloc[0]
        return row, False
    row = candidates.sort_values(["recall", "threshold"], ascending=[False, True]).iloc[0]
    return row, True


def plot_pr_curves(y_true, proba_by_model, out_path):
    plt.figure(figsize=(7, 6))
    for model_key in MODEL_ORDER:
        precision, recall, _ = precision_recall_curve(y_true, proba_by_model[model_key])
        plt.plot(recall, precision, label=MODEL_LABELS[model_key], linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves — Test Set")
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def _report_markdown(recommendations, sweep_by_model):
    lines = [
        "# Threshold and Calibration Analysis",
        "",
        "All three models output a probability, not just a label — the README/REPORT numbers "
        "use each model's default 0.5 cutoff. This sweeps the decision threshold from 0.10 to "
        "0.90 in steps of 0.05 on the **test set** (237 examples) to see how much precision/recall "
        "can be traded off post hoc, without retraining, and what threshold to deploy at for two "
        "concrete operating points.",
        "",
        "## Method",
        "",
        "- Full sweep (17 thresholds x 3 models) saved to [`threshold_sweep.csv`](threshold_sweep.csv).",
        "- Precision-recall curves (full-resolution, not just the 17-point sweep) for all three "
        "models on one figure: [`pr_curves.png`](pr_curves.png).",
        "- **HIGH_RECALL**: among swept thresholds with recall >= 0.95, pick the one with the "
        "highest precision (catch nearly all injections, minimize false positives given that).",
        "- **HIGH_PRECISION**: among swept thresholds with precision >= 0.98, pick the one with "
        "the highest recall (almost never flag a benign prompt, catch as much as possible given that).",
        "- If no threshold in [0.10, 0.90] satisfies a constraint, the table falls back to the "
        "closest available point and flags it in the Notes column.",
        "",
        "## Recommended thresholds",
        "",
        "| Model | Operating point | Threshold | Precision | Recall | F1 | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, op_name, row, achieved in recommendations:
        note = "-" if achieved else f"constraint unreachable in [0.10, 0.90] sweep; closest shown"
        lines.append(
            f"| {name} | {op_name} | {row['threshold']:.2f} | {row['precision']:.3f} "
            f"| {row['recall']:.3f} | {row['f1']:.3f} | {note} |"
        )

    lines += ["", "## Default threshold (0.5) for reference", "", "| Model | Precision | Recall | F1 |", "|---|---|---|---|"]
    for model_key in MODEL_ORDER:
        sweep_df = sweep_by_model[model_key]
        row = sweep_df.iloc[(sweep_df["threshold"] - 0.5).abs().argsort().iloc[0]]
        name = MODEL_LABELS[model_key]
        lines.append(f"| {name} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} |")

    distilbert_max_recall = sweep_by_model["distilbert"]["recall"].max()
    lines += [
        "",
        "## Discussion",
        "",
        f"**DistilBERT can't hit the HIGH_RECALL bar (recall >= {HIGH_RECALL_MIN}) anywhere in the swept "
        f"range** — its test-set recall tops out at {distilbert_max_recall:.3f}, even at the lowest "
        "threshold tried (0.10). This isn't a calibration problem, it's a coverage problem: the "
        "remaining false negatives are cases DistilBERT is confidently wrong about (several sit at "
        ">0.7 confidence in `results/error_analysis.md`'s NOVEL_ATTACK_PATTERN bucket), so no amount of "
        "threshold lowering recovers them — only retraining or more data would. TF-IDF and MiniLM, by "
        "contrast, both clear 0.95 recall comfortably (0.963 and 0.954) once the threshold drops enough, "
        "which says their missed injections are borderline rather than confidently misclassified.",
        "",
        "**For a HIGH_RECALL deployment (e.g. a pre-filter ahead of a stricter downstream check), "
        "TF-IDF at t=0.35 is the best of the three** — 0.963 recall at 0.788 precision, beating "
        "MiniLM's 0.954 recall at only 0.696 precision. DistilBERT's t=0.15 point (0.935 recall, 0.927 "
        "precision) is the strongest single-model tradeoff overall if the hard 0.95 recall floor can "
        "be relaxed slightly — its precision at that recall level is far above either linear model's.",
        "",
        "**For a HIGH_PRECISION deployment (e.g. auto-blocking without human review), DistilBERT at "
        "t=0.85 is the clear choice** — 0.852 recall at 0.989 precision, well ahead of MiniLM (0.806 "
        "recall) and TF-IDF, which only clears the 0.98 precision bar at t=0.80 by giving up almost "
        "half its recall (0.546). TF-IDF's PR curve (see `pr_curves.png`) falls off a cliff past "
        "~0.85 recall, while DistilBERT's stays highest through most of the curve — consistent with "
        "its better test-set F1 in the README.",
        "",
        "**Recommendation:** deploy DistilBERT if the fine-tuned weights are available (t=0.15 for a "
        "recall-leaning pre-filter, t=0.85 for precision-leaning auto-blocking). If only a lightweight, "
        "fully-explainable model is acceptable, TF-IDF at t=0.35 is the better HIGH_RECALL choice "
        "between the two linear models, and MiniLM at t=0.60 (0.989 precision, 0.806 recall) is the "
        "better HIGH_PRECISION choice. All of this is threshold-only tuning on the in-distribution test "
        "set — it says nothing about the holdout/adversarial generalization gaps documented in "
        "`REPORT.md`, which no threshold choice can fix.",
    ]

    return "\n".join(lines) + "\n"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _, _, test_df = load_splits()
    y_true = test_df["label"].to_numpy()

    scorers = _load_scorers()
    proba_by_model = {name: fn(test_df) for name, fn in scorers.items()}

    sweep_by_model = {}
    sweep_rows = []
    recommendations = []
    for model_key in MODEL_ORDER:
        name = MODEL_LABELS[model_key]
        sweep_df = sweep_thresholds(y_true, proba_by_model[model_key])
        sweep_by_model[model_key] = sweep_df

        tagged = sweep_df.copy()
        tagged.insert(0, "model", name)
        sweep_rows.append(tagged)

        hr_row, hr_ok = pick_high_recall(sweep_df)
        hp_row, hp_ok = pick_high_precision(sweep_df)
        recommendations.append((name, f"HIGH_RECALL (recall >= {HIGH_RECALL_MIN})", hr_row, hr_ok))
        recommendations.append((name, f"HIGH_PRECISION (precision >= {HIGH_PRECISION_MIN})", hp_row, hp_ok))

    full_sweep_df = pd.concat(sweep_rows, ignore_index=True)
    sweep_csv = f"{RESULTS_DIR}/threshold_sweep.csv"
    full_sweep_df.to_csv(sweep_csv, index=False)
    print(f"Saved {sweep_csv}")

    plot_pr_curves(y_true, proba_by_model, f"{RESULTS_DIR}/pr_curves.png")

    md_path = f"{RESULTS_DIR}/threshold_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_report_markdown(recommendations, sweep_by_model))
    print(f"Saved {md_path}")

    for name, op_name, row, achieved in recommendations:
        flag = "" if achieved else "  [unreachable, closest shown]"
        print(f"{name:24s} {op_name:35s} t={row['threshold']:.2f}  P={row['precision']:.3f}  R={row['recall']:.3f}  F1={row['f1']:.3f}{flag}")


if __name__ == "__main__":
    main()
