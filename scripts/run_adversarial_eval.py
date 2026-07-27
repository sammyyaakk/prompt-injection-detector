import os

import pandas as pd
import joblib

from src.evaluate import (
    MODELS_DIR,
    HOLDOUT_PATH,
    evaluate_tfidf,
    evaluate_embedding,
    evaluate_distilbert,
    load_distilbert,
)
from src.adversarial.attacks import ATTACKS

RESULTS_DIR = "results"
BASE_SEED = 42

MODEL_LABELS = {
    "tfidf": "TF-IDF + LR",
    "embedding": "MiniLM Embeddings + LR",
    "distilbert": "DistilBERT",
}


def _load_models():
    tfidf_pipeline = joblib.load(f"{MODELS_DIR}/tfidf_lr.joblib")
    embed_clf = joblib.load(f"{MODELS_DIR}/embed_lr.joblib")
    distilbert_tokenizer, distilbert_model = load_distilbert()
    return {
        "tfidf": lambda df: evaluate_tfidf(tfidf_pipeline, df),
        "embedding": lambda df: evaluate_embedding(embed_clf, df),
        "distilbert": lambda df: evaluate_distilbert(distilbert_tokenizer, distilbert_model, df),
    }


def _perturbed_df(injection_df, attack_fn):
    perturbed = injection_df.copy()
    perturbed["text"] = [
        attack_fn(text, seed=BASE_SEED + i)
        for i, text in enumerate(injection_df["text"])
    ]
    return perturbed


def run(injection_df, evaluators):
    rows = []
    baseline_recall = {name: evaluators[name](injection_df)["recall"] for name in evaluators}
    for model_name, recall in baseline_recall.items():
        rows.append({
            "attack": "baseline (unperturbed)",
            "model": MODEL_LABELS[model_name],
            "baseline_recall": recall,
            "attack_recall": recall,
            "recall_drop": 0.0,
        })

    for attack_name, attack_fn in ATTACKS.items():
        perturbed = _perturbed_df(injection_df, attack_fn)
        for model_name, evaluate_fn in evaluators.items():
            attack_recall = evaluate_fn(perturbed)["recall"]
            rows.append({
                "attack": attack_name,
                "model": MODEL_LABELS[model_name],
                "baseline_recall": baseline_recall[model_name],
                "attack_recall": attack_recall,
                "recall_drop": baseline_recall[model_name] - attack_recall,
            })

    return pd.DataFrame(rows)


def _to_markdown(results_df):
    pivot = results_df.pivot(index="attack", columns="model", values="recall_drop")
    order = ["baseline (unperturbed)"] + list(ATTACKS.keys())
    pivot = pivot.reindex(order)
    pivot = pivot[[MODEL_LABELS[m] for m in ("tfidf", "embedding", "distilbert")]]

    lines = ["# Adversarial Robustness — Recall Drop per Attack Class", ""]
    lines.append("Recall drop = baseline injection recall on the holdout set minus recall on the perturbed version. Higher is worse (more evasion).")
    lines.append("")
    header = "| Attack | " + " | ".join(pivot.columns) + " |"
    sep = "|---|" + "---|" * len(pivot.columns)
    lines.append(header)
    lines.append(sep)
    for attack, row in pivot.iterrows():
        values = " | ".join(f"{v:.3f}" for v in row)
        lines.append(f"| {attack} | {values} |")
    return "\n".join(lines) + "\n"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    holdout_df = pd.read_csv(HOLDOUT_PATH)
    injection_df = holdout_df[holdout_df["label"] == 1].reset_index(drop=True)
    print(f"Injection subset of holdout set: {len(injection_df)} examples")

    evaluators = _load_models()
    results_df = run(injection_df, evaluators)

    csv_path = f"{RESULTS_DIR}/adversarial_robustness.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    md_path = f"{RESULTS_DIR}/adversarial_robustness.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_to_markdown(results_df))
    print(f"Saved {md_path}")

    print(results_df.pivot(index="attack", columns="model", values="recall_drop"))


if __name__ == "__main__":
    main()
