"""Dump every misclassified test/holdout example across all three models and
bucket each error into a rule-based taxonomy (no LLM calls)."""
import os
import re

import joblib
import pandas as pd
import torch

from src.data_loader import load_splits
from src.evaluate import (
    MODELS_DIR,
    HOLDOUT_PATH,
    DISTILBERT_MAX_LENGTH,
    DISTILBERT_BATCH_SIZE,
    load_distilbert,
)
from src.features import get_embeddings

RESULTS_DIR = "results"

MODEL_LABELS = {
    "tfidf": "TF-IDF + LR",
    "embedding": "MiniLM Embeddings + LR",
    "distilbert": "DistilBERT",
}

# Taxonomy categories
NOVEL_ATTACK_PATTERN = "NOVEL_ATTACK_PATTERN"
TRIGGER_WORD_BENIGN = "TRIGGER_WORD_BENIGN"
OBFUSCATION = "OBFUSCATION"
LENGTH_OUTLIER = "LENGTH_OUTLIER"
AMBIGUOUS_INTENT = "AMBIGUOUS_INTENT"
OTHER = "OTHER"

CATEGORY_ORDER = [
    NOVEL_ATTACK_PATTERN, TRIGGER_WORD_BENIGN, OBFUSCATION,
    LENGTH_OUTLIER, AMBIGUOUS_INTENT, OTHER,
]

# Words/phrases heavily represented in injection training examples that also
# show up in innocuous requests ("ignore the noise in the data", "act as a
# resume reviewer") -- overlap here is the classic false-positive driver.
TRIGGER_WORDS = [
    "ignore", "disregard", "forget", "pretend", "act as", "roleplay",
    "role-play", "override", "bypass", "system prompt", "instructions",
    "unrestricted", "no restrictions", "developer mode", "jailbreak",
    "dan", "configuration", "reveal", "hypothetical", "restrictions",
]
_TRIGGER_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in TRIGGER_WORDS) + r")\b",
    re.IGNORECASE,
)

_BASE64ISH_RE = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")

# Cyrillic and Greek code points are the lookalike scripts actually used by
# unicode_lookalike_substitution (src/adversarial/attacks.py HOMOGLYPH_MAP).
# Deliberately narrower than "any non-ASCII letter" so legitimate non-English
# text (German umlauts, French accents, etc. -- Latin-1/Latin Extended) isn't
# mistaken for a homoglyph attack.
def _is_lookalike_script(ch):
    code = ord(ch)
    return (0x0370 <= code <= 0x03FF) or (0x0400 <= code <= 0x04FF)


LENGTH_Z_THRESHOLD = 2.0


def has_trigger_word(text):
    return bool(_TRIGGER_RE.search(text))


def is_obfuscated(text):
    if any(_is_lookalike_script(ch) for ch in text):
        return True
    return bool(_BASE64ISH_RE.search(text))


def make_length_outlier_check(reference_texts):
    lengths = pd.Series([len(t) for t in reference_texts])
    mean, std = lengths.mean(), lengths.std()

    def _check(text):
        if std == 0:
            return False
        z = (len(text) - mean) / std
        return abs(z) >= LENGTH_Z_THRESHOLD

    return _check


def categorize_error(text, true_label, pred_label, is_length_outlier):
    if is_obfuscated(text):
        return OBFUSCATION
    if is_length_outlier(text):
        return LENGTH_OUTLIER

    contains_trigger = has_trigger_word(text)

    if true_label == 0 and pred_label == 1:
        return TRIGGER_WORD_BENIGN if contains_trigger else AMBIGUOUS_INTENT
    if true_label == 1 and pred_label == 0:
        return NOVEL_ATTACK_PATTERN if not contains_trigger else AMBIGUOUS_INTENT
    return OTHER


def _predict_tfidf(pipeline, df):
    proba = pipeline.predict_proba(df["text"])
    preds = proba.argmax(axis=1)
    confidence = proba.max(axis=1)
    return preds, confidence


def _predict_embedding(clf, df):
    embeddings = get_embeddings(df["text"])
    proba = clf.predict_proba(embeddings)
    preds = proba.argmax(axis=1)
    confidence = proba.max(axis=1)
    return preds, confidence


def _predict_distilbert(tokenizer, model, df):
    texts = df["text"].tolist()
    preds, confidence = [], []
    with torch.no_grad():
        for i in range(0, len(texts), DISTILBERT_BATCH_SIZE):
            batch = texts[i:i + DISTILBERT_BATCH_SIZE]
            inputs = tokenizer(
                batch, truncation=True, padding=True,
                max_length=DISTILBERT_MAX_LENGTH, return_tensors="pt",
            )
            logits = model(**inputs).logits
            proba = torch.softmax(logits, dim=-1)
            batch_preds = torch.argmax(proba, dim=-1)
            preds.extend(batch_preds.tolist())
            confidence.extend(proba.max(dim=-1).values.tolist())
    return preds, confidence


def collect_errors(df, split_name, predictors):
    error_rows = []
    length_outlier = make_length_outlier_check(df["text"])

    for model_name, predict_fn in predictors.items():
        preds, confidence = predict_fn(df)
        for text, true_label, pred_label, conf in zip(df["text"], df["label"], preds, confidence):
            true_label, pred_label = int(true_label), int(pred_label)
            if true_label == pred_label:
                continue
            error_rows.append({
                "text": text,
                "true_label": true_label,
                "predicted_label": pred_label,
                "model": MODEL_LABELS[model_name],
                "confidence": round(float(conf), 4),
                "split": split_name,
                "error_type": categorize_error(text, true_label, pred_label, length_outlier),
            })
    return error_rows


def _taxonomy_table(errors_df):
    pivot = errors_df.pivot_table(
        index="error_type", columns="model", aggfunc="size", fill_value=0
    )
    pivot = pivot.reindex(CATEGORY_ORDER, fill_value=0)
    model_cols = [MODEL_LABELS[m] for m in ("tfidf", "embedding", "distilbert") if MODEL_LABELS[m] in pivot.columns]
    pivot = pivot[model_cols]
    pivot["Total"] = pivot.sum(axis=1)

    lines = ["| Error type | " + " | ".join(pivot.columns) + " |"]
    lines.append("|---|" + "---|" * len(pivot.columns))
    for error_type, row in pivot.iterrows():
        values = " | ".join(str(int(v)) for v in row)
        lines.append(f"| {error_type} | {values} |")
    return "\n".join(lines), pivot


def _examples_section(errors_df):
    lines = ["## Representative examples", ""]
    for category in CATEGORY_ORDER:
        subset = errors_df[errors_df["error_type"] == category]
        lines.append(f"### {category}")
        lines.append("")
        if subset.empty:
            lines.append("_No errors in this category._")
            lines.append("")
            continue
        sample = subset.head(3)
        for _, row in sample.iterrows():
            lines.append(
                f"- `{row['text']}` — true={row['true_label']}, "
                f"pred={row['predicted_label']}, model={row['model']}, "
                f"confidence={row['confidence']:.3f}, split={row['split']}"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    _, _, test_df = load_splits()
    holdout_df = pd.read_csv(HOLDOUT_PATH)

    tfidf_pipeline = joblib.load(f"{MODELS_DIR}/tfidf_lr.joblib")
    embed_clf = joblib.load(f"{MODELS_DIR}/embed_lr.joblib")
    distilbert_tokenizer, distilbert_model = load_distilbert()

    predictors = {
        "tfidf": lambda df: _predict_tfidf(tfidf_pipeline, df),
        "embedding": lambda df: _predict_embedding(embed_clf, df),
        "distilbert": lambda df: _predict_distilbert(distilbert_tokenizer, distilbert_model, df),
    }

    error_rows = []
    error_rows += collect_errors(test_df, "test", predictors)
    error_rows += collect_errors(holdout_df, "holdout", predictors)

    errors_df = pd.DataFrame(error_rows)

    csv_path = f"{RESULTS_DIR}/error_analysis.csv"
    errors_df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path} ({len(errors_df)} misclassified rows)")

    table_md, pivot = _taxonomy_table(errors_df)
    print(pivot)

    md_lines = [
        "# Error Analysis",
        "",
        "Every misclassified test/holdout example across all three models, categorized "
        "with a rule-based taxonomy (no LLM calls). Full per-row dump in "
        "[`error_analysis.csv`](error_analysis.csv).",
        "",
        f"Total misclassified rows: {len(errors_df)} "
        f"(test set: {len(test_df)} examples x 3 models, holdout: {len(holdout_df)} examples x 3 models).",
        "",
        "## Taxonomy summary (error type x model)",
        "",
        table_md,
        "",
        _examples_section(errors_df),
    ]

    md_path = f"{RESULTS_DIR}/error_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
