"""Evaluate the three trained classifiers, unchanged, on the indirect prompt
injection pilot (data/indirect_pilot/), then re-score the injection documents
with only the embedded payload base64-encoded."""
import os
import time

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import torch

from scripts.build_indirect_pilot import PAYLOAD_SOURCES
from src.adversarial.attacks import base64_encoding
from src.evaluate import DISTILBERT_BATCH_SIZE, MODELS_DIR, _score, load_distilbert
from src.features import get_embedder, get_embeddings

RESULTS_DIR = "results"
PILOT_PATH = "data/indirect_pilot/indirect_pilot.csv"
DIRECT_RESULTS_PATH = f"{RESULTS_DIR}/threshold_sweep.csv"
CSV_PATH = f"{RESULTS_DIR}/indirect_pilot_results.csv"
MD_PATH = f"{RESULTS_DIR}/indirect_pilot.md"
CHART_PATH = f"{RESULTS_DIR}/indirect_pilot_chart.png"

# The pilot keeps documents under 400 tokens so DistilBERT can read them whole.
# Its training and existing evaluation used 128; 512 is its positional limit.
PILOT_DISTILBERT_MAX_LENGTH = 512

POSITIONS = ["start", "middle", "end"]
MODEL_ORDER = ["tfidf", "embedding", "distilbert"]
MODEL_LABELS = {
    "tfidf": "TF-IDF + LR",
    "embedding": "MiniLM Embeddings + LR",
    "distilbert": "DistilBERT",
}


def _predict_distilbert(tokenizer, model, texts):
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), DISTILBERT_BATCH_SIZE):
            batch = texts[i:i + DISTILBERT_BATCH_SIZE]
            inputs = tokenizer(
                batch, truncation=True, padding=True,
                max_length=PILOT_DISTILBERT_MAX_LENGTH, return_tensors="pt",
            )
            logits = model(**inputs).logits
            preds.extend(torch.argmax(logits, dim=-1).tolist())
    return preds


def _load_predictors():
    tfidf_pipeline = joblib.load(f"{MODELS_DIR}/tfidf_lr.joblib")
    embed_clf = joblib.load(f"{MODELS_DIR}/embed_lr.joblib")
    distilbert_tokenizer, distilbert_model = load_distilbert()
    predictors = {
        "tfidf": lambda texts: list(tfidf_pipeline.predict(texts)),
        "embedding": lambda texts: list(embed_clf.predict(get_embeddings(texts))),
        "distilbert": lambda texts: _predict_distilbert(distilbert_tokenizer, distilbert_model, texts),
    }
    return predictors, distilbert_tokenizer


def load_payload_lookup():
    lookup = {}
    for name, path in PAYLOAD_SOURCES.items():
        texts = pd.read_csv(path)["text"].astype(str).str.strip()
        for i, text in enumerate(texts):
            lookup[f"{name}:{i}"] = text
    return lookup


def payload_of(row, lookup):
    payload = lookup[row.source_id.split("|")[0]]
    if payload not in row.text:
        raise ValueError(f"Payload for {row.source_id} not found in its document")
    return payload


def encode_span(row, lookup):
    payload = payload_of(row, lookup)
    return row.text.replace(payload, base64_encoding(payload), 1)


def payload_cut_off(tokenizer, text, payload, window):
    """True when the payload ends past the model's input window."""
    end = text.index(payload) + len(payload)
    return len(tokenizer(text[:end])["input_ids"]) > window


def encoded_span_tokens(tokenizer, encoded):
    """Tokens the model receives for the base64 blob after "decode this:".

    WordPiece splits base64 at "+" and "/" and maps any fragment longer than
    its character limit (100 for BERT-family tokenizers) to [UNK]."""
    prefix = tokenizer.tokenize("decode this:")
    return tokenizer.tokenize(encoded)[len(prefix):]


def direct_test_recall():
    sweep = pd.read_csv(DIRECT_RESULTS_PATH)
    at_default = sweep[sweep["threshold"].round(2) == 0.5].set_index("model")["recall"]
    return {key: float(at_default[MODEL_LABELS[key]]) for key in MODEL_ORDER}


def _row(model, section, subset, n, metric, value):
    return {
        "model": MODEL_LABELS[model], "section": section, "subset": subset,
        "n": int(n), "metric": metric, "value": float(value),
    }


def evaluate_model(key, predict, pilot, injected, attacked_texts, payload_texts):
    rows = []
    start = time.perf_counter()
    pilot_preds = pd.Series(predict(pilot["text"].tolist()), index=pilot.index)
    attacked_preds = pd.Series(predict(attacked_texts), index=injected.index)
    standalone_preds = predict(payload_texts)
    seconds = time.perf_counter() - start

    for metric, value in _score(pilot["label"], pilot_preds).items():
        rows.append(_row(key, "pilot_overall", "all", len(pilot), metric, value))

    for position in POSITIONS:
        subset = injected[injected["position"] == position]
        recall = pilot_preds[subset.index].mean()
        rows.append(_row(key, "recall_by_position", position, len(subset), "recall", recall))

    negatives = pilot[pilot["label"] == 0]
    for subset_name, mask in (("embedded_benign", negatives["position"] != "none"),
                              ("plain_paragraph", negatives["position"] == "none")):
        subset = negatives[mask]
        rows.append(_row(key, "fpr_by_negative", subset_name, len(subset), "fpr", pilot_preds[subset.index].mean()))

    baseline = pilot_preds[injected.index].mean()
    attacked = attacked_preds.mean()
    rows.append(_row(key, "base64_span_attack", "injected", len(injected), "baseline_recall", baseline))
    rows.append(_row(key, "base64_span_attack", "injected", len(injected), "attack_recall", attacked))
    rows.append(_row(key, "base64_span_attack", "injected", len(injected), "recall_drop", baseline - attacked))

    standalone_recall = sum(standalone_preds) / len(standalone_preds)
    rows.append(_row(key, "direct_reference", "same_payloads_standalone", len(payload_texts), "recall", standalone_recall))

    print(f"{MODEL_LABELS[key]:24s} pilot recall={baseline:.3f}  base64 recall={attacked:.3f}  ({seconds:.0f}s)")
    return rows, seconds


def coverage_rows(injected, attacked_texts, payloads, minilm_tokenizer, distilbert_tokenizer):
    """Count injected payloads that fall past each model's input window."""
    rows = []
    checks = (
        ("embedding", minilm_tokenizer, get_embedder().max_seq_length),
        ("distilbert", distilbert_tokenizer, PILOT_DISTILBERT_MAX_LENGTH),
    )
    for key, tokenizer, window in checks:
        for position in POSITIONS:
            subset = injected[injected["position"] == position]
            clean = sum(payload_cut_off(tokenizer, r.text, payloads[i], window) for i, r in subset.iterrows())
            encoded = sum(
                payload_cut_off(tokenizer, attacked_texts[injected.index.get_loc(i)],
                                base64_encoding(payloads[i]), window)
                for i in subset.index
            )
            rows.append(_row(key, "payload_past_window", position, len(subset), f"clean_window_{window}", clean))
            rows.append(_row(key, "payload_past_window", position, len(subset), f"base64_window_{window}", encoded))

        spans = [encoded_span_tokens(tokenizer, base64_encoding(payloads[i])) for i in injected.index]
        any_unk = sum(tokenizer.unk_token in span for span in spans)
        only_unk = sum(span == [tokenizer.unk_token] for span in spans)
        rows.append(_row(key, "base64_span_tokenization", "injected", len(injected), "encoded_span_any_unk", any_unk))
        rows.append(_row(key, "base64_span_tokenization", "injected", len(injected), "encoded_span_single_unk", only_unk))
    return rows


def plot_chart(direct, indirect, n_injected, out_path):
    labels = [MODEL_LABELS[k] for k in MODEL_ORDER]
    x = range(len(MODEL_ORDER))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bars_direct = ax.bar([i - width / 2 for i in x], [direct[k] for k in MODEL_ORDER], width,
                         color="#b8b8b8", label="Direct: test set (results/threshold_sweep.csv)")
    bars_indirect = ax.bar([i + width / 2 for i in x], [indirect[k] for k in MODEL_ORDER], width,
                           color="#2f6fdb", label=f"Indirect pilot ({n_injected} injected documents)")
    for bars in (bars_direct, bars_indirect):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Injection recall (threshold 0.5)")
    ax.set_title("Injection recall: direct prompts vs. injections embedded in documents")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def _metric(results, key, section, subset, metric):
    match = results[(results["model"] == MODEL_LABELS[key]) & (results["section"] == section)
                    & (results["subset"] == subset) & (results["metric"] == metric)]
    return match["value"].iloc[0]


def _report_markdown(results, pilot, injected, direct, seconds_by_model):
    n_neg = int((pilot["label"] == 0).sum())
    lines = [
        "# Indirect Prompt Injection Pilot",
        "",
        f"The three trained classifiers, with no retraining, scored on `{PILOT_PATH}`: "
        f"{len(pilot)} documents built by `scripts/build_indirect_pilot.py`, in which "
        f"{len(injected)} injection prompts and {n_neg - (pilot['position'] == 'none').sum()} benign prompts "
        "from the test split and hand-crafted holdout are embedded at the start, middle or end of a short "
        f"SQuAD Wikipedia paragraph, plus {(pilot['position'] == 'none').sum()} unmodified paragraphs. "
        "All numbers use the default 0.5 decision threshold.",
        "",
        "![Direct vs indirect recall](indirect_pilot_chart.png)",
        "",
        "## Overall",
        "",
        "| Model | Precision | Recall | F1 | FPR | Direct test recall | Same payloads, standalone |",
        "|---|---|---|---|---|---|---|",
    ]
    for key in MODEL_ORDER:
        lines.append(
            f"| {MODEL_LABELS[key]} "
            f"| {_metric(results, key, 'pilot_overall', 'all', 'precision'):.3f} "
            f"| {_metric(results, key, 'pilot_overall', 'all', 'recall'):.3f} "
            f"| {_metric(results, key, 'pilot_overall', 'all', 'f1'):.3f} "
            f"| {_metric(results, key, 'pilot_overall', 'all', 'fpr'):.3f} "
            f"| {direct[key]:.3f} "
            f"| {_metric(results, key, 'direct_reference', 'same_payloads_standalone', 'recall'):.3f} |"
        )

    counts = injected["position"].value_counts()
    lines += [
        "",
        "## Recall by injection position",
        "",
        "| Model | " + " | ".join(f"{p} (n={counts[p]})" for p in POSITIONS) + " |",
        "|---|" + "---|" * len(POSITIONS),
    ]
    for key in MODEL_ORDER:
        values = " | ".join(f"{_metric(results, key, 'recall_by_position', p, 'recall'):.3f}" for p in POSITIONS)
        lines.append(f"| {MODEL_LABELS[key]} | {values} |")

    negatives = pilot[pilot["label"] == 0]
    n_embedded = int((negatives["position"] != "none").sum())
    n_plain = int((negatives["position"] == "none").sum())
    lines += [
        "",
        "## False-positive rate by negative type",
        "",
        f"| Model | Benign prompt in paragraph (n={n_embedded}) | Plain paragraph (n={n_plain}) |",
        "|---|---|---|",
    ]
    for key in MODEL_ORDER:
        lines.append(
            f"| {MODEL_LABELS[key]} "
            f"| {_metric(results, key, 'fpr_by_negative', 'embedded_benign', 'fpr'):.3f} "
            f"| {_metric(results, key, 'fpr_by_negative', 'plain_paragraph', 'fpr'):.3f} |"
        )

    lines += [
        "",
        "## Base64-encoding only the injected span",
        "",
        "Each injection document is re-scored with its payload replaced by "
        "`base64_encoding(payload)` from `src/adversarial/attacks.py` (`\"decode this: <base64>\"`); "
        "the host paragraph is left as is. Recall drop = pilot recall minus recall after encoding.",
        "",
        "| Model | Pilot recall | Recall after encoding | Recall drop |",
        "|---|---|---|---|",
    ]
    for key in MODEL_ORDER:
        lines.append(
            f"| {MODEL_LABELS[key]} "
            f"| {_metric(results, key, 'base64_span_attack', 'injected', 'baseline_recall'):.3f} "
            f"| {_metric(results, key, 'base64_span_attack', 'injected', 'attack_recall'):.3f} "
            f"| {_metric(results, key, 'base64_span_attack', 'injected', 'recall_drop'):.3f} |"
        )

    minilm_window = get_embedder().max_seq_length
    lines += [
        "",
        "## How much of each payload the models can see",
        "",
        f"MiniLM truncates input at {minilm_window} tokens (the `all-MiniLM-L6-v2` default, unchanged); "
        f"DistilBERT is run here at {PILOT_DISTILBERT_MAX_LENGTH}. TF-IDF reads the whole document. "
        "Count of injected documents whose payload ends past the window:",
        "",
        "| Model | Window | " + " | ".join(POSITIONS) + " | After base64, all positions |",
        "|---|---|" + "---|" * len(POSITIONS) + "---|",
    ]
    for key, window in (("embedding", minilm_window), ("distilbert", PILOT_DISTILBERT_MAX_LENGTH)):
        clean = [int(_metric(results, key, "payload_past_window", p, f"clean_window_{window}")) for p in POSITIONS]
        encoded = sum(int(_metric(results, key, "payload_past_window", p, f"base64_window_{window}")) for p in POSITIONS)
        cells = " | ".join(f"{c} / {counts[p]}" for c, p in zip(clean, POSITIONS))
        lines.append(f"| {MODEL_LABELS[key]} | {window} | {cells} | {encoded} / {len(injected)} |")

    lines += [
        "",
        "Neither transformer gets readable text back from the encoded span. WordPiece splits base64 at `+` "
        "and `/` and maps every fragment over 100 characters to `[UNK]`, leaving the rest as meaningless "
        "subword pieces:",
        "",
        "| Model | Encoded span contains `[UNK]` | Encoded span is a single `[UNK]` |",
        "|---|---|---|",
    ]
    for key in ("embedding", "distilbert"):
        any_unk = int(_metric(results, key, "base64_span_tokenization", "injected", "encoded_span_any_unk"))
        only_unk = int(_metric(results, key, "base64_span_tokenization", "injected", "encoded_span_single_unk"))
        lines.append(f"| {MODEL_LABELS[key]} | {any_unk} / {len(injected)} | {only_unk} / {len(injected)} |")

    def drop_range(key_from, key_to):
        gaps = [_metric(results, k, *key_from) - _metric(results, k, *key_to) for k in MODEL_ORDER]
        return min(gaps), max(gaps)

    standalone = ("direct_reference", "same_payloads_standalone", "recall")
    overall = ("pilot_overall", "all", "recall")
    embed_lo, embed_hi = drop_range(standalone, overall)
    precisions = [_metric(results, k, "pilot_overall", "all", "precision") for k in MODEL_ORDER]
    fprs = [_metric(results, k, "pilot_overall", "all", "fpr") for k in MODEL_ORDER]
    plain_fprs = [_metric(results, k, "fpr_by_negative", "plain_paragraph", "fpr") for k in MODEL_ORDER]
    attack_recalls = [_metric(results, k, "base64_span_attack", "injected", "attack_recall") for k in MODEL_ORDER]
    minilm_end = _metric(results, "embedding", "recall_by_position", "end", "recall")
    minilm_end_cut = int(_metric(results, "embedding", "payload_past_window", "end", f"clean_window_{minilm_window}"))
    bert_gap = (_metric(results, "distilbert", *standalone) - _metric(results, "distilbert", *overall))
    lines += [
        "",
        "## Reading this",
        "",
        f"**Embedding a payload in an ordinary paragraph roughly halves what the models catch.** The same "
        f"{len(injected)} injection prompts lose {embed_lo:.2f}–{embed_hi:.2f} of recall once they sit inside a "
        "Wikipedia paragraph, with no change to the payload itself. Against the existing direct test-set "
        "results the gap is wider still, partly because the pilot dropped the long jailbreaks that are "
        "easiest to detect.",
        "",
        f"**The models miss injections; they do not start flagging documents.** Precision stays at "
        f"{min(precisions):.2f}–{max(precisions):.2f} and FPR at {min(fprs):.3f}–{max(fprs):.3f}; plain paragraphs "
        f"are flagged at a rate of {max(plain_fprs):.3f} at most. The surrounding benign text pulls scores toward "
        "\"benign\" rather than making the classifiers jumpy.",
        "",
        f"**Seeing the whole document does not save DistilBERT.** It reads every pilot document in full at "
        f"{PILOT_DISTILBERT_MAX_LENGTH} tokens yet loses {bert_gap:.2f} of recall, the largest drop of the three: "
        "the problem is dilution, not truncation. Truncation does matter for MiniLM: its end-position recall "
        f"is its lowest ({minilm_end:.3f}), and {minilm_end_cut} of those payloads end past its "
        f"{minilm_window}-token window. Other position differences are a few documents each and not consistent "
        "across models.",
        "",
        f"**Base64-encoding just the injected span is a complete bypass.** Recall after encoding is "
        f"{min(attack_recalls):.3f}–{max(attack_recalls):.3f} for all three models, matching the direct-setting "
        "result in `REPORT.md`. The transformers receive the span as `[UNK]` tokens and meaningless subword "
        "fragments (table above), and TF-IDF is left with \"decode this\" plus base64 fragments.",
    ]

    runtime =", ".join(f"{MODEL_LABELS[k]} {seconds_by_model[k]:.0f}s" for k in MODEL_ORDER)
    lines += [
        "",
        "## Method notes",
        "",
        f"- All {len(pilot)} documents were scored by all three models, DistilBERT included; no subset "
        f"was needed. CPU runtime for pilot + encoded + standalone passes: {runtime}.",
        f"- **Direct test recall** is the existing test-set result at threshold 0.5 from "
        f"`{DIRECT_RESULTS_PATH}` (DistilBERT there ran at 128 tokens). **Same payloads, standalone** "
        f"scores the exact {len(injected)} injection prompts used in the pilot without a host paragraph, "
        "so the gap between it and pilot recall isolates the effect of embedding.",
        f"- DistilBERT was fine-tuned at 128 tokens; reading {PILOT_DISTILBERT_MAX_LENGTH} here is itself a "
        "shift from its training conditions.",
        f"- Small sample: {len(injected)} injected documents (about {counts.min()}–{counts.max()} per position), "
        "so one flipped prediction moves a per-position recall by roughly "
        f"{1 / counts.min():.2f}. Read differences as directional.",
        "- The pilot is biased toward short injections: payloads that could not fit under 400 tokens with "
        "any host (mostly long DAN-style jailbreaks) were dropped when the pilot was built.",
        "",
        "Full long-format numbers: [`indirect_pilot_results.csv`](indirect_pilot_results.csv).",
    ]
    return "\n".join(lines) + "\n"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pilot = pd.read_csv(PILOT_PATH)
    injected = pilot[pilot["label"] == 1]
    lookup = load_payload_lookup()

    payloads = {i: payload_of(r, lookup) for i, r in injected.iterrows()}
    attacked_texts = [encode_span(r, lookup) for r in injected.itertuples()]
    payload_texts = [payloads[i] for i in injected.index]
    print(f"Pilot: {len(pilot)} documents, {len(injected)} with an injected payload")

    predictors, distilbert_tokenizer = _load_predictors()
    rows, seconds_by_model = [], {}
    for key in MODEL_ORDER:
        model_rows, seconds = evaluate_model(key, predictors[key], pilot, injected, attacked_texts, payload_texts)
        rows += model_rows
        seconds_by_model[key] = seconds

    rows += coverage_rows(injected, attacked_texts, payloads,
                          get_embedder().tokenizer, distilbert_tokenizer)

    results = pd.DataFrame(rows)
    results.to_csv(CSV_PATH, index=False)
    print(f"Saved {CSV_PATH}")

    direct = direct_test_recall()
    indirect = {k: _metric(results, k, "pilot_overall", "all", "recall") for k in MODEL_ORDER}
    plot_chart(direct, indirect, len(injected), CHART_PATH)

    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(_report_markdown(results, pilot, injected, direct, seconds_by_model))
    print(f"Saved {MD_PATH}")


if __name__ == "__main__":
    main()
