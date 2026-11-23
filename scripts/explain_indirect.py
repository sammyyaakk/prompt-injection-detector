"""Explainability for the indirect setting.

1. TF-IDF + LR (indirect-trained): the n-grams with the largest positive and
   negative weights, and per-word contributions inside the flagged window.
2. Window score profiles (any model): each ~400-char window's P(injection), so
   a reviewer sees *where* in a document the detector fired.
3. DistilBERT attention (if weights are present): last-layer [CLS] attention,
   averaged over heads, on the top window; reports how much attention mass
   falls on payload tokens versus carrier tokens.

Writes results/indirect/explain_*.png, explain_examples.html, explain_attention_<model>.csv
Run: python -m scripts.explain_indirect
"""
import html
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.indirect.chunking import overlap, windows
from src.indirect.models import MODEL_LABELS, load_all, load_scorer

OUT = "results/indirect"
EXAMPLE_IDS = None  # picked automatically: one detected injection per carrier + one false positive


def tfidf_terms(pipe, k=15):
    names = pipe["tfidf"].get_feature_names_out()
    coef = pipe["clf"].coef_[0]
    order = np.argsort(coef)
    pos = [(names[i], coef[i]) for i in order[::-1][:k]]
    neg = [(names[i], coef[i]) for i in order[:k]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, terms, color, title in [(axes[0], pos, "#b5402a", "Pushes towards INJECTION"),
                                    (axes[1], neg, "#2b6a4f", "Pushes towards BENIGN")]:
        ax.barh([t for t, _ in terms][::-1], [abs(w) for _, w in terms][::-1], color=color)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("TF-IDF + LR (indirect-trained): largest n-gram weights", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/explain_tfidf_terms.png", dpi=200)
    plt.close(fig)
    return pos, neg


def word_contributions(pipe, text):
    vec, coef = pipe["tfidf"], pipe["clf"].coef_[0]
    X = vec.transform([text])
    vocab = vec.vocabulary_
    contrib = {}
    for tok in set(vec.build_analyzer()(text)):
        j = vocab.get(tok)
        if j is not None and " " not in tok:
            contrib[tok] = X[0, j] * coef[j]
    return contrib


def render_html_window(text, pipe):
    contrib = word_contributions(pipe, text)
    vmax = max([abs(v) for v in contrib.values()] + [1e-6])
    out = []
    for token in text.replace("\n", " \n ").split(" "):
        key = "".join(ch for ch in token.lower() if ch.isalnum())
        v = contrib.get(key, 0.0)
        a = min(1.0, abs(v) / vmax)
        color = f"rgba(200,60,40,{a:.2f})" if v > 0 else f"rgba(40,140,90,{a:.2f})"
        out.append("<br>" if token == "\n" else f'<span style="background:{color}">{html.escape(token)}</span>')
    return " ".join(out)


def pick_examples(df, p_col):
    picks = []
    for carrier in ["email", "table", "code"]:
        g = df[(df.label == 1) & (df.carrier == carrier) & (df[p_col] >= 0.5) & (df.position == "middle")]
        if len(g):
            picks.append(g.iloc[0])
    fp = df[(df.label == 0) & (df[p_col] >= 0.5) & (df.category == "benign_imperative")]
    if len(fp):
        picks.append(fp.iloc[0])
    return picks


def window_profiles(rows, scorers):
    fig, axes = plt.subplots(len(rows), 1, figsize=(11, 2.3 * len(rows)))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, rows):
        spans = windows(r.text)
        s = r.text.find(r.payload) if isinstance(r.payload, str) and r.payload else -1
        pay = (s, s + len(r.payload)) if s >= 0 else None
        x = np.arange(len(spans))
        width = 0.8 / len(scorers)
        for k, (name, sc) in enumerate(scorers.items()):
            vals = sc([r.text[a:b] for a, b in spans])
            ax.bar(x + k * width - 0.4 + width / 2, vals, width, label=MODEL_LABELS[name])
        for i, w in enumerate(spans):
            if pay and overlap(w, pay) >= 0.5 * min(w[1] - w[0], pay[1] - pay[0]):
                ax.axvspan(i - 0.45, i + 0.45, color="#f4c542", alpha=0.25, lw=0)
        ax.axhline(0.5, color="grey", ls="--", lw=0.8)
        ax.set_ylim(0, 1)
        ax.set_xticks(x, [f"w{i + 1}" for i in x], fontsize=7)
        kind = "injected" if r.label == 1 else f"benign ({r.category})"
        ax.set_title(f"{r.carrier} document, {kind}"
                     + (f", payload at {r.position}" if r.label == 1 else ""), fontsize=9, loc="left")
        ax.set_ylabel("P(injection)", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False, ncol=3, loc="upper right")
    fig.suptitle("Window score profiles (yellow = window containing the injected payload; dashed = 0.5 threshold)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/explain_window_profiles.png", dpi=200)
    plt.close(fig)


def distilbert_attention(name, df):
    sc = load_scorer(name)
    if sc is None:
        print(f"[skip] attention: {name} not available")
        return None
    import torch
    from transformers import AutoModelForSequenceClassification
    from src.indirect.models import MODEL_FILES
    tok = sc.tokenizer
    # attention weights are only returned by the "eager" implementation
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_FILES[name], attn_implementation="eager").to(sc.model.device).eval()
    inj = df[(df.label == 1) & df.payload.notna()].reset_index(drop=True)
    rows = []
    for _, r in inj.iterrows():
        spans = windows(r.text)
        texts = [r.text[a:b] for a, b in spans]
        best = int(np.argmax(sc(texts)))
        a, b = spans[best]
        window = r.text[a:b]
        ps = r.text.find(r.payload) - a
        pe = ps + len(r.payload)
        enc = tok(window, return_offsets_mapping=True, truncation=True, max_length=512, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            out = model(**{k: v.to(model.device) for k, v in enc.items()}, output_attentions=True)
        att = out.attentions[-1][0].mean(0)[0].cpu().numpy()  # [CLS] row, last layer
        in_pay = np.array([(s < pe and e > ps and e > s) for s, e in offsets])
        real = np.array([e > s for s, e in offsets])
        if real.sum() == 0 or in_pay.sum() == 0:
            continue
        mass = att[in_pay].sum() / att[real].sum()
        rows.append({"doc_id": r.doc_id, "carrier": r.carrier, "attention_on_payload": mass,
                     "payload_token_share": in_pay.sum() / real.sum()})
    d = pd.DataFrame(rows)
    print(f"attention on payload tokens: mean {d.attention_on_payload.mean():.3f} "
          f"vs payload share of tokens {d.payload_token_share.mean():.3f} (n={len(d)})")
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv("data/indirect/test.csv")
    scorers = load_all()
    tf = load_scorer("tfidf_ind")
    pos, neg = tfidf_terms(tf.pipeline)
    df["p_tfidf_ind"] = [max(tf([t[a:b] for a, b in windows(t)])) for t in df.text]
    examples = pick_examples(df, "p_tfidf_ind")
    window_profiles(examples, {k: v for k, v in scorers.items() if k.endswith("_ind")} or scorers)

    parts = ["<html><meta charset='utf-8'><body style='font-family:sans-serif;max-width:900px'>",
             "<h2>TF-IDF + LR (indirect-trained): word contributions in the top-scoring window</h2>",
             "<p>Red pushes towards injection, green towards benign; darker = stronger.</p>"]
    for r in examples:
        spans = windows(r.text)
        vals = tf([r.text[a:b] for a, b in spans])
        a, b = spans[int(np.argmax(vals))]
        parts.append(f"<h3>{r.carrier}, label={r.label}, top window score {vals.max():.2f}</h3>")
        parts.append(f"<div style='border:1px solid #ccc;padding:8px'>{render_html_window(r.text[a:b], tf.pipeline)}</div>")
    with open(f"{OUT}/explain_examples.html", "w", encoding="utf-8") as f:
        f.write("\n".join(parts) + "</body></html>")

    for name in ["distilbert_ind", "distilbert"]:
        res = distilbert_attention(name, df)
        if res is not None:
            res.to_csv(f"{OUT}/explain_attention_{name}.csv", index=False)
    print("top injection terms:", ", ".join(t for t, _ in pos))
    print("top benign terms:", ", ".join(t for t, _ in neg))


if __name__ == "__main__":
    main()
