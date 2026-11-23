"""Explain the TF-IDF + LR model: its strongest terms in each direction, and
word-level contributions for three indirect-pilot documents (one caught
injection, one missed injection, one false positive)."""
import html
import math
import os
import re
import textwrap
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.eval_indirect_pilot import PILOT_PATH, load_payload_lookup, payload_of
from src.evaluate import MODELS_DIR

RESULTS_DIR = "results"
FEATURES_CHART_PATH = f"{RESULTS_DIR}/tfidf_top_features.png"
EXAMPLES_HTML_PATH = f"{RESULTS_DIR}/explain_examples.html"
EXAMPLES_PNG_PATH = f"{RESULTS_DIR}/explain_examples.png"
TOP_N = 20

INJECTION_RGB = (214, 39, 40)
BENIGN_RGB = (31, 119, 180)


def load_model():
    pipeline = joblib.load(f"{MODELS_DIR}/tfidf_lr.joblib")
    return pipeline, pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]


def top_features(vectorizer, clf, n=TOP_N):
    terms = vectorizer.get_feature_names_out()
    coef = clf.coef_[0]
    order = np.argsort(coef)
    benign = [(terms[i], coef[i]) for i in order[:n]]
    injection = [(terms[i], coef[i]) for i in order[::-1][:n]]
    return injection, benign


def plot_top_features(injection, benign, out_path):
    # Benign terms at the bottom, injection terms at the top, each sorted
    # so the strongest term sits furthest from the zero line.
    rows = benign + injection[::-1]
    labels = [term for term, _ in rows]
    values = [weight for _, weight in rows]
    colors = ["#%02x%02x%02x" % (INJECTION_RGB if v > 0 else BENIGN_RGB) for v in values]

    fig, ax = plt.subplots(figsize=(7.5, 10))
    ax.barh(range(len(rows)), values, color=colors)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.axhline(len(benign) - 0.5, color="#cccccc", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Logistic regression coefficient (positive = toward injection)")
    ax.set_title(f"TF-IDF + LR: top {len(injection)} terms toward injection and toward benign")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def token_contributions(text, vectorizer, clf):
    """Split the document's score over its word tokens.

    For a linear model on TF-IDF features the score is intercept + sum over
    terms of tfidf(term) * coef(term). Each term's share is divided evenly
    across its occurrences, and each bigram's share is split between its two
    words, so the token contributions sum exactly to score - intercept."""
    preprocess = vectorizer.build_preprocessor()
    matches = list(re.finditer(vectorizer.token_pattern, text))
    tokens = [preprocess(m.group()) for m in matches]
    bigrams = [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
    counts = Counter(tokens + bigrams)

    row = vectorizer.transform([text])
    weights = dict(zip(row.indices, row.data))
    vocab = vectorizer.vocabulary_
    coef = clf.coef_[0]

    def share(term):
        j = vocab.get(term)
        return weights.get(j, 0.0) * coef[j] / counts[term] if j is not None else 0.0

    per_token = np.array([share(t) for t in tokens], dtype=float)
    for i, bigram in enumerate(bigrams):
        half = share(bigram) / 2
        per_token[i] += half
        per_token[i + 1] += half
    return matches, per_token


def pick_examples(pilot, proba, lookup, pipeline):
    """Most confident catch, most confident false positive, and the missed
    injection with the lowest score among those the model catches when the
    same payload is scored on its own (falling back to any miss)."""
    pred = (proba >= 0.5).astype(int)
    caught = pilot[(pilot["label"] == 1) & (pred == 1)]
    missed = pilot[(pilot["label"] == 1) & (pred == 0)]
    false_pos = pilot[(pilot["label"] == 0) & (pred == 1)]

    standalone = pipeline.predict_proba([payload_of(r, lookup) for r in missed.itertuples()])[:, 1]
    diluted = missed[standalone >= 0.5]
    missed_pool = diluted if not diluted.empty else missed

    return [
        ("Caught injection", proba[caught.index].idxmax()),
        ("Missed injection", proba[missed_pool.index].idxmin()),
        ("False positive", proba[false_pos.index].idxmax()),
    ]


def build_example(kind, row, proba, lookup, pipeline, vectorizer, clf):
    text = row.text
    matches, per_token = token_contributions(text, vectorizer, clf)
    intercept = clf.intercept_[0]
    logit = pipeline.decision_function([text])[0]
    error = abs(intercept + per_token.sum() - logit)
    if error > 1e-6:
        raise ValueError(f"Contributions do not reproduce the score for {row.source_id} (off by {error:.2e})")

    span = None
    payload_alone = None
    if row.position != "none":
        payload = payload_of(row, lookup)
        start = text.index(payload)
        span = (start, start + len(payload))
        payload_alone = pipeline.predict_proba([payload])[0, 1]

    in_payload = np.array([span is not None and span[0] <= m.start() < span[1] for m in matches])
    return {
        "kind": kind,
        "row": row,
        "proba": proba,
        "logit": logit,
        "intercept": intercept,
        "matches": matches,
        "per_token": per_token,
        "span": span,
        "payload_sum": per_token[in_payload].sum(),
        "host_sum": per_token[~in_payload].sum(),
        "payload_alone": payload_alone,
    }


def _rgba(value, scale):
    rgb = INJECTION_RGB if value > 0 else BENIGN_RGB
    alpha = min(1.0, abs(value) / scale) * 0.85 if scale else 0.0
    return rgb, alpha


def _summary_lines(ex):
    row = ex["row"]
    lines = [
        f"source_id: {row.source_id} | true label: {'injection' if row.label == 1 else 'benign'} "
        f"| position: {row.position} | P(injection) = {ex['proba']:.3f}",
        f"score {ex['logit']:+.3f} = intercept {ex['intercept']:+.3f} + embedded prompt {ex['payload_sum']:+.3f} "
        f"+ host paragraph {ex['host_sum']:+.3f}",
    ]
    if ex["payload_alone"] is not None:
        lines.append(f"Embedded prompt scored on its own: P(injection) = {ex['payload_alone']:.3f}")
    return lines


def _html_tokens(text, start, end, ex, scale):
    out, cursor = [], start
    for match, value in zip(ex["matches"], ex["per_token"]):
        if match.start() < start or match.start() >= end:
            continue
        out.append(html.escape(text[cursor:match.start()]))
        (r, g, b), alpha = _rgba(value, scale)
        out.append(
            f'<span class="tok" style="background: rgba({r},{g},{b},{alpha:.2f})" '
            f'title="{value:+.4f}">{html.escape(match.group())}</span>'
        )
        cursor = match.end()
    out.append(html.escape(text[cursor:end]))
    return "".join(out)


def render_html(examples, scale, out_path):
    sections = []
    for ex in examples:
        text, span = ex["row"].text, ex["span"]
        if span:
            body = (_html_tokens(text, 0, span[0], ex, scale)
                    + '<span class="payload">' + _html_tokens(text, span[0], span[1], ex, scale) + "</span>"
                    + _html_tokens(text, span[1], len(text), ex, scale))
        else:
            body = _html_tokens(text, 0, len(text), ex, scale)
        summary = "".join(f"<p class=\"meta\">{html.escape(line)}</p>" for line in _summary_lines(ex))
        sections.append(f"<section><h2>{html.escape(ex['kind'])}</h2>{summary}<div class=\"doc\">{body}</div></section>")

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TF-IDF explanations</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 860px; margin: 32px auto; padding: 0 16px;
         color: #222; background: #fff; line-height: 1.5; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ font-size: 1.1rem; margin-bottom: 4px; }}
  section {{ border-top: 1px solid #ddd; padding-top: 12px; margin-top: 24px; }}
  .meta {{ margin: 2px 0; font-size: 0.9rem; color: #555; }}
  .doc {{ margin-top: 10px; font-size: 0.95rem; line-height: 1.9; }}
  .tok {{ border-radius: 3px; padding: 1px 1px; }}
  .payload {{ outline: 2px dashed #444; outline-offset: 2px; }}
  .legend span {{ padding: 2px 6px; border-radius: 3px; margin-right: 8px; }}
</style>
</head>
<body>
<h1>TF-IDF + LR: word contributions on indirect-pilot documents</h1>
<p class="legend"><span style="background: rgba({','.join(map(str, INJECTION_RGB))},0.6)">toward injection</span>
<span style="background: rgba({','.join(map(str, BENIGN_RGB))},0.6)">toward benign</span>
<span style="outline: 2px dashed #444">embedded prompt</span></p>
<p class="meta">Colour strength is shared across all three documents (full colour = |contribution| &ge; {scale:.3f}).
Hover a word for its exact contribution to the logit. A word's contribution includes its unigram weight
and half of each bigram it belongs to; contributions plus the intercept equal the model's score.</p>
{''.join(sections)}
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"Saved {out_path}")


def _wrapped_chunks(ex, width):
    """Whitespace-separated chunks of the document, wrapped to lines, each
    carrying the summed contribution of the tokens it contains."""
    text, span = ex["row"].text, ex["span"]
    chunks = []
    for chunk in re.finditer(r"\S+", text):
        value = sum(v for m, v in zip(ex["matches"], ex["per_token"])
                    if chunk.start() <= m.start() < chunk.end())
        inside = span is not None and span[0] <= chunk.start() < span[1]
        chunks.append((chunk.group(), value, inside))

    lines, line, used = [], [], 0
    for word, value, inside in chunks:
        if line and used + 1 + len(word) > width:
            lines.append(line)
            line, used = [], 0
        line.append((word, value, inside))
        used += len(word) + (1 if used else 0)
    if line:
        lines.append(line)
    return lines


def render_png(examples, scale, out_path, width=100):
    font_size = 8
    char_w = font_size * 0.6 / 72
    line_h = font_size * 1.9 / 72
    margin = 0.3

    blocks = []
    for ex in examples:
        header = [ex["kind"]] + [w for line in _summary_lines(ex) for w in textwrap.wrap(line, width)]
        blocks.append((header, _wrapped_chunks(ex, width)))

    title = textwrap.wrap("TF-IDF + LR word contributions (red = toward injection, blue = toward benign, "
                          "underline = embedded prompt)", width)
    n_lines = sum(len(h) + len(body) + 2 for h, body in blocks) + len(title) + 1
    fig_w = width * char_w + 2 * margin
    fig_h = n_lines * line_h + 2 * margin
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(fig_h, 0)
    ax.axis("off")

    y = margin + line_h
    for line in title:
        ax.text(margin, y, line, fontsize=font_size, family="monospace", weight="bold")
        y += line_h
    y += line_h
    for header, body in blocks:
        for i, line in enumerate(header):
            ax.text(margin, y, line, fontsize=font_size + (1 if i == 0 else 0), family="monospace",
                    weight="bold" if i == 0 else "normal", color="#222" if i == 0 else "#555")
            y += line_h
        y += line_h * 0.4
        for line in body:
            x = margin
            for word, value, inside in line:
                (r, g, b), alpha = _rgba(value, scale)
                ax.text(x, y, word, fontsize=font_size, family="monospace",
                        bbox={"facecolor": (r / 255, g / 255, b / 255, alpha), "edgecolor": "none", "pad": 0.6})
                if inside:
                    ax.plot([x, x + len(word) * char_w], [y + line_h * 0.3] * 2, color="#333", linewidth=0.8)
                x += (len(word) + 1) * char_w
            y += line_h
        y += line_h * 1.2

    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pipeline, vectorizer, clf = load_model()

    injection, benign = top_features(vectorizer, clf)
    print("Top terms toward injection:", ", ".join(t for t, _ in injection))
    print("Top terms toward benign:   ", ", ".join(t for t, _ in benign))
    plot_top_features(injection, benign, FEATURES_CHART_PATH)

    pilot = pd.read_csv(PILOT_PATH)
    proba = pd.Series(pipeline.predict_proba(pilot["text"])[:, 1], index=pilot.index)
    lookup = load_payload_lookup()

    examples = [
        build_example(kind, pilot.loc[idx], proba[idx], lookup, pipeline, vectorizer, clf)
        for kind, idx in pick_examples(pilot, proba, lookup, pipeline)
    ]
    scale = max(np.abs(ex["per_token"]).max() for ex in examples)
    scale = math.ceil(scale * 1000) / 1000

    for ex in examples:
        print(f"{ex['kind']:17s} {ex['row'].source_id}  P={ex['proba']:.3f}  "
              f"prompt {ex['payload_sum']:+.3f}  host {ex['host_sum']:+.3f}")

    render_html(examples, scale, EXAMPLES_HTML_PATH)
    render_png(examples, scale, EXAMPLES_PNG_PATH)


if __name__ == "__main__":
    main()
