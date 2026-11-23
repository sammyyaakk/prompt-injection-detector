"""Two follow-up analyses on the indirect test set.

1. Word-level explanations (RQ5). Inside each injected document's top-scoring
   window, rank words by importance and check whether the 5 most important
   words fall inside the injected payload.
     - MiniLM + LR: word removal (occlusion). Importance of a word = drop in
       P(injection) when that word is deleted from the window.
     - TF-IDF + LR: exact linear contribution (tf-idf weight x coefficient).
   Score = share of the top-5 words that lie inside the payload; baseline =
   share of the window's words that lie inside the payload (what 5 random
   words would give on average).

2. Corrupted-benign control (RQ4). In benign documents with a harmless insert,
   apply the character-typo and homoglyph attacks to the insert only and
   measure the false-positive rate. If FPR rises as recall did on injected
   documents, the recall gain under these attacks is a shortcut ("garbled
   text looks suspicious"), not detection.

Writes results/indirect/word_localisation.csv, corrupted_benign.csv, extra.md
Run: python -m scripts.extra_indirect
"""
import re

import numpy as np
import pandas as pd

from src.adversarial import attacks as A
from src.indirect.chunking import windows
from src.indirect.models import MODEL_LABELS, load_all

OUT = "results/indirect"
THRESHOLD = 0.5
TOP_K = 5
_WORD = re.compile(r"[A-Za-z0-9\u0080-￿']+")


def top_window(scorer, text):
    spans = windows(text)
    scores = scorer([text[a:b] for a, b in spans])
    i = int(np.argmax(scores))
    return spans[i], float(scores[i])


def words_in(window_text):
    return [(m.start(), m.end()) for m in _WORD.finditer(window_text)]


def occlusion_importance(scorer, window_text, words):
    base = scorer([window_text])[0]
    variants = [window_text[:s] + window_text[e:] for s, e in words]
    return base - scorer(variants)


def tfidf_importance(pipe, window_text, words):
    vec, coef = pipe["tfidf"], pipe["clf"].coef_[0]
    X = vec.transform([window_text])
    vocab = vec.vocabulary_
    out = []
    for s, e in words:
        tok = window_text[s:e].lower()
        j = vocab.get(tok)
        out.append(X[0, j] * coef[j] if j is not None else 0.0)
    return np.array(out)


def word_localisation(scorers, df):
    inj = df[(df.label == 1) & df.payload.notna()].reset_index(drop=True)
    rows = []
    for name, sc in scorers.items():
        if not (name.startswith("embedding") or name.startswith("tfidf")):
            continue
        hits, base, n = [], [], 0
        for _, r in inj.iterrows():
            (a, b), _ = top_window(sc, r.text)
            win = r.text[a:b]
            ps = r.text.find(r.payload) - a
            pe = ps + len(r.payload)
            words = words_in(win)
            if len(words) < TOP_K:
                continue
            inside = np.array([(s < pe and e > ps) for s, e in words])
            if inside.sum() == 0:  # top window does not contain the payload: count as a miss
                hits.append(0.0)
                base.append(0.0)
                n += 1
                continue
            imp = (occlusion_importance(sc, win, words) if name.startswith("embedding")
                   else tfidf_importance(sc.pipeline, win, words))
            top = np.argsort(-imp)[:TOP_K]
            hits.append(inside[top].mean())
            base.append(inside.mean())
            n += 1
        rows.append({"model": name, "method": "word removal" if name.startswith("embedding") else "linear weights",
                     "docs": n, "top5_in_payload": float(np.mean(hits)), "random_baseline": float(np.mean(base))})
        print(rows[-1])
    return pd.DataFrame(rows)


def corrupted_benign(scorers, df):
    ben = df[(df.label == 0) & df.payload.notna() & (df.payload.astype(str).str.len() > 0)].reset_index(drop=True)
    rows = []
    for attack, fn in [("none", None), ("character_typos", A.character_perturbation),
                       ("unicode_homoglyphs", A.unicode_lookalike_substitution)]:
        texts = [r.text if fn is None else r.text.replace(r.payload, fn(r.payload, seed=42 + i), 1)
                 for i, r in ben.iterrows()]
        for name, sc in scorers.items():
            p = np.array([max(sc([t[a:b] for a, b in windows(t)])) for t in texts])
            rows.append({"attack": attack, "model": name, "n": len(texts), "fpr": float((p >= THRESHOLD).mean())})
    out = pd.DataFrame(rows)
    adv = pd.read_csv(f"{OUT}/adversarial.csv")
    adv = adv[adv["mode"] == "chunk"].set_index(["attack", "model"]).recall
    out["recall_on_injected"] = [adv.get((a, m), np.nan) for a, m in zip(out.attack, out.model)]
    return out


def main():
    df = pd.read_csv("data/indirect/test.csv")
    scorers = load_all()
    loc = word_localisation(scorers, df)
    loc.to_csv(f"{OUT}/word_localisation.csv", index=False)
    cb = corrupted_benign(scorers, df)
    cb.to_csv(f"{OUT}/corrupted_benign.csv", index=False)
    L = lambda m: MODEL_LABELS[m]
    lines = ["# Follow-up analyses: word-level localisation and corrupted-benign control", "",
             "## Top-5 important words inside the payload (top window of each injected test document)", "",
             "| model | method | docs | top-5 words in payload | random baseline |", "|---|---|---|---|---|"]
    lines += [f"| {L(r.model)} | {r.method} | {r.docs} | {r.top5_in_payload:.3f} | {r.random_baseline:.3f} |"
              for r in loc.itertuples()]
    lines += ["", "## Corrupted-benign control (chunk-level, benign documents with a harmless insert)", "",
              "| attack | model | FPR on benign | recall on injected |", "|---|---|---|---|"]
    lines += [f"| {r.attack} | {L(r.model)} | {r.fpr:.3f} | {r.recall_on_injected:.3f} |" for r in cb.itertuples()]
    with open(f"{OUT}/extra.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
