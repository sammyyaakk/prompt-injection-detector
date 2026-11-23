"""Full indirect-injection evaluation.

For every available model (direct-trained and indirect-trained variants of
TF-IDF, MiniLM and DistilBERT) and two scoring modes:
  doc   - the whole document is scored once
  chunk - overlapping ~400-char windows are scored; the document score is the max
  chunk_decode - base64 runs are decoded and appended first, then chunk scoring
it reports, at threshold 0.5:
  1. main metrics on four test sets: indirect test (BIPIA carriers), the
     Wikipedia pilot (unseen carrier domain), direct test, hand-crafted holdout
  2. indirect-test breakdowns by carrier, position, payload family, framing
     template, and false-positive rate by benign document type
  3. the six adversarial attacks applied to the embedded payload only
  4. localisation: does the top-scoring window contain the injected payload?

Writes results/indirect/*.csv, results/indirect/indirect_results.md and figures.
Run: python -m scripts.eval_indirect
"""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from src.adversarial import attacks as A
from src.indirect.chunking import overlap, windows
from src.indirect.decode import decode_and_append
from src.indirect.models import MODEL_LABELS, load_all

OUT = "results/indirect"
THRESHOLD = 0.5
MODES = ["doc", "chunk", "chunk_decode"]
ATTACKS = {
    "none": None,
    "character_typos": A.character_perturbation,
    "unicode_homoglyphs": A.unicode_lookalike_substitution,
    "synonym_swap": A.synonym_substitution,
    "payload_splitting": A.payload_splitting,
    "base64_encoding": A.base64_encoding,
    "roleplay_framing": A.roleplay_framing,
}


# ---------------------------------------------------------------- data ----
def _pilot_with_payloads():
    pilot = pd.read_csv("data/indirect_pilot/indirect_pilot.csv")
    lookup = {}
    for name, path in {"test": "data/processed/test.csv",
                       "holdout": "data/holdout/handcrafted.csv"}.items():
        for i, t in enumerate(pd.read_csv(path)["text"].astype(str).str.strip()):
            lookup[f"{name}:{i}"] = t
    pilot["payload"] = [lookup.get(s.split("|")[0], "") if s.split("|")[0] in lookup else ""
                        for s in pilot.source_id]
    return pilot


def load_test_sets():
    sets = {
        "indirect_test": pd.read_csv("data/indirect/test.csv"),
        "pilot_wikipedia": _pilot_with_payloads(),
        "direct_test": pd.read_csv("data/processed/test.csv"),
        "holdout": pd.read_csv("data/holdout/handcrafted.csv"),
    }
    for df in sets.values():
        df["text"] = df["text"].astype(str)
    return sets


# ------------------------------------------------------------- scoring ----
class ScoreCache:
    """Scores each unique text once per model; doc and chunk modes share it."""
    def __init__(self, scorer):
        self.scorer, self.cache = scorer, {}

    def score(self, texts):
        new = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if new:
            for t, p in zip(new, self.scorer(new)):
                self.cache[t] = float(p)
        return np.array([self.cache[t] for t in texts])


def doc_scores(cache, texts):
    return cache.score(list(texts))


def chunk_scores(cache, texts, return_windows=False):
    spans = [windows(t) for t in texts]
    flat = [t[s:e] for t, sp in zip(texts, spans) for s, e in sp]
    flat_scores = cache.score(flat)
    out, best, k = [], [], 0
    for sp in spans:
        sc = flat_scores[k:k + len(sp)]
        k += len(sp)
        out.append(sc.max())
        best.append((sp, sc))
    return (np.array(out), best) if return_windows else np.array(out)


def chunk_decode_scores(cache, texts):
    return chunk_scores(cache, [decode_and_append(t) for t in texts])


SCORE_FNS = {"doc": lambda c, t: doc_scores(c, t), "chunk": lambda c, t: chunk_scores(c, t),
             "chunk_decode": lambda c, t: chunk_decode_scores(c, t)}


def metrics(y, p):
    pred = (p >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m = {"n": len(y), "precision": precision_score(y, pred, zero_division=0),
         "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
         "fpr": fp / (fp + tn) if (fp + tn) else np.nan, "tp": tp, "fn": fn, "fp": fp, "tn": tn}
    m["auroc"] = roc_auc_score(y, p) if len(set(y)) == 2 else np.nan
    return m


# ------------------------------------------------------------- reports ----
def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v)
                 else ("-" if isinstance(v, float) and pd.isna(v) else str(v)) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_main(scorers, caches, sets):
    rows = []
    for ts, df in sets.items():
        for name in scorers:
            for mode in MODES:
                p = SCORE_FNS[mode](caches[name], df.text.tolist())
                rows.append({"test_set": ts, "model": name, "mode": mode, **metrics(df.label.values, p)})
                if ts == "indirect_test":
                    df[f"p_{name}_{mode}"] = p
    return pd.DataFrame(rows)


def run_breakdowns(scorers, df):
    rows = []
    inj, ben = df[df.label == 1], df[df.label == 0]
    groups = [("carrier", inj, "recall"), ("position", inj, "recall"),
              ("payload_family", inj, "recall"), ("template", inj[inj.template.notna()], "recall"),
              ("category", ben, "fpr"), ("carrier", ben, "fpr")]
    for col, sub, metric in groups:
        for val, g in sub.groupby(col):
            for name in scorers:
                for mode in MODES:
                    pred = g[f"p_{name}_{mode}"].values >= THRESHOLD
                    rows.append({"dimension": col, "value": str(val) or "(none)", "metric": metric,
                                 "n": len(g), "model": name, "mode": mode, "score": pred.mean()})
    return pd.DataFrame(rows)


def run_adversarial(scorers, caches, df):
    inj = df[(df.label == 1) & df.payload.notna()].reset_index(drop=True)
    rows = []
    for attack, fn in ATTACKS.items():
        texts = []
        for i, r in inj.iterrows():
            new = r.payload if fn is None else fn(r.payload, seed=42 + i)
            texts.append(r.text.replace(r.payload, new, 1))
        for name in scorers:
            for mode in MODES:
                p = SCORE_FNS[mode](caches[name], texts)
                rows.append({"attack": attack, "model": name, "mode": mode,
                             "n": len(texts), "recall": float((p >= THRESHOLD).mean())})
    out = pd.DataFrame(rows)
    base = out[out.attack == "none"].set_index(["model", "mode"]).recall
    out["recall_drop"] = [base[(m, md)] - r for m, md, r in zip(out.model, out["mode"], out.recall)]
    return out


def run_localisation(scorers, caches, df):
    """Among injected docs with 2+ windows, how often is the top window the payload?"""
    inj = df[(df.label == 1) & df.payload.notna()].reset_index(drop=True)
    rows, examples = [], []
    for name in scorers:
        _, best = chunk_scores(caches[name], inj.text.tolist(), return_windows=True)
        hits, chance, n, n_detected, hits_detected = 0, 0.0, 0, 0, 0
        for (spans, sc), (_, r) in zip(best, inj.iterrows()):
            if len(spans) < 2:
                continue
            s = r.text.find(r.payload)
            pay = (s, s + len(r.payload))
            ok = [overlap(w, pay) >= 0.5 * min(w[1] - w[0], pay[1] - pay[0]) for w in spans]
            hit = ok[int(np.argmax(sc))]
            n += 1
            hits += hit
            chance += np.mean(ok)
            if sc.max() >= THRESHOLD:
                n_detected += 1
                hits_detected += hit
        rows.append({"model": name, "docs_with_2plus_windows": n, "top_window_hit_rate": hits / n,
                     "random_window_baseline": chance / n,
                     "hit_rate_when_detected": hits_detected / n_detected if n_detected else np.nan,
                     "detected": n_detected})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- figures ----
def fig_main(main):
    d = main[main.test_set == "indirect_test"]
    models = [m for m in MODEL_LABELS if m in set(d.model)]
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, metric, title in [(axes[0], "recall", "Recall (injected documents caught)"),
                              (axes[1], "fpr", "False-positive rate (benign documents flagged)")]:
        for k, (mode, color) in enumerate([("doc", "#7a8fa6"), ("chunk", "#2b5d8a")]):
            vals = [d[(d.model == m) & (d["mode"] == mode)][metric].iloc[0] for m in models]
            bars = ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, label=f"{mode}-level", color=color)
            ax.bar_label(bars, fmt="%.2f", fontsize=7)
        ax.set_xticks(x, [MODEL_LABELS[m].replace(" (indirect-trained)", "\n(indirect-trained)") for m in models],
                      fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.08)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Indirect test set (800 BIPIA documents, unseen carriers, categories and templates)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_indirect_main.png", dpi=200)
    plt.close(fig)


def fig_adversarial(adv):
    d = adv[adv["mode"] == "chunk"]
    models = [m for m in MODEL_LABELS if m in set(d.model)]
    grid = d.pivot(index="attack", columns="model", values="recall").loc[list(ATTACKS), models]
    fig, ax = plt.subplots(figsize=(1.6 * len(models) + 2, 4))
    im = ax.imshow(grid.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(grid.values):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="white" if v > 0.6 else "black")
    ax.set_xticks(range(len(models)), [MODEL_LABELS[m].replace(" (indirect-trained)", "\n(ind.-trained)")
                                       for m in models], fontsize=7)
    ax.set_yticks(range(len(grid)), grid.index, fontsize=8)
    ax.set_title("Recall after attacking only the payload\n(chunk-level scoring)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_indirect_adversarial.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- main ----
def write_markdown(main, br, adv, loc, runtimes):
    L = lambda m: MODEL_LABELS[m]
    lines = ["# Indirect Prompt Injection: Full Evaluation", "",
             "Generated by `python -m scripts.eval_indirect`. Threshold 0.5 throughout. "
             "`doc` scores the whole document; `chunk` scores ~400-character overlapping windows "
             "and takes the maximum; `chunk_decode` first decodes base64-looking runs and appends "
             "the decoded text, then scores like `chunk`.", ""]
    for ts in ["indirect_test", "pilot_wikipedia", "direct_test", "holdout"]:
        d = main[main.test_set == ts].copy()
        d["model"] = d.model.map(L)
        lines += [f"## {ts} (n={int(d.n.iloc[0])})", "",
                  md_table(d[["model", "mode", "precision", "recall", "f1", "fpr", "auroc"]]), ""]
    for dim, metric in [("carrier", "recall"), ("position", "recall"), ("payload_family", "recall"),
                        ("template", "recall"), ("category", "fpr"), ("carrier", "fpr")]:
        d = br[(br.dimension == dim) & (br.metric == metric) & (br["mode"] == "chunk")]
        piv = d.pivot_table(index=["value", "n"], columns="model", values="score").reset_index()
        piv.columns = [L(c) if c in MODEL_LABELS else c for c in piv.columns]
        lines += [f"### Indirect test, chunk-level {metric} by {dim}", "", md_table(piv), ""]
    for mode in MODES:
        d = adv[adv["mode"] == mode].pivot(index="attack", columns="model", values="recall")
        d = d.loc[list(ATTACKS)].reset_index()
        d.columns = [L(c) if c in MODEL_LABELS else c for c in d.columns]
        lines += [f"## Adversarial: recall on {int(adv.n.iloc[0])} injected test documents ({mode}-level)",
                  "", "Only the embedded payload is attacked; the carrier document is unchanged.", "",
                  md_table(d), ""]
    d = loc.copy()
    d["model"] = d.model.map(L)
    lines += ["## Localisation: is the top-scoring window the injected payload?", "",
              "Injected test documents with two or more windows. A hit means the top window and the "
              "payload overlap by at least half of the shorter of the two.", "", md_table(d), "",
              "## Runtime (s, this machine)", "",
              md_table(pd.DataFrame([{"model": L(k), "seconds": v} for k, v in runtimes.items()]),
                       floatfmt="{:.1f}")]
    with open(f"{OUT}/indirect_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    sets = load_test_sets()
    scorers = load_all()
    caches = {n: ScoreCache(s) for n, s in scorers.items()}
    t0 = time.time()
    main_df = run_main(scorers, caches, sets)
    print("main done", round(time.time() - t0), "s")
    br = run_breakdowns(scorers, sets["indirect_test"])
    adv = run_adversarial(scorers, caches, sets["indirect_test"])
    print("adversarial done", round(time.time() - t0), "s")
    loc = run_localisation(scorers, caches, sets["indirect_test"])
    runtimes = {}
    sample = sets["indirect_test"].text.tolist()[:100]
    for n, s in scorers.items():
        t = time.time()
        s(sample)
        runtimes[n] = time.time() - t
    main_df.to_csv(f"{OUT}/main_results.csv", index=False)
    br.to_csv(f"{OUT}/breakdowns.csv", index=False)
    adv.to_csv(f"{OUT}/adversarial.csv", index=False)
    loc.to_csv(f"{OUT}/localisation.csv", index=False)
    sets["indirect_test"].drop(columns=["text", "payload"]).to_csv(f"{OUT}/indirect_test_scores.csv", index=False)
    fig_main(main_df)
    fig_adversarial(adv)
    write_markdown(main_df, br, adv, loc, runtimes)
    print(f"wrote {OUT}/indirect_results.md")


if __name__ == "__main__":
    main()
