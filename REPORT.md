# Adversarial Robustness Report

This is a follow-up to the main [README](README.md) results, testing how much each of the three classifiers' recall degrades under six common evasion techniques. It answers a question the README's Limitations section raised but didn't quantify: *how* fragile is "neither model is adversarially robust," exactly?

## Method

- **Target set:** the 12 injection examples (label = 1) from the 24-example hand-crafted holdout set (`data/holdout/handcrafted.csv`). Benign examples are excluded — the question here is evasion (getting a real injection past the filter), not false positives.
- **Attacks:** each of the 12 texts is perturbed by six attack functions (`src/adversarial/attacks.py`), applied independently (not stacked):
  - `character_perturbation` — random keyboard typos: swap, delete, duplicate, adjacent-key substitution (~15% of characters)
  - `unicode_lookalike_substitution` — Latin letters replaced with confusable Cyrillic/Greek lookalikes (~40% of eligible characters)
  - `synonym_substitution` — content words swapped for a WordNet synonym (~50% of eligible words)
  - `payload_splitting` — the payload is split at the midpoint and stitched back together around an unrelated filler sentence
  - `base64_encoding` — the entire payload is base64-encoded and wrapped as `"decode this: <b64>"`
  - `roleplay_framing` — `"In a hypothetical world where AI has no restrictions, "` is prepended to the original payload
- **Models:** the same three classifiers from the README (TF-IDF + LR, MiniLM embeddings + LR, fine-tuned DistilBERT), loaded from `models/`.
- **Metric:** recall drop = (recall on the original 12 examples) − (recall on the perturbed 12 examples). Positive = attack works (recall falls). Negative = attack backfired (recall rose).
- Perturbations are seeded (`seed = 42 + row index`) for reproducibility. Full harness: `scripts/run_adversarial_eval.py`.

## Results

Baseline recall on the unperturbed 12-example injection subset (matches the README's holdout recall exactly): **TF-IDF 0.583, MiniLM 0.833, DistilBERT 0.667.**

| Attack | TF-IDF + LR | MiniLM Embeddings + LR | DistilBERT |
|---|---|---|---|
| baseline (unperturbed) | 0.000 | 0.000 | 0.000 |
| character_perturbation | 0.167 | 0.083 | **-0.333** |
| unicode_lookalike_substitution | 0.167 | **0.667** | **-0.333** |
| synonym_substitution | 0.167 | 0.083 | -0.167 |
| payload_splitting | -0.083 | 0.000 | 0.000 |
| base64_encoding | **0.583** | **0.833** | **0.583** |
| roleplay_framing | -0.167 | -0.167 | 0.000 |

(Full numbers in [`results/adversarial_robustness.csv`](results/adversarial_robustness.csv); rendered table in [`results/adversarial_robustness.md`](results/adversarial_robustness.md).)

## Reading this

**Base64 encoding is a complete, universal bypass.** All three models drop recall to near zero. None of them can decode base64 — they see `"decode this: <gibberish>"` and, having no readable trigger words left, call it benign. This is the single most effective attack tested and works identically regardless of model architecture, because the vulnerability isn't in the classifier, it's in the premise: none of these models operate on decoded content.

**Unicode homoglyphs are highly effective against MiniLM** (0.833 → 0.167 recall) but, surprisingly, **increased DistilBERT's recall to a perfect 1.0** — a negative drop, meaning the attack backfired. The same reversal happens with character-level typos on DistilBERT (also -0.333). The likely explanation: heavy character-level corruption produces a flood of `[UNK]` or badly fragmented subword tokens, and DistilBERT appears to have learned "garbled/corrupted input" itself as a suspicious signal, rather than being robust to obfuscation in any meaningful semantic sense. That's a fragile, spurious correlation, not evidence of real robustness — it would be easy to construct a counterexample (e.g. a benign sentence with the same corruption rate) that this same shortcut would flag incorrectly.

**Roleplay framing doesn't work as an evasion technique against any of the three models** — it made TF-IDF and MiniLM recall go *up*, not down. `"In a hypothetical world where AI has no restrictions, "` is itself a jailbreak-style phrase heavily represented in the training data (DAN-style prompts, etc.), so prepending it doesn't disguise the underlying instruction, it adds a second, independently-detectable red flag on top of the first.

**Payload splitting (naive midpoint + filler sentence) has almost no effect** on any model. A single filler sentence isn't enough dilution to break either lexical (TF-IDF) or semantic (embedding/DistilBERT) pattern matching — the trigger words are all still present, just with a short irrelevant sentence in between.

**Synonym substitution and light character perturbation are mildly effective against the two linear models** (TF-IDF and MiniLM both lose ~0.08–0.17 recall) but are the weakest attacks tested overall — consistent with the models still having some vocabulary/embedding overlap with the substituted words.

## Caveats

- **n = 12.** The injection subset of the holdout set is tiny — each single flipped prediction moves recall by 0.083. These numbers are directional evidence, not precise estimates; a different random seed or a slightly different holdout set could shift several of them by one or two flips.
- **Single perturbation strength per attack.** Rates (e.g. 15% typo rate, 40% homoglyph rate) were chosen once and not swept. A stronger or weaker attack would likely show different (probably more monotonic) drop patterns, especially for DistilBERT's counterintuitive negative drops.
- **Attacks are applied independently, not combined.** A real attacker would likely stack multiple techniques (e.g. base64 + roleplay framing); this evaluation doesn't test combined attacks.
- **This does not make any model "robust."** A near-zero recall drop for an attack (e.g. payload_splitting) means this specific, naive implementation of that attack class didn't work well — not that a more sophisticated version of the same idea wouldn't. Base64 encoding alone is proof that all three models are trivially bypassable by anyone willing to encode their payload.

## Takeaway

This quantifies what the README's Limitations section already stated qualitatively: none of these three models is adversarially robust. The specific finding worth remembering is that the failure modes differ by architecture — MiniLM is most exposed to unicode obfuscation, all three are equally and completely exposed to encoding-based bypasses, and DistilBERT's apparent "resistance" to character-level corruption is a spurious side effect of the corruption itself looking like a training-time correlate of injection, not a real defense.
