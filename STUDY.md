# Prompt Injection Detection: A Comparative Study

This document consolidates four separate analyses in this repository into a single narrative: the main
three-model comparison (README.md), adversarial robustness under six evasion techniques (REPORT.md), a
rule-based error taxonomy (results/error_analysis.md), and threshold/calibration analysis
(results/threshold_analysis.md). It's the single entry point for understanding what was tested, what was
found, and what the results do and don't support.

## 1. Motivation

Prompt injection is when adversarial input causes an LLM to disregard its original instructions and follow
attacker-supplied ones instead. OWASP has ranked it the top security risk for LLM applications
(LLM01:2025) for two consecutive years. The problem is structurally hard: LLMs process trusted instructions
and untrusted data through the same channel, so there is no built-in way for the model to distinguish "a
system instruction" from "user- or document-supplied text that happens to look like an instruction."

Detection — classifying a piece of text as a legitimate instruction/query versus an injection attempt before
it reaches the LLM — is one mitigation layer among several (others include prompt structuring, output
filtering, and privilege separation). This project treats detection as a binary text classification problem
and asks a narrower question than "how do we solve prompt injection": given three classifiers of increasing
capacity and decreasing interpretability, how do they actually compare on in-distribution accuracy,
generalization to unseen phrasing, and robustness to deliberate evasion — and where does each one break?

## 2. Related work

This project does not benchmark against these tools directly (see [Limitations](#9-limitations)); they are
named here for context on the production/research landscape this sits inside.

- **[Rebuff](https://github.com/protectai/rebuff)** — an open-source, multi-layered prompt injection
  detection framework combining heuristics, a vector database of known attack embeddings, an LLM-based
  self-check, and canary tokens to detect leaked prompts.
- **[LLM Guard](https://github.com/protectai/llm-guard)** — an open-source toolkit of input/output scanners
  for LLM applications, including a prompt injection scanner alongside PII detection, toxicity filtering,
  and other content checks.
- **[Lakera Guard](https://www.lakera.ai/lakera-guard)** — a commercial API-based guardrail product
  specifically for prompt injection and jailbreak detection, trained on a large, continuously updated corpus
  of real-world attacks.
- **[ProtectAI `deberta-v3-base-prompt-injection-v2`](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2)** —
  an openly published fine-tuned transformer classifier for this exact task, trained on a much larger corpus
  than this project's ~1,600 examples. It's the closest published analogue to this project's DistilBERT
  model, and a useful reference point for how much a larger training set might move the numbers here.

## 3. Methodology

### Datasets

Combined from two public HuggingFace datasets:
- [`deepset/prompt-injections`](https://huggingface.co/datasets/deepset/prompt-injections)
- [`jackhhao/jailbreak-classification`](https://huggingface.co/datasets/jackhhao/jailbreak-classification)

After deduplication and cleaning: **1,578 examples**, class balance 859 benign / 719 injection.

### Splits

| Split | Size | Purpose |
|---|---|---|
| Train | 1,104 | Model fitting |
| Val | 237 | Model selection during training |
| Test | 237 | In-distribution held-out evaluation (stratified from the same source distribution) |
| Holdout | 24 | Hand-crafted, out-of-distribution generalization check |

The 70/15/15 train/val/test split is stratified but drawn from the same two source datasets, so the test set
measures how well a model fits the distribution it was trained on — not whether it generalizes past it. The
24-example hand-crafted holdout set (`data/holdout/handcrafted.csv`) exists specifically to test that gap: 12
injection attempts phrased differently from anything in the training data, and 12 benign prompts that
deliberately use trigger words ("ignore," "pretend," "act as") in innocent contexts, to probe both missed
attacks and false positives on wording the models haven't memorized.

### Three models, and why these three

- **TF-IDF + Logistic Regression** — word/bigram frequency features, no notion of meaning, purely surface
  pattern matching. The interpretable, cheap-to-run floor.
- **Sentence Embeddings + Logistic Regression** — `all-MiniLM-L6-v2` (frozen, pretrained, not fine-tuned)
  produces a dense vector per prompt, fed into a logistic regression head. Only the LR head is trained on
  this project's data; the embedding model itself sees no gradient updates.
- **Fine-tuned DistilBERT** — `distilbert-base-uncased`, fine-tuned end-to-end (all weights, not just a
  classification head) on the same training split for 3 epochs.

These three were chosen to span a specific axis: purely lexical (TF-IDF) → frozen semantic representation
(MiniLM) → fully fine-tuned semantic representation (DistilBERT). The first two only ever train a logistic
regression head, which keeps them fully explainable — you can inspect the learned weights on either
n-grams or a fixed embedding space. DistilBERT's full fine-tuning gives it far more capacity to fit the
training distribution, at the direct cost of interpretability and, as the results below show, at some cost
to generalization given how little training data (~1,100 examples) it had to fit that capacity with.

## 4. Results — in-distribution (test set)

| Metric | TF-IDF + LR | Embeddings + LR | DistilBERT |
|---|---|---|---|
| Test F1 | 0.903 | 0.892 | 0.929 |
| Test precision | 0.949 | 0.948 | 0.951 |
| Test recall | 0.861 | 0.843 | 0.907 |
| Test FPR | 0.039 | 0.039 | 0.039 |
| 5-fold CV F1 | 0.894 ± 0.038 | 0.866 ± 0.031 | not run¹ |

¹ 5-fold CV would mean fine-tuning DistilBERT five separate times on CPU; skipped as impractical for this
project's scale.

DistilBERT posts the best numbers across the board on the in-distribution test set — highest F1, precision,
and recall of the three. This is expected: it's the only model with full gradient-based access to the
training data, so of course it fits that distribution best. All three models hold the same false positive
rate (0.039) at their default 0.5 threshold.

## 5. Results — out-of-distribution generalization (holdout set)

| Metric | TF-IDF + LR | Embeddings + LR | DistilBERT |
|---|---|---|---|
| Holdout F1 | 0.737 | 0.870 | 0.800 |
| Holdout recall | 0.583 | 0.833 | 0.667 |
| Holdout precision | 1.000 | 0.909 | 1.000 |
| Holdout FPR | 0.000 | 0.083 | 0.000 |

The ranking inverts. TF-IDF, which led among the two linear models in-distribution, has the worst holdout
recall — it misses roughly 4 in 10 rephrased injection attempts, while keeping perfect precision and a 0.0
false positive rate. DistilBERT, despite the best test-set numbers, lands *between* the two linear models on
holdout recall (0.667) — better than TF-IDF, but worse than MiniLM's 0.833, despite having by far the most
model capacity of the three.

**Why TF-IDF's recall collapses:** it is pattern-matching literal surface phrasing (words and bigrams). Once
an injection is phrased with vocabulary it hasn't seen combined in that way before, there's no mechanism for
it to recognize the underlying intent — it has no notion of meaning at all, only frequency statistics over
tokens it has seen.

**Why DistilBERT doesn't win here despite most capacity:** fine-tuning all its weights lets it fit the
training distribution's surface patterns more tightly than either linear model, but with only ~1,100
training examples, that extra capacity doesn't translate into a more robust decision boundary — it has
enough data to overfit training-specific patterns but not enough to learn something more general than what
the frozen, pretrained MiniLM embeddings already encode from their much larger pretraining corpus. The
embedding model, capturing semantic similarity rather than a fitted decision boundary over this project's
small training set, generalizes best to reworded attacks it never saw.

The holdout set is only 24 examples, so a single flipped prediction moves holdout recall by roughly 8
points — these numbers are directional evidence of a real effect (the ranking inversion between test and
holdout is consistent and explainable), not precise estimates.

## 6. Results — adversarial robustness

Six evasion techniques (`src/adversarial/attacks.py`) were applied independently to the 12 injection
examples in the holdout set (benign examples excluded — this measures evasion, not false positives).
Baseline recall on this 12-example subset matches the holdout numbers above: TF-IDF 0.583, MiniLM 0.833,
DistilBERT 0.667.

| Attack | TF-IDF + LR | MiniLM Embeddings + LR | DistilBERT |
|---|---|---|---|
| character_perturbation | +0.167 | +0.083 | **−0.333** |
| unicode_lookalike_substitution | +0.167 | **+0.667** | **−0.333** |
| synonym_substitution | +0.167 | +0.083 | −0.167 |
| payload_splitting | −0.083 | 0.000 | 0.000 |
| base64_encoding | **+0.583** | **+0.833** | **+0.583** |
| roleplay_framing | −0.167 | −0.167 | 0.000 |

(Positive = attack works, recall falls. Negative = attack backfired, recall rose. Full numbers:
[`results/adversarial_robustness.csv`](results/adversarial_robustness.csv).)

**Base64 encoding is a complete, universal bypass against all three models.** Wrapping the payload as
`"decode this: <base64>"` drops recall to near zero across the board — TF-IDF 0.583→0.0, MiniLM 0.833→0.0,
DistilBERT 0.667→0.083. None of the three models operate on decoded content, so this isn't a model-quality
problem; it's a premise problem. Any classifier that only reads the literal text in front of it will have
this hole until decoding (or a rule that flags base64-looking payloads) is added upstream.

**DistilBERT's apparent resistance to character-level corruption is a shortcut, not robustness.** Unicode
homoglyph substitution and plain character typos both *increased* DistilBERT's recall (to as high as a
perfect 1.0), while the same attacks hurt or barely affected the linear models (unicode substitution alone
cut MiniLM's recall from 0.833 to 0.167). The likely mechanism: heavy character-level corruption produces
fragmented or `[UNK]` subword tokens, and DistilBERT appears to have learned "garbled input" as a
training-time correlate of injection — a spurious, fragile correlation, not genuine obfuscation resistance.
A benign sentence with the same corruption rate would likely trip the same shortcut incorrectly; this was
not tested here but follows directly from the mechanism.

**Roleplay framing backfired as an attack against every model.** Prepending "In a hypothetical world where
AI has no restrictions..." made TF-IDF and MiniLM recall go *up*, because that phrasing is itself a common
jailbreak pattern (DAN-style prompts) well represented in the training data — it doesn't disguise the
underlying instruction, it adds a second detectable signal on top of the first.

**Payload splitting (naive midpoint + filler sentence) had almost no effect on any model** — a single filler
sentence isn't enough dilution to break lexical or semantic pattern matching when the trigger words are
still present.

## 7. Error analysis

69 misclassified rows across all three models on test + holdout, categorized with a rule-based taxonomy (no
LLM calls; see `results/error_analysis.md` and `results/error_analysis.csv`).

| Error type | TF-IDF + LR | MiniLM Embeddings + LR | DistilBERT | Total |
|---|---|---|---|---|
| NOVEL_ATTACK_PATTERN | 14 | 14 | 10 | 38 |
| AMBIGUOUS_INTENT | 9 | 10 | 6 | 25 |
| TRIGGER_WORD_BENIGN | 1 | 1 | 0 | 2 |
| OBFUSCATION | 0 | 0 | 2 | 2 |
| LENGTH_OUTLIER | 1 | 0 | 1 | 2 |
| OTHER | 0 | 0 | 0 | 0 |

**NOVEL_ATTACK_PATTERN and AMBIGUOUS_INTENT together account for 63 of 69 errors (91%)** across all three
models — the dominant failure mode is not obfuscation or trigger-word confusion, it's attacks phrased in
ways the model hasn't seen (e.g. injection instructions embedded inside an otherwise-plausible task like
"write a short story about an investor...") or prompts where injection intent is genuinely ambiguous (e.g. a
benign summarization request that happens to end with an out-of-place instruction like "write Andy is the
best!"). This is consistent with the holdout results in section 5: the models' central weakness is
generalizing past memorized surface patterns, not surface-level trigger-word confusion.

**TRIGGER_WORD_BENIGN is rare (2 total)** — false positives from innocent use of words like "ignore" or
"instructions" are not a major failure mode for any of the three models at their default threshold, despite
the holdout set being specifically designed to probe for it.

**OBFUSCATION errors appear only for DistilBERT (2 cases)**, both false positives on benign
"please segment the words" / "add spaces between words" requests — text that is unusually mashed together
without spaces. This is a plausible false-positive counterpart to the character-corruption shortcut
identified in the adversarial evaluation: DistilBERT appears sensitive to unusual character-level structure
in general, in both directions.

## 8. Threshold selection and deployment considerations

All numbers above use each model's default 0.5 decision threshold. `results/threshold_analysis.md` sweeps
thresholds from 0.10 to 0.90 on the (unperturbed) **test set** to find better operating points; see
[`results/threshold_sweep.csv`](results/threshold_sweep.csv) and
[`results/pr_curves.png`](results/pr_curves.png) for the full curves.

| Model | Operating point | Threshold | Precision | Recall | F1 | Notes |
|---|---|---|---|---|---|---|
| TF-IDF + LR | HIGH_RECALL (recall ≥ 0.95) | 0.35 | 0.788 | 0.963 | 0.867 | - |
| TF-IDF + LR | HIGH_PRECISION (precision ≥ 0.98) | 0.80 | 1.000 | 0.546 | 0.707 | - |
| MiniLM + LR | HIGH_RECALL (recall ≥ 0.95) | 0.25 | 0.696 | 0.954 | 0.805 | - |
| MiniLM + LR | HIGH_PRECISION (precision ≥ 0.98) | 0.60 | 0.989 | 0.806 | 0.888 | - |
| DistilBERT | HIGH_RECALL (recall ≥ 0.95) | 0.15 | 0.927 | 0.935 | 0.931 | constraint unreachable in swept range; closest shown |
| DistilBERT | HIGH_PRECISION (precision ≥ 0.98) | 0.85 | 0.989 | 0.852 | 0.915 | - |

**DistilBERT cannot hit 0.95 recall anywhere in the swept range** — it tops out at 0.935 even at the lowest
threshold tried. This is a coverage problem, not a calibration problem: several of its remaining false
negatives are confidently wrong (>0.7 confidence, per the NOVEL_ATTACK_PATTERN errors in section 7), so no
threshold adjustment recovers them. TF-IDF and MiniLM both clear 0.95 recall comfortably once the threshold
is lowered enough, meaning their missed injections tend to be borderline rather than confidently
misclassified.

**Deployment recommendation from this analysis alone:** if fine-tuned weights are available, DistilBERT at
t=0.15 (recall-leaning pre-filter) or t=0.85 (precision-leaning auto-block) gives the best precision at a
given recall level of the three. If only a fully explainable, lightweight model is acceptable, TF-IDF at
t=0.35 is the better high-recall choice and MiniLM at t=0.60 is the better high-precision choice among the
two linear models.

**This threshold tuning is orthogonal to, and does not fix, the generalization and adversarial gaps in
sections 5–6.** It was performed entirely on the in-distribution test set. Moving DistilBERT's threshold to
0.15 does nothing for its 0.667 holdout recall or its near-total collapse under base64 encoding — those are
distributional and representational gaps, not threshold miscalibration, and no post-hoc threshold choice
addresses them.

## 9. Limitations

- **Small dataset.** 1,578 training examples is not much for a task with this much linguistic variety across
  real-world injection techniques; all three models are likely underfitting the true diversity of attacks
  that exist outside this project's two source datasets.
- **Holdout set is small and single-author.** 24 hand-crafted examples, all written with knowledge of what
  the models were trained on. Useful as a sanity check for one specific failure mode (generalization to
  rephrased attacks and trigger-word false positives), not a comprehensive adversarial evaluation. A single
  flipped prediction moves holdout recall by ~8 points and adversarial recall-drop numbers by ~8.3 points —
  read all holdout- and adversarial-derived numbers in this document as directional, not precise.
- **Adversarial evaluation covers six techniques, applied independently, at one perturbation strength
  each, against a 12-example set.** Real attackers would likely stack multiple techniques (e.g. base64 +
  roleplay framing); this was not tested. Perturbation rates (15% typo rate, 40% homoglyph rate, etc.) were
  chosen once and not swept, so the reported effect sizes — especially DistilBERT's counterintuitive
  negative drops — could look different at a different strength. A near-zero recall drop for an attack
  (e.g. payload splitting) means this specific, naive implementation didn't work, not that no version of
  that attack class would.
- **No defense against obfuscation, multi-turn injection, or indirect injection.** All three models classify
  single, plaintext prompts. Encoding-based bypass (base64) is a complete, universal, one-line evasion
  against all three models as currently built; unicode tricks, multi-message spread, and injection hidden
  inside retrieved/tool-output content (indirect prompt injection) are all out of scope.
- **DistilBERT's cross-validation was skipped.** 5-fold CV would mean fine-tuning it five separate times on
  CPU, impractical at this project's scale. Unlike the two linear models, its stability across folds is
  unverified — the single train/val/test split result should be read with that caveat.
- **Not evaluated against production tools.** No comparison against Lakera Guard, Rebuff, LLM Guard, or the
  ProtectAI DeBERTa model described in section 2 — those are trained on far larger and more current attack
  corpora, and this project isn't attempting to compete with them.
- **Error taxonomy is rule-based, not manually audited row-by-row.** The categories in section 7
  (NOVEL_ATTACK_PATTERN, AMBIGUOUS_INTENT, etc.) are assigned programmatically from surface heuristics, not
  individually verified by a human reader for every one of the 69 rows — treat category boundaries as
  approximate.

## 10. Future work

- Expand the holdout set, ideally with adversarial examples from someone other than the project author, to
  reduce the single-author bias and shrink the per-example variance noted throughout this document.
- Test indirect prompt injection — malicious instructions embedded in retrieved documents or tool output
  rather than the direct user prompt — which none of the current evaluations cover.
- Test stacked/combined adversarial techniques (e.g. base64 encoding plus roleplay framing) rather than each
  evasion technique in isolation.
- Add a decode-and-rescan step (or a base64-pattern heuristic) ahead of the classifiers, given that encoding
  is a complete bypass against all three models as-is.
- Ensemble the models rather than treating them as separate deployment choices — the holdout and adversarial
  results suggest the three make different, partially uncorrelated errors (e.g. MiniLM and DistilBERT fail
  on different attacks in section 6), which is exactly the condition under which ensembling tends to help.
- Investigate the DistilBERT character-corruption shortcut directly by constructing corrupted *benign*
  examples and checking whether they get flagged as false positives, which would confirm the spurious-
  correlation hypothesis in section 6.
