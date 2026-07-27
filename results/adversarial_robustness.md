# Adversarial Robustness — Recall Drop per Attack Class

Recall drop = baseline injection recall on the holdout set minus recall on the perturbed version. Higher is worse (more evasion).

| Attack | TF-IDF + LR | MiniLM Embeddings + LR | DistilBERT |
|---|---|---|---|
| baseline (unperturbed) | 0.000 | 0.000 | 0.000 |
| character_perturbation | 0.167 | 0.083 | -0.333 |
| unicode_lookalike_substitution | 0.167 | 0.667 | -0.333 |
| synonym_substitution | 0.167 | 0.083 | -0.167 |
| payload_splitting | -0.083 | 0.000 | 0.000 |
| base64_encoding | 0.583 | 0.833 | 0.583 |
| roleplay_framing | -0.167 | -0.167 | 0.000 |
