# Follow-up analyses: word-level localisation and corrupted-benign control

## Top-5 important words inside the payload (top window of each injected test document)

| model | method | docs | top-5 words in payload | random baseline |
|---|---|---|---|---|
| TF-IDF + LR | linear weights | 400 | 0.605 | 0.436 |
| MiniLM + LR | word removal | 400 | 0.594 | 0.425 |
| TF-IDF + LR (indirect-trained) | linear weights | 400 | 0.708 | 0.495 |
| MiniLM + LR (indirect-trained) | word removal | 400 | 0.645 | 0.452 |

## Corrupted-benign control (chunk-level, benign documents with a harmless insert)

| attack | model | FPR on benign | recall on injected |
|---|---|---|---|
| none | TF-IDF + LR | 0.265 | 0.482 |
| none | MiniLM + LR | 0.215 | 0.472 |
| none | DistilBERT | 0.810 | 0.963 |
| none | TF-IDF + LR (indirect-trained) | 0.205 | 0.757 |
| none | MiniLM + LR (indirect-trained) | 0.305 | 0.672 |
| none | DistilBERT (indirect-trained) | 0.135 | 0.800 |
| character_typos | TF-IDF + LR | 0.250 | 0.378 |
| character_typos | MiniLM + LR | 0.255 | 0.505 |
| character_typos | DistilBERT | 0.945 | 0.995 |
| character_typos | TF-IDF + LR (indirect-trained) | 0.185 | 0.550 |
| character_typos | MiniLM + LR (indirect-trained) | 0.360 | 0.665 |
| character_typos | DistilBERT (indirect-trained) | 0.230 | 0.812 |
| unicode_homoglyphs | TF-IDF + LR | 0.260 | 0.305 |
| unicode_homoglyphs | MiniLM + LR | 0.210 | 0.425 |
| unicode_homoglyphs | DistilBERT | 0.955 | 0.998 |
| unicode_homoglyphs | TF-IDF + LR (indirect-trained) | 0.185 | 0.398 |
| unicode_homoglyphs | MiniLM + LR (indirect-trained) | 0.340 | 0.595 |
| unicode_homoglyphs | DistilBERT (indirect-trained) | 0.370 | 0.922 |
