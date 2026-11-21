# Indirect injection benchmark: data licence

`train.csv` and `test.csv` are built by `scripts/build_indirect_benchmark.py` from carrier documents in BIPIA
(Yi et al., "Benchmarking and Defending against Indirect Prompt Injection Attacks on Large Language Models",
KDD 2025), vendored in [`../bipia/`](../bipia) together with BIPIA's [licence file](../bipia/LICENSE).

Licences, as stated in BIPIA's licence file:

| Rows (`carrier` column) | Carrier source | Licence |
|---|---|---|
| `table` | WikiTableQuestions (https://github.com/ppasupat/WikiTableQuestions) | CC BY-SA 4.0 |
| `code` | Stack Exchange answers (https://archive.org/details/stackexchange) | CC BY-SA 4.0 |
| `email` | BIPIA benchmark data | MIT |

Rows built on `table` or `code` carriers are adaptations of CC BY-SA 4.0 material and are shared under
CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/), with attribution to the sources above. The
repository's MIT licence covers the code, not these rows. The injected and benign insert texts come from the
direct-injection datasets in `data/processed/`, the hand-crafted holdout and BIPIA's attack instructions.
