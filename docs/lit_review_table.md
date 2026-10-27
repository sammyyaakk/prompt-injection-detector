# Related Work: Detecting Direct and Indirect Prompt Injection

Literature table for the project "Explainable, Adversarial Robust Detection of Direct and Indirect Prompt
Injection Attacks in Large Language Model Systems Using Lightweight Classifiers". Ten peer-reviewed papers
from 2023–2026, chosen for the related-work section and the mid-semester progress review.

Compiled 27 October 2026. Every citation below was checked against a primary record (DOI metadata,
proceedings index or conference programme); see [Verification log](#verification-log). Numbers about our
own models come from `results/` in this repository.

## Table

| # | Paper | Venue / year | Method | Setting | Relation to our project |
|---|---|---|---|---|---|
| 1 | Greshake et al., *Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection* [1] | ACM AISec (CCS workshop), 2023 | Attack study: introduces indirect prompt injection, a taxonomy of its impacts, and demonstrations on real LLM-integrated applications. No detector. | Indirect | Source of our threat model: the attacker controls retrieved content, not the user's query or the model. Our indirect pilot operationalises this by embedding payloads in Wikipedia paragraphs. |
| 2 | Liu et al., *Formalizing and Benchmarking Prompt Injection Attacks and Defenses* [2] | USENIX Security, 2024 | Formal framework treating existing attacks as special cases; benchmarks 5 attacks and 10 defenses (including detection defenses) across 10 LLMs and 7 tasks; Open-Prompt-Injection toolkit. | Direct and indirect (injected instruction inside task data) | Their "target data + injected instruction" formulation matches how our pilot documents are built (host paragraph + payload). Their detection baselines are LLM-based and heavier than ours; useful as context, not as baselines this semester. |
| 3 | Ayub and Majumdar, *Embedding-based classifiers can detect prompt injection attacks* [3] | CAMLIS, 2024 (CEUR-WS Vol-3920) | Three embedding models plus traditional ML classifiers; Random Forest and XGBoost perform best and outperform open-source encoder-based detectors. | Direct (single prompts) | Closest prior work to our MiniLM embeddings + LR model. They evaluate prompts only; our pilot tests the same idea on documents, where MiniLM + LR recall falls from 0.784 on the payloads alone to 0.477 once embedded (n = 88). |
| 4 | Yi et al., *Benchmarking and Defending against Indirect Prompt Injection Attacks on Large Language Models* (BIPIA) [4] | ACM KDD, 2025 | First indirect prompt injection benchmark (malicious instructions in external content); finds LLMs universally vulnerable; proposes black-box prompting defences and a white-box fine-tuning defence. | Indirect | Closest benchmark design to our pilot: instructions hidden in external content. BIPIA measures LLM vulnerability and LLM-side defences rather than standalone lightweight detectors; a candidate source of realistic host documents when we scale past the 276-document pilot. |
| 5 | Hung et al., *Attention Tracker: Detecting Prompt Injection Attacks in LLMs* [5] | Findings of NAACL, 2025 | Training-free detection: tracks "important heads" whose attention shifts from the original instruction to the injected one (the distraction effect); no extra LLM inference. | Direct | Attention-based detection, relevant to our goal of inspecting DistilBERT attention. The difference: they read the protected LLM's attention, while we explain a separate lightweight detector that needs no access to the LLM. |
| 6 | Jacob et al., *PromptShield: Deployable Detection for Prompt Injection Attacks* [6] | ACM CODASPY, 2025 | Benchmark of conversational and application-structured data for training and evaluating detectors; a fine-tuned detector that is strongest in the low false-positive-rate regime. | Direct and application-structured | Supports reporting detectors at low FPR, as our threshold sweep and pilot do (pilot FPR 0.016–0.021). Their application-structured data is a step toward the indirect setting we target. |
| 7 | Liu et al., *DataSentinel: A Game-Theoretic Detection of Prompt Injection Attacks* [7] | IEEE S&P, 2025 | Fine-tunes an LLM detector, building on known-answer detection, with a minimax objective in which adaptive injections are generated during training. | Injected data (indirect-style) | Shows detectors should be tested against adaptive attackers. Our six-attack suite is fixed and non-adaptive; this is a limitation to state. Their detector is an LLM, far heavier than our classifiers. |
| 8 | Abdelnabi et al., *Get my drift? Catching LLM Task Drift with Activation Deltas* (TaskTracker) [8] | IEEE SaTML, 2025 | Simple linear classifiers on activation deltas (LLM activations before vs. after reading external data) detect task drift without fine-tuning the LLM. | Indirect | A lightweight classifier for indirect injection, but one that needs white-box access to the LLM's activations. Our detectors read only the text, so they are model-agnostic; a useful contrast for the paper. |
| 9 | Li et al., *PIGuard: Prompt Injection Guardrail via Mitigating Overdefense for Free* (arXiv: InjecGuard) [9] | ACL, 2025 | Guard model trained with MOF to reduce trigger-word bias; NotInject benchmark of 339 benign prompts rich in trigger words, on which existing guards fall to about 60% accuracy. | Direct | Same failure mode our holdout probes with 12 benign trigger-word prompts. Our TF-IDF top features (`results/tfidf_top_features.png`) show reliance on function words and German words; NotInject is a larger set we could evaluate on. |
| 10 | Zhong et al., *Attention is All You Need to Defend Against Indirect Prompt Injection Attacks in LLMs* (Rennervate) [10] | NDSS, 2026 | Token-level detection of injected instructions from LLM attention features with two-step attentive pooling, followed by sanitisation; introduces the FIPI dataset. | Indirect | Locates the injected span inside a document, as our word-level TF-IDF explanations do (`results/explain_examples.html`). Theirs needs the LLM's attention; ours explains a text-only classifier. |

## How the papers group

**Threat model and benchmarks [1], [2], [4].** Greshake et al. defined indirect injection. Liu et al. and BIPIA
turned it into benchmarks, but both measure how vulnerable LLMs are and test LLM-side defences, not standalone
detectors.

**Lightweight text classifiers [3], [6], [9].** These are the closest to our models: embeddings plus classical
ML [3], fine-tuned detectors at low FPR [6], and guard models that over-flag trigger words [9]. All three are
evaluated on prompts or application inputs, not on long documents carrying an embedded payload.

**Detectors that use the LLM's internals [5], [8], [10].** Attention patterns [5], [10] and activation deltas [8]
work well in their papers, and [8] and [10] target the indirect setting directly. They need white-box access to the
protected model, which a text-only classifier does not.

**Adaptive attackers [7].** DataSentinel trains against adaptive injections, a stronger threat model than our
fixed attack suite.

## Where our project fits

- **Gap (a), document-level evaluation of lightweight classifiers.** In this set, the text-only lightweight
  detectors [3], [6], [9] are evaluated on prompts, while the indirect-setting detectors [8], [10] use LLM
  internals. Our pilot evaluates text-only classifiers on documents and finds that embedding a payload in a
  paragraph cuts recall from 0.773–0.852 to 0.477–0.511 (n = 88 injected documents, `results/indirect_pilot.md`).
- **Gap (b), explainability.** The attention-based papers [5], [10] explain the protected LLM; we explain the
  detector itself.
- **Gap (c), evasion transfer.** Our pilot finds that base64-encoding only the injected span drops recall to
  0.000–0.045 for all three models. This review read abstracts and bibliographic records only, so it cannot
  say whether these ten papers test encoding-based evasion; check the full texts before claiming this as a gap
  in the paper.

## Scope and limitations of this review

- Narrative review of ten papers, chosen for relevance to classifier-based and lightweight detection; it is
  not a systematic search and does not claim to cover the field.
- Inclusion: peer-reviewed venue, published 2023–2026, detects or benchmarks direct or indirect prompt
  injection. Three papers were required by the project brief: [1], [4], [10].
- Read scope: abstracts, proceedings records and conference programmes. Method descriptions above restate
  what those sources say; no paper's experiments were re-checked, and no method weaknesses are claimed.

## Verification log

Checked 27 October 2026.

| # | Primary record used | Notes |
|---|---|---|
| 1 | Crossref metadata for DOI 10.1145/3605764.3623985 | AISec '23 proceedings, pp. 79–90. Preprint arXiv:2302.12173. |
| 2 | USENIX Security '24 presentation page | pp. 1831–1847. Preprint arXiv:2310.12815. |
| 3 | CEUR-WS Vol-3920 index (CAMLIS 2024, 24–25 Oct 2024, Arlington, VA) | Paper 15, pp. 257–268. The arXiv record (2410.22284) carries no venue field. |
| 4 | Crossref metadata for DOI 10.1145/3690624.3709179 | KDD '25 V.1, pp. 1809–1820. Preprint arXiv:2312.14197. |
| 5 | ACL Anthology 2025.findings-naacl.123 | pp. 2309–2322. A third-party aggregator lists it as NAACL 2024; the Anthology record is authoritative. |
| 6 | Crossref metadata for DOI 10.1145/3714393.3726501 | Proceedings of the 15th ACM CODASPY, pp. 341–352. Crossref's print-date field reads 2024, which conflicts with the 2025 conference; cited as CODASPY 2025. |
| 7 | Penn State and NSF PAR publication records | IEEE S&P 2025, pp. 2190–2208. Preprint arXiv:2504.11358. |
| 8 | arXiv:2406.00799 (comment field "SaTML 2025") | First posted as "Are you still on track!? Catching LLM Task Drift with Activations". |
| 9 | ACL Anthology 2025.acl-long.1468 | pp. 30420–30437. The arXiv version (2410.22770) calls the model InjecGuard. |
| 10 | NDSS 2026 programme (Session 3B, AI Security) and arXiv:2512.08417 | Comment field "Accepted by NDSS Symposium 2026". |

## References

[1] K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, and M. Fritz, "Not what you've signed up for:
Compromising real-world LLM-integrated applications with indirect prompt injection," in *Proc. 16th ACM
Workshop on Artificial Intelligence and Security (AISec)*, 2023, pp. 79–90, doi: 10.1145/3605764.3623985.

[2] Y. Liu, Y. Jia, R. Geng, J. Jia, and N. Z. Gong, "Formalizing and benchmarking prompt injection attacks
and defenses," in *Proc. 33rd USENIX Security Symposium*, 2024, pp. 1831–1847.

[3] M. A. Ayub and S. Majumdar, "Embedding-based classifiers can detect prompt injection attacks," in
*Proc. Conf. Applied Machine Learning in Information Security (CAMLIS)*, CEUR Workshop Proceedings,
vol. 3920, 2024, pp. 257–268.

[4] J. Yi, Y. Xie, B. Zhu, E. Kiciman, G. Sun, X. Xie, and F. Wu, "Benchmarking and defending against
indirect prompt injection attacks on large language models," in *Proc. 31st ACM SIGKDD Conf. Knowledge
Discovery and Data Mining (KDD)*, 2025, pp. 1809–1820, doi: 10.1145/3690624.3709179.

[5] K.-H. Hung, C.-Y. Ko, A. Rawat, I.-H. Chung, W. H. Hsu, and P.-Y. Chen, "Attention Tracker: Detecting
prompt injection attacks in LLMs," in *Findings of the Association for Computational Linguistics: NAACL
2025*, 2025, pp. 2309–2322, doi: 10.18653/v1/2025.findings-naacl.123.

[6] D. Jacob, H. Alzahrani, Z. Hu, B. Alomair, and D. Wagner, "PromptShield: Deployable detection for
prompt injection attacks," in *Proc. 15th ACM Conf. Data and Application Security and Privacy (CODASPY)*,
2025, pp. 341–352, doi: 10.1145/3714393.3726501.

[7] Y. Liu, Y. Jia, J. Jia, D. Song, and N. Z. Gong, "DataSentinel: A game-theoretic detection of prompt
injection attacks," in *Proc. IEEE Symposium on Security and Privacy (S&P)*, 2025, pp. 2190–2208.

[8] S. Abdelnabi, A. Fay, G. Cherubin, A. Salem, M. Fritz, and A. Paverd, "Get my drift? Catching LLM task
drift with activation deltas," in *Proc. IEEE Conf. Secure and Trustworthy Machine Learning (SaTML)*, 2025.

[9] H. Li, X. Liu, N. Zhang, and C. Xiao, "PIGuard: Prompt injection guardrail via mitigating overdefense
for free," in *Proc. 63rd Annual Meeting of the Association for Computational Linguistics (ACL)*, 2025,
pp. 30420–30437, doi: 10.18653/v1/2025.acl-long.1468.

[10] Y. Zhong, Q. Miao, Y. Chen, J. Deng, Y. Cheng, and W. Xu, "Attention is all you need to defend against
indirect prompt injection attacks in LLMs," in *Proc. Network and Distributed System Security Symposium
(NDSS)*, 2026.
