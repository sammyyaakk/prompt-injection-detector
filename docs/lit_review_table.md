# Related Work: Detecting Direct and Indirect Prompt Injection

Literature table for the project "Explainable, Adversarial Robust Detection of Direct and Indirect Prompt
Injection Attacks in Large Language Model Systems Using Lightweight Classifiers". Ten peer-reviewed papers
from 2023–2026, chosen for the related-work section.

First compiled 27 October 2026 from abstracts and bibliographic records; revised 22 November 2026 after
reading the full texts (see [Scope](#scope-and-limitations-of-this-review)). Every citation was checked against
a primary record (DOI metadata, proceedings index or conference programme); see
[Verification log](#verification-log). Numbers about our own models come from `results/indirect/` in this
repository.

## Table

| # | Paper | Venue / year | Method | Data it tests on | Text-only detectors evaluated | Relation to our project |
|---|---|---|---|---|---|---|
| 1 | Greshake et al., *Not What You've Signed Up For* [1] | ACM AISec, 2023 | Attack study and taxonomy of indirect prompt injection; no detector proposed or tested | Real and synthetic LLM-integrated apps (e.g. Bing Chat) with planted injections | None. Calls current input filtering "likely brittle" and shows Base64-encoded injections (§4.3.2) | Source of our threat model. Raises encoding as a filter-evasion tactic without testing any filter; we test it against detectors |
| 2 | Liu et al., *Formalizing and Benchmarking Prompt Injection Attacks and Defenses* [2] | USENIX Security, 2024 | Formal framework; 5 attacks × 10 defenses, 10 LLMs, 7 tasks | Task data (e.g. MRPC, SST2, Gigaword) with an injected instruction | None. All detection defenses (perplexity, LLM-based, response-based, known-answer) need an LLM | Their "data + injected instruction" setup matches our documents; their detectors are far heavier than ours |
| 3 | Ayub and Majumdar, *Embedding-based classifiers can detect prompt injection attacks* [3] | CAMLIS, 2024 | Three embedding models (incl. all-MiniLM-L6-v2) + LR, XGBoost, Random Forest | 467,057 single prompts from six public datasets | ProtectAI v1/v2, deepset and a SetFit model as baselines (AUC 0.50–0.59 vs 0.76 for theirs) | Closest to our MiniLM + LR model, on prompts only; they name indirect injection as future work. No evasion tests, no FPR |
| 4 | Yi et al., *Benchmarking and Defending against Indirect Prompt Injection Attacks* (BIPIA) [4] | ACM KDD, 2025 | Benchmark of indirect injection against LLMs; prompting and fine-tuning defenses on the LLM | Emails, web pages, tables, code and summaries with attacks at start, middle or end; attack types include Base Encoding, ciphers and reversed text | None; the metric is LLM attack success rate | Our benchmark's carriers come from BIPIA. Its encoding attacks target LLMs; we test them against detectors |
| 5 | Hung et al., *Attention Tracker* [5] | Findings of NAACL, 2025 | Training-free detection from the protected LLM's attention heads | Open-Prompt-Injection and deepset: task data with injected instructions | ProtectAI (AUROC 0.69 / 0.90) and PromptGuard (0.97 / 0.75) as baselines | Needs white-box LLM access; ours is text-only. Finds data length has little effect on its own score |
| 6 | Jacob et al., *PromptShield* [6] | ACM CODASPY, 2025 | Fine-tuned detectors from DeBERTa-v3-base up to Llama-3.1-8B (LoRA) | Conversational prompts and "application-structured" prompt + data, with injections appended to the data | PromptGuard, ProtectAI v1/v2, InjecGuard, a DistilBERT detector; at 1% FPR their best model reaches 94.8% TPR vs 1.97% for ProtectAI v2 | Text-only and aimed at indirect injection, but no documents with injections hidden inside, no evasion tests, no explanations |
| 7 | Liu et al., *DataSentinel* [7] | IEEE S&P, 2025 | Mistral-7B detector fine-tuned against adaptive attacks (minimax) | Task data contaminated with injected instructions (7 tasks) | PromptGuard (FPR up to 1.00) and a CLIP-embedding classifier (FPR up to 0.73) as baselines | Runs a 7B model per check. Its adaptive attacks are optimisation-based (GCG); no encoding or character evasions |
| 8 | Abdelnabi et al., *Get my drift?* (TaskTracker) [8] | IEEE SaTML, 2025 | Linear and metric-learning probes on the LLM's activation deltas | SQuAD and HotPotQA paragraphs with instructions at start, middle or end | Prompt Guard ROC AUC 0.974 and Prompt Shields 0.988 vs 0.996 for theirs; Prompt Guard 0.728 on a harder private set | Closest benchmark design to ours, but needs the LLM's activations. Suggests longer injections may dilute their activation signal |
| 9 | Li et al., *PIGuard* (arXiv: InjecGuard) [9] | ACL, 2025 | DeBERTa-v3-base guard trained to reduce trigger-word over-defense; NotInject benign set (339) | NotInject, WildGuard, PINT and BIPIA injections | Deepset, Fmops, PromptGuard, ProtectAI v2, LakeraGuard; on BIPIA injections ProtectAI v2 catches 8.7% and LakeraGuard 12.0% | Over-defense on single prompts; we measure it on documents with benign imperatives. Base64 appears only as a training augmentation |
| 10 | Zhong et al., *Attention is All You Need to Defend Against IPI* (Rennervate) [10] | NDSS, 2026 | Token-level detection from the LLM's attention, then sanitisation | FIPI: 100k documents with injections at random positions | PromptGuard (acc 64.4%, FPR 69.9%) and ProtectAI v2 (acc 75.5%, FNR 46.5%) as baselines | Shows prompt-trained detectors fail on documents in opposite ways, as our prompt-trained models do. Localises injected tokens, but needs the LLM's internals |

## How the papers group

**Threat model and LLM-side benchmarks [1], [2], [4].** These define indirect injection and measure how
vulnerable LLMs are. None evaluates a standalone text classifier.

**Detectors that use an LLM [2], [5], [7], [8], [10].** They either run an LLM to judge the input or read the
protected model's attention or activations. They reach high accuracy, but need white-box access or an extra
LLM call.

**Standalone text classifiers [3], [6], [9].** Ayub and Majumdar test single prompts; PromptShield and PIGuard
also cover data-carrying inputs (application-structured inputs, BIPIA injections).

**Text-only detectors as baselines on injected data [5]–[10].** ProtectAI and PromptGuard appear in almost every
one of these papers, and their results on injected data vary widely: from ROC AUC 0.974 (Prompt Guard,
TaskTracker's test set) down to 8.7% of BIPIA injections caught (ProtectAI v2, PIGuard) and a 69.9% false
positive rate (PromptGuard, Rennervate).

## Where our project fits

Text-only detectors *have* been run on injected documents, mostly as baselines reported with one aggregate
number. What none of the ten papers does:

1. **Explain why they fail on documents.** We measure recall against the share of the document the injection
   fills (dilution) and test window scoring as a fix (`results/indirect/fig_dilution.png`).
2. **Compare prompt-trained and document-trained versions of the same lightweight models** on a test split
   built for generalisation. Indirect-trained DistilBERT reaches F1 0.863 [0.837, 0.889] with FPR 0.055, against
   0.684 and FPR 0.850 for the prompt-trained one.
3. **Test encoding and character evasions against detectors in the document setting.** Base64 on the injected
   span drops indirect-trained DistilBERT recall from 0.800 to 0.055; a corrupted-benign control shows its
   gains under homoglyphs are a shortcut (FPR 0.135 → 0.370).
4. **Measure whether a text-only detector locates the injection.** The top-scoring window is the payload for
   95.1% of injected test documents (random window: 32.8%).
5. **Measure false positives on document-level hard negatives**, such as "Please ignore my previous email".
   Indirect-trained DistilBERT: FPR 0.116 on benign imperatives.

The observation that prompt-trained detectors fail on documents in opposite directions (missing attacks, or
flagging harmless text) is not new; Rennervate reports it for ProtectAI v2 and PromptGuard. Because every paper
that evaluates text-only detectors uses ProtectAI and/or PromptGuard, they are the expected baselines for our
benchmark.

## Scope and limitations of this review

- Narrative review of ten papers, chosen for relevance to classifier-based and lightweight detection; not a
  systematic search.
- Inclusion: peer-reviewed venue, published 2023–2026, detects or benchmarks direct or indirect prompt
  injection. Three papers were required by the project brief: [1], [4], [10].
- Read scope (22 November 2026): the full text of every paper was obtained (arXiv HTML or PDF, or the ACL
  Anthology PDF for [5] and [9]) and read for its method, evaluation data, baselines, evasion tests,
  localisation, false positives and limitations. Sections outside those topics were searched by keyword
  rather than read line by line, so "not found" means not found in those sections and searches. For [9], both
  the ACL version and the earlier arXiv InjecGuard version were used.

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
