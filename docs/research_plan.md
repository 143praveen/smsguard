# Research plan: SMSGuard

**Working title:** *SMSGuard: Normalisation-First, Evasion-Robust SMS Spam Detection with Adversarial Training and an Explainable Web Deployment*

**Base paper:** M. Salman, M. Ikram, M. A. Kaafar, "Investigating Evasive Techniques in SMS Spam Filtering: A Comparative Analysis of Machine Learning Models," *IEEE Access*, vol. 12, pp. 24306–24324, 2024. doi:10.1109/ACCESS.2024.3364671

## 1. Gap this paper fills
The base paper shows that every model and every commercial filter it tested can be evaded by spacing, homoglyph (punycode), charswap and hybrid attacks. It recommends a "smart NLP pipeline" plus adversarial training, but **it never builds or tests one**. Its released code also has evaluation flaws (see §5), so its robustness numbers can't be taken at face value.

**Our contribution:** design, build and rigorously evaluate that missing defence, then ship it as a usable, explainable tool.

## 2. Research questions
- **RQ1:** Under a leakage-free, reproducible protocol, how do the base paper's models actually perform on the Super SMS Dataset?
- **RQ2:** How much evasion robustness comes from input normalisation alone (zero-width removal, homoglyph folding, de-spacing, de-leeting, entity tokens), compared with model choice?
- **RQ3:** Does treating obfuscation as a *feature* (evasion flags) plus adversarial training close the remaining gap, and at what cost in false positives on legitimate messages?
- **RQ4:** Does the defence transfer to *unseen* attacks (train on some attack types, test on held-out ones) and to transformer models?
- **RQ5:** Can the detector run as a lightweight, explainable service suitable for deployment on a phone or gateway?

## 3. Contributions (for the abstract)
1. A reproducible re-evaluation of the base paper, with corrected splits, documented bugs and fixes.
2. A normalisation-first defence pipeline that reverses six evasion families, two of them not covered by the base paper (zero-width injection, leetspeak).
3. A hybrid detector (word + character n-grams + evasion/URL meta-features) trained adversarially, with an ablation of each component.
4. A leave-one-attack-out generalisation study, plus a comparison against fine-tuned DistilBERT/RoBERTa with and without normalisation.
5. An open-source, explainable web application (FastAPI) showing the verdict, detected tricks and the features behind each decision.

## 4. Experiments
| # | Experiment | Status |
|---|---|---|
| E1 | Clean performance: baseline (B0) vs ablations (A1–A4) vs full SMSGuard, with McNemar tests | done (`scripts/train.py`) |
| E2 | Robustness to eight attacks (three from the base paper, five new) at a 30% perturbation rate | done |
| E3 | Perturbation-rate sweep (10–100%), plotted as robustness curves | done (`experiments.py sweep`) |
| E4 | Leave-one-attack-out adversarial training, and training on the paper's attacks only | done (`experiments.py unseen`) |
| E5 | 5-fold stratified CV, mean ± std | done (`experiments.py cv`) |
| E5b | Adaptive black-box attacker (greedy, queries the model) | done (`experiments.py adaptive`) |
| E5c | False-positive stress test: curated tricky ham, ham + emoji, ham + non-Latin names | done (`experiments.py hardham`) |
| E6 | Transformers on Colab: DistilBERT and RoBERTa, raw vs normalised input, with and without adversarial training | TODO (`notebooks/`) |
| E7 | Paraphrase/EDA attacks (TextAttack on Colab) and LLM-generated paraphrases | TODO |
| E8 | Temporal drift: train on older sources, test on newer ones (rebuild the era split from source metadata) | TODO, needs source/date columns |
| E9 | Latency and model size: CPU inference time per message for each model | done (`experiments.py timing`) |
| E10 | Error analysis: inspect false positives and false negatives by category | partly (hard-ham list); TODO for false negatives |
| E11 | URL reputation (trusted/institutional domains, risky TLDs, brand impersonation) | done (`smsguard/urls.py`) |
| E12 | Spelling restoration against charswap/typo attacks, with ablation and adaptive attacker | done (`SpellFixer` in `normalize.py`) |
| E13 | Live URL signals: domain age, PhishTank/OpenPhish feeds | TODO |

## 5. Evaluation protocol (fixes for the base paper's flaws)
- One deduplicated, stratified, seeded 80/20 split shared by every model. A validation split is carved from training data only.
- The test set is never used for early stopping or hyperparameter tuning (the base paper's LSTM/CNN did this).
- A message that becomes empty after preprocessing counts as a miss (the base paper dropped these with `dropna`).
- Report precision, recall, F1, false-positive rate on ham, ROC-AUC, and false-positive rate on ham containing natural typos.
- Base-paper bugs to cite: the one-class SVM's label mismatch (0/1 vs −1/+1), its training on the test set, a "PU" setup that is effectively supervised, and an 85%-spam test slice used for fastText.
- Dataset note: the released CSV is Windows-1252 (single-byte), so it can't store homoglyph examples. Attacked text is generated in memory as Unicode.

## 6. Paper outline (Scopus journal, ~10–14 pages)
1. Introduction: smishing losses, the evasion problem, contributions
2. Related work: SMS datasets, ML/DL detection, adversarial NLP, defences (normalisation, Unicode TS #39 confusables, adversarial training)
3. Re-examining the base paper: reproducibility audit
4. SMSGuard design: normalisation, features, adversarial training, explainability
5. Experimental setup
6. Results: RQ1–RQ5, ablation tables, robustness curves
7. Deployment: web app architecture and latency
8. Discussion and limitations: adaptive attackers, multilingual SMS, false positives
9. Conclusion and future work

## 7. Target venues (all Scopus-indexed; check the current indexing status and APCs before submitting)
- **Journals:** *IEEE Access*, *PeerJ Computer Science*, *Computers & Security* (Elsevier, harder), *Journal of Information Security and Applications* (Elsevier), *Applied Sciences* / *Future Internet* / *Electronics* (MDPI, fast turnaround), *Security and Communication Networks*, *International Journal of Information Security* (Springer).
- **Conferences (proceedings in Scopus):** IEEE TrustCom, ACM ASIA CCS workshops, Springer LNCS security conferences (e.g. ICICS, ISPEC, ACNS workshops), IEEE ICCCNT/ICACCS-tier conferences (easier).
- Avoid predatory "Scopus-guaranteed" journals. Verify each venue on Scopus Sources (scopus.com/sources) before submitting.

## 8. Ethics and licensing
- Cite the base paper and dataset as the authors request in their README.
- Don't redistribute their CSV without checking its licence; point readers to the original repository.
- Attack code serves defensive evaluation only. Follow responsible disclosure if you test real apps or APIs.
