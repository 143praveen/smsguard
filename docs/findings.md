# Key findings so far (for the paper)

All numbers come from `results/` (seed 42; 66,913 unique messages; stratified 80/20 split unless stated). Regenerate with `scripts/train.py` and `scripts/experiments.py all`.

## The SMSGuard pipeline
1. **Normalisation** reverses invisible characters, full-width letters, homoglyphs, emoji/bullet wedges, spacing (with vocabulary re-segmentation) and leetspeak.
2. **Spelling restoration** reverses one-character edits (charswap, insert, delete, substitute) toward much more frequent training words. It never touches words that appear in legitimate training messages.
3. **URL reputation** (offline):
   - trusted and institutional domains are neutralised
   - risky TLDs, IP hosts, shorteners, brand impersonation and suspicious path words become features
   - the domain name is kept as a token
4. **Hybrid features** (word 1–2-grams, character 2–5-grams, evasion and URL flags) feed a logistic regression that is **adversarially trained** on attacked spam.

## Headline numbers
| | Paper-style SVM (B0) | SMSGuard |
|---|---|---|
| F1, 5-fold CV | 0.9716 ± 0.0019 | **0.9900 ± 0.0008** |
| False-positive rate, 5-fold CV | 1.43% ± 0.15 | **0.42% ± 0.03** |
| Hybrid-attack detection, 5-fold CV | 86.9% ± 0.9 | **99.8% ± 0.04** |
| Worst of 9 attacks at 30% perturbation | 85.8% (typo) | **98.6%** (typo) |
| Adaptive attacker success (greedy, queries the model) | 64.0% | **1.0%** |
| Words the adaptive attacker had to change, when successful | 4.2 | 1.7 (succeeds only on very short messages) |
| Spam-with-link recall | 96.9% | **100%** |
| Latency (CPU) / model size | 0.5 ms / 0.8 MB | 2.3 ms / 7.1 MB |

McNemar test, clean test set: SMSGuard vs B0, p = 1.7e-33.

## What each component contributes (ablation, all adversarially trained unless noted)
| Component removed | Effect of removing it |
|---|---|
| Normalisation (A4) | Adaptive success rises 1.0% → 5.3%. Worse on unseen full-width (97.5% vs 98.7%) and zero-width (98.2% vs 99.0%), and on homoglyph/spacing when they are held out. |
| Spelling restoration (A5) | Adaptive success **triples**: 1.0% → 3.0%. Typo and charswap are what the adaptive attacker uses once everything else is closed. Random-attack scores barely change, because character n-grams already absorb random typos; the fixer matters against an attacker who *searches*. |
| URL reputation (A6) | Clean accuracy drops significantly (McNemar p = 0.031: SMSGuard fixes 6 of A6's errors and introduces 0). Real ham with links flagged: 21.9% vs 12.5%. Curated link ham: 50% vs 0%. Curated hard ham: 15% vs 10%. |
| Adversarial training (A3, normalisation only) | Detection under the nine attacks falls from 98.6–100% to 98.5–99.5%, and the adaptive attacker succeeds more often. |
| Word + char n-grams (B0, paper-style) | The largest clean-accuracy loss: F1 0.990 → 0.972, FPR 0.42% → 1.43%. |

## Generalisation to unseen attacks (E4)
- **One attack held out of adversarial training:** SMSGuard still detects 97.5–99.4% of it.
- **Trained only on the base paper's three attacks:** SMSGuard detects 98.5–99.3% of the six other attacks, and is better than or equal to the no-normalisation model on every one (largest gains: full-width 98.8% vs 97.1%, zero-width 99.0% vs 97.8%).
- **Honest exceptions (leave-one-out):** held-out emoji wedge (97.5% vs 98.1%), leet (98.4% vs 98.6%) and charswap/typo (within 0.2 points) are marginally better *without* normalisation. These differences are within about 0.5 points.

## False positives (hard-ham stress test)
- Adding emoji or non-Latin names (Cyrillic, Greek, Devanagari, Chinese) to legitimate messages does **not** raise false positives: 0.3% and 0.2%.
- **Curated tricky ham** (40 messages, written before any URL work): SMSGuard 10%, B0 20%. The remaining failures are an appointment confirmation, a parcel-delivered notice, a bill reminder and "call me on <number>". These are transactional messages that closely resemble scams.
- **Legitimate messages with links in the test set (n = 32):** SMSGuard 12.5% (4 messages), no-normalisation models 6.2% (2 messages). The sample is too small to separate them. Two of the four are arguably promotional (a Flipkart delivery notice, a property-listings ad).

## Bugs found and fixed during the study (worth a "lessons learned" paragraph)
1. **Bullet wedges missed.** Unicode bullets ("•") are punctuation, not symbols, so "C•L•A•I•M" first slipped through.
2. **Non-ASCII ratio fooled by disguise.** Computing it on *raw* text let obfuscation characters make spam look like legitimate foreign text. Computing it after normalisation raised unseen full-width detection from 95.5% to 98.7%.
3. **Foreign names mangled.** Folding every look-alike letter turned "Дмитрий" into "дmиtpий". Now only mixed-script words, or all-look-alike words in mostly-Latin messages, are folded.
4. **Missing spaces read as links.** TLDs that are English words ("so", "at", "be", "im", "in") turned "renovated.so" or "psam.cn u" (SMS-speak for "can") into links, raising false positives on casual ham. Such matches now need a path, a mixed letter/digit or hyphenated name, or a known domain.
5. **Trusted links counted as spam evidence.** Spam often *mentions* trusted domains (paypal.com, amazon.com), so a "trusted link count" feature learned a positive spam weight. Trusted links are now neutralised instead.
6. **Label leakage through the spelling fixer.** Protecting words seen in training ham meant only spam was ever "corrected" during training. A "typos fixed" feature then leaked the label: the false-positive rate jumped from 0.3% to 3.3%. The count was removed as a feature; the fixer now only repairs text.

## Dataset observations
- Only 90 of 40,820 legitimate messages contain a link, against 7,885 of 26,093 spam. Any learned model treats "has a link" as spam, which is why trusted links must be handled by rules, not learning.
- The released CSV is Windows-1252 with a few undefined bytes, so it can't store homoglyph or emoji examples. All attacked text is generated in memory as Unicode.

## Limitations to state in the paper
- **Language and look-alikes:** English only, and the look-alike table is a curated subset of Unicode TS #39.
- **Spelling fixer:** it can still "correct" a rare real word that never appeared in training ham (e.g. "drown" → "down"); the verdict is usually unaffected. Very short messages remain attackable: the adaptive attacker's few successes needed only about 2 edits.
- **URL reputation is offline:** no domain age or threat feeds. The allow-list is small and hand-picked from popular-site lists.
- **Untested attacks and drift:** paraphrase and LLM-rewriting attacks (E7) and temporal drift (E8; the CSV has no dates) are untested.
- **Our own attacks:** the attacks and the adaptive attacker are our implementations. The attacker is greedy and character-level, and doesn't rewrite meaning.
- **Small curated sets:** 40 + 10 messages. The link set was written alongside the URL feature, so its 0% is optimistic.
- **Transformers:** the comparison (E6) is still to run on Colab.
