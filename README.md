# SMSGuard

Evasion-robust SMS spam detection with an explainable web frontend. The project builds on, and corrects the evaluation of, Salman et al., *IEEE Access* 2024 (the Super SMS Dataset). The research plan is in [docs/research_plan.md](docs/research_plan.md).

## How it works
```
raw SMS ──► 1. normalise ─────────────────────► word 1-2gram TF-IDF ┐
            • strip zero-width chars, full-width   char 2-5gram TF-IDF ├─► logistic regression ─► spam probability
            • fold homoglyphs (Cyrillic/Greek)     evasion + URL flags ┘        (adversarially trained)
            • remove emoji/bullet wedges
            • de-space "F R E E" (+ re-segment), de-leet "fr33"
          ► 2. URL reputation
            • trusted / institutional links neutralised
            • risky TLD, shortener, brand impersonation ("auspost-redelivery.com") flagged
          ► 3. spelling restoration: "Cliam your przie" → "claim your prize"
```
Obfuscation is not only undone, it is also counted: each detected trick becomes a feature, because legitimate messages almost never contain look-alike letters or invisible characters. Key results and lessons learned are in [docs/findings.md](docs/findings.md).

## Layout
| Path | Purpose |
|---|---|
| `smsguard/normalize.py` | Defensive normalisation, evasion flags, vocabulary re-segmentation, spelling restoration (`SpellFixer`) |
| `smsguard/urls.py` | Offline URL reputation: trusted/institutional domains, risky TLDs, shorteners, brand impersonation |
| `smsguard/attacks.py` | Nine black-box attacks: the base paper's spacing, homoglyph and charswap, plus typo, zero-width, leet, full-width, emoji-wedge and hybrid |
| `smsguard/model.py` | Base-paper-style baseline and the SMSGuard pipeline |
| `smsguard/experiment.py` | Shared protocol: loading, split, adversarial augmentation, metrics, McNemar |
| `scripts/train.py` | E1/E2: baseline, ablations and SMSGuard on one split; saves every model to `models/` |
| `scripts/experiments.py` | E3 sweep, E4 unseen attacks, E5 cross-validation, adaptive attacker, hard-ham false positives, E9 timing |
| `tests/` | Unit tests for normalisation (`pytest`) |
| `results/` | One `.md` + `.json` per experiment, figures in `results/figures/` |
| `app/` | FastAPI backend (`/api/predict`) and single-page frontend |

## Quick start (Windows)
```bash
python -m venv .venv
```
```bash
.venv/Scripts/python -m pip install -r requirements.txt
```
```bash
.venv/Scripts/python scripts/train.py --data ../spstudy/Data/super_sms_dataset.csv
```
```bash
.venv/Scripts/python scripts/experiments.py all
```
```bash
.venv/Scripts/python -m pytest tests
```
```bash
.venv/Scripts/python -m uvicorn app.main:app --port 8000
```
Then open http://localhost:8000.

## API
`POST /api/predict` with `{"text": "..."}` returns the label, spam probability, normalised text, detected evasion tricks, URLs and the top contributing features.

## Citation of the base work
Salman, M., Ikram, M., Kaafar, M.A. "Investigating Evasive Techniques in SMS Spam Filtering: A Comparative Analysis of Machine Learning Models." *IEEE Access* 12 (2024): 24306–24324.
