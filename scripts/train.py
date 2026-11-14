"""E1 + E2: train baseline, ablations and SMSGuard on one split; evaluate on clean
and attacked test data; McNemar significance tests; save every model.

Usage:  python scripts/train.py [--data path/to/super_sms_dataset.csv]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smsguard.attacks import ALL_ATTACKS, attack  # noqa: E402
from smsguard.experiment import DEFAULT_DATA, SEED, augment, evaluate, load_dataset, mcnemar, split  # noqa: E402
from smsguard.model import build_baseline, build_smsguard, spam_keywords  # noqa: E402

MODELS = {
    # name: (builder, adversarial training?)
    "B0_paper_style_svm": (build_baseline, False),
    "A1_hybrid_no_norm": (lambda: build_smsguard(use_meta=False, normalize_text=False), False),
    "A2_hybrid_norm": (lambda: build_smsguard(use_meta=False), False),
    "A3_hybrid_norm_meta": (build_smsguard, False),
    "A4_no_norm_advtrain": (lambda: build_smsguard(use_meta=False, normalize_text=False), True),
    "A5_no_spellfix_advtrain": (lambda: build_smsguard(spellfix=False), True),
    "A6_no_urlrep_advtrain": (lambda: build_smsguard(url_reputation=False), True),
    "SMSGuard": (build_smsguard, True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    args = ap.parse_args()
    out, mdir = ROOT / "results", ROOT / "models"
    out.mkdir(exist_ok=True)
    mdir.mkdir(exist_ok=True)

    df = load_dataset(args.data)
    X_tr, X_te, y_tr, y_te = split(df)
    print(f"{len(df)} unique messages | spam ratio {df.label.mean():.3f} | train {len(X_tr)} | test {len(X_te)}")
    keywords = spam_keywords(X_tr, list(y_tr))
    X_adv, y_adv = augment(X_tr, y_tr, keywords)

    spam_te = [t for t, lab in zip(X_te, y_te) if lab == 1]
    ham_te = [t for t, lab in zip(X_te, y_te) if lab == 0]
    ham_typos = [attack(t, "charswap", set(), rate=0.2, seed=SEED) for t in ham_te]
    attacked = {a: [attack(t, a, keywords, rate=0.3, seed=SEED) for t in spam_te] for a in ALL_ATTACKS}

    res = {"n_train": len(X_tr), "n_test": len(X_te), "n_test_spam": len(spam_te), "seed": SEED, "models": {}}
    preds = {}
    for name, (build, adv) in MODELS.items():
        model = build()
        t0 = time.time()
        model.fit(*(X_adv, y_adv) if adv else (X_tr, y_tr))
        fit_s = time.time() - t0
        preds[name] = model.predict(X_te)
        r = {"clean": evaluate(model, X_te, y_te, preds[name]), "train_seconds": round(fit_s, 1),
             "fpr_ham_with_typos": round(float(model.predict(ham_typos).mean()), 4),
             "attack_detection_rate": {"none": round(float(model.predict(spam_te).mean()), 4)}}
        for a, texts in attacked.items():
            r["attack_detection_rate"][a] = round(float(model.predict(texts).mean()), 4)
        res["models"][name] = r
        print(f"{name:22s} F1={r['clean']['f1']:.4f} FPR={r['clean']['fpr']:.4f} "
              f"min attack det={min(r['attack_detection_rate'].values()):.3f} ({fit_s:.0f}s)")
        joblib.dump({"model": model, "keywords": sorted(keywords), "metrics": r, "seed": SEED}, mdir / f"{name}.joblib", compress=3)

    # Significance on the clean test set: SMSGuard vs every other model
    res["mcnemar_vs_SMSGuard"] = {n: mcnemar(y_te, preds["SMSGuard"], preds[n]) for n in MODELS if n != "SMSGuard"}
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    write_markdown(res, out / "results.md")
    print(f"Wrote {out / 'results.md'}")


def write_markdown(res, path):
    names = list(res["models"])
    L = [f"# E1/E2 results (seed {res['seed']}, train {res['n_train']}, test {res['n_test']}, test spam {res['n_test_spam']})",
         "", "## Clean test set", "", "| Model | Acc | Prec | Recall | F1 | FPR | ROC-AUC | FPR (ham+typos) | Train s |",
         "|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        c, m = res["models"][n]["clean"], res["models"][n]
        L.append(f"| {n} | {c['accuracy']:.4f} | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1']:.4f} | {c['fpr']:.4f} | "
                 f"{c.get('roc_auc', float('nan')):.4f} | {m['fpr_ham_with_typos']:.4f} | {m['train_seconds']} |")
    attacks = list(res["models"][names[0]]["attack_detection_rate"])
    L += ["", "## Spam detection rate under attack (30% of words perturbed)", "",
          "Attacks after `charswap` were not evaluated by the base paper.", "",
          "| Model | " + " | ".join(attacks) + " |", "|---|" + "---|" * len(attacks)]
    for n in names:
        d = res["models"][n]["attack_detection_rate"]
        L.append(f"| {n} | " + " | ".join(f"{d[a]:.3f}" for a in attacks) + " |")
    L += ["", "## McNemar test vs SMSGuard (clean test set)", "", "| Model | SMSGuard right, model wrong | model right, SMSGuard wrong | p-value |", "|---|---|---|---|"]
    for n, m in res["mcnemar_vs_SMSGuard"].items():
        L.append(f"| {n} | {m['a_right_b_wrong']} | {m['b_right_a_wrong']} | {m['p_value']:.2e} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
