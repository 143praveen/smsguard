"""Shared data loading, splitting, adversarial augmentation and metrics.

Every script uses these, so all experiments share one protocol:
  * deduplicated data, one stratified seeded split
  * the test set is never used for training or model selection
  * a message that ends up empty still gets a prediction (counted, never dropped)
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .attacks import ALL_ATTACKS, attack

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT.parent / "spstudy" / "Data" / "super_sms_dataset.csv"


def load_dataset(path=DEFAULT_DATA) -> pd.DataFrame:
    # The released file is Windows-1252 (single-byte), with a few undefined bytes.
    df = pd.read_csv(path, encoding="cp1252", encoding_errors="replace")
    df.columns = ["text", "label"]
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip() != ""]
    df["label"] = df["label"].astype(int)
    key = df["text"].str.lower().str.split().str.join(" ")
    return df.loc[~key.duplicated()].reset_index(drop=True)


def split(df, seed=SEED, test_size=0.2):
    return train_test_split(df.text.tolist(), df.label.values, test_size=test_size, stratify=df.label.values, random_state=seed)


def augment(X, y, keywords, attacks=ALL_ATTACKS, copies=1, seed=SEED):
    """Adversarial training: add attacked copies of the training spam, using only `attacks`."""
    rng = random.Random(seed)
    pool = [a for a in attacks if a != "hybrid"]
    extra = []
    for i, (t, lab) in enumerate(zip(X, y)):
        if lab != 1:
            continue
        for _ in range(copies):
            a = rng.choice(list(attacks))
            extra.append(attack(t, a, keywords, rate=rng.uniform(0.15, 0.5), seed=seed + i, pool=pool or None))
    return list(X) + extra, np.concatenate([np.asarray(y), np.ones(len(extra), dtype=int)])


def scores(model, X):
    clf = model.steps[-1][1]
    return model.decision_function(X) if hasattr(clf, "decision_function") else model.predict_proba(X)[:, 1]


def evaluate(model, X, y, pred=None):
    y = np.asarray(y)
    p = model.predict(X) if pred is None else pred
    out = dict(accuracy=accuracy_score(y, p), precision=precision_score(y, p, zero_division=0),
               recall=recall_score(y, p), f1=f1_score(y, p), fpr=float(((p == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)))
    try:
        out["roc_auc"] = roc_auc_score(y, scores(model, X))
    except Exception:
        pass
    return {k: round(float(v), 4) for k, v in out.items()}


def mcnemar(y, pred_a, pred_b):
    """Exact McNemar test: do models A and B make significantly different errors?"""
    y, a, b = map(np.asarray, (y, pred_a, pred_b))
    a_only = int(((a == y) & (b != y)).sum())   # A right, B wrong
    b_only = int(((a != y) & (b == y)).sum())   # B right, A wrong
    n = a_only + b_only
    p = binomtest(a_only, n, 0.5).pvalue if n else 1.0
    return {"a_right_b_wrong": a_only, "b_right_a_wrong": b_only, "p_value": float(p)}
