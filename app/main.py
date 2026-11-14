"""SMSGuard web API + frontend.

Run:  .venv/Scripts/python -m uvicorn app.main:app --port 8000
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smsguard.normalize import TRICK_FLAGS, evasion_score, normalize  # noqa: E402

MODEL_PATH = ROOT / "models" / "SMSGuard.joblib"
app = FastAPI(title="SMSGuard", version="0.1.0")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")
_bundle = None


def bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, "Model not trained yet. Run scripts/train.py first.")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


class Message(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


def explain(model, text: str, k: int = 8):
    """Top word/char n-grams pushing the prediction toward spam or ham (linear model weights x tf-idf)."""
    norm, feats, clf = (model.named_steps[s] for s in ("norm", "feats", "clf"))
    normed = norm.transform([text])
    x = feats.transform(normed).tocsr()
    flags = normed[0].flags
    names = []
    for name, pipe in feats.transformer_list:
        if name == "meta":
            from smsguard.model import META_KEYS
            names += [f"[{m}]" for m in META_KEYS]
        else:
            names += [f"{name}:{t}" for t in pipe.named_steps["v"].get_feature_names_out()]
    contrib = x.multiply(clf.coef_[0]).tocsr()
    idx, vals = contrib.indices, contrib.data
    order = np.argsort(-np.abs(vals))[: k * 3]
    out, seen = [], set()
    for i in order:
        label = names[idx[i]]
        # meta features are standardised, so an absent trick still has a value; only show tricks that are present
        if label in seen or (label.startswith("[") and not flags.get(label.strip("[]"))):
            continue
        seen.add(label)
        out.append({"feature": label, "weight": round(float(vals[i]), 3)})
        if len(out) == k:
            break
    return out


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/api/health")
def health():
    b = bundle()
    return {"status": "ok", "model": "SMSGuard", "clean_metrics": b["metrics"]["clean"]}


@app.post("/api/predict")
def predict(msg: Message):
    model = bundle()["model"]
    proba = float(model.predict_proba([msg.text])[0, 1])
    norm = model.named_steps["norm"]
    n = normalize(msg.text, getattr(norm, "vocab_", None), getattr(norm, "fixer_", None))
    tricks = [k for k in TRICK_FLAGS + ("typos_fixed",) if n.flags.get(k)]
    return {
        "label": "spam" if proba >= 0.5 else "ham",
        "spam_probability": round(proba, 4),
        "normalized_text": n.text,
        "evasion_tricks": tricks,
        "evasion_score": evasion_score(n.flags),
        "urls": n.urls,
        "links": [{"url": i.url.replace(" ", ""), "domain": i.registered, "verdict": i.verdict,
                   "risky": bool(i.brand_spoof or i.risky_tld or i.ip_host), "trusted": i.trusted,
                   "suspicious_words": i.suspicious_words} for i in n.url_info],
        "flags": n.flags,
        "explanation": explain(model, msg.text),
    }
