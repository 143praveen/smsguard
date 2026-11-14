"""Model definitions: a base-paper-style baseline and the SMSGuard hybrid model."""
from __future__ import annotations

import re
import string

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

from .normalize import NormResult, SpellFixer, build_vocab, normalize

# typos_fixed is deliberately NOT a feature: the corrector protects words seen in *training ham*, so in training
# only spam gets corrected and the count leaks the label (FPR rose from 0.3% to 3.3% when it was included).
META_KEYS = ["zero_width", "fullwidth", "homoglyph_words", "symbol_wedges", "spaced_runs", "leet_tokens",
             # n_trusted_urls is deliberately NOT a feature: spam often *mentions* trusted brands' domains,
             # so the model would learn "trusted link => spam". Trusted links are neutralised instead.
             "non_ascii_ratio", "n_urls", "n_short_urls", "n_risky_tld", "n_brand_spoof",
             "n_url_suspicious_words", "n_emails", "n_phones", "n_money", "upper_ratio", "length"]
RATIO_KEYS = {"non_ascii_ratio", "upper_ratio"}


# ---------- Baseline: reproduces the base paper's preprocessing ----------
def paper_preprocess(texts):
    """Strip punctuation, lowercase, keep alphabetic non-stopword tokens (as in spstudy notebooks)."""
    table = str.maketrans("", "", string.punctuation)
    return [" ".join(w for w in str(t).translate(table).lower().split()
                     if w.isalpha() and w not in ENGLISH_STOP_WORDS) for t in texts]


def build_baseline():
    return Pipeline([
        ("prep", FunctionTransformer(paper_preprocess)),
        ("tfidf", TfidfVectorizer()),
        ("clf", LinearSVC(C=1.0)),
    ])


# ---------- SMSGuard ----------
class Normalizer(BaseEstimator, TransformerMixin):
    """Runs defensive normalisation once and passes text + flags downstream.

    fit() learns a word-frequency vocabulary from the training messages, used to
    re-segment de-spaced runs ("w o n a f r e e" -> "won a free").
    enabled=False is the ablation: raw lowercased text, no flags.
    """

    def __init__(self, enabled: bool = True, segment: bool = True, spellfix: bool = True, url_reputation: bool = True):
        self.enabled = enabled
        self.segment = segment
        self.spellfix = spellfix
        self.url_reputation = url_reputation

    def fit(self, X, y=None):
        need_vocab = self.enabled and (self.segment or self.spellfix)
        texts = [normalize(t, url_reputation=self.url_reputation).text for t in X] if need_vocab else []
        vocab = build_vocab(texts) if need_vocab else None
        self.vocab_ = vocab if self.segment else None
        protected = set()
        if self.enabled and self.spellfix and y is not None:
            # Only spam is adversarially augmented, so every word in legitimate training messages is genuine.
            for t, lab in zip(texts, y):
                if lab == 0:
                    protected.update(re.findall(r"[a-z]{4,}", t))
        self.fixer_ = SpellFixer(vocab, protected) if (self.enabled and self.spellfix) else None
        return self

    def transform(self, X):
        if self.enabled:
            vocab, fixer = getattr(self, "vocab_", None), getattr(self, "fixer_", None)
            return [normalize(t, vocab, fixer, url_reputation=self.url_reputation) for t in X]
        return [NormResult(text=str(t).lower(), flags={}) for t in X]


def _texts(results):
    return [r.text for r in results]


def _meta(results):
    """Evasion + entity flags. Counts are log-scaled so one extreme message can't dominate."""
    m = np.array([[r.flags.get(k, 0) for k in META_KEYS] for r in results], dtype=float)
    for j, k in enumerate(META_KEYS):
        if k not in RATIO_KEYS:
            m[:, j] = np.log1p(m[:, j])
    return m


def build_smsguard(use_meta: bool = True, normalize_text: bool = True, segment: bool = True,
                   spellfix: bool = True, url_reputation: bool = True, C: float = 4.0):
    parts = [
        ("word", Pipeline([("t", FunctionTransformer(_texts)),
                           ("v", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=300_000))])),
        ("char", Pipeline([("t", FunctionTransformer(_texts)),
                           ("v", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=300_000))])),
    ]
    if use_meta:
        parts.append(("meta", Pipeline([("t", FunctionTransformer(_meta)), ("s", StandardScaler())])))
    return Pipeline([
        ("norm", Normalizer(enabled=normalize_text, segment=segment, spellfix=spellfix, url_reputation=url_reputation)),
        ("feats", FeatureUnion(parts)),
        ("clf", LogisticRegression(C=C, max_iter=3000, solver="liblinear", class_weight="balanced")),
    ])


def spam_keywords(texts, labels, k: int = 200) -> set[str]:
    """Top-k spam-indicative words by smoothed log-odds (the attacker's 'thesaurus')."""
    from collections import Counter
    sp, hm = Counter(), Counter()
    for t, y in zip(texts, labels):
        (sp if y == 1 else hm).update(set(re.findall(r"[a-z]{3,}", str(t).lower())))
    ns, nh = max(sum(labels), 1), max(len(labels) - sum(labels), 1)
    score = {w: np.log((sp[w] + 1) / ns) - np.log((hm[w] + 1) / nh) for w in sp if sp[w] >= 20}
    return set(sorted(score, key=score.get, reverse=True)[:k])
