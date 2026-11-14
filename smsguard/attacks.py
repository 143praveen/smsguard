"""Black-box evasion attacks.

Re-implements the base paper's taxonomy (char-level, word-level, multi-level)
and adds tricks it did not test: zero-width injection, leetspeak, full-width
letters and emoji wedges. Attacks target the most spam-indicative words first,
mirroring the paper's "thesaurus of the 200 most frequent spam keywords".
"""
from __future__ import annotations

import random
import re

HOMOGLYPHS = {"a": "а", "c": "с", "e": "е", "i": "і", "j": "ј", "o": "о", "p": "р", "s": "ѕ", "x": "х", "y": "у",
              "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "I": "І", "K": "К", "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х"}
LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}
ZWSP = "​"
WEDGES = ["🎁", "💰", "★", "✓", "•", "🔥", "✨"]
WORD_RE = re.compile(r"[A-Za-z]{3,}")


def _targets(text: str, keywords: set[str], rate: float, rng: random.Random) -> set[int]:
    words = list(WORD_RE.finditer(text))
    if not words:
        return set()
    k = max(1, round(len(words) * rate))
    keyed = [i for i, m in enumerate(words) if m.group(0).lower() in keywords]
    rest = [i for i in range(len(words)) if i not in keyed]
    rng.shuffle(keyed)
    rng.shuffle(rest)
    return set((keyed + rest)[:k])


def _apply(text: str, fn, keywords, rate, rng) -> str:
    idx = _targets(text, keywords, rate, rng)
    out, last = [], 0
    for i, m in enumerate(WORD_RE.finditer(text)):
        out.append(text[last:m.start()])
        out.append(fn(m.group(0), rng) if i in idx else m.group(0))
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def spacing(w, rng):
    return " ".join(w)


def homoglyph(w, rng):
    pos = [i for i, c in enumerate(w) if c in HOMOGLYPHS]
    for i in rng.sample(pos, max(1, len(pos) // 2)) if pos else []:
        w = w[:i] + HOMOGLYPHS[w[i]] + w[i + 1:]
    return w


def charswap(w, rng):
    i = rng.randrange(1, len(w) - 1) if len(w) > 3 else 0
    return w[:i] + w[i + 1] + w[i] + w[i + 2:]


def typo(w, rng):
    """Random single-character insert, delete or substitute (textattack-style charswap variants)."""
    i = rng.randrange(1, len(w) - 1) if len(w) > 2 else 0
    op = rng.choice(("insert", "delete", "substitute"))
    c = rng.choice("abcdefghijklmnopqrstuvwxyz")
    if op == "insert":
        return w[:i] + c + w[i:]
    if op == "delete":
        return w[:i] + w[i + 1:]
    return w[:i] + c + w[i + 1:]


def zero_width(w, rng):
    return ZWSP.join(w)


def leet(w, rng):
    return "".join(LEET.get(c.lower(), c) if rng.random() < 0.6 else c for c in w)


def fullwidth(w, rng):
    return "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else c for c in w)


def emoji_wedge(w, rng):
    return rng.choice(WEDGES).join(w)


ATTACKS = {
    "spacing": spacing,
    "homoglyph": homoglyph,
    "charswap": charswap,
    "typo": typo,
    "zero_width": zero_width,
    "leet": leet,
    "fullwidth": fullwidth,
    "emoji_wedge": emoji_wedge,
}
# Attacks the base paper evaluated (homoglyph = its punycode attack); the rest are new in this work.
PAPER_ATTACKS = ["spacing", "homoglyph", "charswap"]
NEW_ATTACKS = ["typo", "zero_width", "leet", "fullwidth", "emoji_wedge"]


def attack(text: str, name: str, keywords: set[str], rate: float = 0.3, seed: int = 0, pool: list[str] | None = None) -> str:
    """Perturb `rate` of the words with attack `name`.

    'hybrid' uses a different trick per targeted word, drawn from `pool` (default: all attacks).
    """
    rng = random.Random(f"{seed}:{name}:{text}")
    if name == "hybrid":
        fns = [ATTACKS[a] for a in (pool or ATTACKS)]
        return _apply(text, lambda w, r: r.choice(fns)(w, r), keywords, rate, rng)
    return _apply(text, ATTACKS[name], keywords, rate, rng)


ALL_ATTACKS = list(ATTACKS) + ["hybrid"]
