"""Defensive text normalisation.

Undoes the evasion tricks reported by Salman et al. (IEEE Access 2024), plus
several they did not test, before a message reaches the classifier, and
records *what* it undid. The record is returned as flags: obfuscation itself
is a strong spam signal.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field

from .urls import analyse

# Invisible characters used to split keywords without changing how they look.
ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍‎‏⁠⁡⁢⁣﻿­͏᠎"), None)

# Look-alike (homoglyph) characters -> Latin. A focused subset of the
# Unicode TS #39 confusables table covering Cyrillic, Greek and common symbols.
CONFUSABLES = {
    # Cyrillic lower
    "а": "a", "в": "b", "е": "e", "ё": "e", "к": "k", "м": "m", "н": "h", "о": "o",
    "р": "p", "с": "c", "т": "t", "у": "y", "х": "x", "ѕ": "s", "і": "i", "ї": "i",
    "ј": "j", "ԁ": "d", "ԛ": "q", "ԝ": "w", "һ": "h", "ӏ": "l", "ь": "b", "ɡ": "g",
    # Cyrillic upper
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P",
    "С": "C", "Т": "T", "У": "Y", "Х": "X", "Ѕ": "S", "І": "I", "Ј": "J", "Ԁ": "D",
    # Greek
    "α": "a", "β": "b", "ε": "e", "η": "n", "ι": "i", "κ": "k", "ν": "v", "ο": "o",
    "ρ": "p", "τ": "t", "υ": "u", "χ": "x", "ω": "w", "Α": "A", "Β": "B", "Ε": "E",
    "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P",
    "Τ": "T", "Υ": "Y", "Χ": "X",
    # Armenian / Latin extensions / misc
    "ո": "n", "ս": "u", "օ": "o", "ı": "i", "ȷ": "j", "ɑ": "a", "ʏ": "y", "ℓ": "l",
}
CONFUSABLE_TABLE = str.maketrans(CONFUSABLES)

LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"})

URL_RE = re.compile(
    r"(?:https?\s*:\s*/\s*/\s*\S+|www\.\S+|\b[a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)*\.(?:com|net|org|info|biz|ly|me|co|io|us|uk|au|in|ga|tk|ml|cf|gq|xyz|top|click|link|site|online|app|gl|to|cc|ws|tv|ru|cn|be|de|fr|nz|ca|sg|gov|edu|shop|icu|buzz|live|vip|so|za|ie|nl|jp|br|sbi|im|gy|at|id|gd)\b(?:/\S*)?)",
    re.IGNORECASE,
)
SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "rb.gy", "cutt.ly",
    "shorturl.at", "tiny.cc", "rebrand.ly", "t.ly", "bl.ink", "s.id", "v.gd", "olo.me", "g.tm", "wa.me", "y77.us",
}
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
PHONE_RE = re.compile(r"(?:\+?\d[\d\s\-()]{7,}\d)")
MONEY_RE = re.compile(r"(?:[£$€₹]\s?\d[\d,.]*|\b\d[\d,.]*\s?(?:usd|aud|gbp|eur|rs|inr|dollars?|pounds?|bucks)\b)", re.IGNORECASE)
# Three or more single letters separated by spaces/dots/dashes: "f r e e", "w.i.n"
SPACED_RE = re.compile(r"(?<![A-Za-z])(?:[A-Za-z][ .\-_*]{1,2}){2,}[A-Za-z](?![A-Za-z])")
LEET_TOKEN_RE = re.compile(r"\b(?=\w*[A-Za-z])(?=\w*[0-9@$])[A-Za-z0-9@$]{3,}\b")
# Legitimate alphanumerics that must not be "de-leeted": 1pm, 2nd, 10am, 5gb, mp3, ids
NOT_LEET_RE = re.compile(r"^(?:\d+(?:am|pm|st|nd|rd|th|k|m|g|gb|mb|kb|hrs?|mins?|secs?|days?|yrs?|x|s|p)|[a-z]{1,3}\d|\w*\d{3,}\w*)$", re.IGNORECASE)
# Symbol/emoji categories that can be wedged between letters ("F🎁R🎁E🎁E", "W★I★N")
WEDGE_CATEGORIES = {"So", "Sk", "Sm", "Co", "Cn", "Mn", "Me"}
# Bullet/dot characters are Unicode punctuation (Po), not symbols, but are used the same way
WEDGE_CHARS = set("•·∙‧⋅◦‣⁃●○◆◇■□▪▫♦♥♠♣¦")


# TLDs that are also common English words/abbreviations: "renovated.so", "going.im", "lo.at", "liking.be".
AMBIGUOUS_TLDS = {"so", "im", "at", "be", "to", "in", "id", "me", "us", "it", "is", "no", "do", "go", "my", "he", "am", "gd", "gy", "co",
                  "cn"}  # "cn" = "can" in SMS-speak ("psam.cn u ...")


def is_real_url(s: str) -> bool:
    """Reject scheme-less matches that are really two words joined by a missing space."""
    s = s.strip()
    if re.match(r"(?i)https?\s*:|www\.", s):
        return True
    host = re.split(r"[/?#]", s, maxsplit=1)[0]
    tld = host.rsplit(".", 1)[-1]
    if tld.lower() not in AMBIGUOUS_TLDS:
        return True
    lower = host.lower()
    known = any(lower == d or lower.endswith("." + d) for d in SHORTENERS) or analyse(host).trusted
    has_path = "/" in s[len(host):len(host) + 1]
    label = host.rsplit(".", 1)[0]
    mixed_label = bool(re.search(r"[a-z][\d\-]|[\d\-][a-z]", label, re.IGNORECASE))  # "est1", "cancel-pay", not "199"
    return known or has_path or mixed_label or (tld.islower() and host.count(".") >= 2)


@dataclass
class NormResult:
    text: str
    flags: dict = field(default_factory=dict)
    urls: list = field(default_factory=list)
    url_info: list = field(default_factory=list)


def _script_of(ch: str) -> str:
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return "OTHER"
    return name.split(" ")[0]


def _remove_wedges(text: str) -> tuple[str, int]:
    """Drop symbols/emoji sitting between two letters, e.g. 'F🎁R🎁E🎁E' -> 'FREE'."""
    out, n, i = [], 0, 0
    while i < len(text):
        j = i
        while j < len(text) and (unicodedata.category(text[j]) in WEDGE_CATEGORIES or text[j] in WEDGE_CHARS):
            j += 1
        if j > i:
            # Only between *Latin* letters, so Devanagari/Thai vowel marks etc. are never touched
            prev_alpha = bool(out) and out[-1][-1:].isascii() and out[-1][-1:].isalpha()
            next_alpha = j < len(text) and text[j].isascii() and text[j].isalpha()
            if prev_alpha and next_alpha and j - i <= 3:
                n += 1
            else:
                out.append(text[i:j])
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out), n


def segment(s: str, vocab: dict | None) -> str:
    """Split a run of letters into the most probable word sequence (unigram Viterbi).

    'wonafree' -> 'won a free'. Without a vocabulary the run is returned unchanged.
    """
    if not vocab or len(s) > 80:
        return s
    total = vocab.get("__total__", 1)
    maxlen = 20
    best = [0.0] + [math.inf] * len(s)
    back = [0] * (len(s) + 1)
    for i in range(1, len(s) + 1):
        for j in range(max(0, i - maxlen), i):
            w = s[j:i]
            c = vocab.get(w)
            cost = math.log(total / c) if c else 10.0 + 4.0 * len(w)  # unknown words are very expensive
            if best[j] + cost < best[i]:
                best[i], back[i] = best[j] + cost, j
    words, i = [], len(s)
    while i > 0:
        words.append(s[back[i]:i])
        i = back[i]
    return " ".join(reversed(words))


def normalize(raw: str, vocab: dict | None = None, fixer: "SpellFixer | None" = None, url_reputation: bool = True) -> NormResult:
    text = raw if isinstance(raw, str) else ""
    flags = {}

    # 1. Invisible characters
    no_zw = text.translate(ZERO_WIDTH)
    flags["zero_width"] = len(text) - len(no_zw)
    text = no_zw

    # 2. Full-width / compatibility forms ("ＦＲＥＥ") and homoglyphs
    nfkc = unicodedata.normalize("NFKC", text)
    flags["fullwidth"] = sum(0xFF01 <= ord(c) <= 0xFF5E for c in text)
    text = nfkc
    # Fold a word only if it (a) mixes Latin with another script ("Ϲlаim"), or (b) is made entirely of
    # look-alike letters inside a mostly-Latin message ("рау" for "pay"). Genuine foreign words
    # ("Дмитрий", "Σοφία") are left untouched.
    letters = [c for c in text if c.isalpha()]
    mostly_latin = sum(_script_of(c) == "LATIN" for c in letters) > 0.5 * max(len(letters), 1)
    mixed, changed = 0, 0

    def fold(m):
        nonlocal mixed, changed
        w = m.group(0)
        scripts = {_script_of(c) for c in w if c.isalpha()}
        all_lookalike = "LATIN" not in scripts and all(c in CONFUSABLES for c in w if c.isalpha())
        if ("LATIN" in scripts and len(scripts) > 1) or (mostly_latin and all_lookalike and len(w) > 1):
            mixed += 1
            new = w.translate(CONFUSABLE_TABLE)
            changed += sum(a != b for a, b in zip(w, new))
            return new
        return w

    text = re.sub(r"\w+", fold, text)
    flags["homoglyph_words"], flags["homoglyph_chars"] = mixed, changed
    # 3. Emoji / symbols wedged inside words
    text, flags["symbol_wedges"] = _remove_wedges(text)

    # Non-ASCII share of what is LEFT after undoing obfuscation: measures genuinely foreign text,
    # so disguise characters can't make spam look like a legitimate non-English message.
    flags["non_ascii_ratio"] = round(sum(ord(c) > 127 for c in text) / max(len(text), 1), 4)

    # 4. Entities (before lowercasing so URLs keep their shape). Emails first, so "a@b.com" isn't read as a URL.
    flags["n_emails"] = len(EMAIL_RE.findall(text))
    text = EMAIL_RE.sub(" emailtoken ", text)
    matches = [m for m in URL_RE.finditer(text) if is_real_url(m.group(0))]
    urls = [m.group(0).replace(" ", "") for m in matches]
    short = sum(any(s in u.lower() for s in SHORTENERS) for u in urls)
    infos = [analyse(m.group(0)) for m in matches] if url_reputation else []
    if url_reputation:
        # Trusted links are neutralised: they no longer count as (spam) evidence.
        untrusted = [i for i in infos if not i.trusted]
        flags["n_urls"], flags["n_trusted_urls"] = len(untrusted), len(infos) - len(untrusted)
        flags["n_short_urls"] = sum(i.shortener for i in untrusted)
        flags["n_risky_tld"] = sum(i.risky_tld or i.ip_host for i in untrusted)
        flags["n_brand_spoof"] = sum(bool(i.brand_spoof) for i in untrusted)
        flags["n_url_suspicious_words"] = sum(len(i.suspicious_words) for i in untrusted)

        def token(i):
            if i.trusted:
                return " "
            # keep the registered domain's name as a token so the model can learn per-domain reputation
            dom = "dom" + re.sub(r"[^a-z0-9]", "", i.registered.split(".")[0])
            extra = (" riskytldtoken" if i.risky_tld or i.ip_host else "") + (" brandspooftoken" if i.brand_spoof else "")
            return f" urltoken {dom}{extra} "

        replacements = [token(i) for i in infos]
    else:
        flags["n_urls"], flags["n_short_urls"] = len(urls), short
        replacements = [" urltoken "] * len(matches)
    for m, rep in reversed(list(zip(matches, replacements))):
        text = text[:m.start()] + rep + text[m.end():]
    flags["n_phones"] = len(PHONE_RE.findall(text))
    text = PHONE_RE.sub(" phonetoken ", text)
    flags["n_money"] = len(MONEY_RE.findall(text))
    text = MONEY_RE.sub(" moneytoken ", text)
    if short:
        text += " shorturltoken"

    # 5. De-spacing: "F R E E" -> "free"; runs spanning several words are re-segmented
    spaced = SPACED_RE.findall(text)
    flags["spaced_runs"] = len(spaced)
    text = SPACED_RE.sub(lambda m: " " + segment(re.sub(r"[ .\-_*]", "", m.group(0)).lower(), vocab) + " ", text)

    # 6. Leetspeak inside mixed alphanumeric tokens: "fr33" -> "free"
    leet = [t for t in LEET_TOKEN_RE.findall(text) if not NOT_LEET_RE.match(t)]
    flags["leet_tokens"] = len(leet)
    text = LEET_TOKEN_RE.sub(lambda m: m.group(0) if NOT_LEET_RE.match(m.group(0)) else m.group(0).translate(LEET), text)

    flags["upper_ratio"] = round(sum(c.isupper() for c in raw) / max(sum(c.isalpha() for c in raw), 1), 4)
    flags["length"] = len(raw)
    text = re.sub(r"\s+", " ", text.lower()).strip()

    # 7. Spelling restoration: undo 1-character edits ("cliam" -> "claim") using the training vocabulary
    if fixer is not None:
        text, flags["typos_fixed"] = fixer.fix_text(text)
    return NormResult(text=text, flags=flags, urls=urls, url_info=infos)


class SpellFixer:
    """Noisy-channel style corrector for charswap/typo attacks.

    A word is replaced by a neighbour at edit distance 1 (swap, insert, delete,
    substitute) only when that neighbour is far more frequent in the training
    vocabulary (ratio >= `ratio`). Frequent words, short words and anything the
    vocabulary knows well are left alone, which keeps legitimate slang intact.
    """
    LETTERS = "abcdefghijklmnopqrstuvwxyz"

    def __init__(self, vocab: dict, protected: set | None = None, ratio: float = 50.0, min_target: int = 20,
                 min_len: int = 4, skip_above: int = 200):
        self.vocab, self.ratio, self.min_target, self.min_len, self.skip_above = vocab, ratio, min_target, min_len, skip_above
        # Words seen in legitimate training messages are never "corrected" ("drown" must not become "down").
        self.protected = protected or set()
        self._cache: dict = {}

    def __getstate__(self):  # don't pickle the cache
        d = self.__dict__.copy()
        d["_cache"] = {}
        return d

    def _edits1(self, w):
        splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
        yield from (a + b[1:] for a, b in splits if b)
        yield from (a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1)
        yield from (a + c + b[1:] for a, b in splits if b for c in self.LETTERS)
        yield from (a + c + b for a, b in splits for c in self.LETTERS)

    def correct(self, w: str) -> str:
        if w in self._cache:
            return self._cache[w]
        have = self.vocab.get(w, 0)
        best, best_n = w, 0
        if len(w) >= self.min_len and have < self.skip_above and w not in self.protected:
            for cand in self._edits1(w):
                n = self.vocab.get(cand, 0)
                if n > best_n and n >= self.min_target and n >= self.ratio * max(have, 1):
                    best, best_n = cand, n
        self._cache[w] = best
        return best

    def fix_text(self, text: str) -> tuple[str, int]:
        n = 0

        def sub(m):
            nonlocal n
            c = self.correct(m.group(0))
            n += c != m.group(0)
            return c

        return re.sub(r"(?<![a-z])[a-z]{4,}(?![a-z])", sub, text), n


TRICK_FLAGS = ("zero_width", "homoglyph_words", "fullwidth", "symbol_wedges", "spaced_runs", "leet_tokens")


def evasion_score(flags: dict) -> int:
    """Number of distinct obfuscation tricks detected (0 = none)."""
    return sum(bool(flags.get(k)) for k in TRICK_FLAGS)


def build_vocab(texts, min_count: int = 3) -> dict:
    """Word-frequency table used by `segment`, built from (normalised) training text."""
    from collections import Counter
    c = Counter()
    for t in texts:
        c.update(w for w in re.findall(r"[a-z]+", t) if len(w) > 1 or w in ("a", "i"))
    vocab = {w: n for w, n in c.items() if n >= min_count}
    vocab["__total__"] = sum(vocab.values())
    return vocab
