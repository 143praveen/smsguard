import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smsguard.attacks import ALL_ATTACKS, attack  # noqa: E402
from smsguard.normalize import evasion_score, normalize, segment  # noqa: E402

KW = {"urgent", "won", "free", "prize", "claim"}
SPAM = "URGENT! You have WON a free prize. Claim now"
VOCAB = {"urgent": 50, "you": 900, "have": 500, "won": 80, "a": 2000, "free": 300, "prize": 90, "claim": 70, "now": 400, "__total__": 4390}


def words(s):
    return re.findall(r"\w+", s)


# charswap/typo (and hybrid, which may use them) change spelling; the vocabulary-based SpellFixer handles them (tested below)
@pytest.mark.parametrize("name", [a for a in ALL_ATTACKS if a not in ("charswap", "typo", "hybrid")])
def test_normalisation_reverses_reversible_attacks(name):
    attacked = attack(SPAM, name, KW, rate=0.6, seed=1)
    assert words(normalize(attacked, VOCAB).text) == words(normalize(SPAM, VOCAB).text)


@pytest.mark.parametrize("name", ["zero_width", "homoglyph", "fullwidth", "emoji_wedge", "spacing", "leet"])
def test_tricks_are_flagged(name):
    attacked = attack(SPAM, name, KW, rate=0.6, seed=1)
    assert evasion_score(normalize(attacked).flags) >= 1


def test_segmentation():
    assert segment("wonafree", VOCAB) == "won a free"
    assert segment("wonafree", None) == "wonafree"


@pytest.mark.parametrize("text", [
    "lunch at 1pm on the 2nd", "mp3 and 5gb data", "Dinner with Дмитрий and Ольга on Friday",
    "My grandma says नमस्ते to everyone", "Happy birthday!! 🎉🎂", "Café with Zoë and José",
])
def test_legitimate_text_has_no_evasion_flags(text):
    assert evasion_score(normalize(text).flags) == 0, normalize(text).flags


def test_foreign_words_are_not_mangled():
    assert "дмитрий" in normalize("Dinner with Дмитрий and Ольга").text
    assert "σοφία" in normalize("Σοφία says hi").text


def test_all_lookalike_word_in_english_message_is_folded():
    r = normalize("Please рау the fee today")  # Cyrillic р, а, у
    assert "pay the fee" in r.text and r.flags["homoglyph_words"] == 1


def test_bullet_wedges_removed():
    assert normalize("C•L•A•I•M your gift").text.startswith("claim")


FIX_VOCAB = {"claim": 3000, "prize": 900, "free": 5000, "your": 9000, "from": 8000, "form": 300, "cliam": 30, "__total__": 1}


def test_spellfix_restores_one_edit_typos():
    from smsguard.normalize import SpellFixer
    r = normalize("Cliam your przie now", fixer=SpellFixer(FIX_VOCAB))
    assert r.text.startswith("claim your prize") and r.flags["typos_fixed"] == 2


def test_spellfix_keeps_real_words():
    from smsguard.normalize import SpellFixer
    assert "form" in normalize("fill in the form", fixer=SpellFixer(FIX_VOCAB)).text


def test_spellfix_never_touches_protected_ham_words():
    from smsguard.normalize import SpellFixer
    v = {"down": 5000, "__total__": 1}
    assert "drown" in normalize("he can drown a fish", fixer=SpellFixer(v, protected={"drown"})).text
    assert "down" in normalize("he can drown a fish", fixer=SpellFixer(v)).text  # without protection it would be wrong


def test_institution_domains_are_trusted():
    from smsguard.urls import analyse
    assert analyse("staff.science.nus.edu.sg/~x/teaching").trusted
    assert analyse("lms.unimelb.edu.au").trusted


@pytest.mark.parametrize("url,verdict", [
    ("docs.google.com/document/d/abc", "trusted domain"), ("ato.gov.au", "trusted domain"),
    ("auspost-redelivery.com", "impersonates auspost"), ("paypa1-secure.xyz/login", "impersonates paypal"),
    ("netfIix-billing.com", "impersonates netflix"), ("win-cash.top/x", "high-risk domain"),
    ("bit.ly/x", "link shortener (destination hidden)"),
])
def test_url_verdicts(url, verdict):
    from smsguard.urls import analyse
    assert analyse(url).verdict == verdict


@pytest.mark.parametrize("text,is_url", [
    ("renovated.so", False), ("going.im", False), ("LIKING.BE", False), ("199.To", False), ("psam.cn", False),
    ("wa.me/123", True), ("kredt.be/2U1Q", True), ("cancel-payments.co", True), ("gift-claims.top", True), ("youtu.be/x", True),
])
def test_missing_space_is_not_a_link(text, is_url):
    from smsguard.normalize import is_real_url
    assert is_real_url(text) == is_url


def test_trusted_links_are_neutralised():
    r = normalize("Slides here: docs.google.com/presentation/d/1")
    assert "urltoken" not in r.text and r.flags["n_urls"] == 0 and r.flags["n_trusted_urls"] == 1


def test_entities():
    r = normalize("Pay at bit.ly/x1 or call +44 7911 123456, $50 fee")
    assert r.flags["n_short_urls"] == 1 and r.flags["n_phones"] == 1 and r.flags["n_money"] == 1
    assert "urltoken" in r.text and "phonetoken" in r.text and "moneytoken" in r.text
