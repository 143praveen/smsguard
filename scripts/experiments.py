"""Robustness experiments E3, E4, E5 and E9, plus the adaptive attacker and a hard-ham stress test.

Run after scripts/train.py (which saves the models these load).

  python scripts/experiments.py sweep      # E3  detection vs perturbation rate  -> figure
  python scripts/experiments.py unseen     # E4  leave-one-attack-out + paper-attacks-only training
  python scripts/experiments.py cv         # E5  5-fold stratified CV, mean +/- std
  python scripts/experiments.py adaptive   #     greedy black-box attacker that queries the model
  python scripts/experiments.py hardham    #     false positives on tricky legitimate messages
  python scripts/experiments.py timing     # E9  latency, throughput, model size
  python scripts/experiments.py all
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smsguard.attacks import ALL_ATTACKS, ATTACKS, NEW_ATTACKS, PAPER_ATTACKS, WORD_RE, attack  # noqa: E402
from smsguard.experiment import SEED, augment, evaluate, load_dataset, split  # noqa: E402
from smsguard.model import build_baseline, build_smsguard, spam_keywords  # noqa: E402

OUT = ROOT / "results"
FIG = OUT / "figures"
MAIN = ["B0_paper_style_svm", "A1_hybrid_no_norm", "A4_no_norm_advtrain", "A3_hybrid_norm_meta",
        "A5_no_spellfix_advtrain", "A6_no_urlrep_advtrain", "SMSGuard"]


def save(name, obj, md_lines):
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=2))
    (OUT / f"{name}.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print("\n".join(md_lines))


def data():
    df = load_dataset()
    X_tr, X_te, y_tr, y_te = split(df)
    kw = spam_keywords(X_tr, list(y_tr))
    return df, X_tr, X_te, y_tr, y_te, kw


def load_models(names=MAIN):
    return {n: joblib.load(ROOT / "models" / f"{n}.joblib")["model"] for n in names}


# ---------------------------------------------------------------- E3
def sweep():
    _, _, X_te, _, y_te, kw = data()
    spam = [t for t, y in zip(X_te, y_te) if y == 1]
    rng = random.Random(SEED)
    spam = rng.sample(spam, min(2000, len(spam)))
    models = load_models()
    rates = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    res = {m: {a: {} for a in ALL_ATTACKS} for m in models}
    for a in ALL_ATTACKS:
        for r in rates:
            texts = [attack(t, a, kw, rate=r, seed=SEED) for t in spam]
            for m, model in models.items():
                res[m][a][str(r)] = round(float(model.predict(texts).mean()), 4)
        print(f"sweep {a} done")
    md = ["# E3: spam detection rate vs share of words perturbed (2,000 test spam)", ""]
    for a in ALL_ATTACKS:
        md += [f"### {a}", "", "| Model | " + " | ".join(f"{int(r*100)}%" for r in rates) + " |", "|---|" + "---|" * len(rates)]
        md += [f"| {m} | " + " | ".join(f"{res[m][a][str(r)]:.3f}" for r in rates) + " |" for m in models] + [""]
    save("e3_sweep", {"rates": rates, "results": res}, md)
    plot_sweep(res, rates)


def plot_sweep(res, rates):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    attacks = ALL_ATTACKS
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), sharex=True, sharey=True)
    colors = {"B0_paper_style_svm": "#c62f3b", "A1_hybrid_no_norm": "#e08a2b", "A4_no_norm_advtrain": "#8e6bbf",
              "A3_hybrid_norm_meta": "#3a86c8", "SMSGuard": "#1f7a4d"}
    for ax, a in zip(axes.flat, attacks):
        for m, d in res.items():
            ax.plot([r * 100 for r in rates], [d[a][str(r)] for r in rates], marker="o", ms=3, lw=1.8,
                    color=colors.get(m), label=m)
        ax.set_title(a)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=.3)
    for ax in axes[1]:
        ax.set_xlabel("% of words perturbed")
    for ax in axes[:, 0]:
        ax.set_ylabel("spam detection rate")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "e3_robustness_curves.png", dpi=200)
    print(f"figure -> {FIG / 'e3_robustness_curves.png'}")


# ---------------------------------------------------------------- E4
def unseen():
    """Adversarially train WITHOUT an attack, then test on it. Normalisation must carry the defence."""
    _, X_tr, X_te, y_tr, y_te, kw = data()
    spam = [t for t, y in zip(X_te, y_te) if y == 1]
    singles = list(ATTACKS)
    settings = {f"without_{a}": [x for x in singles if x != a] for a in singles}
    settings["paper_attacks_only"] = PAPER_ATTACKS
    tests = singles + ["hybrid"]
    attacked = {a: [attack(t, a, kw, rate=0.3, seed=SEED) for t in spam] for a in tests}
    res = {}
    for name, train_attacks in settings.items():
        for variant, build in (("SMSGuard", build_smsguard),
                               ("no_norm_advtrain", lambda: build_smsguard(use_meta=False, normalize_text=False))):
            Xa, ya = augment(X_tr, y_tr, kw, attacks=train_attacks)
            t0 = time.time()
            model = build().fit(Xa, ya)
            r = {"train_attacks": train_attacks, "clean": evaluate(model, X_te, y_te),
                 "detection": {a: round(float(model.predict(attacked[a]).mean()), 4) for a in tests}}
            res.setdefault(name, {})[variant] = r
            print(f"{name:22s} {variant:17s} {r['detection']}  ({time.time() - t0:.0f}s)")
    md = ["# E4: generalisation to attacks never seen in adversarial training", "",
          "Each row trains with every attack except the one named (or only the base paper's three attacks), "
          "then measures detection of the held-out attack. `no_norm` = same model without normalisation.", "",
          "| Held-out attack | SMSGuard | no_norm + adv. training |", "|---|---|---|"]
    for a in singles:
        r = res[f"without_{a}"]
        md.append(f"| {a} | {r['SMSGuard']['detection'][a]:.3f} | {r['no_norm_advtrain']['detection'][a]:.3f} |")
    p = res["paper_attacks_only"]
    md += ["", "## Trained only on the base paper's attacks (spacing, homoglyph, charswap), tested on new ones", "",
           "| Attack | SMSGuard | no_norm + adv. training |", "|---|---|---|"]
    md += [f"| {a} | {p['SMSGuard']['detection'][a]:.3f} | {p['no_norm_advtrain']['detection'][a]:.3f} |" for a in NEW_ATTACKS + ["hybrid"]]
    save("e4_unseen_attacks", res, md)


# ---------------------------------------------------------------- E5
def cv(folds=5):
    df = load_dataset()
    X, y = np.array(df.text.tolist(), dtype=object), df.label.values
    builders = {"B0_paper_style_svm": (build_baseline, False), "A1_hybrid_no_norm": (lambda: build_smsguard(use_meta=False, normalize_text=False), False),
                "SMSGuard": (build_smsguard, True)}
    rows = {n: [] for n in builders}
    for k, (tr, te) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=SEED).split(X, y)):
        kw = spam_keywords(X[tr], list(y[tr]))
        spam = [t for t, lab in zip(X[te], y[te]) if lab == 1]
        hyb = [attack(t, "hybrid", kw, rate=0.3, seed=SEED) for t in spam]
        Xa, ya = augment(list(X[tr]), y[tr], kw, seed=SEED + k)
        for n, (build, adv) in builders.items():
            model = build().fit(*(Xa, ya) if adv else (list(X[tr]), y[tr]))
            r = evaluate(model, list(X[te]), y[te])
            r["hybrid_detection"] = round(float(model.predict(hyb).mean()), 4)
            rows[n].append(r)
        print(f"fold {k + 1}/{folds} done")
    keys = ["accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "hybrid_detection"]
    summary = {n: {k: {"mean": float(np.mean([r[k] for r in rs])), "std": float(np.std([r[k] for r in rs]))} for k in keys} for n, rs in rows.items()}
    md = [f"# E5: {folds}-fold stratified cross-validation (mean ± std)", "", "| Model | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    md += [f"| {n} | " + " | ".join(f"{s[k]['mean']:.4f} ± {s[k]['std']:.4f}" for k in keys) + " |" for n, s in summary.items()]
    save("e5_cv", {"folds": rows, "summary": summary}, md)


# ---------------------------------------------------------------- adaptive attacker
CHAR_OPS = ["charswap", "typo", "homoglyph", "spacing", "leet", "zero_width", "emoji_wedge", "fullwidth"]


def greedy_attack(model, text, budget_frac=0.3, ops=CHAR_OPS):
    """Black-box greedy attack: each round, try every op on every untouched word, keep the change that
    lowers the spam score most; stop once the model says ham or the word budget is spent."""
    words = list(WORD_RE.finditer(text))
    if not words:
        return text, False, 0, []
    budget = max(1, round(len(words) * budget_frac))
    spans = [(m.start(), m.end()) for m in words]
    tokens = [m.group(0) for m in words]
    touched, used = set(), []

    def render(toks):
        out, last = [], 0
        for (s, e), t in zip(spans, toks):
            out.append(text[last:s]); out.append(t); last = e
        out.append(text[last:])
        return "".join(out)

    rng = random.Random(text)
    for _ in range(budget):
        cands, meta = [], []
        for i, w in enumerate(tokens):
            if i in touched:
                continue
            for op in ops:
                t2 = tokens.copy()
                t2[i] = ATTACKS[op](w, rng)
                cands.append(render(t2)); meta.append((i, op, t2[i]))
        if not cands:
            break
        s = model.decision_function(cands)  # < 0 means "ham" for both LinearSVC and LogisticRegression
        j = int(np.argmin(s))
        i, op, new = meta[j]
        tokens[i] = new; touched.add(i); used.append(op)
        if s[j] < 0:
            return render(tokens), True, len(used), used
    return render(tokens), False, len(used), used


def adaptive(n=300):
    _, _, X_te, _, y_te, _ = data()
    models = load_models(["B0_paper_style_svm", "A1_hybrid_no_norm", "A4_no_norm_advtrain", "A5_no_spellfix_advtrain", "SMSGuard"])
    rng = random.Random(SEED)
    pool = [t for t, y in zip(X_te, y_te) if y == 1]
    res = {}
    for name, model in models.items():
        detected = [t for t in pool if model.predict([t])[0] == 1]
        sample = rng.sample(detected, min(n, len(detected)))
        t0 = time.time()
        out = [greedy_attack(model, t) for t in sample]
        ops = {}
        for _, ok, _, used in out:
            if ok:
                for o in used:
                    ops[o] = ops.get(o, 0) + 1
        succ = [o for o in out if o[1]]
        res[name] = {"n": len(sample), "evasion_success_rate": round(len(succ) / len(sample), 4),
                     "mean_words_changed_when_successful": round(float(np.mean([o[2] for o in succ])), 2) if succ else None,
                     "ops_used_in_successful_attacks": dict(sorted(ops.items(), key=lambda x: -x[1])),
                     "examples": [o[0] for o in succ[:5]], "seconds": round(time.time() - t0, 1)}
        print(f"{name:22s} success={res[name]['evasion_success_rate']:.3f} ops={res[name]['ops_used_in_successful_attacks']}")
    md = ["# Adaptive black-box attacker (greedy, queries the model, may change up to 30% of words)", "",
          f"Sample: {n} test spam messages each model initially detects.", "",
          "| Model | Evasion success | Words changed (successful) | Most-used tricks |", "|---|---|---|---|"]
    for m, r in res.items():
        top = ", ".join(f"{k} ({v})" for k, v in list(r["ops_used_in_successful_attacks"].items())[:3])
        md.append(f"| {m} | {r['evasion_success_rate']:.3f} | {r['mean_words_changed_when_successful']} | {top} |")
    save("adaptive_attack", res, md)


# ---------------------------------------------------------------- hard ham
HARD_HAM = [
    # transactional / service messages that look spammy
    "Your OTP for login is 482913. Do not share it with anyone.",
    "Your verification code is 771204. It expires in 10 minutes.",
    "Your appointment with Dr Patel is confirmed for Tue 14 May at 10:30am. Reply C to cancel.",
    "Your parcel has been delivered to your front door. Thanks for shopping with us.",
    "Reminder: your electricity bill of $84.20 is due on 28 Feb. Pay in the app as usual.",
    "Your table for 4 at Luigi's is booked for 7pm tonight. See you soon!",
    "Flight QF12 to Sydney is now boarding at gate 34.",
    "Your prescription is ready for collection at the pharmacy.",
    "Payment of $25.00 received. Thank you.",
    "Your Uber is arriving in 3 minutes. Look for a white Toyota.",
    # personal, with emoji
    "Happy birthday!! 🎉🎂 Have the best day ❤️",
    "On my way 🚗 be there in 10",
    "Thanks so much for dinner last night 😊",
    "Lol that's hilarious 😂😂",
    "Good luck with the exam tomorrow 👍 you've got this 💪",
    "Can you grab milk and bread on the way home? 🙏",
    # non-Latin names and words used normally (whole words, no mixing inside a word)
    "Dinner with Дмитрий and Ольга on Friday, are you coming?",
    "Σοφία says hi, she landed safely",
    "Meeting moved to 3pm, Анна will send the slides",
    "Just finished reading Tolstoy in Russian: Война и мир. Amazing.",
    "My grandma says नमस्ते to everyone, she loved the photos",
    "We had biryani at the place near the station, आप कब आ रहे हो?",
    "Café with Zoë and José at 5? Señor Díaz recommended it",
    "Got the 東京 tickets! Leaving on the 12th",
    # spaced/abbreviated legitimately
    "I a m so tired today",
    "OMG did you see the game? GG",
    "U.S.A trip is booked for July",
    "lunch at 1pm on the 2nd floor, room B4",
    "Can u call me back asap pls",
    "Need 2 tickets for sat, can you get them?",
    "Call me on 0412 345 678 when you're free",
    "My new email is sam.lee@example.com",
    "Here's the link to the doc: docs.google.com/document/d/abc",
    "Check this video youtube.com/watch?v=dQw4w9WgXcQ haha",
    "The meeting link is zoom.us/j/123456789",
    "Mum I'm at the station, can you pick me up?",
    "Hi, it's Sarah from the dentist, just confirming tomorrow at 9.",
    "Congrats on the new job!! So proud of you",
    "Free tonight? Want to watch a movie?",
    "Win or lose, great effort at the match today",
]


# Written together with the URL-reputation feature, so results on this set are optimistic; it is reported
# separately from HARD_HAM (which was written before any URL work).
LINK_HAM = [
    "Here are the slides from today: docs.google.com/presentation/d/1x2y3z",
    "Join the standup here: zoom.us/j/9876543210",
    "Saw this and thought of you youtu.be/abc123",
    "Your tax return status is available at ato.gov.au, log in with myGov as usual",
    "Repo is up at github.com/sam/thesis-code, have a look",
    "The reading for week 3 is on en.wikipedia.org/wiki/Spam_(electronic)",
    "Photos from the trip: dropbox.com/sh/abc/def",
    "Booked the hotel, details on booking.com in my account",
    "Bus timetable changed, see transport.nsw.gov.au for details",
    "Lecture recording is up at the uni site: lms.unimelb.edu.au",
]


def hardham():
    _, _, X_te, _, y_te, _ = data()
    models = load_models()
    ham = [t for t, y in zip(X_te, y_te) if y == 0]
    rng = random.Random(SEED)
    emoji = [t + " " + rng.choice(["🙂", "👍", "😂", "❤️", "🎉"]) for t in ham]
    names = [t + " " + rng.choice(["- Дмитрий", "- Σοφία", "- राहुल", "- 李明", "- Zoë"]) for t in ham]
    from smsguard.normalize import URL_RE
    ham_url = [t for t in ham if URL_RE.search(t)]
    spam_url = [t for t, y in zip(X_te, y_te) if y == 1 and URL_RE.search(t)]
    sets = {f"curated_hard_ham (n={len(HARD_HAM)})": HARD_HAM, f"curated_link_ham (n={len(LINK_HAM)})": LINK_HAM,
            "test_ham": ham, f"test_ham with link (n={len(ham_url)})": ham_url,
            "test_ham + emoji": emoji, "test_ham + non-Latin name": names}
    res = {m: {s: round(float(model.predict(texts).mean()), 4) for s, texts in sets.items()} for m, model in models.items()}
    spam_link_recall = {m: round(float(model.predict(spam_url).mean()), 4) for m, model in models.items()}
    flagged = {m: [t for t in HARD_HAM + LINK_HAM if model.predict([t])[0] == 1] for m, model in models.items()}
    md = ["# False-positive stress test (share of legitimate messages wrongly flagged as spam)", "",
          "`curated_link_ham` was written alongside the URL-reputation feature, so treat it as optimistic.", "",
          "| Model | " + " | ".join(sets) + f" | spam-with-link recall (n={len(spam_url)}) |", "|---|" + "---|" * (len(sets) + 1)]
    md += [f"| {m} | " + " | ".join(f"{r[s]:.3f}" for s in sets) + f" | {spam_link_recall[m]:.3f} |" for m, r in res.items()]
    md += ["", "## Curated messages wrongly flagged by SMSGuard", ""] + [f"- {t}" for t in flagged["SMSGuard"]]
    save("hard_ham", {"fpr": res, "spam_with_link_recall": spam_link_recall, "flagged_curated": flagged}, md)


# ---------------------------------------------------------------- E9
def timing():
    _, _, X_te, _, _, _ = data()
    models = load_models()
    sample = X_te[:500]
    res = {}
    for m, model in models.items():
        t0 = time.perf_counter()
        for t in sample[:200]:
            model.predict([t])
        single = (time.perf_counter() - t0) / 200 * 1000
        t0 = time.perf_counter()
        model.predict(sample)
        batch = len(sample) / (time.perf_counter() - t0)
        res[m] = {"ms_per_message": round(single, 2), "messages_per_second_batch": round(batch), "size_mb": round((ROOT / "models" / f"{m}.joblib").stat().st_size / 1e6, 1)}
    md = ["# E9: inference cost on CPU (this machine, single process)", "", "| Model | ms / message | msgs/s (batch) | Model file (MB) |", "|---|---|---|---|"]
    md += [f"| {m} | {r['ms_per_message']} | {r['messages_per_second_batch']} | {r['size_mb']} |" for m, r in res.items()]
    save("e9_timing", res, md)


if __name__ == "__main__":
    jobs = {"sweep": sweep, "unseen": unseen, "cv": cv, "adaptive": adaptive, "hardham": hardham, "timing": timing}
    which = sys.argv[1:] or ["all"]
    for j in (list(jobs) if which == ["all"] else which):
        print(f"\n===== {j} =====")
        jobs[j]()
