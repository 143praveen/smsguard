"""Offline URL reputation for SMS links.

No network calls: everything is decided from the URL string, so it works on a
phone and is reproducible. Live signals (domain age, threat feeds) are future work.

Design note: in the Super SMS Dataset only 90 of ~40.8K legitimate messages
contain a link, versus ~7.9K spam messages. A classifier therefore learns
"link => spam". Trusted links are neutralised (they stop counting as evidence)
rather than learned, because there is too little legitimate link data to learn from.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Widely used legitimate domains (chosen from public popularity lists, not from our test messages)
TRUSTED_DOMAINS = {
    "google.com", "youtube.com", "youtu.be", "gmail.com", "zoom.us", "microsoft.com", "office.com", "live.com",
    "outlook.com", "apple.com", "icloud.com", "amazon.com", "amazon.co.uk", "amazon.in", "amazon.com.au",
    "facebook.com", "instagram.com", "whatsapp.com", "linkedin.com", "github.com", "wikipedia.org",
    "dropbox.com", "spotify.com", "netflix.com", "paypal.com", "twitter.com", "x.com", "reddit.com",
    "uber.com", "airbnb.com", "booking.com", "yahoo.com", "bing.com", "duckduckgo.com", "slack.com",
    "notion.so", "canva.com", "teams.microsoft.com", "maps.app.goo.gl", "flipkart.com", "ebay.com",
    "hotmail.com", "nhs.uk", "bbc.co.uk", "bbc.com", "nytimes.com", "theguardian.com",
}
TRUSTED_SUFFIXES = (".gov", ".gov.uk", ".gov.au", ".gov.in", ".gov.sg", ".edu", ".edu.au", ".ac.uk", ".ac.in", ".nhs.uk", ".mil")
# Free/cheap TLDs heavily abused for phishing (Spamhaus / Interisle reports)
RISKY_TLDS = {"xyz", "top", "click", "link", "site", "online", "info", "tk", "ml", "ga", "cf", "gq", "icu", "buzz",
              "shop", "live", "vip", "work", "support", "rest", "fit", "monster", "cyou", "sbs", "cfd", "ru", "cn", "cc", "ws"}
SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "rb.gy", "cutt.ly", "shorturl.at",
    "tiny.cc", "rebrand.ly", "t.ly", "bl.ink", "s.id", "v.gd", "olo.me", "g.tm", "y77.us", "tny.im", "short.gy",
}
# Brands commonly impersonated in smishing. Short or ambiguous names are matched only as whole labels.
BRANDS = {
    "paypal", "amazon", "apple", "icloud", "netflix", "microsoft", "google", "facebook", "whatsapp", "instagram",
    "linkedin", "dhl", "fedex", "usps", "royalmail", "auspost", "hmrc", "mygov", "medicare", "centrelink", "irs",
    "chase", "wellsfargo", "citibank", "hsbc", "barclays", "lloyds", "natwest", "santander", "commbank", "westpac",
    "anz", "sbi", "hdfc", "icici", "paytm", "telstra", "optus", "vodafone", "verizon", "tmobile", "evri", "hermes",
    "coinbase", "binance", "steam", "ups", "ato", "nab", "linkt", "etoll", "ezpass", "dvla",
}
OFFICIAL = {b: {f"{b}.com"} for b in BRANDS}
OFFICIAL.update({"auspost": {"auspost.com.au"}, "royalmail": {"royalmail.com"}, "hmrc": {"hmrc.gov.uk"}, "mygov": {"my.gov.au"},
                 "commbank": {"commbank.com.au"}, "westpac": {"westpac.com.au"}, "anz": {"anz.com", "anz.com.au"},
                 "nab": {"nab.com.au"}, "ato": {"ato.gov.au"}, "telstra": {"telstra.com.au", "telstra.com"},
                 "optus": {"optus.com.au"}, "sbi": {"onlinesbi.sbi", "sbi.co.in"}, "dvla": {"gov.uk"}, "usps": {"usps.com"},
                 "irs": {"irs.gov"}, "medicare": {"medicare.gov", "servicesaustralia.gov.au"}, "linkt": {"linkt.com.au"}})
SUSPICIOUS_WORDS = ("login", "log-in", "signin", "verify", "verification", "secure", "account", "update", "confirm", "wallet",
                    "unlock", "suspend", "refund", "redeliver", "redelivery", "parcel", "delivery", "billing", "payment",
                    "claim", "prize", "bonus", "reward", "gift", "toll", "fine", "penalty", "track")
TWO_LEVEL_SUFFIXES = {"co.uk", "org.uk", "gov.uk", "ac.uk", "com.au", "net.au", "org.au", "gov.au", "edu.au", "co.in",
                      "gov.in", "ac.in", "co.nz", "com.sg", "gov.sg", "co.za", "com.br", "co.jp"}
# Second-level labels reserved for institutions in most country domains: *.edu.sg, *.gov.nz, *.ac.jp ...
INSTITUTION_LABELS = {"edu", "gov", "ac", "mil"}
LEET = str.maketrans("013457@$", "oieastas")
IP_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


@dataclass
class UrlInfo:
    url: str
    host: str
    registered: str
    trusted: bool = False
    shortener: bool = False
    risky_tld: bool = False
    ip_host: bool = False
    brand_spoof: str | None = None
    suspicious_words: list = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.brand_spoof:
            return f"impersonates {self.brand_spoof}"
        if self.trusted:
            return "trusted domain"
        if self.ip_host or self.risky_tld:
            return "high-risk domain"
        if self.shortener:
            return "link shortener (destination hidden)"
        return "unknown domain"


def _edit1(a: str, b: str) -> bool:
    """True if a and b differ by exactly one insert, delete, substitution or adjacent swap."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
    s, l = (a, b) if len(a) < len(b) else (b, a)
    return any(l[:i] + l[i + 1:] == s for i in range(len(l)))


def registered_domain(host: str) -> str:
    parts = host.split(".")
    if len(parts) >= 3 and (".".join(parts[-2:]) in TWO_LEVEL_SUFFIXES or (parts[-2] in INSTITUTION_LABELS | {"com", "co", "org", "net"} and len(parts[-1]) == 2)):
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def analyse(url: str) -> UrlInfo:
    u = re.sub(r"\s+", "", url).lower()
    u = re.sub(r"^[a-z]+://", "", u)
    host = re.split(r"[/?#:]", u, maxsplit=1)[0].strip(".,;!)(")
    host = host[4:] if host.startswith("www.") else host
    path = u[len(host):]
    reg = registered_domain(host)
    info = UrlInfo(url=url, host=host, registered=reg)
    info.ip_host = bool(IP_RE.match(host))
    parts = host.split(".")
    institution = len(parts) >= 3 and parts[-2] in INSTITUTION_LABELS and len(parts[-1]) == 2  # nus.edu.sg, ato.gov.au
    info.trusted = (reg in TRUSTED_DOMAINS or host in TRUSTED_DOMAINS or host.endswith(TRUSTED_SUFFIXES) or institution) and not info.ip_host
    info.shortener = reg in SHORTENERS or host in SHORTENERS
    info.risky_tld = host.rsplit(".", 1)[-1] in RISKY_TLDS
    info.suspicious_words = [w for w in SUSPICIOUS_WORDS if w in u]
    # Brand impersonation: a brand (or a 1-edit / leetspeak variant of it) appears in the host,
    # but the registered domain is not one of the brand's official domains.
    labels = [t for t in re.split(r"[.\-_]", host) if t]
    for brand in BRANDS:
        official = OFFICIAL.get(brand, set())
        if reg in official or host in official or info.trusted:
            continue
        for lab in labels:
            folded = lab.translate(LEET)
            if lab == brand or folded == brand or (len(brand) >= 5 and (_edit1(folded, brand) or brand in folded)):
                info.brand_spoof = brand
                break
        if info.brand_spoof:
            break
    return info
