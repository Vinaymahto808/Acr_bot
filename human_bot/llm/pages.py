"""'Page dikhao': which website page the customer wants opened, from the persona's page catalog.

Pure functions, no network. The server (server/app.py /chat) calls:

    page = find_page(user_text, persona, kb_hits)    # -> {"path", "title", "exact"} or None
    line = page_line(page)                           # what Neha says while the page opens

Rules:
  * Only a request to SEE something opens a page: "insurance page dikhao", "offers dikha do", "open the contact
    page". "Bonnet khol ke dekha" or "brake khul gaya" never navigates: open/khol need the word page/website.
  * Only paths from persona["pages"]["list"] or a page of the persona's own website found in the knowledge base.
    Never a URL the customer typed.
  * Nothing matches but the customer clearly asked for a page ("parts ka page dikhao"): the fallback page
    opens and Neha says honestly that there is no separate page for it.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

SHOW_VERB = re.compile(r"\b(dikha\w*|dikhla\w*|show|le\s*chalo|le\s*chaliye|le\s*jao)\b", re.I)
OPEN_VERB = re.compile(r"\b(khol\w*|open\w*|kholiye|jana\s*hai|jaana\s*hai|chalo|navigate|visit)\b", re.I)
PAGE_WORD = re.compile(r"\b(page|pej|paij|section|website|site|webpage|link)\b", re.I)


OVERVIEW = re.compile(
    r"\b(overview|walk\s*(me\s*)?through|introduction|intro)\b|\b(website|site)\s+tour\b|\btour\s+(of|karao|karwao|karaiye)\b|"
    r"\b(about|explain|describe)\s+(this|the|your)?\s*(website|site|company|business|acr|workshop)\b|"
    r"\b(website|site|company|acr|workshop)\s+(ke|ki|ka)\s+(baare|bare)\b|"
    r"\bwhat\s+(all\s+)?(services|do\s+you\s+(do|offer)|you\s+offer)\b|"
    r"\b(kya|kaun)\s*(kya|kaun)?\s*(services?|kaam)\b|\bkya\s+kya\s+karte\b|\baap\s*(log)?\s+kya\s+karte\b|"
    r"\b(sari|saari|sab|all)\s+(services|sevayen)\b", re.I)


def wants_overview(text: str) -> bool:
    """'Can you walk through this website', 'website ka overview do', 'what services do you offer', 'aap log kya
    karte ho': a general introduction, answered with the persona's fixed overview (and its page opens)."""
    return bool(OVERVIEW.search(text or ""))


def overview(text: str, persona: dict) -> dict | None:
    """-> {"say": [sentences], "path", "title"} from persona["pages"]["overview"], or None."""
    ov = ((persona or {}).get("pages") or {}).get("overview") or {}
    if not ov.get("say") or not wants_overview(text):
        return None
    say = ov["say"] if isinstance(ov["say"], list) else [ov["say"]]
    return {"say": say, "path": ov.get("path") or "", "title": "Services"}


def wants_page(text: str) -> bool:
    t = text or ""
    if SHOW_VERB.search(t):
        return True
    return bool(OPEN_VERB.search(t) and PAGE_WORD.search(t))


def _has(text: str, phrase: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(phrase.lower()) + r"(?![a-z0-9])", text) is not None


def match_catalog(text: str, pages: list[dict]) -> dict | None:
    """Best catalog entry: the most (and longest) keyword hits; ties go to the earlier entry."""
    low = " " + (text or "").lower() + " "
    best, best_score = None, 0
    for pg in pages:
        hits = [w for w in pg.get("words", []) if _has(low, w)]
        score = len(hits) * 100 + sum(len(w) for w in hits)
        if score > best_score:
            best, best_score = pg, score
    return best


STOP = {"page", "pej", "paij", "site", "website", "webpage", "section", "link", "wala", "wali", "wale", "dikhao", "dikha",
         "dikhaiye", "show", "open", "kholo", "khol", "mujhe", "please", "aapka", "aapki", "aapke", "apna", "apni", "the",
         "and", "kya", "hai", "hain", "karo", "kar", "dijiye", "chalo", "chaliye"}
# the site's default <title>, carried by pages that have none of their own: never shown as a page name
GENERIC_TITLE = re.compile(r"used\s*&\s*out-of-warranty|^auto car repair$", re.I)


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", (text or "").lower()) if w not in STOP}


def from_kb(hits: list[dict] | None, site: str, text: str = "") -> dict | None:
    """A real page of this website from the knowledge results (no Q&A anchors, no other hosts) whose URL or
    title carries a word the customer said. A loosely related result ("parts" -> /windshield-repair) is
    not a page they asked for, so it is never opened."""
    host = urlparse(site or "").netloc.lower().removeprefix("www.")
    asked = _words(text)
    for h in hits or []:
        u = urlparse(h.get("url") or "")
        if not host or u.netloc.lower().removeprefix("www.") != host or u.fragment or h.get("page_type") == "qa":
            continue
        title = (h.get("title") or "").split("|")[0].strip()
        if not u.path or u.path == "/":
            continue
        named = _words(u.path.replace("-", " ").replace("/", " ")) | (set() if GENERIC_TITLE.search(title) else _words(title))
        if asked and not any(a in n or n in a for a in asked for n in named if min(len(a), len(n)) >= 4):
            continue
        if not title or GENERIC_TITLE.search(title):
            title = u.path.strip("/").split("/")[-1].replace("-", " ").title()
        return {"path": u.path, "title": title}
    return None


def find_page(text: str, persona: dict, kb_hits: list[dict] | None = None) -> dict | None:
    cfg = persona.get("pages") or {}
    pages = cfg.get("list") or []
    if not pages or not wants_page(text):
        return None
    pg = match_catalog(text, pages)
    if pg:
        return {"path": pg["path"], "title": pg["title"], "exact": True}
    if not PAGE_WORD.search(text or ""):
        return None          # "price dikhao": a question for Neha, not a page
    site = (persona.get("knowledge") or {}).get("website", "")
    kb = from_kb(kb_hits, site, text)
    if kb:
        return {**kb, "exact": True}
    fb = next((p for p in pages if p["path"] == cfg.get("fallback")), None)
    return {"path": fb["path"], "title": fb["title"], "exact": False} if fb else None


# "AC service book karni hai", "denting karwana hai", "brake repair chahiye", "I want a car wash": the customer asks
# for a service to be DONE (not just about it). Then that service's page of the website opens by itself while Neha
# carries on booking (2026-11-24 owner: "jab bhi koi bole ki service book karna, website ka service page open kar de").
# A question ("AC service kitne ka hai?") or a problem description ("brake se awaaz aa rahi hai") never navigates.
BOOK_INTENT = re.compile(
    r"\b(book\w*|appointment|slot|schedule|karwa\w*|karva\w*|karana|karani|karaani|karane|chahiye|chahte|chahti|"
    r"karn[ai]\s+(?:hai|h)|badal\w*|dalwa\w*|check\s+karwa\w*|lagwa\w*|badalwa\w*|change\s+kar\w*|replace\w*|theek\s+kar\w*|thik\s+kar\w*|repair\s+kar\w*|need|want|"
    r"get\s+(it|my|the)?\s*\w*\s*(done|fixed|repaired|serviced))\b", re.I)
QUESTION_ONLY = re.compile(r"\b(kitne|kitna|kitni|price|cost|rate|charges?|kab\s+tak|how\s+much)\b", re.I)


def wants_service(text: str) -> bool:
    t = text or ""
    return bool(BOOK_INTENT.search(t)) and not (QUESTION_ONLY.search(t) and not re.search(r"\bbook\w*\b", t, re.I))


def service_page(text: str, persona: dict) -> dict | None:
    """The service page to open for a booking request, or None. Only catalog entries with "kind": "service"."""
    cfg = (persona or {}).get("pages") or {}
    pages = [p for p in cfg.get("list") or [] if p.get("kind") == "service"]
    if not pages or not wants_service(text):
        return None
    pg = named_service(text, persona)      # a specific job wins over the generic "servicing" words
    if pg:
        return {**pg, "exact": True}
    if wants_some_service(text):           # "car ki servicing karani hai": all services, while Neha asks which one
        allp = next((p for p in cfg.get("list") or [] if p["path"] == "/services"), None)
        if allp:
            return {"path": allp["path"], "title": allp["title"], "exact": True}
    return None


# "car service / servicing karani hai" names no specific job: Neha first asks which (2026-11-24 owner). A plain
# service word matches the generic /car-service page; these words make it a specific periodic service.
GENERIC_PATHS = {"/car-service", "/services"}
PERIODIC = re.compile(r"\b(regular|periodic|general\s+service|oil\s+change|full\s+service|scheduled|free\s+service)\b", re.I)


_NEG_AFTER = re.compile(r"^\W{0,3}(nahi|nahin|nai|not|mat)\b", re.I)


def named_services(text: str, persona: dict) -> list[dict]:
    """Every specific service the turn names, ordered by where it is last mentioned ("AC nahi, battery chahiye" ->
    [battery]; "AC aur battery dono" -> [AC, battery]). Generic "servicing" words are not a service."""
    cfg = (persona or {}).get("pages") or {}
    pages = [p for p in cfg.get("list") or [] if p.get("kind") == "service" and p["path"] not in GENERIC_PATHS]
    low = " " + (text or "").lower() + " "
    found = []
    for pg in pages:
        last, negated, spans = -1, False, []
        for w in pg.get("words", []):
            for m in re.finditer(r"(?<![a-z0-9])" + re.escape(w.lower()) + r"(?![a-z0-9])", low):
                spans.append((m.start(), m.end()))
                if m.start() > last:
                    last, negated = m.start(), bool(_NEG_AFTER.match(low[m.end():]))
        if last >= 0:
            found.append((last, negated, pg, spans))
    # "wheel alignment": the alignment page's phrase covers the brake page's "wheel" -> only the specific one counts
    def covered(f):
        return all(any(o is not f and a0 <= s0 and e0 <= a1 and (a1 - a0) > (e0 - s0) for o in found for a0, a1 in o[3])
                   for s0, e0 in f[3])
    found = [f for f in found if not covered(f)]
    if any(not f[1] for f in found):             # a negated one only drops when something else is named
        found = [f for f in found if not f[1]]
    found.sort(key=lambda f: f[0])
    out, seen = [], set()
    for _, _, pg, _ in found:
        if pg["path"] not in seen:
            seen.add(pg["path"])
            out.append({"path": pg["path"], "title": pg["title"], "about": pg.get("about", "")})
    return out


def named_service(text: str, persona: dict) -> dict | None:
    """The specific service the customer named ({"path", "title"}), or None for none / only a generic "service".
    A specific job always wins over the generic service words ("battery ki servicing" is battery, not servicing)."""
    cfg = (persona or {}).get("pages") or {}
    pages = [p for p in cfg.get("list") or [] if p.get("kind") == "service"]
    named = named_services(text, persona)
    if named:
        return named[-1]                         # the service mentioned last ("AC nahi, battery" -> battery)
    if PERIODIC.search(text or ""):
        reg = next((p for p in pages if "regular" in p["path"]), None)
        if reg:
            return {"path": reg["path"], "title": reg["title"], "about": reg.get("about", "")}
    return None


SERVICE_TALK = re.compile(r"\b(servic\w*|kaam\b|sarvis\w*|repair\w*|maintenance|kaam\s+karwa\w*|theek\s+karwa\w*|thik\s+karwa\w*)\b", re.I)


def wants_some_service(text: str) -> bool:
    """'Car ki servicing karani hai', 'service chahiye', 'repair karwana hai': a service is wanted, which one is open."""
    return bool(SERVICE_TALK.search(text or "")) and wants_service(text)


# Devanagari typing / STT ("मेरा कार का servicing कराना है", "बैटरी की सर्विसिंग"): the matchers above read Roman
# Hinglish, so the common words are mapped to their Roman form and appended before matching (2026-11-24 owner test).
DEVA_WORDS = {
    "सर्विसिंग": "servicing", "सर्विस": "service", "सर्विसेज": "services", "रिपेयर": "repair", "मरम्मत": "repair",
    "कराना": "karana", "करानी": "karani", "कराने": "karane", "करवाना": "karwana", "करवानी": "karwani", "करवाने": "karwane",
    "चाहिए": "chahiye", "चाहिये": "chahiye", "बुक": "book", "बुकिंग": "booking", "अपॉइंटमेंट": "appointment",
    "बैटरी": "battery", "इंजन": "engine", "टायर": "tyre", "ब्रेक": "brake", "एसी": "ac", "क्लच": "clutch",
    "सस्पेंशन": "suspension", "डेंटिंग": "denting", "पेंटिंग": "painting", "पेंट": "paint", "डेंट": "dent",
    "इंश्योरेंस": "insurance", "इन्शुरन्स": "insurance", "बीमा": "bima", "क्लेम": "claim", "वॉश": "wash", "वाश": "wash",
    "धुलाई": "washing", "सफाई": "safai", "कोटिंग": "coating", "सिरेमिक": "ceramic", "पॉलिश": "polish", "लाइट": "light",
    "हेडलाइट": "headlight", "शीशा": "glass", "विंडशील्ड": "windshield", "वाइपर": "wiper", "अलाइनमेंट": "alignment",
    "बैलेंसिंग": "balancing", "वारंटी": "warranty", "गियर": "gear", "रेगुलर": "regular", "पीरियोडिक": "periodic",
    "गाड़ी": "gaadi", "गाडी": "gaadi", "कार": "car", "है": "hai", "की": "ki", "का": "ka", "के": "ke", "मेरा": "mera",
    "मेरी": "meri", "मेरे": "mere", "मुझे": "mujhe", "हमको": "humko", "हमें": "hume", "करना": "karna", "करनी": "karni",
    "ठीक": "theek", "कितना": "kitna", "कितने": "kitne", "कितनी": "kitni", "क्या": "kya",
    "ठंडा": "thanda", "ठंडी": "thandi", "नहीं": "nahi", "नही": "nahi", "तेल": "oil", "ऑयल": "oil", "आवाज़": "awaaz",
    "आवाज": "awaaz", "स्टार्ट": "start", "बदलना": "badalna", "बदलवाना": "badalwana", "लगवाना": "lagwana",
    "हाँ": "haan", "हां": "haan",
    # 2026-11-25 QA: price / time questions and goodbyes in Devanagari ("कितना खर्चा आएगा?" got "didn't catch")
    "खर्चा": "kharcha", "खर्च": "kharch", "खर्चे": "kharche", "आएगा": "aayega", "आयेगा": "aayega", "आएगी": "aayegi",
    "समय": "samay", "टाइम": "time", "लगेगा": "lagega", "लगेगी": "lagegi", "लगता": "lagta", "इसका": "iska", "इसकी": "iski",
    "कब": "kab", "प्राइस": "price", "रेट": "rate", "पैसे": "paise", "धन्यवाद": "dhanyavaad", "शुक्रिया": "shukriya",
    "बाय": "bye", "कैंसल": "cancel", "रहने": "rehne", "दो": "do", "छोड़ो": "chhodo",
    # car names, so the website form gets brand + model ("वैगनआर" was dropped and the form never submitted)
    "स्विफ्ट": "swift", "क्रेटा": "creta", "बलेनो": "baleno", "डिज़ायर": "dzire", "डिजायर": "dzire", "ऑल्टो": "alto",
    "आल्टो": "alto", "वैगनआर": "wagon r", "सिटी": "city", "वेन्यू": "venue", "नेक्सन": "nexon", "ब्रेज़ा": "brezza",
    "ब्रेजा": "brezza", "अर्टिगा": "ertiga", "एर्टिगा": "ertiga", "इनोवा": "innova", "फॉर्च्यूनर": "fortuner", "वर्ना": "verna",
    "सेल्टोस": "seltos", "थार": "thar", "स्कॉर्पियो": "scorpio", "बोलेरो": "bolero", "सोनेट": "sonet", "पंच": "punch",
    "टियागो": "tiago", "अमेज़": "amaze", "अमेज": "amaze", "सेलेरियो": "celerio", "किआ": "kia", "होंडा": "honda",
    "हुंडई": "hyundai", "ह्युंडई": "hyundai", "मारुति": "maruti", "मारुती": "maruti", "टाटा": "tata", "महिंद्रा": "mahindra",
    "टोयोटा": "toyota",
}
# Devanagari spellings that span a space or carry digits: replaced in place (a name like "सुरेश" is untouched)
DEVA_PHRASES = [(r"आई\s?(\d+)", r"i\1"), (r"वैगन\s?आर", "wagon r"), (r"एक्सयूवी\s?(\d+)", r"xuv\1"), (r"ग्रैंड\s?आई\s?10", "grand i10")]
_DEVA_RE = re.compile(r"[ऀ-ॿ]+")


TYPOS = [(r"\b(sevice|servis|sarvis|sarvice|servise|servic|sirvice)(ing)?\b", r"service\2"),
         (r"\b(tayar|tyer|tair|taayar|tyre?s?)\b", "tyre"), (r"\b(batri|battary|baitri|bettery|batery)\b", "battery"),
         (r"\b(brek|braek|break)\b", "brake"), (r"\b(enjan|injan|engin)\b", "engine"), (r"\b(clatch|klach)\b", "clutch"),
         (r"\b(insurence|insurans|inshurance)\b", "insurance"), (r"\b(dainting|denting)\b", "denting"),
         (r"\b(oil\s+change|engine\s+oil)\b", "oil change"),
         # 2026-11-25 QA: "sarvis krni h gadi ki" got "didn't catch"
         (r"\bkrni\b", "karni"), (r"\bkrna\b", "karna"), (r"\bkrwani\b", "karwani"), (r"\bkrwana\b", "karwana"),
         (r"\bgadi\b", "gaadi"), (r"\brukh\b", "ruk"), (r"\bgya\b", "gaya"), (r"\bgyi\b", "gayi"), (r"\bgai\b", "gayi"),
         (r"\bmai\b", "mein"), (r"\b(h|hy|hai+)(?=\s*[?.!,]*\s*$)", "hai"), (r"\bkr\s+do\b", "kar do")]


def with_roman(text: str) -> str:
    """The text with common typos fixed, plus a Roman rendering of its known Devanagari words, for the matchers."""
    t = text or ""
    for rx, to in TYPOS:
        t = re.sub(rx, to, t, flags=re.I)
    for rx, to in DEVA_PHRASES:
        t = re.sub(rx, to, t)
    words = _DEVA_RE.findall(t)
    if not words:
        return t
    return t + " " + " ".join(DEVA_WORDS.get(w, "") for w in words if DEVA_WORDS.get(w))


def about_line(page: dict) -> str:
    """What the service covers, for Neha to say while its page is open ("" if the catalog has no line)."""
    return str((page or {}).get("about") or "")


def page_line(page: dict) -> str:
    if page.get("exact"):
        return f"Ji, ye dekhiye, {page['title']} ka page aapke saamne khul gaya hai. Ismein kuch poochna ho to boliye."
    return (f"Iska alag page website par nahi hai, isliye main {page['title']} ka page khol rahi hoon. "
            "Aap bataiye kya chahiye, main yahin bata deti hoon.")
