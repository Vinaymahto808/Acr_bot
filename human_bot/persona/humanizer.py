"""Post-processing that turns an LLM reply into something a human would actually say.

Everything here is deterministic and cheap (no model call), so it runs in the 20 ms
between token stream end and TTS start. This is where most "robotic" tells die:
markdown, written numbers, 90-word monologues, stacked questions.
"""

from __future__ import annotations

import re

_MARKUP = [
    (re.compile(r"```[\s\S]*?```"), ""),
    (re.compile(r"^\s*#{1,6}\s*", re.M), ""),
    (re.compile(r"\*\*(.+?)\*\*"), r"\1"),
    (re.compile(r"\*(.+?)\*"), r"\1"),
    (re.compile(r"^\s*[-*•]\s+", re.M), ""),
    (re.compile(r"^\s*\d+[.)]\s+", re.M), ""),
    (re.compile(r"\[(.*?)\]\((.*?)\)"), r"\1"),
    (re.compile(r"[`~>|]"), ""),
    (re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]"), ""),
]

_BANNED_OPENERS = re.compile(
    r"^(certainly|sure thing|of course|absolutely|great question|that'?s a great question|"
    r"as an ai( language model)?|i'?d be happy to|i am happy to help)[!,.\s]*",
    re.I,
)

# 16,46,000 / 1,23,45,678 -> Indian spoken form
_INDIAN_GROUP = re.compile(r"\b(\d{1,2}(?:,\d{2})*,\d{3})\b")
# boundary before "rs" so "filters, brakes" is not money; the amount must start with a digit; the
# optional unit owns its leading space so a following word keeps its own ("rupees lagega")
_CURRENCY = re.compile(r"(?<![A-Za-z])(?:₹|rs\.?|inr)\s*(\d[\d,]*(?:\.\d+)?)(?:\s*(lakh|lac|cr|crore|k|thousand)\b)?", re.I)
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")

# Hindi verbs agree with the speaker. Neha is female, but LLMs drift into masculine first person
# ("kar deta hoon", "bataunga", "samajh gaya"). Only first-person-only shapes are rewritten, so a third-person
# "brake pad ghis gaya tha" is left alone.
_M_TA_HOON = re.compile(r"\b(\w+?)ta\s+(hoon|hun|hu)\b", re.I)          # karta hoon -> karti hoon
_M_RAHA_HOON = re.compile(r"\braha\s+(hoon|hun|hu)\b", re.I)             # raha hoon  -> rahi hoon
_M_UNGA = re.compile(r"\b(\w+?)(u|oo)nga\b", re.I)                        # bataunga   -> bataungi
_M_SAMAJH = re.compile(r"\b(samajh|samjh)\s+gaya\b", re.I)               # samajh gaya -> samajh gayi
_M_MAIN_GAYA = re.compile(r"\b(main|mai|mein)\s+(\w+\s+)?gaya\b", re.I)  # main bhool gaya -> gayi

# Spoken Delhi NCR Hinglish uses the English word where textbook Hindi has a Sanskrit one. Nobody in
# Gurgaon says "dhanyavaad" to a customer on a call; they say "thank you".
_DELHI = [
    (re.compile(r"\bdhanyava+d\b", re.I), "thank you"),
    (re.compile(r"\bkripya\b", re.I), "please"),
    (re.compile(r"\bavashya\b", re.I), "zaroor"),
    (re.compile(r"\bsahayata\b", re.I), "help"),
    (re.compile(r"\bsamasya\b", re.I), "problem"),
    (re.compile(r"\bsampark\b", re.I), "contact"),
    (re.compile(r"\bkshama\s+(kijiye|karein|karen|kare)\b", re.I), "sorry"),
    (re.compile(r"\bkshama\b", re.I), "maaf"),
    (re.compile(r"\bprateeksha\b", re.I), "wait"),
    (re.compile(r"\bupalabdh\b", re.I), "available"),
    (re.compile(r"\bnishulk\b", re.I), "free"),
    (re.compile(r"\bshulk\b", re.I), "charge"),
    (re.compile(r"\bmulya\b", re.I), "price"),
    (re.compile(r"\bvaa?han\s+(ki|ka|ke|mein|me)\b", re.I), r"gaadi \1"),
    (re.compile(r"\batirikt\b", re.I), "extra"),
    (re.compile(r"\bsuchna\b", re.I), "information"),
    (re.compile(r"\bnamaskar\b", re.I), "namaste"),
]


def _keep_case(src: str, out: str) -> str:
    return out[:1].upper() + out[1:] if src[:1].isupper() else out


def fix_gender(text: str, female: bool = True) -> str:
    """Rewrite masculine first-person verb endings to feminine (no-op for a male persona)."""
    if not female:
        return text
    t = _M_TA_HOON.sub(lambda m: f"{m.group(1)}ti {m.group(2)}", text)
    t = _M_RAHA_HOON.sub(lambda m: f"rahi {m.group(1)}", t)
    t = _M_UNGA.sub(lambda m: f"{m.group(1)}{m.group(2)}ngi", t)
    t = _M_SAMAJH.sub(lambda m: f"{m.group(1)} gayi", t)
    t = _M_MAIN_GAYA.sub(lambda m: f"{m.group(1)} {m.group(2) or ''}gayi", t)
    return t


def delhi_register(text: str) -> str:
    """Textbook Hindi -> the English loanwords Delhi NCR actually speaks."""
    t = text
    for pat, out in _DELHI:
        t = pat.sub(lambda m, out=out: _keep_case(m.group(0), m.expand(out) if "\\" in out else out), t)
    return t


def strip_markup(text: str) -> str:
    out = text
    for pat, rep in _MARKUP:
        out = pat.sub(rep, out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n{2,}", "\n", out)
    return out.strip()


def _indian_words(n: int) -> str:
    """12345678 -> '1 crore 23 lakh 45 thousand 678'."""
    if n == 0:
        return "zero"
    units = [(10_000_000, "crore"), (100_000, "lakh"), (1_000, "thousand")]
    parts: list[str] = []
    rest = n
    for size, name in units:
        if rest >= size:
            q, rest = divmod(rest, size)
            parts.append(f"{q} {name}")
    if rest:
        parts.append(str(rest))
    return " ".join(parts)


def spoken_numbers(text: str) -> str:
    def _curr(m: re.Match) -> str:
        num, unit = m.group(1), (m.group(2) or "").lower()
        num = num.replace(",", "")
        if unit in ("k", "thousand"):
            return f"{num} thousand rupees"
        if unit in ("lakh", "lac"):
            return f"{num} lakh rupees"
        if unit in ("cr", "crore"):
            return f"{num} crore rupees"
        try:
            return f"{_indian_words(int(float(num)))} rupees"
        except ValueError:
            return f"{num} rupees"

    text = _CURRENCY.sub(_curr, text)

    def _grp(m: re.Match) -> str:
        try:
            return _indian_words(int(m.group(1).replace(",", "")))
        except ValueError:
            return m.group(1)

    text = _INDIAN_GROUP.sub(_grp, text)
    text = _PERCENT.sub(r"\1 percent", text)
    text = re.sub(r"\bkmpl\b", "km per litre", text, flags=re.I)
    text = re.sub(r"\bkm/h\b", "km per hour", text, flags=re.I)
    return text


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?।])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


# "…hota hai aur washing…": the joiner starts the next clause, so the head can end before it.
_CLAUSE_JOINERS = ("aur", "to", "lekin", "magar", "kyunki", "isliye", "phir", "jab", "agar", "ya")


LONG_FIRST_WORDS = 12   # a complete first sentence longer than this is still cut at its first clause


def split_first_phrase(text: str, min_words: int = 3, max_words: int = 10) -> tuple[str, str] | None:
    """First speakable clause of the first sentence, for TTS to start before the reply is finished.

    Returns (head, rest) once the buffer holds a natural break: the last comma/semicolon that still
    leaves >= min_words in the head, else the first conjunction inside the window. Returns None while
    there is nothing safe to say yet (too short, no break, or the first sentence is already complete
    and the normal sentence flush will handle it). Never cuts mid-phrase just to hit max_words: a
    wrong pause is worse than a later first clip.
    """
    if not text:
        return None
    sents = split_sentences(text)
    if len(sents) > 1:
        # A complete first sentence normally goes out whole, but a long one (a fast-path or cached answer
        # arrives as one block) is one long fresh Edge clip: 3.0 s to the first byte for an 11 s sentence
        # (2026-11-24, "Periodic service mein air filter, oil filter badalte hain, ..."). Cut its first clause.
        if len(sents[0].split()) <= LONG_FIRST_WORDS:
            return None
        cut = split_first_phrase(sents[0], min_words, max_words)
        if not cut:
            return None
        return cut[0], (cut[1] + " " + " ".join(sents[1:])).strip()
    # last punctuation break inside the window
    cut = None
    for m in re.finditer(r"[,;:]\s+", text):
        head = text[: m.end()].strip()
        if min_words <= len(head.split()) <= max_words:
            cut = m.end()
    if cut:
        head, rest = text[:cut].strip(), text[cut:].strip()
        return (head, rest) if rest else None
    # no punctuation: a joiner may start the rest ("… hota hai aur washing …")
    cut = None
    for m in re.finditer(r"\b(" + "|".join(_CLAUSE_JOINERS) + r")\b", text, re.I):
        head = text[: m.start()].strip()
        if min_words <= len(head.split()) <= max_words:
            cut = m.start()
    if cut is not None:
        head, rest = text[:cut].strip(), text[cut:].strip()
        if head and rest:
            return head, rest
    return None


def enforce_budget(text: str, max_words: int = 40) -> str:
    """Trim to the word budget without cutting mid-sentence and without losing the ask."""
    sents = split_sentences(text)
    if not sents:
        return text
    kept: list[str] = []
    words = 0
    for s in sents:
        n = len(s.split())
        if kept and words + n > max_words:
            break
        kept.append(s)
        words += n
    if len(kept) < len(sents) and not any("?" in s for s in kept):
        q = next((s for s in sents[len(kept):] if "?" in s), None)
        if q:
            kept.append(q)
    return " ".join(kept)


_DETAIL = re.compile(
    r"\b(in detail|detail mein|detail me|details?|explain|elaborate|step by step|poora batao|pura batao|poori baat|"
    r"sab kuch batao|samjhao|samjha do|samjha dijiye|kaise hota hai|kaise karte ho|process kya hai|vistaar)\b",
    re.I,
)


def wants_detail(user_text: str) -> bool:
    """True when the customer asked for an explanation, so the turn may use the larger word budget
    (persona.conversation.max_words_detail) instead of the normal cap."""
    return bool(user_text) and bool(_DETAIL.search(user_text))


def humanize(reply: str, max_words: int = 40, language: str = "hinglish", female: bool = True) -> str:
    t = strip_markup(reply)
    t = _BANNED_OPENERS.sub("", t).strip()
    t = spoken_numbers(t)
    if language in ("hinglish", "hindi"):
        t = fix_gender(t, female)
    if language == "hinglish":
        t = delhi_register(t)
    t = re.sub(r"\s+([,.!?])", r"\1", t)
    # "confirm karegi.Har kaam" (2026-11-25 loop): a lost space between two sentences; lower-case before the dot
    # so "3.5", "OEM/OES" and initials stay as they are
    t = re.sub(r"(?<=[a-z])([.!?])(?=[A-Z][a-z])", r"\1 ", t)
    t = enforce_budget(t, max_words)
    # one question max: keep the first, demote the rest
    if t.count("?") > 1:
        first = t.index("?")
        t = t[: first + 1] + t[first + 1 :].replace("?", ".")
    if language == "hinglish":
        t = re.sub(r"\bOkay\b", "Theek hai", t)
        t = re.sub(r"\bYes\b", "Haan", t)
    return t.strip()


_CONCERN_CUES = ("problem", "issue", "expensive", "mehnga", "mehenga", "nahi", "dikkat", "kharab", "awaaz",
                 "noise", "leak", "garam", "overheat", "band ho", "accident", "complaint", "shikayat", "pareshan")
_FEMALE_BACKCHANNELS = ["haan ji", "hmm", "right", "samajh gayi", "acha", "okay"]


def pick_backchannel(user_text: str, index: int = 0, options: list[str] | None = None) -> str | None:
    """Short acknowledgment to play while the customer is still talking.

    `options` comes from persona.conversation.backchannels so the words match the speaker (female Delhi
    NCR by default: "samajh gayi", never "samajh gaya")."""
    t = (user_text or "").lower()
    if len(t.split()) < 4:
        return None
    if any(w in t for w in _CONCERN_CUES):
        return "hmm"
    opts = [o for o in (options or _FEMALE_BACKCHANNELS) if o]
    return opts[index % len(opts)] if opts else None


def pick_gesture(gestures: dict, emotion: str, index: int = 0) -> str:
    """Rotate through the persona's gestures for an emotion so consecutive sentences don't repeat one move."""
    opts = gestures.get(emotion) or gestures.get("neutral") or ["idle"]
    return opts[index % len(opts)]


_E_REASSURE = ("koi dikkat nahi", "koi baat nahi", "tension mat", "no problem", "ho jayega", "fikar mat", "don't worry",
               "chinta mat", "chinta na")
_E_THINK = ("ek second", "ek minute", "check karti", "dekhti hoon", "confirm karke", "let me check")
_E_CONCERN = ("problem", "issue", "delay", "nahi ho paya", "dikkat", "kharab", "awaaz", "noise", "leak",
              "overheat", "garam", "accident", "complaint", "shikayat", "arre", "oh no", "band ho",
              "start nahi", "nahi chal", "atak", "jhatka", "smoke", "dhuan", "warning light", "chot", "injur")
# Avatar face table (owner spec 2026-11-25): an apology / "I understand" line to an upset customer is EMPATHETIC (soft
# concerned brows, gentle mouth), the accident / breakdown itself is CONCERNED. Concern cues win when both appear.
_E_EMPATHY = ("sorry", "maaf", "afsos", "samajh sakti", "samajh sakta", "i understand", "i can understand", "apologi",
              "pareshani ke liye", "bura laga", "chinta", "jayaz", "naraz", "naraaz", "frustrat", "galti", "regret")
# booking / request confirmed (face: smile). Never with a negation ("booking confirm nahi hui").
_E_BOOKED = ("booking ho gayi", "booking ho gaya", "booking confirm", "book ho gayi", "book ho gaya", "booking pakki",
             "request file", "file kar di", "file ho gayi", "confirm ho gayi", "confirm ho gaya", "confirm kar di",
             "booked", "is confirmed", "has been confirmed", "request note ho gayi", "registered")
_E_NEG = ("nahi", "nahin", "not", "couldn't", "could not", "failed", "na ho")
_E_PRICE = ("kharcha", "price", "prices", "cost", "costs", "rupaye", "rupees", "rs", "charges", "charge", "estimate",
            "quotation", "rate", "rates", "fees")
_E_ENTHUS = ("congrat", "great", "badhiya", "perfect", "best", "zaroor", "bilkul", "done", "book kar d", "mubarak",
             "wah", "mast", "shandaar", "excellent", "sahi")
_E_WARM = ("welcome", "namaste", "dhanyavaad", "thank", "shukriya", "hello", "hi ", "haan ji", "ji haan", "acha",
           "achha", "accha", "theek hai", "good", "nice",
           # service / insurance enquiries get the friendly face
           "insurance", "claim", "cashless", "service", "servicing", "pickup", "pick-up", "madad", "help")


_E_WH = ("kaise", "kya", "kaun", "kab", "kitna", "kitni", "kahan", "kyun", "kaunsi", "how", "why", "which", "when", "where")

# The CUSTOMER's situation in a turn (situation_of): an angry customer gets the empathetic face, an accident /
# breakdown the concerned one, while she listens, thinks and answers, even when her answer itself is upbeat
# ("Ji zaroor, insurance claim ..."). A plain service problem ("brake se awaaz") is an enquiry: no override.
_S_ANGRY = ("bekaar", "bekar", "bakwas", "bakwaas", "ghatiya", "worst", "pathetic", "useless", "fraud", "cheat", "loot",
            "chor", "gussa", "naraz", "naraaz", "angry", "frustrat", "disgust", "kharab kar d", "bigaad", "bigad",
            "disappointed", "never again", "consumer court", "bewakoof", "pagal", "shame", "rubbish", "nonsense",
            "kya mazaak", "dhokha", "terrible", "horrible")
_S_CONCERN = ("accident", "takkar", "thok di", "thok diya", "crash", "collision", "breakdown", "break down",
              "band pad", "beech raste", "smoke", "dhuan", "aag", "fire", "brake fail", "tow", "towing", "stuck",
              "phas gay", "phans gay", "chot", "injur", "hurt", "complaint", "shikayat", "stolen", "chori")


_PREFIX_CUES = {"book kar d", "apologi", "frustrat", "kharab kar d", "injur", "phas gay", "phans gay", "disgust"}   # stems that continue ("book kar deti / dungi"); everything else is a whole word


def _cue_re(cues: tuple[str, ...]) -> re.Pattern:
    """Whole-word cues, so "namaste" never hits "mast" and "achanak" never hits "acha"."""
    parts = [re.escape(c.strip()) + ("" if c in _PREFIX_CUES else r"\b") for c in cues]
    return re.compile(r"\b(?:" + "|".join(parts) + r")", re.I)


_RE_REASSURE, _RE_THINK, _RE_CONCERN = _cue_re(_E_REASSURE), _cue_re(_E_THINK), _cue_re(_E_CONCERN)
_RE_ENTHUS, _RE_WARM, _RE_WH = _cue_re(_E_ENTHUS), _cue_re(_E_WARM), _cue_re(_E_WH)
_RE_EMPATHY, _RE_BOOKED, _RE_NEG, _RE_PRICE = _cue_re(_E_EMPATHY), _cue_re(_E_BOOKED), _cue_re(_E_NEG), _cue_re(_E_PRICE)
_RE_S_ANGRY, _RE_S_CONCERN = _cue_re(_S_ANGRY), _cue_re(_S_CONCERN)


def emotion_of(text: str) -> str:
    """Emotion tag per sentence -> persona.avatar.gestures, TTS prosody and the card's face. Hinglish and English
    cues, reassurance before concern so "koi dikkat nahi" reads as warm, not worried.

    Face table (2026-11-25): booking confirmed -> enthusiastic (smile); accident / breakdown / complaint -> concerned;
    an apology or "samajh sakti hoon" -> empathetic; a price line -> neutral; service / insurance -> warm (friendly)."""
    t = text.lower()
    if _RE_REASSURE.search(t):
        return "warm"
    if _RE_THINK.search(t):
        return "thinking"
    if _RE_BOOKED.search(t) and not _RE_NEG.search(t):
        return "enthusiastic"
    if "?" in t and _RE_WH.search(t):
        return "thinking"
    if _RE_CONCERN.search(t):
        return "concerned"
    if _RE_EMPATHY.search(t):
        return "empathetic"
    if _RE_PRICE.search(t) or "₹" in t:
        return "neutral"
    if _RE_ENTHUS.search(t):
        return "enthusiastic"
    if _RE_WARM.search(t):
        return "warm"
    if "?" in t:
        return "warm"
    return "neutral"


def situation_of(user_text: str) -> str | None:
    """The customer's situation for one turn, for the card's face while she listens, thinks and answers:
    "empathetic" for an angry / upset customer, "concerned" for an accident, breakdown or complaint, else None
    (her own sentences then pick the face). Anger wins: "accident ke baad bhi bakwas service" is empathetic."""
    t = (user_text or "").lower()
    if _RE_S_ANGRY.search(t):
        return "empathetic"
    if _RE_S_CONCERN.search(t):
        return "concerned"
    return None


def delivery_chunks(text: str, max_words: int = 14) -> list[str]:
    """Chunks of ~14 words at clause boundaries -> TTS starts sooner, sounds less read."""
    chunks: list[str] = []
    for s in split_sentences(text):
        if len(s.split()) <= max_words:
            chunks.append(s)
            continue
        parts = re.split(r"(?<=[,;:])\s+", s)
        buf = ""
        for p in parts:
            if buf and len((buf + " " + p).split()) > max_words:
                chunks.append(buf.strip())
                buf = p
            else:
                buf = f"{buf} {p}".strip()
        if buf:
            chunks.append(buf)
    return [c for c in chunks if c]
