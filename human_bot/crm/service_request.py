"""Service request flow: what Neha has collected, what she still needs, when to submit.

Pure functions (no network), unit-tested. The server (server/app.py) calls:

    slots = collect(transcript)                      # facts from the customer's own words
    block = prompt_block(slots, persona)              # instructions for the model this turn
    if is_confirmation(user_text) and pending: ...    # the server submits, never the model

Rules: ask one missing thing at a time; never invent a field; the request is sent only after the customer
says yes to the read-back; the customer hears the real reference from the backend or an honest "could
not send, the team will call" (never a fabricated confirmation).
"""
from __future__ import annotations

import re

REQUIRED = ("name", "phone", "car", "problem", "area")
OPTIONAL = ("registration", "when", "pickup")
LABEL_HI = {"name": "aapka naam", "phone": "mobile number", "car": "car ka model aur number (ask both in the one question: \"Aapki car ka model aur number kya hai?\"; if only the model comes back, do not insist on the number)",
            "problem": "kya kaam karwana hai", "area": "aap kis area mein hain", "registration": "gaadi ka registration number", "when": "kab convenient rahega", "pickup": "pickup chahiye ya workshop aayenge"}

# spoken service -> ACR website service category (lookups/service-categories, 2026-11-03)
SERVICE_SLUG = {
    "AC repair": "car-ac-service-repair", "battery and tyre service": "car-battery", "brakes and clutch work": "car-brake-wheel-maintenance",
    "detailing": "car-care-detailing", "denting and painting": "car-denting-painting", "insurance claim repair": "car-insurance-claim",
    "lights and glass work": "car-lights-and-glass-work", "suspension check": "car-suspension-work", "engine and gearbox diagnosis": "car-repairs-inspection",
    "periodic service": "car-repairs-inspection",
}

PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?([6-9]\d{9})(?!\d)")
REG_RE = re.compile(r"\b([A-Z]{2}[\s-]?\d{1,2}[\s-]?[A-Z]{1,3}[\s-]?\d{3,4})\b", re.I)
NAME_RE = re.compile(r"(?:mera naam|my name is|naam hai|name hai|main|i am|i'm|this is|bol raha hoon|bol rahi hoon)\s+([A-Za-z][a-z]{1,15}(?:\s+[A-Za-z][a-z]{1,15})?)", re.I)
NAME_STOP = {"neha", "auto", "car", "gurgaon", "delhi", "noida", "bol", "baat", "kar", "ek", "the", "ok", "haan", "nahi", "service", "sir", "madam",
             "hai", "hoon", "hu", "ji", "se", "raha", "rahi", "bolta", "bolti", "and", "aur", "mera", "meri"}
# A one/two-word reply that is only a name shape (letters, no verb/noise words) can be the name itself:
# the question "Aapka naam kya hai?" asked just before makes "vinay" a name, not noise (2026-11-04).
NAME_HINT = re.compile(r"\b(naam|name)\b", re.I)
# Refusals and hedges are not names. Checked on the RAW user turn, not the filtered words: "nahi"
# itself is a NAME_STOP fill, so "nahi batana" would otherwise leave only "batana" and slip through.
NAME_DECLINE = re.compile(r"\b(nahi|nahin|nahi bataunga|nahi batana|nahi bataungi|mat\b|pata nahi|baad mein|skip)\b", re.I)
# Spoken yes as STT writes it: haa / haan / haaan / ha ji / hanji / ji haan / हाँ / जी (2026-11-05: "haa" was missed,
# so the read-back was never confirmed and the customer heard an apology).
YES_RE = re.compile(r"^\W*(?:(?:h+a+n*|ha+n+ji)(?:\s*ji)?\b|ji(?:\s*h+a+n*)?\b|हाँ|हां|हा|जी|"
                    r"(?:yes|yeah|yep|ok|okay|theek|thik|sahi|bilkul|kar\s*do|kardo|karo|bhej\s*do|bhejo|bhej\s*dijiye|bhej\s*dena|"
                    r"send|confirm|done|sure|go ahead|chalega)\b)", re.I)
NO_RE = re.compile(r"^\W*(nahi|nahin|na|no|mat|ruko|wait|cancel|galat|wrong|change)\b|\b(nahi|nahin|galat|wrong|cancel|mat karo)\b|नहीं|नही", re.I)
WHEN_RE = re.compile(r"\b(aaj|today|abhi|kal|tomorrow|parso|weekend|saturday|sunday|monday|tuesday|wednesday|thursday|friday|subah|morning|dopahar|afternoon|shaam|evening)\b", re.I)


def _bare_name(transcript: list[dict]) -> str:
    """The name from a short turn that is only a name: "vinay", "vinay kumar", "vinay hai".

    Used when NAME_RE found nothing: the bot asked "Aapka naam kya hai?" and the customer answered
    with the name alone (STT also returns "VInay"/"VINAY" — the check is case-insensitive). Fills
    ("hai", "ji", "sir"...) are dropped, at most two letters-only words of 3-16 chars count, and a
    question mark anywhere disqualifies the turn ("kya charges hain?" is never a name)."""
    turns = transcript or []
    # find the assistant turn that asked for the name, then take the user turn that answered it
    name_asked_at = None
    for i, t in enumerate(turns):
        if t.get("role") == "assistant" and NAME_HINT.search(t.get("text", "")):
            name_asked_at = i
    if name_asked_at is None:
        return ""  # the name was never asked: a bare word stays noise
    # the customer's answer is the first user turn after the name question
    src = ""
    for t in turns[name_asked_at + 1:]:
        if t.get("role") == "user" and t.get("text", "").strip():
            src = t["text"].strip()
            break
    if not src or "?" in src or NAME_DECLINE.search(src):
        return ""
    if len(re.findall(r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ek|do|teen|char|paanch|panch|chhah|cheh|saat|aath|nau)\b", src, re.I)) >= 3:
        return ""  # spoken digits = the phone number, never a name
    words = [w.strip(".,!?;:()'\"") for w in src.split()]
    words = [w for w in words if w and w.lower() not in NAME_STOP]
    if not words or len(words) > 2:
        return ""
    if any(not w.isalpha() or not (3 <= len(w) <= 16) for w in words):
        return ""
    return " ".join(words).title()


# How speech-to-text spells the popular models (Whisper / Deepgram, Roman and Devanagari): "Meri kreta ka AC" must
# count as a Creta, or Neha asks for the car the customer just named (2026-11-13 mic test).
CAR_SPELLINGS = {
    "kreta": "creta", "creata": "creta", "क्रेटा": "creta", "क्रेता": "creta",
    "swif": "swift", "suift": "swift", "स्विफ्ट": "swift", "dezire": "dzire", "dizire": "dzire", "डिजायर": "dzire",
    "baleeno": "baleno", "balino": "baleno", "बलेनो": "baleno", "breeza": "brezza", "brezaa": "brezza", "ब्रेज़ा": "brezza",
    "ertica": "ertiga", "artiga": "ertiga", "अर्टिगा": "ertiga", "inova": "innova", "inowa": "innova", "इनोवा": "innova",
    "fortunar": "fortuner", "फॉर्च्यूनर": "fortuner", "nexan": "nexon", "nekson": "nexon", "नेक्सॉन": "nexon", "नेक्सन": "nexon",
    "seltose": "seltos", "सेल्टोस": "seltos", "scorpion": "scorpio", "स्कॉर्पियो": "scorpio", "thaar": "thar", "थार": "thar",
    "i 20": "i20", "i 10": "i10", "वेन्यू": "venue", "वरना": "verna", "सिटी": "city", "आल्टो": "alto", "अल्टो": "alto",
}


def _spellfix(low: str) -> str:
    for k, v in CAR_SPELLINGS.items():
        low = re.sub(r"(?<![\wऀ-ॿ])" + re.escape(k) + r"(?![\wऀ-ॿ])", v, low)
    return low


def collect(transcript: list[dict]) -> dict[str, str]:
    """Facts from user turns only. Uses the same car/service/area tables as the offline script."""
    from llm.client import OfflineServiceBrain  # noqa: PLC0415
    out: dict[str, str] = {}
    user_turns = [t.get("text", "") for t in transcript if t.get("role") == "user"]
    user = " ".join(user_turns)
    low = _spellfix(" " + user.lower() + " ")
    m = PHONE_RE.search(user)
    if m:
        out["phone"] = m.group(1)
    else:
        # spoken digits: "nine eight one one two two..." -> 9811223344 (the way customers actually
        # say a number on a call; STT writes it as words more often than as numerals)
        word_digits = re.findall(
            r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ek|do|teen|char|paanch|panch|chhah|cheh|saat|aath|nau)\b", user, re.I)
        table = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                 "six": "6", "seven": "7", "eight": "8", "nine": "9", "ek": "1", "do": "2",
                 "teen": "3", "char": "4", "paanch": "5", "panch": "5", "chhah": "6", "cheh": "6",
                 "saat": "7", "aath": "8", "nau": "9"}
        digits = "".join(table.get(w.lower(), "") for w in word_digits)
        if len(digits) >= 10 and digits[-10] in "6789":
            out["phone"] = digits[-10:]
    m = REG_RE.search(user)
    if m:
        out["registration"] = re.sub(r"[\s-]", "", m.group(1)).upper()
    for t in user_turns:
        m = NAME_RE.search(t)
        if not m:
            continue
        words = [w for w in m.group(1).split() if w.lower() not in NAME_STOP]  # "mera naam Rahul hai" -> Rahul
        if words:
            out["name"] = " ".join(words).title()
        elif (m.group(1) or "").lower() in ("hai", "hoon"):
            pass  # "mera naam hai": the name came before or comes next; nothing to store yet
    if not out.get("name"):
        # the bare-name answer: "Aapka naam kya hai?" -> "vinay". STT returns "VInay"/"VINAY" as
        # often as "Vinay", so the check is case-insensitive; "vinay hai" keeps only the real word.
        name = _bare_name(transcript)
        if name:
            out["name"] = name
    for name in OfflineServiceBrain._MODELS:
        if f" {name} " in low or f" {name}," in low:
            out["car"] = name.title(); break
    else:
        for make in OfflineServiceBrain._MAKES:
            if f" {make} " in low or f" {make}," in low:
                out["car"] = make.title(); break
    for pat, label in OfflineServiceBrain._SERVICES:
        if re.search(pat, low):
            out["problem"] = label; break
    for key, label in OfflineServiceBrain._AREAS.items():
        if key in low:
            out["area"] = label; break
    m = WHEN_RE.search(user)
    if m:
        out["when"] = m.group(1).lower()
    if re.search(r"\b(pickup|pick up|uthwa|ghar se|doorstep)\b", low):
        out["pickup"] = "yes"
    elif re.search(r"\b(workshop (aa|visit)|khud aa|main aa jaunga|main aa jaungi|le aaunga|drop kar dunga)", low):
        out["pickup"] = "no"
    return out


def missing(slots: dict) -> list[str]:
    return [k for k in REQUIRED if not slots.get(k)]


def wants_request(transcript: list[dict], slots: dict) -> bool:
    """A service/repair/booking conversation, not a price or timing enquiry only."""
    user = " ".join(t.get("text", "") for t in transcript if t.get("role") == "user").lower()
    if slots.get("problem") or slots.get("car"):
        return True
    return bool(re.search(r"\b(service|servicing|repair|book|booking|appointment|pickup|karwana|karwani|karwa|request|problem|dikkat)\b", user))


# --------------------------------------------------------------- on-screen request form (card overlay)
# The card opens a small form once a request is real: an insurance claim, or the customer asking to get work
# done / book. A price or timing question alone does not open it ("AC service kitne ka hai?").
INSURANCE_RE = re.compile(r"\b(insurance|insurence|insurans|bima|beema|claim|clam|claime|clame|cashless)\b", re.I)
BOOK_RE = re.compile(r"\b(book|booking|appointment|slot|pickup|pick up|karwana|karwani|karwane|karana|karani|karwa\s*(do|dena|dijiye)|"
                     r"repair\s+(karna|karwa|chahiye)|theek\s+karwa|thik\s+karwa|service\s+chahiye|request)\b", re.I)
FORM_FIELDS = ("name", "phone", "registration", "car", "service", "area")
# the customer asks for the form itself ("form open kar dijiye", "form fill kara lijiye", "फॉर्म खोल दो"): it opens
# again even after an earlier one was closed (found live 2026-11-13: Neha said "form khul gaya" and nothing opened)
FORM_ASK = re.compile(r"(\bform\b|\bfarm\b|फ़?ॉर्म|फार्म|फ़ॉर्म)", re.I)


def form_asked(text: str) -> bool:
    return bool(FORM_ASK.search(text or ""))
NAME_FORM_RE = re.compile(r"^[A-Za-zऀ-ॿ][A-Za-zऀ-ॿ .'-]{1,39}$")


def form_reason(transcript: list[dict]) -> str | None:
    """"insurance" | "service" | None, from the customer's own words so far."""
    user = " ".join(t.get("text", "") for t in transcript if t.get("role") == "user")
    if INSURANCE_RE.search(user):
        return "insurance"
    if BOOK_RE.search(user):
        return "service"
    return None


def form_prefill(slots: dict, reason: str | None) -> dict[str, str]:
    """What the form starts with: only facts the customer said (never a guess)."""
    # a claim is a claim even when an earlier question mentioned the AC or a dent
    service = "insurance claim repair" if reason == "insurance" else slots.get("problem", "")
    return {k: v for k, v in {"name": slots.get("name", ""), "phone": slots.get("phone", ""),
                              "registration": slots.get("registration", ""), "car": slots.get("car", ""),
                              "service": service, "area": slots.get("area", "")}.items() if v}


def validate_form(data: dict) -> tuple[dict[str, str], dict[str, str]]:
    """Form values -> (slots, errors by field). Server-side: the card's checks are only a convenience."""
    v = {k: re.sub(r"\s+", " ", str(data.get(k) or "")).strip()[:60] for k in FORM_FIELDS}
    errors: dict[str, str] = {}
    if not NAME_FORM_RE.match(v["name"]):
        errors["name"] = "Naam likhiye"
    digits = re.sub(r"\D", "", v["phone"])
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    if not re.fullmatch(r"[6-9]\d{9}", digits):
        errors["phone"] = "10 digit mobile number likhiye"
    reg = re.sub(r"[\s-]", "", v["registration"]).upper()
    if reg and not REG_RE.fullmatch(reg):
        errors["registration"] = "Number aise likhiye: DL01AB1234"
    if not (2 <= len(v["car"]) <= 40):
        errors["car"] = "Car ka model likhiye"
    if v["service"] not in SERVICE_SLUG:
        errors["service"] = "Service chuniye"
    slots = {"name": v["name"].title(), "phone": digits, "car": v["car"], "problem": v["service"],
             "registration": reg, "area": v["area"]}
    return {k: s for k, s in slots.items() if s}, errors


def summary_line(slots: dict) -> str:
    bits = [slots.get("name"), slots.get("car"), slots.get("registration"), slots.get("problem"), slots.get("area") and f"{slots['area']} se", slots.get("when"), "free pickup" if slots.get("pickup") == "yes" else None]
    return ", ".join(b for b in bits if b)


def prompt_block(slots: dict, pending_confirm: bool, form_open: bool = False) -> str:
    """System-prompt section: what is known, what to ask next, when to read back and ask for a yes.
    form_open: the card is showing the short request form, so pointing at it is allowed (and only at it)."""
    need = missing(slots)
    known = "; ".join(f"{k}: {v}" for k, v in slots.items() if v) or "nothing yet"
    rule = ("A short request form was shown on the customer's screen. They can fill it and press Submit, or tell you the "
            "details; both work. Only the system opens the form: never say on your own that a form has opened or is open. "
            "If they ask for the form, just ask them to say \"form kholiye\"." if form_open else
            "You CAN take the request on this call. Never send the customer to the website form or ask them to fill anything.")
    lines = ["## Service request (you file it yourself; the system sends it to the workshop after the customer says yes)",
             rule, f"Known so far: {known}."]
    if need:
        lines.append(f"Still needed, ask ONLY the first one now, in one short question: {', '.join(LABEL_HI[k] for k in need)}.")
        lines.append("Do not invent any of these. A phone number must be read back digit by digit once you have it.")
    elif not pending_confirm:
        lines.append("Everything is known. Read it back in one sentence and ask exactly: \"Main yeh request bhej doon?\" Then wait.")
    else:
        lines.append("You already asked for confirmation. If the customer agrees, say nothing about sending (the system does it); if they correct something, update it and read back again.")
    return "\n".join(lines)


def is_yes(text: str) -> bool:
    return bool(YES_RE.search(text or "")) and not NO_RE.search(text or "")


def is_no(text: str) -> bool:
    return bool(NO_RE.search(text or ""))


def asked_confirmation(assistant_text: str) -> bool:
    return bool(re.search(r"bhej\s*d(oo|u)n|request (bhej|send)|confirm kar", assistant_text or "", re.I))
