"""AIVA-shaped calling loop, stdlib only (no FastAPI/websockets install needed).

    python human_bot/server/app.py --port 8010

Endpoints
  GET  /health                     -> {"ok":true,"provider":"...","persona":"..."}
  POST /session                    -> {session_id, greeting, chunks[]}
  POST /chat      (SSE)            -> events: backchannel, sentence, final
  POST /analyze                    -> AIVA post-call JSON contract
  GET  /                           -> browser demo (mic, streaming, TTS, avatar panel)

Why SSE and not WebSocket: the transport is the only stdlib-free part of a real
build. Swapping this for LiveKit/Agora is a transport change, not a brain change;
everything below (session context, barge-in, sentence-level TTS, emotion tags,
post-call analysis) stays identical.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import signal
import sys
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from llm.client import (  # noqa: E402
    OfflineBrain,
    offline_brain,
    STTUnavailable,
    TTSUnavailable,
    EDGE_VOICE_EN,
    EDGE_VOICE_HI,
    EMOTIONS,
    stt_status,
    transcribe,
    provider_name,
    stream_reply,
    cached_reply,
    cache_reply,
    synthesize,
    synthesize_stream,
    tts_status,
)
from llm.knowledge import KB_TIMEOUT_S, knowledge_block, retrieve  # noqa: E402
from memory.store import ConversationStore  # noqa: E402
from crm.outbox import Outbox  # noqa: E402
from crm.bitrix24 import sink_from_env as bitrix_sink_from_env  # noqa: E402
from crm import acr_leads  # noqa: E402
from crm import site_form  # noqa: E402
from crm.service_request import SERVICE_GUIDE, SERVICE_SLUG, form_asked, form_prefill, form_reason, is_hold, is_question, validate_form, wants_booking  # noqa: E402
from llm.pages import about_line, find_page, named_service, named_services, page_line, service_page, wants_service, wants_some_service, with_roman, overview as pages_overview  # noqa: E402
from llm import intents as meaning  # noqa: E402  (meaning-based service / intent matcher, an extra signal)
NAV_ON = os.getenv("HUMAN_BOT_NAV", "1").strip().lower() not in ("0", "false", "no", "off")    # "page dikhao" navigation
FORM_ON = os.getenv("HUMAN_BOT_FORM", "1").strip().lower() not in ("0", "false", "no", "off")  # on-screen request form
FORM_LINES = {   # what Neha says as the form slides up (spoken; the server adds it to the /tts allow-list)
    "insurance": "Ji, insurance claim mein main poori madad karti hoon. Screen par ek chhota form khul gaya hai, "
                 "details bhar ke Submit kar dijiye, ya mujhe bol kar bata dijiye.",
    "service": "Ji zaroor. Screen par ek chhota form khul gaya hai, details bhar ke Submit kar dijiye, "
               "ya mujhe bol kar bata dijiye.",
}
# The same lines for a customer speaking English (llm/language.py), spoken in the Indian English voice
# 2026-11-24 owner: "servicing karani hai" -> first WHICH service; once a specific one is named the booking form opens
ASK_WHICH = "Ji zaroor. Kis cheez ki servicing karwani hai? Jaise regular service, AC, battery, brakes, engine, ya denting-painting?"
ASK_WHICH_EN = "Sure. Which service do you need? For example regular service, AC, battery, brakes, engine, or denting and painting?"
SITE_FORM_LINE = "Ji, {title} ke liye maine website ka booking form khol diya hai. Aap ye fill kar dijiye, aapka confirmation aa jayega."
SITE_FORM_LINE_EN = "Sure, I've opened the booking form for {title} on the website. Please fill it in and you'll get your confirmation."
CARD_FORM_LINE = "Ji, {title} ke liye screen par booking form khul gaya hai. Aap ye fill kar dijiye, aapka confirmation aa jayega."
CARD_FORM_LINE_EN = "Sure, a booking form for {title} is open on your screen. Please fill it in and you'll get your confirmation."
# 2026-11-24 owner, step 2: the service's page opens, Neha says what is done there, then asks model + registration
EXPLAIN_LINE = ("Ji zaroor, screen par {title} ka page khul gaya hai, aap dekh sakte hain. Ismein hum {about} karte hain. "
                "Aapki car ka model aur registration number bata dijiye?")
EXPLAIN_LINE_EN = ("Sure, the {title} page is open on your screen. We do {about}. "
                   "Could you tell me your car's model and registration number?")
# step 3: with the car details the booking form opens; the mobile number comes next
DETAILS_FORM_LINE = ("Shukriya ji. {title} ke liye maine website ka booking form khol diya hai. "
                     "Aapka mobile number bata dijiye, servicing ke liye aapko call aa jayega.")
DETAILS_FORM_LINE_EN = ("Thank you. I've opened the booking form for {title} on the website. "
                        "Please share your mobile number; you'll get a call for the servicing.")
DETAILS_CARD_LINE = ("Shukriya ji. {title} ke liye screen par booking form khul gaya hai. "
                     "Aapka mobile number bata dijiye, servicing ke liye aapko call aa jayega.")
DETAILS_CARD_LINE_EN = ("Thank you. A booking form for {title} is open on your screen. "
                        "Please share your mobile number; you'll get a call for the servicing.")
# the question of each step, asked again after Neha answers a question in between
STEP_Q = {"which": ("Kis cheez ki servicing karwani hai?", "Which service do you need?"),
          "car": ("Aapki car ka model aur registration number bata dijiye?", "Could you tell me your car's model and registration number?"),
          "phone": ("Aapka mobile number bata dijiye, servicing ke liye aapko call aa jayega.", "Please share your mobile number; you'll get a call for the servicing."),
          "name": ("Aapka naam kya hai?", "May I have your name?"),
          "confirm": ("Main yeh request bhej doon?", "Shall I send this request?")}
PHONE_AGAIN = ("Mobile number 10 digit ka bata dijiye, jaise 98 1234 5678.", "Please share a 10-digit mobile number.")
HOLD_LINE = ("Ji zaroor, aaram se. Jab ready hon, bata dijiye.", "Sure, take your time. Tell me when you're ready.")
TWO_LINE = ("Ji, dono ho jayenge. Pehle {a} ya {b}, kaunsa note karun?", "Sure, we do both. Shall I note {a} or {b} first?")
CHANGE_LINE = ("Theek hai ji, {title} ke liye note kar liya, page aur form dono update kar diye hain.",
               "Okay, noted {title}; I've updated the page and the form.")
HUMAN_LINE = ("Ji zaroor, hamari team aapko call karegi. Aapka mobile number bata dijiye?",
              "Sure, our team will call you. Please share your mobile number.")
READBACK = ("Toh main note kar leti hoon: {title}{car}, number {digits}, naam {name}. Main yeh request bhej doon?",
            "Let me note it: {title}{car}, number {digits}, name {name}. Shall I send this request?")
NOT_YES = ("Koi baat nahi. Bataiye, kya badalna hai?", "No problem. What should I change?")
OUT_OF_SCOPE_RE = re.compile(r"\b(nayi\s+gaadi|nayi\s+car|new\s+car|gaadi\s+kh?a?ree?d\w*|car\s+kh?a?ree?d\w*|kharee?d\w*\s+(?:ni|na)|"
                             r"buy\s+a\s+car|konsi\s+(?:car|gaadi)\s+(?:loon|lu|lein|kharid\w*)|loan|emi\b|finance|"
                             r"bechni|sell\s+my\s+car|job|naukri|vacancy)\b", re.I)
# 2026-11-25 QA fixes ------------------------------------------------------------------------------------------
CANCEL_RE = re.compile(r"\b(cancel\w*|rehne\s+do|rahne\s+do|chhod\s*do|chhodo|mat\s+karo|nahi\s+karwan\w*|band\s+karo|"
                       r"don'?t\s+(?:book|want)|not\s+interested|forget\s+it)\b", re.I)
CANCEL_LINE = ("Theek hai ji, maine booking cancel kar di. Kabhi bhi zaroorat ho toh bas bata dijiye.",
               "Alright, I've cancelled the booking. Just tell me whenever you need us.")
NOT_INTERESTED_RE = re.compile(r"\b(nahi|nahin|no|mat|not|don'?t)\b|नहीं|नही", re.I)   # not "na": "kar do na" is a yes
BYE_RE = re.compile(r"^\W*(?:ok(?:ay)?\s+|achha\s+|theek\s+hai\s+)?(?:bye|bye\s*bye|thank\s*you|thanks|thankyou|thnx|shukriya|"
                    r"dhanyavaad|dhanyawad|alvida|tata)(?:\s+(?:ji|bye|so\s+much|neha|very\s+much|aapka))*\W*$", re.I)
BYE_LINE = ("Aapka bahut shukriya ji! Jab bhi zaroorat ho, main yahin hoon. Aapka din shubh rahe.",
            "Thank you! I'm here whenever you need us. Have a great day.")
COMPLAINT_RE = re.compile(r"\b(loot\w*|mehe?nga|mahenga|expensive|costly|bakwas|fraud|cheat\w*|ghatiya|bekaar|bekar|rip\s*off)\b", re.I)
COMPLAINT_LINE = ("Main samajh sakti hoon ji, aapki chinta bilkul jayaz hai. Team pehle aapko poora estimate batayegi, "
                  "phir aap aaram se decide kar lijiye.",
                  "I understand, that's a fair concern. The team will give you the full estimate first, then you can decide.")
# a complaint about OUR past work (2026-11-25 loop: "tumhari service bahut kharab thi, gaadi me scratch aa gaye" opened
# the Denting page cheerfully): apologise and hand to a senior advisor, never sell
SERVICE_COMPLAINT_RE = re.compile(
    r"\b(?:tumhar\w*|tumne|aapk\w*|aapne|your|you\s+guys|last\s+time|pichh?l[ie]\s+baar|aap\s+logo?n?|tum\s+log|aapke\s+yahan)\b"
    r"[^.?!]{0,50}\b(?:kharab|kharaab|bekaar|bekar|bura|buri|ghatiya|bakwas|worst|bad|poor|scratch\w*|damage\w*|toot\w*|tod\s+di\w*)\b"
    r"|\b(?:complaint|shikayat|not\s+(?:happy|satisfied)|khush\s+nahi|satisfied\s+nahi|bad\s+experience|poor\s+service)\b"
    # a repeat fault after OUR work: "pichle hafte aapse service karwayi thi, ab phir se awaaz" (loop round 5)
    r"|\b(?:aapse|aapke\s+yahan|tumse|aap\s+logon\s+se|from\s+you|with\s+you)\b[^.?!]{0,40}\b(?:karwa\w*|kiya|ki\s+thi|done|serviced|repaired)\b"
    r"[^.?!]{0,50}\b(?:phir|fir|dobara|again|still|abhi\s+bhi|wapas)\b", re.I)
SORRY_LINE = ("Mujhe sach mein afsos hai ji ki aapka experience accha nahi raha. Main yeh senior team tak pahuncha deti hoon, "
              "woh aapko call karke ise theek karenge. Aapka mobile number bata dijiye?",
              "I'm really sorry your experience wasn't good. I'll take this to our senior team and they'll call you to "
              "make it right. Could you share your mobile number?")
CHANGE_NUMBER_RE = re.compile(r"\b(?:number|mobile|phone)\b[^.?!]{0,25}\b(?:badal\w*|change\w*|update\w*|galat)\b"
                              r"|\b(?:badal\w*|change|update)\b[^.?!]{0,25}\b(?:number|mobile|phone)\b", re.I)
CHANGE_NUMBER_LINE = ("Ji, naya mobile number bata dijiye, main team ko note karwa deti hoon taaki booking update ho jaaye.",
                      "Sure, tell me the new mobile number and I'll have the team update your booking.")
# "kya aap log genuine ho? fraud to nahi" is a trust question, not a price complaint (loop round 7): reassure with
# facts the KB already states (centres, OEM parts, written warranty, estimate approved first)
TRUST_RE = re.compile(r"\b(genuine\s+ho|genuine\s+company|fraud\s+(?:to\s+)?nahi|fraud\s+ho|trust\s+kar\w*|bharosa|bharosemand|"
                      r"legit\w*|is\s+this\s+(?:a\s+)?(?:real|genuine)|are\s+you\s+(?:legit|genuine|real\s+company)|scam)\b", re.I)
TRUST_LINE = ("Ji, bilkul bharosa kar sakte hain. Auto Car Repair ke Delhi NCR mein 4 centres hain, genuine OEM parts lagte hain, "
              "har kaam ke saath written warranty card milta hai, aur kaam aapke approve kiye estimate ke baad hi shuru hota hai.",
              "Yes, you can trust us. Auto Car Repair has 4 centres in Delhi NCR, uses genuine OEM parts, gives a written "
              "warranty card with every job, and starts work only after you approve the estimate.")
ACCIDENT_RE = re.compile(r"\b(accident|takkar|thok|thuk|collision|crash\w*|thokar)\b|टक्कर|एक्सीडेंट", re.I)
PROBLEM_RE = re.compile(r"\b(nahi|nahin|kharab|kharaab|awaaz|aawaz|noise|sound|leak\w*|tapak\w*|smell|badbu|problem|issue|dikkat|ho\s+gay[ai]|band|toot\w*|dhundli|ganda|gandi|bhaari|bhari|garam|overheat\w*|not\s+working|stopped|broken|grinding|vibrat\w*|dhua\w*|dhuan|dhuaan|puncture|dead|flat|chatak\w*)\b|नहीं|खराब|आवाज|धुआ", re.I)   # a symptom told ("AC se smell", "steering bhaari"), not a plain request
ACCIDENT_LEAD = ("Oh, yeh sun ke bura laga ji, umeed hai aap theek hain.", "Oh, I'm sorry to hear that, I hope you're okay.")
IDENTITY_RE = re.compile(r"\b(aap\s+kaun\s+ho|tum\s+kaun\s+ho|kaun\s+ho\s+aap|who\s+are\s+you|robot|chatbot|bot\s+ho|ai\s+ho|insaan\s+ho|insan\s+ho|real\s+ho|machine\s+ho|"
                         r"are\s+you\s+(?:a\s+)?(?:human|real|robot|bot|ai))\b", re.I)
IDENTITY_LINE = ("Main Neha hoon, Auto Car Repair ki AI service advisor. Gaadi ki service mein poori madad karungi, aur agar team se "
                 "baat karni ho toh bas bata dijiye.",
                 "I'm Neha, Auto Car Repair's AI service advisor. I can help with anything about your car's service, and if "
                 "you'd like to talk to the team, just tell me.")
PAGE_OPEN_LINE = ("Ji, screen par {title} ka page khul gaya hai.", "Sure, the {title} page is open on your screen.")
ABOUT_Q_RE = re.compile(r"\b(batao|bataiye|bataye|kya\s+kya|kya\s+hai|kya\s+hota|kya\s+karte|kaise\s+hot\w*|about|details?|"
                        r"jaankari|jankari|what\s+(?:is|do|does|are)|tell\s+me)\b", re.I)
URGENT_PATHS = ("/category/car-emergency-services", "/car-roadside-assistance")
URGENT_LINE = ("Ji, pareshaan mat hoiye, main madad karti hoon. Screen par {title} ka page khul gaya hai, ismein hum {about} karte hain. "
               "Jaldi madad ke liye aapki car ka model aur registration number bata dijiye?",
               "Don't worry, I'll help you. The {title} page is open on your screen; we do {about}. "
               "To get help to you quickly, could you tell me your car's model and registration number?")
FORM_NAME_LINE = ("Shukriya ji. {title} ke liye maine website ka booking form khol diya hai. Aapka naam kya hai?",
                  "Thank you. I've opened the booking form for {title}. May I have your name?")
ATTR_RE = re.compile(r"\b(warranty|guarantee|gaurantee|fark|difference|compare|vs)\b", re.I)
NAME_STOP = {"naam", "name", "galat", "hai", "h", "mera", "meri", "my", "is", "wrong", "sahi", "ji", "actually", "its", "it's",
             "the", "correct", "spelling", "nahi", "not", "se", "ka", "toh", "to", "please", "kar", "do", "likho", "likhiye",
             "change", "badlo", "naam:", "नाम", "गलत", "है", "मेरा", "सही", "जी"}


def name_correction(text: str) -> str:
    """"naam galat hai, Sunil Kumar hai" / "my name is Rahul, not Rohit" -> the corrected name ("" if none)."""
    if not re.search(r"\b(naam|name)\b|नाम", text or "", re.I):
        return ""
    t = re.sub(r"\b(?:not|nahi)\s+\S+", " ", text, flags=re.I)          # "Rahul, not Rohit": drop the wrong one
    words = [w.strip(" .,!?।:;\"'") for w in t.split()]
    keep = [w for w in words if w and w.lower() not in NAME_STOP and NAME_WORD.match(w)]
    if 1 <= len(keep) <= 3:
        return " ".join(w.title() if w.isascii() else w for w in keep)
    return ""


def english_about(about: str) -> str:
    a = about.replace(" aur ", " and ").replace(", yaani ", ", i.e. ")
    a = re.sub(r"\s+ki\s+safai\b", " cleaning", a)
    a = re.sub(r"\s+ka\s+kaam\b", " work", a)
    return re.sub(r"\s+k[aei]\s+", " ", a)
HUMAN_RE = re.compile(r"\b(insaan|human|real\s+person|agent|manager|executive|kisi\s+se\s+baat|aadmi\s+se|team\s+se\s+baat|"
                      r"call\s+(?:me\s+)?back|callback|call\s+kar\s+(?:do|dena|lena|dijiye|lijiye)|mujhe\s+call)\b", re.I)
LATER_RE = re.compile(r"\b(baad\s+mein|later|kuch\s+der\s+mein|abhi\s+nahi|bas\s+pooch\w*|bas\s+puch\w*|"
                      r"just\s+(?:asking|checking|enquiring)|abhi\s+zaroorat\s+nahi)\b", re.I)   # "bas pooch raha tha" (loop)
PRICE_RE = re.compile(r"\b(price|rate|cost|kitne\s+ka|kitne\s+ki|kitna\s+lag\w*|kitna\s+pad\w*|kitne\s+lag\w*|charges?|how\s+much(?!\s+(?:time|long))|"
                      r"kharch\w*|paise|paisa|fees?|estimate)\b", re.I)
NAME_WORD = re.compile(r"^[A-Za-z\u0900-\u097F]{2,20}$")


ASK_DETAILS_RE = re.compile(r"\b(mobile|phone|number|naam|name|model|registration|contact)\b.{0,50}(bata|share|dijiye|dena|chahiye|\?)|"
                            r"\b(bata\s+dijiye|share\s+kar)\b.{0,30}\b(mobile|number|naam|model)\b", re.I)


def drop_detail_asks(tokens):
    """The model's answer inside the booking flow, minus any sentence that asks for personal / car details: the
    step machine asks the right one itself (2026-11-24: the model asked for the mobile at the car step)."""
    # 2026-11-25: also any question of the model's own ("Kya aap booking karna chahenge?", "kaunsi insurance?"):
    # the step question follows, two questions in a row is one too many. "rahe.Kya" (no space) is split too.
    def keep(sent: str) -> bool:
        return not ASK_DETAILS_RE.search(sent) and not sent.rstrip().endswith("?")

    buf = ""
    for tok in tokens:
        buf += tok
        while True:
            m = re.search(r"[.?!।](\s+|$|(?=[A-Zऀ-ॿ]))", buf)
            if not m or (m.end() == len(buf) and not m.group(1)):
                break
            sent, buf = buf[:m.end()], buf[m.end():]
            if keep(sent):
                yield sent.rstrip() + " "
    if buf.strip() and keep(buf):
        yield buf


def booking_step(s: "Session", slots: dict) -> str | None:
    """Which question the booking flow is waiting on (None outside the flow)."""
    if s.service_request:
        return None
    if s.sr_pending:
        return "confirm"
    if s.form_sent:
        return "phone" if not slots.get("phone") else ("name" if not slots.get("name") else "confirm")
    if s.chosen:
        return "car" if s.interested else "offer"
    if s.asked_which:
        return "which"
    return None


OFFER_Q = ("Iske baare mein aur kuch jaanna hai, ya main aapki booking schedule kar doon?",
           "Would you like to know anything else, or shall I schedule your booking?")
OFFER_Q_INS = ("Claim ke baare mein aur kuch jaanna hai, ya main aapka insurance claim book kar doon?",
               "Would you like to know anything else about the claim, or shall I book your insurance claim?")
INTEREST_LEAD = ("Zaroor ji, main booking schedule kar deti hoon. ", "Sure, I'll schedule it for you. ")
NO_MORE = ("Theek hai ji. Jab bhi booking karni ho, bas bata dijiye.", "Alright. Whenever you'd like to book, just tell me.")
PRICE_UNKNOWN = (" Price aapki car aur kaam par depend karta hai, isliye exact estimate team inspection ke baad deti hai.",
                 " The price depends on your car and the work, so the team gives the exact estimate after an inspection.")
INTEREST_RE = re.compile(r"\b(book\w*|schedule\w*|appointment|slot|kar\s*do|kardo|kar\s+dijiye|karwa\s*do|karwa\s+dijiye|"
                         r"karwa(?:na|ni)\s+(?:hai|h)|kara(?:na|ni)\s+(?:hai|h)|aage\s+badh\w*|proceed|go\s+ahead|"
                         r"let'?s\s+do|i\s+want|chahiye|interested)\b|बुक|करवा\s*दो|कर\s*दो", re.I)
PRICES_FILE = HERE.parent / "persona" / "acr_prices.json"
_PRICES: dict = {"t": 0.0, "data": {}}


def service_price(path: str, en: bool) -> str:
    """The owner's price line for a service page (persona/acr_prices.json, re-read when edited), or ""."""
    try:
        mt = PRICES_FILE.stat().st_mtime
        if mt != _PRICES["t"]:
            _PRICES["data"] = json.loads(PRICES_FILE.read_text(encoding="utf-8")).get("prices") or {}
            _PRICES["t"] = mt
    except (OSError, ValueError):
        return ""
    row = _PRICES["data"].get(path) or {}
    return (row.get("en") if en else row.get("hi")) or "" if isinstance(row, dict) else ""


def price_line(spec: dict, en: bool) -> str:
    p = service_price(spec.get("path", ""), en).strip()
    if p:
        return " " + p.rstrip(".") + "."
    if "insurance" in (spec.get("path") or ""):
        return ""                                  # a claim has no "price"; the questions get answered from the page
    return PRICE_UNKNOWN[1 if en else 0]


def offer_question(spec: dict | None, en: bool) -> str:
    ins = "insurance" in ((spec or {}).get("path") or "")
    return (OFFER_Q_INS if ins else OFFER_Q)[1 if en else 0]


def shows_interest(text: str, question: bool) -> bool:
    """After the explanation: the customer wants to go ahead ("haan kar do", "book kar do", "theek hai")."""
    t = text or ""
    if NOT_INTERESTED_RE.search(t) or CANCEL_RE.search(t):
        return False          # "nahi chahiye" / "not interested" started a booking (2026-11-25 QA, 4/4)
    if INTEREST_RE.search(t) and (not question or re.search(r"\bbook\w*|बुक", t, re.I)):
        return True
    return not question and sr_is_yes(t)


CAR_REG_Q = ("Aapki {car} ka registration number bata dijiye?", "What's the registration number of your {car}?")
CAR_DONE_Q = ("Aapki {car} ki details mil gayi hain. Main booking form khol doon?", "I have your {car} details. Shall I open the booking form?")


def car_question(slots: dict | None, en: bool) -> str:
    """Step 2's question, minus what the customer already said ("meri creta ke brake...": only the registration)."""
    car = ((slots or {}).get("car") or "").strip()
    if car and (slots or {}).get("registration"):
        return CAR_DONE_Q[1 if en else 0].format(car=car)
    if car:
        return CAR_REG_Q[1 if en else 0].format(car=car)
    return STEP_Q["car"][1 if en else 0]


def step_question(step: str | None, en: bool, slots: dict | None = None) -> str:
    if step == "car":
        return car_question(slots, en)
    if step == "offer":
        return OFFER_Q[1 if en else 0]
    return STEP_Q[step][1 if en else 0] if step in STEP_Q else ""
FORM_LINES_EN = {
    "insurance": "Sure, I'll help you with the insurance claim. A short form is open on your screen; fill it in "
                 "and press Submit, or just tell me the details.",
    "service": "Sure. A short form is open on your screen; fill it in and press Submit, or just tell me the details.",
}
EN_BACKCHANNELS = ["right", "okay", "I see", "sure", "got it"]
from llm import fastpath  # noqa: E402
# Fast path: a question the customer-call knowledge already answers is answered in ~1 ms without the model
# (llm/fastpath.py, thresholds from tools/eval_fastpath.py). HUMAN_BOT_FAST=0 switches it off.
FAST_ON = os.getenv("HUMAN_BOT_FAST", "1").strip().lower() not in ("0", "false", "no", "off")
# after a fast answer Neha asks for the next missing booking detail, in the customer's language
# 2026-11-24 owner: a quick answer is followed by "which servicing?", never by car model / area / name / phone (those
# come only after the service is confirmed, from the service-request prompt).
FAST_FOLLOW = (("problem", "Gaadi mein kis cheez ki servicing karwani hai?", "Which service does the car need?"),)


def fast_follow_up(slots: dict, english: bool) -> str:
    for key, hi, en in FAST_FOLLOW:
        if not slots.get(key):
            return en if english else hi
    return "Anything else you'd like to know?" if english else "Aur kuch jaanna chahenge?"


def static_pieces(text: str, persona: dict, language: str, female: bool, max_words: int, en: bool) -> list[str]:
    """The clips the /chat loop produces for this whole text, in order: opener, first clause, sentences, tail.
    Mirrors that loop (one token = the whole text) so a warmed clip is the exact text the card will ask for."""
    buf, out = text, []
    m = _OPENER.match(buf)
    if m:
        opener, buf = m.group(1), buf[m.end(1):].lstrip(" ,!")
        word = opener.lower().strip(" ,!")
        out.append((EN_OPENER.get(word, word.capitalize()) + opener[-1]) if en else cached_opener(opener, persona))
    cut = split_first_phrase(buf)
    if cut:
        head, buf = cut
        p = humanize(head, max_words=max_words, language=language, female=female)
        if p:
            out.append(p)
    while True:
        parts = split_sentences(buf)
        if len(parts) <= 1:
            break
        head, buf = parts[0], " ".join(parts[1:])
        p = humanize(head, max_words=max_words, language=language, female=female)
        if p:
            out.append(p)
    tail = humanize(buf, max_words=max_words, language=language, female=female)
    if tail:
        out.append(tail)
    return out


# Few and paced on purpose: the Edge endpoint throttles bursts ("NoAudioReceived" after ~150 clips on 2026-11-14),
# and a throttled provider would silence live customers. Stops at the first failure or slow clip.
FAST_WARM_N = int(os.getenv("HUMAN_BOT_FAST_WARM", "1000"))   # every fast answer: clips persist on disk, so this resumes each start


_WARM_FAILS = {"n": 0}


def _warm_piece(piece: str, voice: str | None, tag: str) -> bool:
    """Synthesize one clip for the cache with back-off: up to 3 tries 15/30/60 s apart. False only after 6 clips
    in a row could not be made (the provider is really down), so the warm-up does not hammer it."""
    for wait in (0, 15, 30, 60):
        if wait:
            time.sleep(wait)
        try:
            tts_get_or_make(piece, voice=voice, emotion=emotion_of(piece), wait_s=10)
            _WARM_FAILS["n"] = 0
            return True
        except Exception as ex:  # noqa: BLE001
            err = str(ex)[:80]
    _WARM_FAILS["n"] += 1
    print(f"[{tag}] could not warm one clip ({err}); moving on", file=sys.stderr, flush=True)
    return _WARM_FAILS["n"] < 6


def warm_flow(persona: dict) -> None:
    """Pre-synthesize the booking-flow lines (which-service, every service's page + explanation, the form lines) as the
    /chat loop will cut them, so each step plays at once. Clips persist on disk: later starts skip them in ms."""
    female = is_female(persona)
    max_words = persona["conversation"]["max_words_turn"]
    lines = [ASK_WHICH]
    for pg in ((persona.get("pages") or {}).get("list") or []):
        if pg.get("kind") != "service":
            continue
        ex = EXPLAIN_LINE.format(title=pg["title"], about=about_line(pg) or pg["title"])
        lines.append(ex)
        lines.append(ex.replace("Ji zaroor, screen", "Ji, samajh gayi. Screen"))   # a problem told (PROBLEM_RE)
        lines.append(PAGE_OPEN_LINE[0].format(title=pg["title"]))
        lines += [DETAILS_FORM_LINE.format(title=pg["title"]), DETAILS_CARD_LINE.format(title=pg["title"])]
    # the fixed replies of the booking machine and the rule branches (2026-11-25 latency: each was a fresh synthesis)
    lines += [INTEREST_LEAD[0] + car_question({}, False), car_question({}, False), STEP_Q["phone"][0], STEP_Q["name"][0],
              OFFER_Q[0], OFFER_Q_INS[0]]
    lines += [x[0] for x in (NO_MORE, CANCEL_LINE, SORRY_LINE, TRUST_LINE, ACCIDENT_LEAD, HUMAN_LINE, CHANGE_NUMBER_LINE,
                             BYE_LINE, IDENTITY_LINE, COMPLAINT_LINE, HOLD_LINE)]
    lines += [URGENT_LINE[0].format(title=pg["title"], about=about_line(pg) or pg["title"])
              for pg in ((persona.get("pages") or {}).get("list") or []) if pg.get("path") in URGENT_PATHS]
    t0, n, fresh = time.perf_counter(), 0, 0
    for line in lines:
        for piece in static_pieces(line, persona, "hinglish", female, max_words, False):
            remember_tts_text(piece)
            n += 1
            if tts_cached(tts_key(piece, None, emotion_of(piece))):
                continue
            if not _warm_piece(piece, None, "flow"):
                print(f"[flow] warm-up paused for good after {n} clips (provider keeps failing)", file=sys.stderr, flush=True)
                return
            fresh += 1
            time.sleep(0.8)               # leave the provider to live customers
    print(f"[flow] {n} flow clips ready ({fresh} new) in {time.perf_counter() - t0:.0f}s", file=sys.stderr, flush=True)


def warm_fast(persona: dict) -> None:
    """Pre-synthesize the most asked fast answers (Hinglish and English) as a new customer would get them, so
    their first audio is a cache hit (~5 ms) instead of a fresh synthesis (~1 s). Runs after the kb answers load."""
    for _ in range(120):                      # the index loads in the background; wait up to ~4 min
        fp = fastpath.get()
        if fp:
            break
        time.sleep(2)
    else:
        return
    female = is_female(persona)
    max_words = persona["conversation"]["max_words_turn"]
    t0, n = time.perf_counter(), 0
    for row in fp.popular(FAST_WARM_N):
        for language, voice, text in (("hinglish", None, row.answer), ("english", EDGE_VOICE_EN, row.answer_en)):
            if not text:
                continue
            en = language == "english"
            full = text if text.strip().endswith("?") else text.strip() + " " + fast_follow_up({}, en)
            for piece in static_pieces(full, persona, language, female, max_words, en):
                if tts_cached(tts_key(piece, voice, emotion_of(piece))):
                    n += 1
                    continue                  # made on an earlier start (disk cache): nothing to do, no pause
                # 2026-11-25 latency: an uncached answer costs 1-2 s of silence, so the warm-up no longer gives up on the
                # first Edge timeout (after a cache wipe it stopped at 203 clips): it backs off and carries on
                if not _warm_piece(piece, voice, "fast"):
                    print(f"[fast] warm-up paused for good after {n} clips (provider keeps failing)", file=sys.stderr, flush=True)
                    return
                n += 1
                time.sleep(1.0)               # leave the provider to live customers
    print(f"[fast] warmed {n} answer clips in {time.perf_counter() - t0:.0f}s", file=sys.stderr, flush=True)
EN_OPENER = {"haan": "Yes", "haan ji": "Yes", "ji haan": "Yes", "ji": "Yes", "acha": "Okay", "achha": "Okay", "accha": "Okay",
             "theek hai": "Okay", "bilkul": "Sure", "dekhiye": "See", "arre": "Oh", "okay": "Okay", "right": "Right",
             "samajh gayi": "I understand"}
from llm.language import detect as detect_language  # noqa: E402
from crm.service_request import asked_confirmation, collect as sr_collect, is_no as sr_is_no, is_yes as sr_is_yes, missing as sr_missing, prompt_block as sr_prompt_block, summary_line as sr_summary, wants_request as sr_wants  # noqa: E402
from llm.knowledge import status as kb_status  # noqa: E402
from llm import did_avatar, simli_avatar  # noqa: E402


def avatar_status() -> dict:
    return simli_avatar.status() if simli_avatar.enabled() else did_avatar.status()
from persona.humanizer import (  # noqa: E402
    delivery_chunks,
    emotion_of,
    humanize,
    pick_backchannel,
    pick_gesture,
    situation_of,
    split_first_phrase,
    split_sentences,
    wants_detail,
)
from persona.prompting import build_compact_prompt, build_system_prompt, load_persona  # noqa: E402
# "full" (default): the 1,540-word production prompt. "compact": the 104-word prompt a model fine-tuned on
# training/acr_sft (human_bot/train/build_acr_sft.py) was trained with.
PROMPT_STYLE = os.getenv("HUMAN_BOT_PROMPT", "full").strip().lower()
from postcall.analyze import analyze  # noqa: E402

# End-of-turn Transformer (human_bot/turn): scores each transcript so the card can end a finished turn early
# and keep listening (with a soft "hmm") through a mid-thought pause. Optional: no numpy / no weights -> the
# cards keep their built-in rule.
try:
    from turn import model as turn_model  # noqa: E402
except Exception as _ex:  # noqa: BLE001
    turn_model = None
    print(f"[turn] end-of-turn model unavailable: {str(_ex)[:120]}", file=sys.stderr)


# What the cards may say *while the customer is still talking* (Moshi-style full duplex). One fixed set for every
# brand, the same as DEFAULT_CUES in avatar_widget/src/duplex.js; when to say it is the turn model's call.
LISTEN_CUES = ("hmm", "haan ji", "acha")


def eot_fields(text: str) -> dict:
    """{"eot": p, "eot_threshold": t} for an /stt reply, or {} when the model is not available."""
    if not turn_model or not text:
        return {}
    try:
        p = turn_model.end_of_turn(text)
    except Exception as ex:  # noqa: BLE001
        print(f"[turn] scoring failed: {str(ex)[:120]}", file=sys.stderr)
        return {}
    return {} if p is None else {"eot": round(p, 3), "eot_threshold": round(turn_model.threshold(), 3)}

PERSONA_DIR = HERE.parent / "persona"
WEB = HERE.parent / "web"
WEB_ROOT = WEB.resolve()                      # static files are confined to this directory
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
                ".css": "text/css", ".jpg": "image/jpeg", ".png": "image/png", ".svg": "image/svg+xml",
                ".wav": "audio/wav", ".mp3": "audio/mpeg", ".mp4": "video/mp4", ".json": "application/json; charset=utf-8",
                ".ico": "image/x-icon", ".webp": "image/webp"}
# The only voices a client may request: female Indian (project rule). Anything else is ignored.
ALLOWED_VOICES = {EDGE_VOICE_EN, EDGE_VOICE_HI, "en-IN-NeerjaNeural", "en-IN-NeerjaExpressiveNeural", "hi-IN-SwaraNeural"}


class BodyTooLarge(Exception):
    pass


def puppet_version() -> str:
    """Changes whenever the portrait or any puppet layer is rebuilt (newest mtime across the asset folders)."""
    base = HERE.parents[1] / "tran_moment"
    newest = 0
    for d in (base / "prepared", base / "path_b" / "mouths", base / "path_b" / "expressions", base / "path_b" / "reactions"):
        if d.is_dir():
            for f in d.rglob("*"):  # recursive: the morph frames live in expressions/<name>/fNN.webp
                if f.is_file():
                    newest = max(newest, f.stat().st_mtime_ns)
    return "%x" % (newest // 1_000_000)


from llm.client import OfflineServiceBrain as _OSB  # noqa: E402
KNOWN_CARS = frozenset(w for n in _OSB._MAKES + _OSB._MODELS for w in n.split()) | frozenset(
    ("glanza", "rumion", "taisor", "hyryder", "innova", "crysta", "fortuner", "hilux", "camry", "vellfire", "urban", "cruiser",
     "creta", "venue", "verna", "exter", "alcazar", "tucson", "ioniq", "aura", "grand", "i10", "i20", "nios"))
MAX_TURNS_KEPT = 16


def load_personas() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in sorted(PERSONA_DIR.glob("persona_*.json")):
        p = load_persona(f)
        out[p.get("id") or f.stem.replace("persona_", "")] = p
    return out


# every persona_*.json in persona/ is servable; --persona only picks the default
PERSONAS = load_personas()
PERSONA = next(iter(PERSONAS.values()))


def is_female(persona: dict) -> bool:
    return (persona.get("gender") or "female").lower().startswith("f")


class Session:
    def __init__(self, persona: dict) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.persona = persona
        self.transcript: list[dict] = []
        self.context: dict = {}
        self.brain: OfflineBrain = offline_brain(persona)  # service-advisor script for workshops, sales script for showrooms
        self.generation = 0
        self.lock = threading.Lock()
        self.language = persona["languages"]["primary"]
        self.started = time.time()
        self.last_seen = self.started
        # conversation memory (memory/store.py): who this is, what was said before, what is being recorded
        self.visitor_id = ""
        self.recall_block = ""
        self.recalled: list[dict] = []
        self.consent = False
        self.audio_dir: Path | None = None
        self.audio_files: list[str] = []
        self.pending_audio: str | None = None
        self.analysis: dict | None = None
        self.save_lock = threading.Lock()
        # service request (crm/service_request.py): has Neha asked "bhej doon?", and what was filed
        self.sr_pending = False
        self.form_sent = False   # the on-screen request form was opened in the card (once per conversation)
        self.booking = False     # the customer asked to book: details are collected from here on (before: explain + ask which service)
        self.site_form_opened = False   # the website's own booking form was opened for this customer
        self.asked_which = False        # Neha asked "kis cheez ki servicing?": the next service named is the choice
        self.chosen: dict | None = None  # the service the customer chose (its page is open, model/registration asked)
        self.bk: dict = {}               # details the step machine captured itself (phone, name)
        self.asked_name = False
        # 2026-11-25 owner: explain the service + price, answer every question, and only when the customer shows
        # interest in booking ask model + registration. `interested` = that moment has come.
        self.interested = False
        self.crm_early = False           # the lead went to Bitrix24 at booking time (the end-of-call push updates it)
        self.site_fill: dict | None = None   # values for the website's own booking form (crm/site_form.py)
        self.nav_opened: set[str] = set()     # service pages already opened for a booking request (once each)
        # knowledge prefetch: /stt starts retrieval for the transcript it just produced, so /chat for the same
        # text finds the passages ready instead of spending 30-300 ms on the kb call in the critical path
        self.kb_prefetch: dict | None = None   # {"text", "done": Event, "result": (block, hits)}
        self.service_request: dict | None = None

    def messages(self, extra: str = "") -> list[dict]:
        ctx = json.dumps(self.context, ensure_ascii=False) if self.context else "{}"
        if self.recall_block:
            extra = self.recall_block + ("\n\n" + extra if extra else "")
        if PROMPT_STYLE == "compact":   # a model fine-tuned on training/acr_sft learned the long rules; send what it saw
            system = build_compact_prompt(self.persona, self.language) + ("\n\n" + extra if extra else "")
        else:
            system = build_system_prompt(self.persona, self.language, extra=extra).replace("{SESSION_CONTEXT}", ctx)
        history = [{"role": t["role"], "content": t["text"]} for t in self.transcript[-MAX_TURNS_KEPT:]]
        return [{"role": "system", "content": system}] + history

    def knowledge(self, query: str) -> tuple[str, list[dict]]:
        """Retrieve website facts for this turn (persona.knowledge decides if/where). Fail-soft.
        Uses the /stt prefetch when it was started for exactly this text."""
        kcfg = self.persona.get("knowledge") or {}
        wid = kcfg.get("website_id")
        if not wid:
            return "", []
        pre = self.kb_prefetch
        if pre and pre["text"] == query.strip().lower():
            pre["done"].wait(KB_TIMEOUT_S)
            self.kb_prefetch = None
            if pre.get("result") is not None:
                return pre["result"]
        hits = retrieve(wid, query, k=int(kcfg.get("top_k", 5)), min_score=kcfg.get("min_score"))
        return knowledge_block(hits, self.persona.get("company", "the company")), hits

    def prefetch_knowledge(self, text: str) -> None:
        """Start retrieval for `text` in the background (called by /stt as soon as the transcript exists)."""
        if not (self.persona.get("knowledge") or {}).get("website_id") or not text.strip():
            return
        job = {"text": text.strip().lower(), "done": threading.Event(), "result": None}
        self.kb_prefetch = job

        def run() -> None:
            try:
                kcfg = self.persona.get("knowledge") or {}
                hits = retrieve(kcfg["website_id"], text, k=int(kcfg.get("top_k", 5)), min_score=kcfg.get("min_score"))
                job["result"] = (knowledge_block(hits, self.persona.get("company", "the company")), hits)
            except Exception:  # noqa: BLE001  (fail-soft: /chat retrieves itself)
                job["result"] = None
            finally:
                job["done"].set()

        threading.Thread(target=run, daemon=True, name="kb-prefetch").start()


SESSIONS: dict[str, Session] = {}
SESSIONS_LOCK = threading.Lock()

TTS_CACHE: dict[str, bytes] = {}
# Clips (and their lip cues) also go to disk: a sentence synthesized once is instant for every later customer, also
# after a restart (fresh Edge synthesis costs 1.1-3.3 s, measured 2026-11-24). human_bot/data is private + git-ignored.
TTS_DISK = HERE.parent / "data" / "tts_cache"
TTS_DISK_ON = os.getenv("HUMAN_BOT_TTS_DISK", "1").lower() not in ("0", "false", "no", "off")


def _disk_name(key: str) -> Path:
    import hashlib
    return TTS_DISK / hashlib.sha1(key.encode("utf-8")).hexdigest()


def _disk_put(key: str, audio: bytes, cues: list, words: list | None = None) -> None:
    if not TTS_DISK_ON or not audio:
        return
    try:
        TTS_DISK.mkdir(parents=True, exist_ok=True)
        base = _disk_name(key)
        tmp = base.with_suffix(".tmp")
        tmp.write_bytes(audio)
        tmp.replace(base.with_suffix(".mp3"))
        meta = {"key": key, "cues": cues}
        if words:   # Edge word timing: avatar v2 builds its finer viseme timeline from it (llm/visemes_v2.py)
            meta["words"] = [[round(s, 3), round(d, 3), w] for s, d, w in words]
        base.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    except OSError as ex:
        print(f"[tts] disk cache write failed: {ex}", file=sys.stderr)


def tts_cached(key: str) -> bytes | None:
    """A finished clip from memory, else from disk (then kept in memory too); None when it was never made."""
    hit = TTS_CACHE.get(key)
    if hit or not TTS_DISK_ON:
        return hit
    base = _disk_name(key)
    try:
        audio = base.with_suffix(".mp3").read_bytes()
    except OSError:
        return None
    try:
        meta = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
        if meta.get("key") == key:
            TTS_CUES[key] = meta.get("cues") or []
            if meta.get("words"):
                TTS_WORDS[key] = [(float(s), float(d), str(w)) for s, d, w in meta["words"]]
    except (OSError, ValueError):
        pass
    with TTS_LOCK:
        while len(TTS_CACHE) >= TTS_CACHE_MAX:
            TTS_CACHE.pop(next(iter(TTS_CACHE)))
        TTS_CACHE[key] = audio
    return audio
TTS_LOCK = threading.Lock()
TTS_LAST: dict[str, str] = {}
TTS_INFLIGHT: dict[str, "TtsJob"] = {}   # key -> synthesis in progress (chunks appear as Edge sends them)
# 400 was FIFO by insertion, so the start-up warm set (backchannels, openers, fast answers) was the first to go
# once a few hundred sentences were spoken. ~20-30 KB a clip: 2000 clips ~ 50 MB. Hits move to the back (LRU).
TTS_CACHE_MAX = int(os.getenv("HUMAN_BOT_TTS_CACHE_MAX", "2000"))

# --------------------------------------------------------------------------- chat response cache
# Exact-match cache for customer questions served by the LLM. The first time a
# question is asked it hits the model; every repeat is answered from RAM with zero
# network latency. Keyed by normalised user text (the session matters little for
# short questions like "AC service kitna hai?" or "Namaste").
_CHAT_CACHE: dict[str, tuple[float, str]] = {}
_CHAT_CACHE_TTL = 3600.0


def _chat_cache_get(text: str) -> str | None:
    rec = _CHAT_CACHE.get(text.strip().lower())
    if not rec:
        return None
    ts, reply = rec
    if time.time() - ts > _CHAT_CACHE_TTL:
        _CHAT_CACHE.pop(text.strip().lower(), None)
        return None
    return reply


_CACHE_UNSAFE = re.compile(r"\d|[०-९]")


def chat_cacheable(user_text: str, reply: str | None = None, in_request: bool = False, names: tuple = ()) -> bool:
    """The reply cache is shared by EVERY session and ignores the conversation, so only a context-free question
    may use it (2026-11-05: "haa" served one customer's read-back, name and phone number to the next caller).
    Short answers (haa / ok / nahi / a name / a number) mean something only in context; digits and the
    customer's name are personal data; a service request in progress is always that customer's own."""
    if in_request:
        return False
    words = re.findall(r"\w+", user_text or "")
    if len(words) < 3 or _CACHE_UNSAFE.search(user_text or ""):
        return False
    if reply is not None:
        if _CACHE_UNSAFE.search(reply):
            return False
        low = reply.lower()
        if any(n and len(n) > 1 and n.lower() in low for n in names):
            return False
    return True


def _chat_cache_put(text: str, reply: str) -> None:
    key = text.strip().lower()
    if len(key) > 200:
        return
    _CHAT_CACHE[key] = (time.time(), reply)


def tts_key(text: str, voice: str | None = None, emotion: str = "neutral") -> str:
    return f"{voice or 'default'}|{emotion or 'neutral'}|{text}"


TTS_CUES: dict[str, list[dict]] = {}   # tts_key -> [{"s": start_s, "e": end_s, "v": Rhubarb letter}] for the card's lips
TTS_WORDS: dict[str, list[tuple[float, float, str]]] = {}   # tts_key -> Edge word boundaries (avatar v2 visemes)
# Avatar v2 (tran_moment/avatar_v2: viseme reference pack + crossfading runtime) as the card's default. Off: the card
# uses v1 unless the page asks for ?avatar=v2. /tts_cues adds the v2 timeline when on or when asked with &v2=1.
AVATAR_V2 = os.getenv("HUMAN_BOT_AVATAR_V2", "0").lower() in ("1", "true", "yes", "on")
AVATAR_V2_DIR = HERE.parents[1] / "tran_moment" / "avatar_v2" / "pack"


def viseme_v2(key: str, cues: list[dict]) -> list[dict]:
    """15-viseme timeline (llm/visemes_v2.py) from the clip's word timing; from the v1 letters when that is gone."""
    try:
        from llm import visemes_v2
        words = TTS_WORDS.get(key)
        return visemes_v2.timeline(words) if words else visemes_v2.from_v1(cues)
    except Exception:  # noqa: BLE001 - the card falls back to the v1 letters
        return []
# letter -> Rhubarb mouth letter (A closed, B teeth, C open, D wide, F pucker, G f/v, H l, X rest); same map as
# tran_moment/synth_talk_clips.py so the live card and the rendered clips move their lips the same way
_LETTER_VISEME = {"a": "C", "e": "B", "i": "D", "o": "F", "u": "G", "m": "A", "b": "A", "p": "A", "f": "G", "v": "G",
                  "l": "H", "w": "F", "y": "D", "s": "B", "z": "B", "t": "B", "d": "B", "n": "B", "r": "B", "k": "C",
                  "g": "C", "h": "C", "j": "B", "c": "B"}


def _mouth_map() -> dict:
    """cue letter -> puppet mouth image (O.png for round vowels, FV.png for f/v; both were unused before)"""
    try:
        from llm.visemes import MOUTH_MAP
        return MOUTH_MAP
    except Exception:  # noqa: BLE001
        return {"A": "closed.png", "B": "E.png", "C": "A.png", "D": "I.png", "E": "E.png", "F": "U.png", "G": "U.png",
                "H": "L.png", "X": "X.png"}


def viseme_cues(text: str, words: list[tuple[float, float, str]]) -> list[dict]:
    """Lip-shape timeline for a clip from Edge's word boundaries: the letters of each (Roman) word are spread over
    that word's time, rests (X) fill the pauses, micro-cues and repeats merge so the mouth does not flicker.
    Returns [] when the provider gave no word timing (the card then falls back to the loudness analyser)."""
    if not words:
        return []
    try:                                     # phoneme-level aligner on Edge's own spoken words (llm/visemes.py)
        from llm import visemes
        return [{"s": c["s"], "e": c["e"], "v": c["v"]} for c in visemes.cues(words)]
    except Exception:  # noqa: BLE001 - keep the old letter spread as a fallback
        pass
    roman = re.findall(r"[A-Za-z]+", text)
    cues: list[dict] = []
    t_prev = 0.0
    for i, (start, dur, _spoken) in enumerate(words):
        if start - t_prev > 0.12:
            cues.append({"s": round(t_prev, 3), "e": round(start, 3), "v": "X"})
        letters = [c for c in (roman[i] if i < len(roman) else "a").lower() if c.isalpha()] or ["a"]
        keep: list[str] = []
        for c in letters:                               # consonant clusters collapse to their first consonant
            if c in "aeiou" or not keep or keep[-1] in "aeiou":
                keep.append(c)
        step = max(dur, 0.08) / len(keep)
        for k, c in enumerate(keep):
            cues.append({"s": round(start + k * step, 3), "e": round(start + (k + 1) * step, 3), "v": _LETTER_VISEME.get(c, "B")})
        t_prev = start + max(dur, 0.08)
    cues.append({"s": round(t_prev, 3), "e": round(t_prev + 0.3, 3), "v": "X"})
    merged: list[dict] = []
    for c in cues:
        if merged and ((c["e"] - c["s"]) < 0.09 or merged[-1]["v"] == c["v"]):
            merged[-1]["e"] = c["e"]
        else:
            merged.append(dict(c))
    return merged


# Edge pads every clip: ~0.25 s of silence before the first word and 0.9-1.3 s after the last (measured
# 2026-11-24 with ffmpeg silencedetect on 16 clips; "haan ji" is 1.87 s of which 0.55 s is voice). Played
# back to back that is ~1.2 s of dead air at every sentence boundary. Edge's MP3 (24 kHz, 48 kbit/s, mono)
# is a run of fixed 144-byte frames of 24 ms, so trimming is a byte slice on frame boundaries, guided by
# the WordBoundary times Edge already sends. Anything that is not that exact format is returned untouched.
_EDGE_FRAME, _EDGE_FRAME_S = 144, 0.024
TTS_LEAD_KEEP_S = float(os.getenv("HUMAN_BOT_TTS_LEAD_KEEP_S", "0.06"))
TTS_TAIL_KEEP_S = float(os.getenv("HUMAN_BOT_TTS_TAIL_KEEP_S", "0.18"))


def trim_edge_silence(audio: bytes, words: list) -> tuple[bytes, float]:
    """(trimmed audio, seconds cut from the front). Needs word times; keeps a small margin on both sides."""
    n = len(audio) // _EDGE_FRAME
    if not words or n < 4 or len(audio) % _EDGE_FRAME:
        return audio, 0.0
    for i in range(0, len(audio), _EDGE_FRAME):          # every frame must be a 144-byte MPEG-2 L3 frame
        if audio[i] != 0xFF or audio[i + 1] & 0xE0 != 0xE0 or (audio[i + 2] >> 1) & 1:
            return audio, 0.0
    first = min(s for s, _d, _w in words)
    last = max(s + d for s, d, _w in words)
    # one extra frame at the front: the first kept frame may borrow bits (bit reservoir) from the one before
    lead_f = max(0, int((first - TTS_LEAD_KEEP_S) / _EDGE_FRAME_S) - 1)
    tail_f = min(n, int((last + TTS_TAIL_KEEP_S) / _EDGE_FRAME_S) + 1)
    if tail_f - lead_f < 4:
        return audio, 0.0
    return audio[lead_f * _EDGE_FRAME: tail_f * _EDGE_FRAME], lead_f * _EDGE_FRAME_S


class TtsJob:
    """One synthesis. Readers can stream the chunks while the producer thread is still receiving them,
    so /tts starts sending audio ~1 s after the request instead of after the whole clip (1.5-4.6 s)."""

    def __init__(self, text: str, voice: str | None, emotion: str = "neutral") -> None:
        self.text, self.voice, self.emotion = text, voice, emotion
        self.chunks: list[bytes] = []
        self.words: list[tuple[float, float, str]] = []   # Edge WordBoundary events (start_s, dur_s, word)
        self.done = False
        self.error: str | None = None
        self.cv = threading.Condition()

    def run(self) -> None:
        key = tts_key(self.text, self.voice, self.emotion)
        try:
            for chunk in synthesize_stream(self.text, self.voice, emotion=self.emotion, words=self.words):
                with self.cv:
                    self.chunks.append(chunk)
                    self.cv.notify_all()
            audio, lead = trim_edge_silence(b"".join(self.chunks), self.words)
            if lead:   # cues are built from word times: shift them by what was cut from the front
                self.words[:] = [(max(0.0, s - lead), d, w) for s, d, w in self.words]
            with TTS_LOCK:
                while len(TTS_CACHE) >= TTS_CACHE_MAX:
                    TTS_CACHE.pop(next(iter(TTS_CACHE)))  # oldest first
                TTS_CACHE[key] = audio
                while len(TTS_CUES) >= TTS_CACHE_MAX:
                    TTS_CUES.pop(next(iter(TTS_CUES)))
                TTS_CUES[key] = viseme_cues(self.text, self.words)
                while len(TTS_WORDS) >= TTS_CACHE_MAX:
                    TTS_WORDS.pop(next(iter(TTS_WORDS)))
                TTS_WORDS[key] = list(self.words)
            _disk_put(key, audio, TTS_CUES.get(key) or [], self.words)
            TTS_LAST.pop("error", None)
        except Exception as ex:  # noqa: BLE001
            self.error = str(ex)[:220]
            TTS_LAST["error"] = self.error
        finally:
            with TTS_LOCK:
                if TTS_INFLIGHT.get(key) is self:
                    TTS_INFLIGHT.pop(key, None)
            with self.cv:
                self.done = True
                self.cv.notify_all()

    def stream(self, timeout_s: float = 30.0) -> Iterator[bytes]:
        """Yield chunks in order, waiting for new ones; raises TTSUnavailable on failure or timeout."""
        i = 0
        deadline = time.monotonic() + timeout_s
        while True:
            with self.cv:
                while i >= len(self.chunks) and not self.done:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise TTSUnavailable("tts timeout")
                    self.cv.wait(left)
                if i < len(self.chunks):
                    chunk = self.chunks[i]
                    i += 1
                elif self.error:
                    raise TTSUnavailable(self.error)
                else:
                    return
            yield chunk


def tts_job(text: str, voice: str | None = None, emotion: str = "neutral") -> TtsJob:
    """The job for this text: finished (from cache), already running (joined), or started now.

    The server starts a job the moment a sentence is produced (see prefetch_tts), so the card's later
    /tts request usually finds the clip ready or half done. Two callers never synthesize the same text twice.
    """
    key = tts_key(text, voice, emotion)
    if key not in TTS_CACHE:
        tts_cached(key)                     # pulls a clip made earlier (also before a restart) off the disk
    with TTS_LOCK:
        hit = TTS_CACHE.pop(key, None)
        if hit:
            TTS_CACHE[key] = hit            # re-insert: most recently used goes last, eviction takes the front
            job = TtsJob(text, voice, emotion)
            job.chunks, job.done = [hit], True
            return job
        job = TTS_INFLIGHT.get(key)
        if job is None:
            job = TTS_INFLIGHT[key] = TtsJob(text, voice, emotion)
            threading.Thread(target=job.run, daemon=True, name="tts-job").start()
        return job


def tts_get_or_make(text: str, voice: str | None = None, wait_s: float = 30.0, emotion: str = "neutral") -> bytes:
    """Whole clip (cache, joined in-flight job, or a fresh synthesis)."""
    remember_tts_text(text)
    try:
        return b"".join(tts_job(text, voice, emotion).stream(wait_s))
    except TTSUnavailable:
        with TTS_LOCK:
            hit = tts_cached(tts_key(text, voice, emotion))
        if hit:
            return hit
        raise


# An opening interjection followed by a comma and more text: "Acha, ", "Haan ji, ", "Bilkul, ". Only matched once
# per turn, only when text follows (so a bare "Acha" sentence is left to the normal splitter).
_OPENER = re.compile(r"^\s*((?:haan ji|ji haan|haan|acha|achha|accha|bilkul|theek hai|dekhiye|arre|right|okay|samajh gayi)[,!])\s+\S", re.I)


def cached_opener(opener: str, persona: dict) -> str:
    """The interjection as the warm backchannel clip when one exists (exact text = cache key), else as spoken."""
    bare = opener.rstrip(",!").strip()
    for cand in (bare.lower(), bare.capitalize(), bare):
        if cached_clip(cand):
            return cand
    for bc in (persona.get("conversation") or {}).get("backchannels") or []:
        if bc.lower() == bare.lower():
            return bc
    return bare.capitalize() + "."


def prefetch_tts(texts: list[str], voice: str | None = None, emotion: str = "neutral") -> None:
    """Start synthesis now so playback does not wait. Failures are recorded; /tts will retry.
    Every text that passes through here is one the server produced, which is what /tts may speak."""
    for t in texts:
        t = (t or "").strip()
        if t:
            remember_tts_text(t)
            tts_job(t, voice, emotion)


def cached_clip(text: str, emotion: str = "neutral") -> bytes | None:
    with TTS_LOCK:
        return tts_cached(tts_key(text, None, emotion)) or tts_cached(tts_key(text, None, "neutral"))


def warm_tts(personas: dict) -> None:
    """At start-up: backchannels, fillers, guardrail/handoff lines and the opening interjections for every
    persona, so the first turn never pays the Edge cold start (7.6 s measured) and 'haan ji' / 'acha' play
    the instant the customer stops talking."""
    phrases: list[str] = []
    for p in personas.values():
        conv = p.get("conversation") or {}
        phrases += list(conv.get("backchannels") or [])
        phrases += list(conv.get("fillers") or [])
        g = p.get("guardrails") or {}
        phrases += [g.get("handoff_line"), g.get("out_of_scope_line"), g.get("handoff_unavailable_line")]
    # The opening-interjection vocabulary from _OPENER: cached_opener looks up the bare form and its
    # capitalised twin, so both are warmed. The multi-word ones in the personas are already above.
    for w in ("haan", "ji haan", "acha", "achha", "accha", "bilkul", "theek hai", "dekhiye", "arre", "okay"):
        phrases += [w, w.capitalize()]
    phrases += ["Ek second, main check karti hoon.", "Haan ji, boliye.", "Samajh gayi."]
    phrases += list(FORM_LINES.values()) if FORM_ON else []   # spoken as the request form opens
    if FORM_ON:   # "kis cheez ki servicing?" and the booking-form line of every service page: cached, spoken at once
        phrases += [ASK_WHICH] + [line.format(title=pg["title"]) for p in personas.values()
                                  for pg in ((p.get("pages") or {}).get("list") or []) if pg.get("kind") == "service"
                                  for line in (SITE_FORM_LINE, CARD_FORM_LINE)]
    phrases += list(LISTEN_CUES)  # mid-turn "hmm" cues (turn model + avatar_widget/src/duplex.js), not persona text
    seen: list[str] = []
    for t in phrases:
        if t and t not in seen:
            seen.append(t)
    t0 = time.perf_counter()
    failed = 0
    for t in seen:
        remember_tts_text(t)
        try:
            tts_get_or_make(t)
        except Exception as ex:  # noqa: BLE001
            failed += 1
            print(f"[tts] warm-up failed for {t!r}: {str(ex)[:120]}", file=sys.stderr)
            if failed >= 3:
                print("[tts] warm-up stopped after 3 failures (provider down?); clips synthesize on demand", file=sys.stderr)
                return
    print(f"[tts] warmed {len(seen) - failed}/{len(seen)} clips in {time.perf_counter() - t0:.1f}s", file=sys.stderr, flush=True)
    # English-speaking customers hear these in the Indian English voice (llm/language.py)
    for t in EN_BACKCHANNELS + (list(FORM_LINES_EN.values()) + [ASK_WHICH_EN] if FORM_ON else []):
        remember_tts_text(t)
        try:
            tts_job(t, EDGE_VOICE_EN, "neutral")
        except Exception:  # noqa: BLE001
            break
    warm_greetings(personas)


def warm_greetings(personas: dict) -> None:
    """The two generic /session greetings (no car model on the page) and the recording disclosure, chunked and
    emotion-tagged exactly as /session builds them, so the first caller after a restart or a Render cold start
    hears Neha without waiting on fresh TTS. Personalised openers (model / returning customer) stay on demand."""
    t0, made = time.perf_counter(), 0
    for p in personas.values():
        try:
            base = f"Namaste! Main {p['display_name']}, {p['company']} se. "
            for tail in ("Aap kaunsi car dekh rahe hain?", "Aapki car mein kya problem hai?"):
                greeting = humanize(base + tail, max_words=p["conversation"]["max_words_first_turn"],
                                    language=p["languages"]["primary"], female=is_female(p))
                for c in delivery_chunks(greeting):
                    remember_tts_text(c)
                    tts_get_or_make(c, emotion=emotion_of(c))
                    made += 1
        except Exception as ex:  # noqa: BLE001
            print(f"[tts] greeting warm-up failed for {p.get('id')!r}: {str(ex)[:120]}", file=sys.stderr)
    if STORE and SPOKEN_DISCLOSURE:
        for line in DISCLOSURE.values():
            try:
                remember_tts_text(line)
                tts_get_or_make(line)
                made += 1
            except Exception as ex:  # noqa: BLE001
                print(f"[tts] disclosure warm-up failed: {str(ex)[:120]}", file=sys.stderr)
    print(f"[tts] warmed {made} greeting clips in {time.perf_counter() - t0:.1f}s", file=sys.stderr, flush=True)


def wake_kb() -> None:
    """At start-up, touch the kb once. On Render free both services sleep; when a visitor wakes the bot, the kb is
    usually asleep too, and its ~33 s cold start would otherwise land on the first knowledge lookup."""
    import urllib.request
    from llm.knowledge import KB_URL
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(f"{KB_URL}/health", timeout=60) as r:
            r.read(256)
        print(f"[kb] awake at {KB_URL} in {time.perf_counter() - t0:.1f}s", file=sys.stderr, flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"[kb] wake-up failed ({KB_URL}): {str(ex)[:120]}", file=sys.stderr, flush=True)


def get_session(sid: str) -> Session | None:
    with SESSIONS_LOCK:
        s = SESSIONS.get(sid)
    if s:
        s.last_seen = time.time()
    return s


def new_session(context: dict | None, persona: dict) -> Session:
    s = Session(persona)
    s.context = context or {}
    s.last_seen = time.time()
    with SESSIONS_LOCK:
        SESSIONS[s.id] = s
        while len(SESSIONS) > SESSION_MAX:
            SESSIONS.pop(next(iter(SESSIONS)), None)
    return s


def clean_context(ctx) -> dict:
    """Only the page facts the prompt needs, each capped: the context comes from the host page / URL."""
    if not isinstance(ctx, dict):
        return {}
    page = ctx.get("page") if isinstance(ctx.get("page"), dict) else {}
    keep = {}
    for k in ("url", "title", "model", "lang"):
        v = page.get(k)
        if isinstance(v, str) and v.strip():
            keep[k] = v.strip()[:200]
    if page.get("site_form") is True:   # widget.js found the site's own booking form on this page
        keep["site_form"] = True
    out = {"page": keep} if keep else {}
    for k in ("lang", "language", "source"):
        v = ctx.get(k)
        if isinstance(v, str) and v.strip():
            out[k] = v.strip()[:40]
    v = ctx.get("visitor")  # long-lived id the card keeps in localStorage: how a returning customer is recognised
    if isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9_-]{8,64}", v):
        out["visitor"] = v
    return out


# ------------------------------------------------------------------ conversation memory
RECORD = os.getenv("HUMAN_BOT_RECORD", "1").lower() in ("1", "true", "yes")   # store transcripts + audio
# Returning-customer recall (earlier conversations by visitor id -> prompt + "wapas swagat hai" opener). Off since
# 2026-11-06 at the owner's request ("remove application memory"): every call starts fresh. Recording and the CRM
# feed are unaffected; the visitor id is still stored so the CRM can link a customer's calls. HUMAN_BOT_RECALL=1 restores it.
RECALL = os.getenv("HUMAN_BOT_RECALL", "0").lower() in ("1", "true", "yes")
ADMIN_KEY = os.getenv("HUMAN_BOT_ADMIN_KEY", "")                               # /conversations off loopback
DISCLOSURE = {"service": "Yeh baat-cheet quality ke liye record hoti hai.",
              "sales": "Yeh baat-cheet quality ke liye record hoti hai."}
# Spoken disclosure off by default since 2026-11-13 (owner: "esse nhi bolna h"): the card's footer note
# ("Yeh baat-cheet record hoti hai") is the written notice instead. HUMAN_BOT_SPOKEN_DISCLOSURE=1 speaks it again.
SPOKEN_DISCLOSURE = os.getenv("HUMAN_BOT_SPOKEN_DISCLOSURE", "0").lower() in ("1", "true", "yes")
STORE: ConversationStore | None = ConversationStore() if RECORD else None
# CRM: every ended conversation is queued in the outbox (same SQLite file) and pushed to Bitrix24 by a worker
# with retries. Without BITRIX24_WEBHOOK_URL the rows wait; they flush once it is configured.
CRM_SINK = bitrix_sink_from_env()
OUTBOX: Outbox | None = Outbox(STORE.path, CRM_SINK) if STORE else None
# Recording off (HUMAN_BOT_RECORD=0) must not stop the leads (2026-11-25): a filed booking still goes to Bitrix24,
# with the booking details only (no transcript, no audio), through a small outbox of its own.
LEAD_OUTBOX: Outbox | None = None
if not STORE and CRM_SINK:
    (HERE.parent / "data").mkdir(parents=True, exist_ok=True)
    LEAD_OUTBOX = Outbox(HERE.parent / "data" / "crm_leads_only.sqlite", CRM_SINK)


def booking_lead(s: "Session") -> dict:
    """The lead record of a filed booking without the conversation (recording off)."""
    sl = dict(((s.service_request or {}).get("slots")) or {})
    sl.update({k: v for k, v in (s.bk or {}).items() if v})
    page = ((s.context or {}).get("page") or {}).get("url", "")
    return {"id": s.id, "turns": sum(1 for t in s.transcript if t.get("role") == "user"), "started": getattr(s, "started", None),
            "phone": sl.get("phone"), "name": sl.get("name"), "car": sl.get("car"), "registration": sl.get("registration"),
            "problem": (s.chosen or {}).get("title") or sl.get("problem"), "page_url": page, "intent": "booking",
            "summary": "Booking taken by Neha (recording off: booking details only)",
            "next_step": "service_request:" + str((s.service_request or {}).get("status", "")), "consent": False}
if STORE and not CRM_SINK:
    print("[crm] BITRIX24_WEBHOOK_URL not set: conversations queue in the outbox, nothing is pushed yet", file=sys.stderr, flush=True)
# Only a conversation the team can act on becomes a Bitrix24 lead: a phone or a name, a filed service request, or a
# real exchange (CRM_MIN_TURNS customer turns). A visitor who said "hello" and left is kept in the conversation
# store, not sent (2026-11-14: 70 of 104 queued records were one-line visits).
CRM_MIN_TURNS = int(os.getenv("CRM_MIN_TURNS", "3"))


def crm_worthy(rec: dict) -> bool:
    if not rec.get("turns"):
        return False
    return bool(rec.get("phone") or rec.get("name") or str(rec.get("next_step") or "").startswith("service_request:")
                or rec["turns"] >= CRM_MIN_TURNS)


def persist(s: Session, ended: bool = False) -> None:
    """Write the structured record (upsert). Called after every turn and at the end; never raises."""
    if not STORE:
        if LEAD_OUTBOX and s.crm_early and s.service_request and not getattr(s, "crm_early_sent", False):
            try:
                LEAD_OUTBOX.enqueue(booking_lead(s))
                s.crm_early_sent = True
            except Exception as ex:  # noqa: BLE001
                sys.stderr.write(f"[crm] booking lead not queued for {s.id}: {ex}\n")
        return
    with s.save_lock:  # the per-turn save thread and /end must not interleave (duplicate clip files)
        _persist(s, ended)


def _persist(s: Session, ended: bool) -> None:
    try:
        if ended and s.analysis is None:
            try:
                s.analysis = analyze(s.transcript)
            except Exception:  # noqa: BLE001
                s.analysis = None
        if s.audio_dir and RECORD:  # keep Neha's side too: clips are in the TTS cache by text
            for i, t in enumerate(s.transcript):
                if t.get("role") != "assistant" or t.get("audio"):
                    continue
                names = []
                for k, piece in enumerate(delivery_chunks(t["text"])):
                    clip = cached_clip(piece, emotion_of(piece))
                    if not clip and ended:
                        # the recording must be complete: wait for the clip still being made (or make it)
                        try:
                            clip = tts_get_or_make(piece, emotion=emotion_of(piece), wait_s=20)
                        except Exception:  # noqa: BLE001
                            clip = None
                    if clip:
                        name = f"a{i:03d}_{k}.mp3"
                        (s.audio_dir / name).write_bytes(clip)
                        names.append(name)
                if names:
                    t["audio"] = names
                    s.audio_files += [n for n in names if n not in s.audio_files]
        rec = STORE.record(s, ended=ended)
        if s.service_request:  # what was filed with the workshop (dry/queued or the live lead id)
            rec["analysis"] = {**(rec.get("analysis") or {}), "service_request": s.service_request}
            rec["next_step"] = "service_request:" + (s.service_request.get("lead_id") or s.service_request.get("status", ""))
        for k in ("phone", "name", "car", "registration"):   # the step machine's own captures (Devanagari names...)
            if (s.bk or {}).get(k) and not rec.get(k):
                rec[k] = s.bk[k]
        if s.chosen:
            rec["problem"] = s.chosen.get("title") or rec.get("problem")
        STORE.save(rec)
        # 2026-11-25 owner: the lead reaches Bitrix24 the moment the booking is filed (crm_early); the end-of-call
        # push then updates the same lead (matched by conversation id) with the whole transcript.
        early = s.crm_early and not getattr(s, "crm_early_sent", False)
        if (ended or early) and OUTBOX and crm_worthy(rec):
            OUTBOX.enqueue(rec, again=ended and s.crm_early)  # pushed by the worker, never on the request path
            if early:
                s.crm_early_sent = True
        if ended and s.audio_dir and s.audio_files:
            ordered = [f for t in s.transcript for f in ([t["audio"]] if isinstance(t.get("audio"), str) else (t.get("audio") or []))]
            def _merge():
                name = STORE.merge_audio(s.audio_dir, ordered)
                if name:
                    rec["audio_files"] = list(rec["audio_files"]) + [name]
                    STORE.save(rec)
            threading.Thread(target=_merge, daemon=True, name="audio-merge").start()
    except Exception as ex:  # noqa: BLE001
        sys.stderr.write(f"[memory] save failed for {s.id}: {type(ex).__name__}: {ex}\n")


def persona_brief(p: dict) -> dict:
    return {
        "id": p.get("id"),
        "name": p.get("display_name"),
        "company": p.get("company"),
        "location": p.get("location"),
        "business_type": p.get("business_type") or "sales",
    }



# ------------------------------------------------------------------ production guards (2026-11-02 audit)
# Origins allowed to call the API from a browser (dealer sites). "*" keeps the dev behaviour.
ALLOWED_ORIGINS = [o.strip().rstrip("/") for o in os.getenv("HUMAN_BOT_ALLOWED_ORIGINS", "*").split(",") if o.strip()]
BEHIND_PROXY = os.getenv("HUMAN_BOT_BEHIND_PROXY", "").lower() in ("1", "true", "yes")
MAX_BODY = 64 * 1024                      # JSON bodies; audio goes through /stt with its own 8 MB cap
SESSION_IDLE_S = 30 * 60                  # a session unused this long is dropped
SESSION_MAX = 5000                        # hard cap, oldest first
RATE_PER_MIN = int(os.getenv("HUMAN_BOT_RATE_PER_MIN", "240"))      # API calls per IP per minute
SESSION_PER_MIN = int(os.getenv("HUMAN_BOT_SESSIONS_PER_MIN", "20"))  # /session per IP per minute
TTS_OPEN = os.getenv("HUMAN_BOT_TTS_OPEN", "").lower() in ("1", "true", "yes")  # dev: /tts speaks any text
RATE: dict[str, list] = {}                # ip -> [window_start, api_calls, session_calls]
RATE_LOCK = threading.Lock()
KNOWN_TTS: "OrderedDict[str, None]" = OrderedDict()  # texts this server produced: the only ones /tts will speak
KNOWN_TTS_MAX = 20000
KNOWN_LOCK = threading.Lock()
LOOPBACK = ("127.0.0.1", "::1", "::ffff:127.0.0.1")


def remember_tts_text(text: str) -> None:
    with KNOWN_LOCK:
        KNOWN_TTS[text] = None
        KNOWN_TTS.move_to_end(text)
        while len(KNOWN_TTS) > KNOWN_TTS_MAX:
            KNOWN_TTS.popitem(last=False)


def tts_text_allowed(text: str, voice: str | None) -> bool:
    if TTS_OPEN:
        return True
    with KNOWN_LOCK:
        return text in KNOWN_TTS


def rate_check(ip: str, session_call: bool = False) -> bool:
    """Fixed one-minute window per client IP. Loopback (the dev page, tests) is exempt."""
    if ip in LOOPBACK and not BEHIND_PROXY:
        return True
    now = time.time()
    with RATE_LOCK:
        rec = RATE.get(ip)
        if not rec or now - rec[0] > 60:
            rec = RATE[ip] = [now, 0, 0]
        rec[1] += 1
        if session_call:
            rec[2] += 1
        if len(RATE) > 10000:  # forget old windows
            for k in [k for k, v in RATE.items() if now - v[0] > 60][:5000]:
                RATE.pop(k, None)
        return rec[1] <= RATE_PER_MIN and rec[2] <= SESSION_PER_MIN


def reap_sessions() -> None:
    """Drop idle sessions so memory is bounded (transcripts + offline brains live per session).
    An idle session is closed properly first: its record is finished and saved."""
    while True:
        time.sleep(60)
        now = time.time()
        with SESSIONS_LOCK:
            idle = [SESSIONS.pop(k) for k in [k for k, s in SESSIONS.items() if now - getattr(s, "last_seen", now) > SESSION_IDLE_S]]
            while len(SESSIONS) > SESSION_MAX:
                idle.append(SESSIONS.pop(next(iter(SESSIONS))))
        for s in idle:
            if s.transcript and any(t.get("role") == "user" for t in s.transcript):
                persist(s, ended=True)


class Handler(BaseHTTPRequestHandler):
    server_version = "HumanBot/1.0"
    protocol_version = "HTTP/1.1"  # keep-alive + chunked /tts, so a proxy can stream and a cut clip is detectable

    def log_message(self, fmt, *args):  # quieter logs: date, client, message
        sys.stderr.write("[%s] %s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), self.client_ip(), fmt % args))

    def send_response(self, code, message=None):
        self._responded = True
        super().send_response(code, message)

    def end_headers(self):
        # embeddable from the dealer sites (all sites when the allow-list is "*") and from this origin
        # itself: the site mirror at "/" frames /embed same-origin, which a bare allow-list would block
        ancestors = "*" if "*" in ALLOWED_ORIGINS else " ".join(["'self'"] + ALLOWED_ORIGINS)
        self.send_header("Content-Security-Policy", f"frame-ancestors {ancestors}")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    # ------------------------------------------------------------------ helpers
    def client_ip(self) -> str:
        if BEHIND_PROXY:
            fwd = self.headers.get("X-Forwarded-For") if hasattr(self, "headers") else None
            if fwd:
                return fwd.split(",")[0].strip()
        return self.client_address[0] if self.client_address else "-"

    def _cors(self) -> None:
        """Reflect the caller's origin when it is allowed; no header otherwise (the browser then blocks)."""
        origin = (self.headers.get("Origin") or "").rstrip("/")
        if "*" in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", "*")
        elif origin and origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if n < 0 or n > MAX_BODY:
            self.close_connection = True  # the unread body would poison the keep-alive connection
            raise BodyTooLarge()
        if not n:
            return {}
        try:
            obj = json.loads((self.rfile.read(n) or b"{}").decode("utf-8-sig") or "{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}
        return obj if isinstance(obj, dict) else {}

    def _sse_start(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")  # no length: the close marks the end of the stream
        self.send_header("X-Accel-Buffering", "no")
        self._cors()
        self.end_headers()
        self.close_connection = True

    def _sse(self, event: str, data) -> None:
        try:
            self.wfile.write(f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            raise

    # ------------------------------------------------------------ site preview
    # /site shows the persona's real website with the launcher injected, before the client adds
    # the one-line embed. The site's own paths (/assets/..., SPA routes) are proxied through this
    # origin because its module bundles are served without CORS headers and would not load
    # cross-origin. Their analytics tags are stripped so previews never hit the real GA property.
    def _site_origin(self) -> str:
        p = getattr(self.server, "persona", PERSONA)
        return ((p.get("knowledge") or {}).get("website") or p.get("website") or "").rstrip("/")

    def _public_base(self) -> str:
        """This server as the browser reaches it. X-Forwarded-* is trusted only behind a configured proxy."""
        host = (self.headers.get("X-Forwarded-Host") if BEHIND_PROXY else None) or self.headers.get("Host") or "127.0.0.1"
        proto = (self.headers.get("X-Forwarded-Proto") if BEHIND_PROXY else None) or "http"
        return f"{proto}://{host}"

    def _inject_widget(self, html: str) -> str:
        import re
        p = getattr(self.server, "persona", PERSONA)
        base = self._public_base()
        html = re.sub(r'<script[^>]+googletagmanager\.com[^>]*>\s*</script>', "", html, flags=re.I)
        html = re.sub(r'<script>(?:(?!</script>).)*?\b(?:gtag|fbq)\((?:(?!</script>).)*?</script>', "", html, flags=re.S)
        html = re.sub(r'<noscript>(?:(?!</noscript>).)*?facebook\.com/tr(?:(?!</noscript>).)*?</noscript>', "", html, flags=re.S)
        # preview only: hide the site's existing chat/WhatsApp launcher so ours owns the corner
        tag = ('<style data-hhbot-preview>.fixed.right-4.bottom-24.z-40{display:none!important}</style>'
               f'<script src="{base}/widget.js" data-persona="{p.get("id")}" '
               f'data-accent="#0b2f6b" data-position="{os.getenv("HUMAN_BOT_WIDGET_SIDE", "left")}" defer></script>')
        # owner 2026-11-25: bubble on the left (clear of the site's chat pill on phones); INSTALL_ON_WEBSITE.md uses the same
        return html.replace("</body>", tag + "</body>", 1) if "</body>" in html else html + tag

    def _rewrite_origin(self, text: str, origin: str) -> str:
        """The SPA bundle hardcodes its own absolute origin for API calls; point those at this
        server so they are same-origin and proxied instead of failing CORS."""
        mine = self._public_base()
        bare = origin.split("://", 1)[-1].removeprefix("www.")
        for o in (f"https://www.{bare}", f"https://{bare}", f"http://www.{bare}", f"http://{bare}"):
            text = text.replace(o, mine)
        return text

    def _proxy_site(self, path: str, query: str) -> bool:
        """GET fallback: fetch `path` from the persona website and relay it. Returns False when
        no website is configured so the caller can 404 as before."""
        import urllib.error
        import urllib.request
        origin = self._site_origin()
        if not origin:
            return False
        url = origin + path + (f"?{query}" if query else "")
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (human_bot preview)",
                "Accept": self.headers.get("Accept") or "*/*",
            })
            with urllib.request.urlopen(req, timeout=20) as r:
                body = r.read()
                ctype = r.headers.get("Content-Type") or "application/octet-stream"
        except urllib.error.HTTPError as ex:
            self._json(ex.code, {"error": f"upstream {ex.code} for {path}"})
            return True
        except (urllib.error.URLError, TimeoutError, OSError) as ex:
            self._json(502, {"error": f"could not fetch {url}: {ex}"})
            return True
        kind = ctype.split(";")[0].strip()
        if kind == "text/html":
            html = self._rewrite_origin(body.decode("utf-8", "replace"), origin)
            body = self._inject_widget(html).encode("utf-8")
            ctype = "text/html; charset=utf-8"
        elif kind in ("application/x-javascript", "application/javascript", "text/javascript"):
            body = self._rewrite_origin(body.decode("utf-8", "replace"), origin).encode("utf-8")
            ctype = "text/javascript; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if kind == "text/html":
            # 2026-11-25 demo safety: the preview must not report fake visits / "conversions" to the dealer's Google
            # Ads, Analytics or Facebook pixel. Only the site's own origin (images, rewritten API) and this server
            # are allowed; trackers are blocked by the browser. The live site is untouched.
            self.send_header("Content-Security-Policy",
                             f"default-src 'self' {origin} data: blob:; "
                             f"script-src 'self' 'unsafe-inline' 'unsafe-eval' {origin}; "
                             f"style-src 'self' 'unsafe-inline' {origin} https://fonts.googleapis.com; "
                             f"font-src 'self' {origin} https://fonts.gstatic.com data:; "
                             f"img-src 'self' {origin} data: blob:; media-src 'self' {origin} data: blob:; "
                             f"connect-src 'self' {origin}; frame-src 'self'")
        self.end_headers()
        self.wfile.write(body)
        return True

    def _site_mirror(self) -> None:
        if not self._proxy_site("/", ""):
            self._json(404, {"error": "persona has no website url (knowledge.website)"})

    def _mirror_on(self) -> bool:
        """The site preview/proxy is a dev tool: on by default only on loopback, never behind a proxy
        unless HUMAN_BOT_SITE_MIRROR=1 (it relays the dealer site through this origin)."""
        return bool(getattr(self.server, "mirror", False))

    # ------------------------------------------------------------------ routes
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Max-Age", "600")
        # a public https site embedding a bot on 127.0.0.1 is a "local network" request in Chrome;
        # the preflight must opt in (the user still sees a one-time Allow prompt)
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        self._responded = False
        try:
            return self._get()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True
        except Exception as ex:  # noqa: BLE001
            sys.stderr.write(f"[error] GET {self.path}: {type(ex).__name__}: {ex}\n")
            traceback.print_exc()
            self.close_connection = True
            if not self._responded:
                self._json(500, {"error": "internal error"})

    def do_POST(self):
        self._responded = False
        try:
            return self._post()
        except BodyTooLarge:
            self._json(413, {"error": f"body too large (max {MAX_BODY // 1024} KB)"})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True
        except Exception as ex:  # noqa: BLE001
            sys.stderr.write(f"[error] POST {self.path}: {type(ex).__name__}: {ex}\n")
            traceback.print_exc()
            self.close_connection = True
            if not self._responded:
                self._json(500, {"error": "internal error"})

    def _get(self):
        path = urlparse(self.path).path
        if path == "/health":
            p = getattr(self.server, "persona", PERSONA)
            provider = provider_name()
            degraded = []
            if provider == "offline":
                degraded.append("no live LLM provider (offline script)")
            if TTS_LAST.get("error"):
                degraded.append("tts error")
            body = {"ok": not degraded, "degraded": degraded, "persona_id": p.get("id"), "brand": p.get("company"),
                    # The card picks its microphone path from these two: with server STT it records and
                    # posts to /stt; without it, it needs the browser recogniser, which iOS does not have
                    # and Android drops after one utterance. They were loopback-only, so every public
                    # deployment silently fell back to the browser and the mic died after the first turn.
                    # Provider names only; no keys and no operational counters.
                    "stt": stt_status(), "tts": tts_status(),
                    # Avatar provider/ready status is public so the widget knows whether to initialise
                    # Simli live streaming or stay in puppet mode. No keys are exposed.
                    "avatar": {"provider": avatar_status().get("provider"), "ready": avatar_status().get("ready", False),
                               "face_id": avatar_status().get("face_id")}}
            if self.client_ip() in LOOPBACK:  # operational detail stays on the box
                body.update({
                    "provider": provider, "persona": p["display_name"], "sessions": len(SESSIONS),
                    "tts_error": TTS_LAST.get("error"),
                    "kb": kb_status(), "intents": meaning.status(), "avatar": avatar_status(),
                    "crm": OUTBOX.status() if OUTBOX else {"sink": None, "note": "recording off"},
                    "leads": {"mode": acr_leads.mode(), "url": acr_leads.LEADS_URL, "dry_runs": len(acr_leads.LOG)},
                })
            # Always 200: "degraded" is information for the operator, not a reason for a host's health check
            # to kill the process (Render restarted the bot mid-conversation every time the free LLM keys hit
            # 429, which the offline script covers anyway). ?strict=1 keeps the 503 for a monitoring probe.
            strict = (parse_qs(urlparse(self.path).query).get("strict") or ["0"])[0] in ("1", "true")
            return self._json(503 if strict and not body["ok"] else 200, body)
        if path == "/conversations" or path.startswith("/conversations/"):
            return self._conversations(path)
        if path == "/personas":
            default = getattr(self.server, "persona", PERSONA)
            return self._json(200, {
                "default": default.get("id"),
                "personas": [persona_brief(p) for p in PERSONAS.values()],
            })
        if path == "/tts":
            return self._tts()
        if path == "/tts_cues":
            # Lip timeline for a clip the server produced (same text/emo/voice as /tts). Waits briefly for a
            # synthesis in flight; [] means "no word timing, use the analyser".
            q = parse_qs(urlparse(self.path).query)
            text = (q.get("text") or [""])[0].strip()
            voice = (q.get("voice") or [""])[0].strip() or None
            emotion = (q.get("emo") or ["neutral"])[0].strip().lower()
            if emotion not in EMOTIONS:
                emotion = "neutral"
            if voice and voice not in ALLOWED_VOICES:
                voice = None
            if not text or len(text) > 600 or not tts_text_allowed(text, voice):
                return self._json(404, {"cues": []})
            key = tts_key(text, voice, emotion)
            deadline = time.monotonic() + 4.0
            while True:
                if key not in TTS_CUES:
                    tts_cached(key)                # a clip from the disk cache brings its cues along
                with TTS_LOCK:
                    cues = TTS_CUES.get(key)
                    inflight = TTS_INFLIGHT.get(key)
                if cues is not None or inflight is None or inflight.done or time.monotonic() > deadline:
                    break
                time.sleep(0.05)
            out = {"cues": cues or [], "map": _mouth_map()}
            if AVATAR_V2 or (q.get("v2") or [""])[0] in ("1", "true"):
                out["v2"] = viseme_v2(key, cues or [])
            return self._json(200, out)
        if path == "/site":
            # the SPA router must see "/" or it renders its own 404; the mirror lives at "/"
            self.send_response(302)
            self.send_header("Location", "/")
            self.end_headers()
            return None
        if path == "/" and self._site_origin() and self._mirror_on():
            # persona has a website: "/" previews the real site with the launcher injected;
            # the legacy full-page UI stays reachable at /index.html
            return self._site_mirror()
        if path in ("/widget.js",):
            f = WEB / "widget.js"
        elif path in ("/embed", "/embed.html"):
            f = WEB / "embed.html"
        elif path in ("/demo", "/demo.html"):
            f = WEB / "demo.html"
        elif path in ("/clips", "/clips.html"):
            f = WEB / "clips.html"
        elif path in ("/", "/index.html"):
            f = WEB / "index.html"
        elif path.startswith("/avatar_v2/"):
            return self._avatar_v2(path[len("/avatar_v2/"):])
        elif path == "/avatar.jpg":   # launcher bubble + header thumbnail: the portrait on black when it exists
            f = HERE.parents[1] / "tran_moment" / "prepared" / "face_black.jpg"
            if not f.is_file():
                f = HERE.parents[1] / "tran_moment" / "prepared" / "face.jpg"
        elif path.startswith("/avatar_puppet/"):
            # 2D puppet assets (tran_moment/path_b): aligned mouth-shape layers + mouths.json, and the
            # landmark file for eyelids. Whitelisted names only.
            name = path[len("/avatar_puppet/"):]
            # <name>.png | <name>.json | <expression>/fNN.webp (morph frames)
            # jpg: the lip grid atlas; webp: the same layers at ~1-2% of the PNG size (2026-11-26 speed)
            if not re.fullmatch(r"[A-Za-z0-9_]+\.(png|json|jpg|webp)|[A-Za-z0-9_]+/f\d{2}\.webp", name):
                return self._json(404, {"error": "no such asset"})
            base = HERE.parents[1] / "tran_moment"
            if name == "face.json":
                f = base / "prepared" / "face.json"
            elif name in ("mouths.json", "reactions.json", "expressions.json"):
                f = base / "path_b" / name
            else:  # mouth shapes first, then expression layers (from the grid), then the warp reaction layers
                for sub in ("mouths", "expressions", "reactions"):
                    f = base / "path_b" / sub / name
                    if f.is_file():
                        break
            if not f.is_file():
                if name == "expressions.json":   # optional manifest (no expression grid for this portrait): no console 404
                    return self._json(200, {})
                return self._json(404, {"error": "no such asset"})
            body = f.read_bytes()
            if name.endswith(".json"):
                # The manifests carry the asset version: the card appends ?v=<version> to every image URL, so
                # a rebuilt portrait is never mixed with images a browser cached from the previous one
                # (seen 2026-11-03: old base.png under new expression layers).
                try:
                    obj = json.loads(body.decode("utf-8"))
                    obj["version"] = puppet_version()
                    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                except ValueError:
                    pass
            self.send_response(200)
            self.send_header("Content-Type", "image/png" if name.endswith(".png") else "image/webp" if name.endswith(".webp") else "image/jpeg" if name.endswith(".jpg") else "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            if name.endswith(".json") or not parse_qs(urlparse(self.path).query).get("v"):
                self.send_header("Cache-Control", "no-cache, must-revalidate")   # manifests and unversioned images: revalidate
                self.send_header("ETag", '"%x-%x"' % (f.stat().st_mtime_ns, len(body)))
            else:
                self.send_header("Cache-Control", "public, max-age=31536000, immutable")  # versioned image URL
            self._cors()  # the voice widget on another origin reads face.json for its blink/mouth placement
            self.end_headers()
            return self.wfile.write(body)
        elif path.startswith("/avatar_clips/"):
            # reaction clips + manifest (tran_moment/prepared/clips); names are whitelisted, no traversal
            name = path[len("/avatar_clips/"):]
            if not re.fullmatch(r"[a-z0-9_]+\.(mp4|json)", name):
                return self._json(404, {"error": "no such clip"})
            # ?set=neha (default: synthesized from the CGI portrait) | ?set=sample (presenter footage, test only)
            clip_set = parse_qs(urlparse(self.path).query).get("set", ["neha"])[0]
            folder = {"neha": "clips_neha", "neha_video": "clips_neha", "neha_render": "clips_neha_render", "sample": "clips"}.get(clip_set)
            if not folder:
                return self._json(404, {"error": "unknown clip set"})
            f = HERE.parents[1] / "tran_moment" / "prepared" / folder / name
            if not f.is_file():
                return self._json(404, {"error": "no such clip"})
            body = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4" if name.endswith(".mp4") else "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            return self.wfile.write(body)
        elif path == "/avatar_idle.mp4":
            # optional pre-rendered idle loop (breathing, blinks, sway) of the same portrait;
            # the card falls back to CSS motion on the still when this 404s
            f = HERE.parents[1] / "tran_moment" / "prepared" / "idle.mp4"
            if not f.is_file():
                return self._json(404, {"error": "no idle loop rendered yet"})  # never fall through to the site proxy
        elif path == "/avatar_bubble.mp4":
            # the launcher bubble's living portrait: a 200x200 muted ping-pong loop of the same Neha portrait
            # (tran_moment/prepared/bubble.mp4, ~40 KB). widget.js shows the still /avatar.jpg when this 404s.
            f = HERE.parents[1] / "tran_moment" / "prepared" / "bubble.mp4"
            if not f.is_file():
                return self._json(404, {"error": "no bubble loop"})
            return self._audio(f.read_bytes(), "video/mp4", cache="public, max-age=3600")   # a re-render shows within the hour
        else:
            # Anything else is a file inside human_bot/web only. The path is resolved and confined:
            # "/web/../../voice_clone/.env" used to be served (2026-11-02 audit).
            rel = path.lstrip("/").removeprefix("human_bot/").removeprefix("web/")
            f = (WEB_ROOT / rel).resolve()
            if not f.is_relative_to(WEB_ROOT) or f.suffix not in STATIC_TYPES or f.name.startswith("."):
                f = None
        if f is not None and f.is_file():
            body = f.read_bytes()
            if AVATAR_V2 and f.name == "embed.html":   # HUMAN_BOT_AVATAR_V2=1: v2 is the card's default (?avatar=v1 opts out)
                body = body.replace(b"/*AV2_DEFAULT*/false", b"true")
            ctype = STATIC_TYPES.get(f.suffix, "application/octet-stream")
            text = f.suffix in (".html", ".js", ".css", ".json", ".svg")
            etag = '"%x-%x"' % (f.stat().st_mtime_ns, len(body))
            # Widget code and pages change often during development: browsers revalidate on every load
            # (ETag -> 304, no body) so a reload always shows the current card. Media stays cacheable,
            # except the portrait itself, which must follow a rebuild immediately.
            if text or path == "/avatar.jpg":
                # weak compare: a proxy (Render) turns our ETag into W/"..." and the browser echoes that form
                inm = [t.strip().removeprefix("W/") for t in (self.headers.get("If-None-Match") or "").split(",")]
                if etag in inm:
                    self.send_response(304)
                    self.send_header("ETag", etag)
                    self.send_header("Cache-Control", "no-cache, must-revalidate")
                    self._cors()
                    self.end_headers()
                    return None
            # text assets gzip 3-4x (the 93 KB card is ~25 KB on the wire): less time on a phone network
            encoding = None
            if text and len(body) > 2048 and "gzip" in (self.headers.get("Accept-Encoding") or ""):
                body = gzip.compress(body, 6)
                encoding = "gzip"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            if encoding:
                self.send_header("Content-Encoding", encoding)
                self.send_header("Vary", "Accept-Encoding")
            self._cors()
            if text or path == "/avatar.jpg":
                self.send_header("Cache-Control", "no-cache, must-revalidate")
                self.send_header("ETag", etag)
            else:
                self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            return self.wfile.write(body)
        if self._mirror_on() and self._proxy_site(path, urlparse(self.path).query):
            return None
        return self._json(404, {"error": "not found"})

    def _avatar_v2(self, name: str) -> None:
        """Avatar v2 reference pack (tran_moment/avatar_v2/pack): manifest, atlas, own base/landmarks for a new
        portrait, the per-viseme reference frames. Whitelisted names only; versioned image URLs are immutable."""
        if not re.fullmatch(r"(pack\.json|atlas\.jpg|base\.png|face\.json|sheet\.png|frames/[A-Za-z]{1,3}\.png)", name):
            return self._json(404, {"error": "no such asset"})
        f = AVATAR_V2_DIR / name
        if not f.is_file():
            return self._json(404, {"error": "no such asset"})
        body = f.read_bytes()
        if name == "pack.json":
            try:   # the card asks for the default only through here: HUMAN_BOT_AVATAR_V2
                obj = json.loads(body.decode("utf-8"))
                obj["default"] = AVATAR_V2
                body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            except ValueError:
                pass
        ctype = {".json": "application/json; charset=utf-8", ".jpg": "image/jpeg", ".png": "image/png"}[f.suffix]
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if f.suffix == ".json" or not parse_qs(urlparse(self.path).query).get("v"):
            self.send_header("Cache-Control", "no-cache, must-revalidate")
        else:
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        self._cors()
        self.end_headers()
        return self.wfile.write(body)

    def _avatar(self, path: str, body: dict) -> None:
        """Browser ↔ avatar-vendor relay for the live talking-head stream (keys stay here)."""
        if simli_avatar.enabled():
            # Simli: server mints a session token; the browser SDK does WebRTC and we stream PCM audio.
            # The face is the server's: a client-chosen face_id would bill minutes on any face.
            try:
                if path == "/avatar/start":
                    return self._json(200, simli_avatar.start_stream(None))
                if path == "/avatar/faces":
                    return self._json(200, {"faces": simli_avatar.faces(), "default": simli_avatar.FACE_ID})
                if path in ("/avatar/stop", "/avatar/say", "/avatar/sdp", "/avatar/ice"):
                    return self._json(200, {"ok": True, "note": "handled in the browser SDK for simli"})
            except simli_avatar.AvatarUnavailable as ex:
                return self._json(503, {"error": str(ex)[:300], "fallback": "puppet"})
            return self._json(404, {"error": "not found"})
        if not did_avatar.enabled():
            return self._json(503, {"error": "avatar stream disabled", "fallback": "puppet"})
        sid, sess = str(body.get("id") or ""), str(body.get("session_id") or "")
        try:
            if path == "/avatar/start":
                return self._json(200, did_avatar.start_stream())
            if not sid:
                return self._json(400, {"error": "id required"})
            if path == "/avatar/sdp":
                return self._json(200, did_avatar.send_answer(sid, sess, body.get("answer") or {}))
            if path == "/avatar/ice":
                return self._json(200, did_avatar.send_ice(sid, sess, body.get("candidate") or {}))
            if path == "/avatar/say":
                text = str(body.get("text") or "").strip()
                if not text:
                    return self._json(400, {"error": "text required"})
                if not tts_text_allowed(text, None):
                    return self._json(403, {"error": "only text produced by this server can be spoken"})
                return self._json(200, did_avatar.say(sid, sess, text, None))  # server-side voice only (female)
            if path == "/avatar/stop":
                return self._json(200, did_avatar.stop_stream(sid, sess))
        except did_avatar.AvatarUnavailable as ex:
            return self._json(503, {"error": str(ex)[:300], "fallback": "puppet"})
        return self._json(404, {"error": "not found"})

    def _stt(self) -> None:
        """Raw audio body (webm/opus, mp4 or wav) -> {"text": ...}; 503 tells the card to use the browser recogniser."""
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        mime = (self.headers.get("Content-Type") or "audio/webm").split(";")[0].strip() or "audio/webm"
        if not n or n > 8_000_000:
            self.close_connection = True  # body not consumed
            return self._json(400, {"error": "audio body required (max 8 MB)"})
        audio = self.rfile.read(n)
        t0 = time.perf_counter()
        try:
            text = transcribe(audio, mime)
        except STTUnavailable as ex:
            return self._json(503, {"error": str(ex)[:200]})
        # Recording: the raw utterance is kept with the conversation when the card tells us which one it is.
        sid = (parse_qs(urlparse(self.path).query).get("session_id") or [""])[0]
        s = get_session(sid) if sid else None
        if s and text.strip():
            s.prefetch_knowledge(text)  # the kb lookup runs while the card is still deciding the turn has ended
        if s and s.audio_dir and text.strip():
            ext = "webm" if "webm" in mime else "mp4" if "mp4" in mime else "wav"
            name = f"u{len(s.audio_files):03d}.{ext}"
            try:
                (s.audio_dir / name).write_bytes(audio)
                s.audio_files.append(name)
                s.pending_audio = name
            except OSError:
                pass
        return self._json(200, {"text": text, "ms": int((time.perf_counter() - t0) * 1000), **eot_fields(text)})

    def _admin_ok(self) -> bool:
        return self.client_ip() in LOOPBACK or (bool(ADMIN_KEY) and self.headers.get("X-Admin-Key") == ADMIN_KEY)

    def _conversations(self, path: str) -> None:
        """Structured records for the CRM / dashboard. Loopback or X-Admin-Key only."""
        if not STORE:
            return self._json(404, {"error": "recording is off (HUMAN_BOT_RECORD=0)"})
        if not self._admin_ok():
            return self._json(403, {"error": "admin only"})
        q = parse_qs(urlparse(self.path).query)
        rest = path[len("/conversations"):].strip("/")
        if not rest:
            rows = STORE.list(limit=int((q.get("limit") or ["50"])[0] or 50), visitor_id=(q.get("visitor") or [None])[0],
                              phone=(q.get("phone") or [None])[0])
            return self._json(200, {"conversations": rows, "count": len(rows)})
        parts = rest.split("/")
        conv = STORE.get(parts[0])
        if not conv:
            return self._json(404, {"error": "no such conversation"})
        if len(parts) == 1:
            return self._json(200, conv)
        if len(parts) == 3 and parts[1] == "audio" and re.fullmatch(r"[A-Za-z0-9_.-]+", parts[2]) and conv.get("audio_dir"):
            f = Path(conv["audio_dir"]) / parts[2]
            if f.is_file():
                body = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", {"mp3": "audio/mpeg", "webm": "audio/webm", "mp4": "audio/mp4", "m4a": "audio/mp4", "wav": "audio/wav"}.get(f.suffix[1:], "application/octet-stream"))
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "private, no-store")
                self.end_headers()
                return self.wfile.write(body)
        return self._json(404, {"error": "not found"})

    def _post(self):
        path = urlparse(self.path).path
        ip = self.client_ip()
        if not rate_check(ip, session_call=(path == "/session")):
            self.close_connection = True
            return self._json(429, {"error": "too many requests"})
        if path == "/stt":
            return self._stt()
        body = self._body()

        if path == "/api/v1/leads" and self._mirror_on():
            # 2026-11-25 demo: in the LOCAL preview the dealer site's own form posts here (its origin is rewritten),
            # never to the live API. Answer like the site's API so the demo shows its real "request sent" screen;
            # nothing is forwarded (Neha's own lead goes through crm/acr_leads.py in its configured mode).
            demo_id = "DEMO-" + uuid.uuid4().hex[:8].upper()
            sys.stderr.write(f"[mirror] site form lead kept locally, NOT sent to the live site ({demo_id}, {len(body)} bytes)\n")
            return self._json(200, {"success": True, "lead_id": demo_id, "message": "Demo: request received locally (not sent)"})

        if path.startswith("/avatar/"):
            return self._avatar(path, body)
        if path == "/session":
            default = getattr(self.server, "persona", PERSONA)
            pid = body.get("persona")
            persona = (PERSONAS.get(pid) if isinstance(pid, str) else None) or default
            context = clean_context(body.get("sessionContext") or body.get("context") or {})
            s = new_session(context, persona)
            model = ((context.get("page") or {}).get("model") or "").strip()
            model = model.replace("Hyundai ", "").replace("hyundai ", "")
            # the loader derives "model" from the page slug; only a real car name may enter the greeting
            # ("Aapki Demo.Html mein kya kaam karwana hai?" was heard on the demo page)
            if model and not any(w in KNOWN_CARS for w in model.lower().split()):
                model = ""
            service = (persona.get("business_type") or "sales") == "service"
            # returning customer: earlier conversations by visitor id feed the prompt and the greeting
            s.visitor_id = context.get("visitor", "")
            if STORE and RECALL and s.visitor_id:
                s.recalled = STORE.recall(visitor_id=s.visitor_id, limit=3)
                s.recall_block = STORE.recall_block(s.recalled)
            last = s.recalled[0] if s.recalled else None
            if last and (last.get("car") or last.get("problem")):
                car = last.get("car") or "gaadi"
                if service and last.get("problem"):
                    opener = (f"Namaste, wapas swagat hai! Main {persona['display_name']}. Pichli baar aapki {car} ki "
                              f"{last['problem']} ki baat hui thi. Kya woh theek ho gayi?")
                elif service:
                    opener = (f"Namaste, wapas swagat hai! Main {persona['display_name']}. Pichli baar aapki {car} ki baat hui thi. "
                              "Aaj kya karwana hai?")
                else:
                    opener = (f"Namaste, wapas swagat hai! Main {persona['display_name']}. Pichli baar aap {car} dekh rahe the. "
                              "Wahin se aage baat karein?")
            elif model and service:
                opener = (f"Namaste! Main {persona['display_name']}, {persona['company']} se. "
                          f"Aapki {model} mein kya kaam karwana hai?")
            elif model:
                opener = (f"Namaste! Main {persona['display_name']}, {persona['company']} se. "
                          f"{model} ke baare mein hi baat karni hai?")
            elif service:
                opener = (f"Namaste! Main {persona['display_name']}, {persona['company']} se. "
                          "Aapki car mein kya problem hai?")
            else:
                opener = (f"Namaste! Main {persona['display_name']}, {persona['company']} se. "
                          "Aap kaunsi car dekh rahe hain?")
            greeting = humanize(
                opener,
                # a returning-customer opener carries the recap sentence; the 14-word cap would cut it
                max_words=32 if last else persona["conversation"]["max_words_first_turn"],
                language=s.language,
                female=is_female(persona),
            )
            chunks = [{"text": c, "emotion": emotion_of(c),
                       "gesture": pick_gesture(persona["avatar"]["gestures"], emotion_of(c), i)}
                      for i, c in enumerate(delivery_chunks(greeting))]
            if STORE:
                # Recording is disclosed in the first breath (DPDP): a separate short chunk after the greeting.
                if SPOKEN_DISCLOSURE:
                    line = DISCLOSURE["service" if service else "sales"]
                    chunks.append({"text": line, "emotion": "neutral", "gesture": "small_nod"})
                    greeting = greeting + " " + line
                s.consent = True  # disclosed (spoken, or the card's footer note); continuing is the customer's consent
                s.audio_dir = STORE.audio_dir_for(s.id)
            s.transcript.append({"role": "assistant", "text": greeting, "t": time.time()})
            # start the greeting's clips now; the card asks for them a few hundred ms from here
            for c in chunks:
                prefetch_tts([c["text"]], emotion=c["emotion"])
            return self._json(200, {
                "session_id": s.id,
                "greeting": greeting,
                "chunks": chunks,
                "provider": provider_name(),
                "persona": persona_brief(persona),
                "returning": bool(s.recalled),
                "recording": bool(STORE),
            })

        if path == "/end":
            sid = body.get("session_id")
            s = get_session(sid) if isinstance(sid, str) else None
            if not s:
                return self._json(404, {"error": "unknown session"})
            with SESSIONS_LOCK:
                SESSIONS.pop(s.id, None)
            if any(t.get("role") == "user" for t in s.transcript):
                try:
                    s.analysis = analyze(s.transcript)
                except Exception:  # noqa: BLE001
                    s.analysis = None
                # the record (clips, merge) completes in the background; the card is not kept waiting
                threading.Thread(target=persist, args=(s, True), daemon=True, name="memory-end").start()
                return self._json(200, {"ok": True, "conversation_id": s.id, "turns": sum(1 for t in s.transcript if t.get("role") == "user"),
                                        "lead": (s.analysis or {}).get("lead_status")})
            return self._json(200, {"ok": True, "conversation_id": None, "turns": 0})

        if path == "/chat":
            return self._chat(body)

        if path == "/lead":
            return self._lead(body)
        if path == "/lead/site":
            return self._lead_site(body)

        if path == "/analyze":
            sid = body.get("session_id")
            s = get_session(sid) if isinstance(sid, str) else None
            if not s:
                return self._json(404, {"error": "unknown session"})
            return self._json(200, analyze(s.transcript))

        return self._json(404, {"error": "not found"})

    # ------------------------------------------------------------------ voice
    def _tts(self) -> None:
        q = parse_qs(urlparse(self.path).query)
        text = (q.get("text") or [""])[0].strip()
        voice = (q.get("voice") or [""])[0].strip() or None
        if not text or len(text) > 600:
            return self._json(400, {"error": "text required (max 600 chars)"})
        if not rate_check(self.client_ip()):
            return self._json(429, {"error": "too many requests"})
        if voice and voice not in ALLOWED_VOICES:
            voice = None  # female Indian voices only; an unknown voice falls back to the server's
        emotion = (q.get("emo") or ["neutral"])[0].strip().lower()
        if emotion not in EMOTIONS:
            emotion = "neutral"
        if not tts_text_allowed(text, voice):
            # Only sentences this server produced are synthesized: /tts is not an open text-to-speech API.
            return self._json(403, {"error": "only text produced by this server can be spoken"})
        job = tts_job(text, voice, emotion)
        if (q.get("prefetch") or [""])[0]:
            return self._json(202, {"started": True})  # the card warms a sentence; it will play it from /tts later
        with TTS_LOCK:
            cached = tts_cached(tts_key(text, voice, emotion))
        if cached:
            return self._audio(cached)  # whole clip: Content-Length + byte ranges (iOS probes with Range)
        # Stream: the first chunk arrives ~1 s in; the browser's <audio> starts playing it while the rest
        # is still being synthesized. Chunked framing lets a proxy pass it through and the browser tell a
        # finished clip from a cut one.
        chunks = job.stream()
        try:
            first = next(chunks)
        except (TTSUnavailable, StopIteration) as ex:
            return self._json(503, {"error": str(ex)[:220] or "empty audio", "fallback": "browser"})
        self.send_response(200)
        self.send_header("Content-Type", self._audio_type(first))
        self.send_header("Cache-Control", "private, max-age=86400")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("X-Accel-Buffering", "no")
        self._cors()
        self.end_headers()
        try:
            self._chunk(first)
            for chunk in chunks:
                self._chunk(chunk)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TTSUnavailable):
            self.close_connection = True  # the card cut in (barge-in) or Edge died mid-clip: the stream ends unterminated

    def _chunk(self, data: bytes) -> None:
        self.wfile.write(b"%x\r\n" % len(data))
        self.wfile.write(data)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    @staticmethod
    def _audio_type(audio: bytes) -> str:
        return "audio/wav" if audio[:4] == b"RIFF" else "audio/mpeg" if audio[:3] == b"ID3" or audio[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2") else "application/octet-stream"

    def _audio(self, audio: bytes, ctype: str | None = None, cache: str = "private, max-age=86400") -> None:
        """A complete clip, with single-range support (iOS Safari probes media with Range: bytes=0-1).
        Also serves the launcher's bubble video (ctype video/mp4): iOS plays no <video> without ranges."""
        ctype = ctype or self._audio_type(audio)
        start, end = 0, len(audio) - 1
        rng = self.headers.get("Range") or ""
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
        partial = False
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), len(audio) - 1)
            else:  # suffix range: last N bytes
                start = max(0, len(audio) - int(m.group(2)))
            if start > end or start >= len(audio):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(audio)}")
                self.send_header("Content-Length", "0")
                self._cors()
                return self.end_headers()
            partial = True
        body = audio[start:end + 1]
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(audio)}")
        self.send_header("Cache-Control", cache)
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------------ chat
    def _file_service_request(self, s: Session, slots: dict) -> str:
        """Submit the collected request (dry or live) and return exactly what Neha says about it.
        Never a made-up reference: a number is spoken only when the backend returned one."""
        page = ((s.context or {}).get("page") or {}).get("url", "")
        en = s.language == "english"
        phone = slots.get("phone", "")
        digits = " ".join(phone) if phone else ""
        fill = site_form.payload(slots)
        s.crm_early = True     # Bitrix24 lead now, not at the end of the call
        t = threading.Timer(1.0, persist, args=(s,))   # after s.service_request is set below (same request, ~ms)
        t.daemon = True
        t.start()
        if site_form.usable(s.context) and site_form.complete(fill):
            # The launcher fills the website's own booking form and submits it (the site posts its usual lead);
            # the server does not post a second copy. /lead/site reports the outcome (or asks for the fallback).
            # Only with a known brand + model: the site cannot pass its vehicle step without them.
            s.site_fill = fill
            s.service_request = {"status": "site_form", "mode": "site_form", "slots": slots, "t": time.time()}
            s.sr_pending = False
            if en:
                return (f"Sure, I'm filling in the website's booking form with your details: {sr_summary(slots, english=True)}. "
                        f"Our team will call you on {digits} shortly.")
            return (f"Ji, main website ka booking form aapki details se bhar ke submit kar rahi hoon: {sr_summary(slots)}. "
                    f"Team aapko {digits} par jaldi call karegi.")
        try:
            res = acr_leads.submit(slots, s.id, page)
        except acr_leads.LeadError as ex:
            sys.stderr.write(f"[leads] submit failed for {s.id}: {ex}\n")
            s.service_request = {"status": "failed", "error": str(ex)[:200], "slots": slots, "t": time.time()}
            s.sr_pending = False
            if en:
                return "Sorry, the request didn't go through just now. I'll have the team call you; your number is noted."
            return "Maaf kijiye, abhi request system mein nahi ja paayi. Main team se aapko call karwa deti hoon, aapka number note hai."
        s.service_request = {"status": "sent" if res["sent"] else "queued", "lead_id": res.get("lead_id"), "mode": res["mode"], "slots": slots, "t": time.time()}
        s.sr_pending = False
        if en:
            ref = f" Your request number is {res['lead_id']}." if res.get("lead_id") else ""
            return (f"Done.{ref} Your request is noted: {sr_summary(slots, english=True)}. "
                    f"Our team will call you on {digits} shortly.")
        if res.get("lead_id"):
            return (f"Done ji. Aapki request number {res['lead_id']} hai, note kar lijiye. "
                    f"{sr_summary(slots)}. Team aapko {digits} par jaldi call karegi.")
        return (f"Done ji. Aapki request note ho gayi: {sr_summary(slots)}. "
                f"Team aapko {digits} par jaldi call karegi.")

    def _lead(self, body: dict) -> None:
        """The card's request form was submitted: validate, file it like a spoken request, return Neha's line."""
        sid = body.get("session_id")
        s = get_session(sid) if isinstance(sid, str) else None
        if not s:
            return self._json(404, {"error": "unknown session"})
        if s.service_request:
            en = s.language == "english"
            return self._json(409, {"error": "already filed", "voice": EDGE_VOICE_EN if en else None,
                                    "say": "Your request is already noted; our team will call you shortly." if en else
                                           "Aapki request pehle hi note ho chuki hai, team aapko jaldi call karegi."})
        form, errors = validate_form(body if isinstance(body, dict) else {})
        if errors:
            return self._json(400, {"error": "invalid", "fields": errors})
        slots = {**sr_collect(s.transcript), **form}   # the form wins; facts said aloud fill the rest (when, pickup)
        s.last_seen = time.time()
        with s.lock:
            s.generation += 1                             # a reply still streaming is superseded by the submission
        say = self._file_service_request(s, slots)
        remember_tts_text(say)          # /tts speaks only server-produced text
        prefetch_tts([say], voice=EDGE_VOICE_EN if s.language == "english" else None, emotion="warm")
        s.transcript.append({"role": "user", "text": "[form] " + sr_summary(slots), "t": time.time()})
        s.transcript.append({"role": "assistant", "text": say, "t": time.time()})
        req = s.service_request or {}
        return self._json(200, {"ok": req.get("status") != "failed", "say": say, "mode": req.get("mode"),
                                "voice": EDGE_VOICE_EN if s.language == "english" else None,
                                "lead_id": req.get("lead_id"), "site_form": s.site_fill if req.get("status") == "site_form" else None})

    def _lead_site(self, body: dict) -> None:
        """The launcher's outcome for the website form: done (the site took the lead), or failed -> the server
        files the lead itself so the customer's request is never lost."""
        sid = body.get("session_id")
        s = get_session(sid) if isinstance(sid, str) else None
        if not s or not s.service_request or s.service_request.get("status") != "site_form":
            return self._json(404, {"error": "no website form pending"})
        if body.get("ok") is True:
            s.service_request.update(status="sent", via="site_form", ref=str(body.get("ref") or "")[:40])
            persist(s)
            return self._json(200, {"ok": True})
        slots = s.service_request.get("slots") or {}
        err = str(body.get("error") or "")[:120]
        sys.stderr.write(f"[leads] website form failed for {s.id} ({err}); filing it from the server\n")
        page = ((s.context or {}).get("page") or {}).get("url", "")
        try:
            res = acr_leads.submit(slots, s.id, page)
            s.service_request = {"status": "sent" if res["sent"] else "queued", "lead_id": res.get("lead_id"), "mode": res["mode"],
                                 "slots": slots, "t": time.time(), "site_form_error": err}
        except acr_leads.LeadError as ex:
            s.service_request = {"status": "failed", "error": str(ex)[:200], "slots": slots, "t": time.time(), "site_form_error": err}
        persist(s)
        return self._json(200, {"ok": s.service_request["status"] != "failed", "fallback": True})

    def _chat(self, body: dict) -> None:
        sid = body.get("session_id")
        s = get_session(sid) if isinstance(sid, str) else None
        if not s:
            return self._json(404, {"error": "unknown session"})
        text_in = body.get("text")
        user_text = (text_in if isinstance(text_in, str) else "").strip()[:2000]
        if not user_text:
            return self._json(400, {"error": "text required"})
        s.last_seen = time.time()

        with s.lock:
            s.generation += 1
            my_gen = s.generation
        turn: dict = {"role": "user", "text": user_text, "t": time.time()}
        if s.pending_audio:  # the recording /stt just transcribed belongs to this utterance
            turn["audio"] = s.pending_audio
            s.pending_audio = None
        s.transcript.append(turn)

        self._sse_start()
        t0 = time.perf_counter()
        self._sse("meta", {"generation": my_gen, "provider": provider_name()})

        persona = s.persona
        female = is_female(persona)
        # Mirror the customer's language every turn: English in -> English out (Indian English voice), Hinglish ->
        # Hinglish, Hindi -> Hindi. A turn too short to tell ("Rahul", "ok", a number) keeps the current one.
        lang = detect_language(user_text)
        # "grand i10, UP16 BX 2211" / "Swift DL01AB1234" inside the booking flow is not a switch to English
        if lang and not (getattr(s, "chosen", None) and len(re.findall(r"[A-Za-z]{4,}", user_text)) < 3):
            s.language = lang
        en = s.language == "english"
        voice = EDGE_VOICE_EN if en else None
        s.brain.en = en                      # the offline script (models down) answers in the same language
        mood = situation_of(user_text)       # the card's face for this turn: accident -> concerned, angry -> empathetic
        if mood:
            self._sse("mood", {"emotion": mood})
        bc = pick_backchannel(user_text, len(s.transcript),
                              options=EN_BACKCHANNELS if en else persona["conversation"].get("backchannels"))
        if bc:
            self._sse("backchannel", {"text": bc, "voice": voice})
            prefetch_tts([bc], voice=voice)  # normally already warm; this covers a persona added after start-up

        buf, spoken, idx = "", [], 0
        clause_done = False  # the one early clause flush has happened (or a sentence made it unnecessary)
        # "explain in detail" lifts the per-sentence cap for this one turn; the prompt tells the model the same
        max_words = (persona["conversation"].get("max_words_detail", 70) if wants_detail(user_text)
                     else persona["conversation"]["max_words_turn"])
        try:
            # Fast path first (~1 ms): a question the customer-call answers already cover skips the knowledge lookup
            # and the model. Not while Neha waits for the customer's yes/no to a read-back, and an English customer
            # only gets an answer that has an English version.
            fast = None
            # (a Hindi-script customer gets Devanagari from the model; the stored answers are Roman Hinglish)
            fp = fastpath.get() if FAST_ON and not s.sr_pending and s.language != "hindi" else None
            # a Hindi-script customer still gets the model's Devanagari answer; the saved answer is kept only as the
            # offline brain's reply for when every AI provider fails (2026-11-25 loop: "आपका ऑफिस कहाँ है" got
            # "samajh nahi paayi" while the free quota was out)
            s.brain.fast_hint = None
            if FAST_ON and not s.sr_pending and s.language == "hindi" and fastpath.get():
                _fh = fastpath.get().match(user_text)
                s.brain.fast_hint = _fh.text("hinglish") if _fh else None
            if fp:
                fast = fp.match(user_text)
                if fast and s.language == "english" and not fast.row.answer_en:
                    fast = None
            if fast:
                self._sse("meta", {"path": "fast", "kind": fast.kind, "score": round(fast.score, 3), "match_ms": round(fast.ms, 3)})
                kb_extra, hits = "", [{"url": fast.row.url, "title": fast.row.questions[0], "score": round(fast.score, 3),
                                       "page_type": "qa", "heading": " | ".join(fast.row.questions), "text": fast.row.answer}]
            else:
                # website knowledge for this turn (only personas with a `knowledge` block; empty block otherwise)
                kb_extra, hits = s.knowledge(user_text)
            if hits:
                self._sse("sources", {"results": [
                    {"url": h.get("url"), "title": h.get("title"), "score": h.get("score"), "page_type": h.get("page_type")}
                    for h in hits
                ]})
            s.brain.context = hits  # offline script quotes the website passages instead of guessing
            # Service request: the server collects the facts from the customer's words, tells the model what
            # is still missing, and files the request itself once the customer has said yes to the read-back.
            source = None
            # "insurance page dikhao": open that page of the website (llm/pages.py), the call keeps going
            # "walk me through the website" / "overview" / "what services do you offer": the persona's fixed overview,
            # with the services page opening. Correct even when the LLM is rate-limited or the KB is down.
            ov = pages_overview(user_text, persona)
            if ov:
                if NAV_ON and ov["path"]:
                    site = ((persona.get("knowledge") or {}).get("website") or "").rstrip("/")
                    self._sse("navigate", {"path": ov["path"], "title": ov["title"], "url": site + ov["path"]})
                source = iter([" ".join(ov["say"])])   # one text: the sentence splitter keeps the spaces
            page = find_page(user_text, persona, hits) if NAV_ON and source is None else None
            if page:
                site = ((persona.get("knowledge") or {}).get("website") or "").rstrip("/")
                self._sse("navigate", {"path": page["path"], "title": page["title"], "url": site + page["path"]})
                source = iter([page_line(page)])
                try:   # later answers know which page the customer is looking at
                    s.context.setdefault("page", {})["url"] = site + page["path"]
                    s.context["page"]["title"] = page["title"]
                except Exception:  # noqa: BLE001
                    pass
            # "AC service book karni hai" / "denting karwana hai": the service's own page opens by itself and Neha
            # carries on with the booking (no extra sentence; the model's answer follows). Once per page per call,
            # and not when the customer is already on that page.
            sp = service_page(with_roman(user_text), persona) if NAV_ON and source is None and not page else None
            if sp:
                site = ((persona.get("knowledge") or {}).get("website") or "").rstrip("/")
                here = str(((s.context or {}).get("page") or {}).get("url") or "").rstrip("/")
                if sp["path"] not in s.nav_opened and here != (site + sp["path"]).rstrip("/"):
                    s.nav_opened.add(sp["path"])
                    self._sse("navigate", {"path": sp["path"], "title": sp["title"], "url": site + sp["path"]})
                    try:
                        s.context.setdefault("page", {})["url"] = site + sp["path"]
                        s.context["page"]["title"] = sp["title"]
                    except Exception:  # noqa: BLE001
                        pass
            in_request = False   # a service request is being collected: replies are this customer's own, never cached
            step_q = ""
            lead_in = ""         # "Screen par X ka page khul gaya hai." before the answer to a question about X
            service = (persona.get("business_type") or "sales") == "service"
            # a new service named after a filed request starts a new booking (it used to loop on the old read-back)
            if service and s.service_request and not is_question(with_roman(user_text)) and named_services(with_roman(user_text), persona):
                s.service_request = None; s.form_sent = False; s.sr_pending = False; s.booking = False
                s.chosen = None; s.asked_which = False; s.bk = {}; s.asked_name = False; s.interested = False
            if service and not s.service_request:
                slots = sr_collect(s.transcript)
                if not slots.get("car"):   # "वैगनआर DL8CAF2345": the Roman rendering carries the model the form needs
                    rs = sr_collect([{"role": "user", "text": with_roman(t.get("text", ""))}
                                     for t in s.transcript if t.get("role") == "user"])
                    if rs.get("car"):
                        slots["car"] = rs["car"]
                for k in ("name", "phone", "car", "registration"):   # once known, kept: the extractors re-read the whole
                    if slots.get(k) and not s.bk.get(k):            # transcript and the read-back ("...naam Rahul.") hid
                        s.bk[k] = slots[k]                           # the name, so "haan" never filed (2026-11-24 sim)
                slots.update({k: v for k, v in s.bk.items() if v})
                if s.chosen:
                    slots["problem"] = s.chosen["title"]   # the chosen service is the job (read-back, form, CRM)
                was_pending = s.sr_pending        # the read-back was waiting (a "galat"/"cancel" clears the flag below)
                if s.sr_pending and sr_is_no(user_text):
                    s.sr_pending = False
                if s.sr_pending and not sr_missing(slots) and sr_is_yes(user_text):
                    source = iter([self._file_service_request(s, slots)])
                    if (s.service_request or {}).get("status") == "site_form":
                        self._sse("site_form", s.site_fill)   # the launcher fills + submits the website's own form
                # 2026-11-24 owner flow, a server-side step machine (same with the LLM up, rate-limited or down):
                #  which service -> its page + what we do + "model aur registration?" -> booking form + "mobile number?"
                #  -> "naam?" -> read-back -> "haan" files it. A question in between is answered (fast path / LLM) and
                #  the current step's question is asked again right after (step_q below).
                L = 1 if en else 0
                mt = with_roman(user_text)
                asking = is_question(mt) and not wants_booking(mt)
                specs = named_services(mt, persona)
                if len(specs) >= 2 and not re.search(r"\bextended\b", mt, re.I):   # "paint ki warranty" = the paint job's
                    specs = [x for x in specs if x["path"] != "/extended-warranty"] or specs   # warranty (2026-11-25)
                spec = specs[-1] if specs else named_service(mt, persona)   # "general service" = regular service (QA)
                # 2026-11-25 owner: the meaning matcher (llm/intents.py, persona/acr_intents.json) reads phrasings the
                # word lists miss ("road pe rukh gya h", "AC se garam hawa"). Only an extra signal: its service is used
                # only when the rules named none, its intent only when no rule regex fired; None below its thresholds.
                mres = meaning.classify(user_text) if not spec else None
                # a rule's own reading of the turn (cancel / bye / complaint / out of scope / human / wait) keeps its branch
                service_complaint = bool(SERVICE_COMPLAINT_RE.search(mt))
                rule_turn = bool(service_complaint or CANCEL_RE.search(mt) or BYE_RE.search(user_text) or COMPLAINT_RE.search(mt)
                                 or OUT_OF_SCOPE_RE.search(mt) or HUMAN_RE.search(mt) or is_hold(user_text) or LATER_RE.search(mt))
                if (mres and mres.service and not rule_turn and not s.form_sent and not s.sr_pending and not (s.chosen and asking)):
                    pg = next((p for p in (persona.get("pages") or {}).get("list") or []
                               if p.get("path") == mres.service.label and p.get("kind") == "service"), None)
                    if pg:
                        spec = {"path": pg["path"], "title": pg["title"], "about": pg.get("about", "")}
                        specs = [spec]
                if (spec and spec["path"] == "/extended-warranty"
                        and not re.search(r"\b(extend\w*|plan|badha\w*|extra|lambi)\b", mt, re.I)):
                    spec = None                 # "repair pe warranty milegi?" is a question, not the plan (2026-11-25 loop)
                    specs = [x for x in specs if x["path"] != "/extended-warranty"]
                    spec = specs[-1] if specs else None
                if (any(x["path"] == "/category/car-lights-and-glass-work" for x in specs) and len(specs) >= 2
                        and re.search(r"\b(dashboard|warning|indicator|wali\s+light|light\s+jal\w*|light\s+aa\s+rahi)\b", mt, re.I)):
                    # "dashboard pe battery wali light jal rahi hai" is a warning lamp, not a headlight job (loop round 6)
                    specs = [x for x in specs if x["path"] != "/category/car-lights-and-glass-work"]
                    spec = specs[-1] if specs else None
                if (spec and spec["path"] == "/insurance-claim"
                        and re.search(r"\b(renew\w*|naya\s+insurance|new\s+insurance|policy\s+(?:lena|leni|kharid\w*))\b", mt, re.I)
                        and not ACCIDENT_RE.search(mt)):
                    specs = [x for x in specs if x["path"] != "/insurance-claim"]   # a renewal is not a claim (loop)
                    spec = specs[-1] if specs else None
                rule_hit = rule_turn or bool(PRICE_RE.search(mt) or wants_booking(mt) or sr_is_yes(user_text) or sr_is_no(user_text))
                mi = (mres.intent.label if mres and mres.intent and not spec and not rule_hit
                      and mres.intent.label in meaning.SERVER_INTENTS else "")
                if mi:
                    self._sse("meta", {"path": "intent", "intent": mi, "score": round(mres.intent.score, 3), "ms": round(mres.ms, 1)})
                elif mres and mres.service and spec:
                    self._sse("meta", {"path": "intent", "service": spec["path"], "score": round(mres.service.score, 3), "ms": round(mres.ms, 1)})
                if mi in ("price_question", "time_question"):
                    asking = True                 # "kitni der lagegi" without a "?": answered, the flow does not advance
                holding = (is_hold(user_text) or bool(LATER_RE.search(mt)) or mi == "hold") and not spec
                booking_now = (wants_booking(mt) or mi == "book") and not holding
                if not spec and not s.chosen and (booking_now or wants_service(mt)) and not asking:
                    for t in reversed(s.transcript):               # "theek hai, karwani hai": the last service named
                        if t.get("role") == "user" and t.get("text") != user_text:
                            spec = named_service(with_roman(t.get("text", "")), persona)
                            if spec:
                                break
                site = ((persona.get("knowledge") or {}).get("website") or "").rstrip("/")
                # phone / name captured by the machine itself (Devanagari names, bare names, digits in any form)
                if s.form_sent and not slots.get("phone"):
                    ph = site_form.mobile(user_text)
                    if ph:
                        s.bk["phone"] = slots["phone"] = ph
                if s.form_sent and s.asked_name and not slots.get("name") and not asking:
                    words = user_text.strip(" .!?,।").split()
                    if (1 <= len(words) <= 3 and all(NAME_WORD.match(w) for w in words)
                            and not sr_is_yes(user_text) and not sr_is_no(user_text) and not holding):
                        nm = " ".join(w.title() if w.isascii() else w for w in words)
                        s.bk["name"] = slots["name"] = nm

                def open_form(choice: dict) -> None:
                    s.booking = True
                    s.form_sent = True
                    reason = "insurance" if "insurance" in choice["path"] else "service"
                    if site_form.usable(s.context):
                        s.site_form_opened = True
                        fill = site_form.payload({**slots, "problem": choice["title"], "service": choice["title"]})
                        self._sse("site_form_open", {k: v for k, v in fill.items() if v})   # widget.js opens + pre-fills it
                    else:
                        self._sse("form", {"reason": reason, "prefill": form_prefill({**slots, "problem": choice["title"]}, reason),
                                           "services": list(SERVICE_SLUG)})

                def form_line(choice: dict) -> str:
                    if site_form.usable(s.context):
                        line = (DETAILS_FORM_LINE_EN if en else DETAILS_FORM_LINE).format(title=choice["title"])
                    else:
                        line = (DETAILS_CARD_LINE_EN if en else DETAILS_CARD_LINE).format(title=choice["title"])
                    if "insurance" in choice["path"]:          # 2026-11-25 QA: the claim said "servicing ke liye"
                        line = line.replace("servicing ke liye aapko call aa jayega", "claim ke liye team aapko call karegi")
                        line = line.replace("you'll get a call for the servicing", "the team will call you about the claim")
                    elif choice["path"] in URGENT_PATHS:
                        line = line.replace("servicing ke liye aapko call aa jayega", "team aapko turant call karegi")
                        line = line.replace("you'll get a call for the servicing", "the team will call you right away")
                    return line

                def after_form(choice: dict) -> str:
                    """The form just opened: ask only what is still missing (car + mobile in one message used to be
                    asked the mobile again, then every name was refused, 2026-11-25 QA)."""
                    if not slots.get("phone"):
                        return form_line(choice)
                    if not slots.get("name"):
                        s.asked_name = True
                        return (FORM_NAME_LINE[L]).format(title=choice["title"])
                    return form_line(choice).split(". ")[0] + ". " + readback()

                def show_page(choice: dict) -> None:
                    if NAV_ON and choice["path"] not in s.nav_opened:
                        s.nav_opened.add(choice["path"])
                        self._sse("navigate", {"path": choice["path"], "title": choice["title"], "url": site + choice["path"]})

                def readback() -> str:
                    s.sr_pending = True
                    car = (" " + " ".join(x for x in (slots.get("car", ""), slots.get("registration", "")) if x)) if slots.get("car") else ""
                    return READBACK[L].format(title=slots.get("problem") or (s.chosen or {}).get("title", ""), car=car,
                                              digits=" ".join(slots.get("phone", "")), name=slots.get("name", ""))

                confirm_stage = was_pending or (s.form_sent and slots.get("phone") and slots.get("name"))
                fixed_name = name_correction(user_text) if confirm_stage else ""
                fixed_phone = site_form.mobile(user_text) if confirm_stage else ""
                # a side question about the chosen service ("warranty milti hai?", "seat ki dhulai bhi?", "PPF aur ceramic
                # mein fark?") stays on its page; only a different service on its own switches (2026-11-25 QA)
                chosen_words = {w for x in ((s.chosen or {}).get("words") or []) for w in str(x).lower().split()
                                if len(w) >= 4 and w not in ("service", "servicing", "karwani", "karwana", "change", "check", "repair")}
                stay = bool(asking and s.chosen and (len(specs) >= 2 or ATTR_RE.search(mt)
                                                     or chosen_words & set(re.findall(r"[a-z]+", mt.lower()))))
                if source is None and FORM_ON:
                    if (was_pending or s.form_sent or s.interested) and (CANCEL_RE.search(mt) or mi == "cancel") and not spec:
                        s.sr_pending = False; s.form_sent = False; s.booking = False; s.interested = False
                        s.asked_name = False
                        source = iter([CANCEL_LINE[L]])
                    elif confirm_stage and ((fixed_phone and fixed_phone != slots.get("phone")) or
                                            (fixed_name and fixed_name != slots.get("name"))):
                        if fixed_phone and fixed_phone != slots.get("phone"):
                            s.bk["phone"] = slots["phone"] = fixed_phone
                        if fixed_name and fixed_name != slots.get("name"):
                            s.bk["name"] = slots["name"] = fixed_name
                        source = iter([readback()])            # read back the corrected details, then "haan" files them
                    elif service_complaint and not s.form_sent and not s.sr_pending:
                        s.booking = True                    # the mobile number goes to a senior advisor (HUMAN flow)
                        source = iter([SORRY_LINE[L]])
                    elif (not s.form_sent and not s.sr_pending and not site_form.mobile(user_text)
                          and CHANGE_NUMBER_RE.search(mt)):
                        s.booking = True                    # "booking mein number badalna hai": the next number is kept
                        source = iter([CHANGE_NUMBER_LINE[L]])
                    elif (BYE_RE.search(user_text) or mi == "goodbye") and not s.form_sent:
                        source = iter([BYE_LINE[L]])
                    elif TRUST_RE.search(mt) and not spec and not service_complaint:
                        source = iter([TRUST_LINE[L]])
                    elif (COMPLAINT_RE.search(mt) or mi == "complaint") and not spec:
                        source = iter([COMPLAINT_LINE[L]])
                    elif not spec and (OUT_OF_SCOPE_RE.search(mt) or mi == "out_of_scope"):
                        source = iter([(persona.get("guardrails") or {}).get("out_of_scope_line") or HOLD_LINE[L]])
                    elif not spec and IDENTITY_RE.search(mt):    # "aap insaan ho ya robot?" is not a handoff request
                        source = iter([IDENTITY_LINE[L]])
                    elif not spec and (HUMAN_RE.search(mt) or mi == "human_handoff"):
                        s.booking = True
                        s.form_sent = s.form_sent or False
                        source = iter([HUMAN_LINE[L]])
                    elif holding:
                        source = iter([HOLD_LINE[L]])
                    elif was_pending and sr_is_no(user_text):
                        s.sr_pending = False
                        source = iter([NOT_YES[L]])
                    elif len(specs) >= 2 and not asking and not (s.chosen and s.chosen["path"] in {x["path"] for x in specs}):
                        s.asked_which = True
                        source = iter([TWO_LINE[L].format(a=specs[-2]["title"], b=specs[-1]["title"])])
                    elif spec and asking and s.form_sent and spec["path"] != (s.chosen or {}).get("path"):
                        show_page(spec)     # mid-booking question about another service: its page, the model answers
                    elif (spec and not stay and (not asking or not s.form_sent)
                          and (not s.chosen or spec["path"] != s.chosen["path"])):
                        # 2026-11-25 owner: a QUESTION about a service ("clutch ka kharcha kitna?", "suspension mein
                        # kya check karte ho?") also opens that service's page and explains it, the same as naming it
                        changed = s.form_sent
                        s.chosen = spec
                        slots["problem"] = spec["title"]
                        show_page(spec)
                        if changed:                                  # a correction after the form opened
                            open_form(spec)
                            source = iter([CHANGE_LINE[L].format(title=spec["title"]) + " " + step_question(booking_step(s, slots), en, slots)])
                        elif booking_now and slots.get("car"):     # knows the car and asks to book: form at once
                            s.interested = True
                            open_form(spec)
                            source = iter([after_form(spec)])
                        else:
                            # step 2 (2026-11-25 owner): the page opens, Neha says what is done there and the price,
                            # then asks if there is anything else to know. Model + registration only once the customer
                            # wants to go ahead ("book karni hai" in the same breath counts).
                            about = about_line(spec) or spec["title"]
                            if en:
                                about = english_about(about)       # "flywheel ka kaam" leaked into English (QA)
                            line = (EXPLAIN_LINE_EN if en else EXPLAIN_LINE).format(title=spec["title"], about=about)
                            line = line.replace(" " + STEP_Q["car"][L], "") + price_line(spec, en)
                            if not asking and ACCIDENT_RE.search(mt):    # 2026-11-25 loop: "accident ho gaya" got a cheerful "Ji zaroor"
                                line = ACCIDENT_LEAD[L] + " " + line.replace("Ji zaroor, screen", "Screen").replace("Sure, the", "The")
                            elif not asking and PROBLEM_RE.search(mt):  # a problem told, not a request: "samajh gayi", not "zaroor"
                                line = line.replace("Ji zaroor, screen", "Ji, samajh gayi. Screen").replace("Sure, the", "I see. The")
                            said_now = sr_collect([{"role": "user", "text": with_roman(user_text)}])
                            if spec["path"] in URGENT_PATHS and not asking:   # breakdown: help first, no price talk (QA);
                                # a question ("roadside mein kya milta hai?", "fuel delivery karte ho?") gets its answer
                                s.interested = True
                                line = URGENT_LINE[L].format(title=spec["title"], about=about)
                            elif said_now.get("car") and said_now.get("registration") and not asking:
                                s.interested = True                 # "meri Alto hai DL1CX9999, AC thanda nahi": ready to book
                                line += " " + car_question(slots, en)
                            elif (booking_now or s.interested) and not asking:
                                s.interested = True
                                line += " " + car_question(slots, en)
                            else:
                                s.interested = False     # a question about it: offer again, "haan" then asks the car
                                line += " " + offer_question(spec, en)
                            if (asking and not PRICE_RE.search(mt)
                                    and not ABOUT_Q_RE.search(mt)):
                                # a specific question ("Full AC service mein kitna time lagta hai?"): the page opens and
                                # the real answer follows (fast path / model), then the offer; the template used to
                                # replace the answer (2026-11-25 KB check)
                                lead_in = PAGE_OPEN_LINE[L].format(title=spec["title"])
                            else:
                                source = iter([line])
                    elif (s.chosen and not s.form_sent and (PRICE_RE.search(mt) or mi == "price_question")
                          and (not spec or spec["path"] == s.chosen["path"])):
                        # the price of the chosen service: the owner's line or "estimate after inspection", never a
                        # number from another service (2026-11-25: battery got the annual-service price)
                        pl = price_line(s.chosen, en).strip() or PRICE_UNKNOWN[L].strip()
                        source = iter([pl + " " + (car_question(slots, en) if s.interested else offer_question(s.chosen, en))])
                    elif s.chosen and not s.form_sent and not s.interested and not asking:
                        # explained, waiting: questions go to the model below (the offer is asked again after the
                        # answer); car details, "haan" / "book kar do" = interest; "nahi" = no push
                        said = sr_collect([{"role": "user", "text": with_roman(user_text)}])
                        ph_now = site_form.mobile(user_text)
                        if ph_now:                                   # a volunteered mobile is kept (QA: it was ignored)
                            s.bk["phone"] = slots["phone"] = ph_now
                        if ph_now and not (said.get("car") or said.get("registration") or slots.get("car")):
                            s.interested = True
                            source = iter([INTEREST_LEAD[L] + car_question(slots, en)])
                        elif said.get("car") or said.get("registration") or (ph_now and slots.get("car")):
                            s.interested = True
                            show_page(s.chosen)
                            open_form(s.chosen)
                            source = iter([after_form(s.chosen)])
                        elif (shows_interest(mt, False) or (spec and spec["path"] == s.chosen["path"])
                              or (mi == "book" and not NOT_INTERESTED_RE.search(mt))):
                            s.interested = True
                            if slots.get("car") and slots.get("registration"):
                                show_page(s.chosen)
                                open_form(s.chosen)
                                source = iter([after_form(s.chosen)])
                            else:
                                lead = (("Zaroor ji, main aapka insurance claim book kar deti hoon. ", "Sure, I'll book your insurance claim. ")[L]
                                        if "insurance" in s.chosen.get("path", "") else INTEREST_LEAD[L])
                                source = iter([lead + car_question(slots, en)])
                        elif (sr_is_no(user_text) or NOT_INTERESTED_RE.search(mt) or CANCEL_RE.search(mt)
                              or mi in ("not_interested", "cancel")):
                            source = iter([NO_MORE[L]])      # "not interested" got "didn't catch" (QA)
                    elif (s.chosen and not s.form_sent and not asking and spec and spec["path"] == s.chosen["path"]
                          and not slots.get("car") and not slots.get("registration")):
                        # "haan, AC service karwani hai" only confirms the service: the car details are still needed
                        source = iter([car_question(slots, en)])
                    elif (s.chosen and not s.form_sent and not asking and sr_is_yes(user_text)
                          and len(mt.split()) <= 3 and not slots.get("car") and not slots.get("registration")):
                        # "yes" to "your car's model and registration?" gives neither (2026-11-25 loop: the form
                        # opened on a bare "yes"): ask again instead of moving on
                        source = iter([car_question(slots, en)])
                    elif s.chosen and not s.form_sent and not asking:
                        show_page(s.chosen)                          # step 3: car details (or "pata nahi") -> form
                        open_form(s.chosen)
                        source = iter([after_form(s.chosen)])
                    elif s.form_sent and not s.sr_pending and not asking:
                        if not slots.get("phone"):
                            said_digits = bool(re.search(r"\d{3,}", user_text))
                            source = iter([PHONE_AGAIN[L] if said_digits else STEP_Q["phone"][L]])
                        elif not slots.get("name"):
                            s.asked_name = True
                            source = iter([STEP_Q["name"][L]])
                        elif not s.sr_pending:
                            source = iter([readback()])
                    elif (not s.chosen and not spec and not asking
                          and not (fast and fast.kind == "exact")   # a word-for-word FAQ answer beats "which service?" (loop)
                          and (wants_some_service(mt) or booking_now or mi == "generic_service_wanted")):
                        again = s.asked_which
                        s.asked_which = True                         # step 1
                        source = iter([(STEP_Q["which"][L] + (" Engine, battery, AC, brakes, tyres ya denting-painting?" if not en else
                                                              " Engine, battery, AC, brakes, tyres, or denting and painting?"))
                                       if again else (ASK_WHICH_EN if en else ASK_WHICH)])
                step = booking_step(s, slots)
                step_q = step_question(step, en, slots) if source is None else ""
                if step_q and step == "offer":
                    step_q = offer_question(s.chosen, en)        # "...insurance claim book kar doon?" on that page
                # the offline script (LLM down) follows the machine: nothing of its own when the step question is appended
                s.brain.pending_ask = "-" if step_q else (step_question("which", en) if not s.chosen and not s.form_sent else "")
                if s.chosen:
                    s.brain.slots["service"] = s.chosen["title"]
                if source is None and step:
                    kb_extra = (kb_extra + "\n\n" if kb_extra else "") + (
                        "## Booking in progress\nAnswer the customer's question in one or two short sentences from the "
                        "passages only. Do NOT ask any question and do not ask for name, number, car or mobile: the system "
                        "asks the next booking question itself right after your answer.")
                    if s.chosen:
                        known = service_price(s.chosen.get("path", ""), en)
                        kb_extra += (f"\nThe customer is looking at {s.chosen['title']}. "
                                     + (f"Owner's price for it: {known}. " if known else
                                        "There is no fixed price list for it: never invent a number; say the exact "
                                        "estimate depends on the car and the work and the team gives it after inspection. ")
                                     + "Explain what is included, how long it takes and why it matters, from the passages.")
                if source is None and not step and (s.booking or s.sr_pending or s.form_sent):
                    kb_extra = (kb_extra + "\n\n" if kb_extra else "") + sr_prompt_block(slots, s.sr_pending, form_open=s.form_sent)
                elif source is None and not step and sr_wants(s.transcript, slots):
                    kb_extra = (kb_extra + "\n\n" if kb_extra else "") + SERVICE_GUIDE
                in_request = bool(s.sr_pending or s.form_sent or s.booking or sr_wants(s.transcript, slots))
            if source is None and fast:
                answer = fast.text(s.language).strip().replace("\ufffd", "")
                in_flow = bool(step_q) or bool(getattr(s, "chosen", None)) or bool(s.form_sent)
                if in_flow and not is_question(with_roman(user_text)):
                    fast = None                 # "number yaad nahi" is a booking answer, not an FAQ (2026-11-24 sim)
                elif re.search(r"rupe|rupa|₹|\brs\.?\s?\d|\binr\b|thousand|hazaar|\b\d+\s*k\b", answer, re.I):   # money only (QA:
                    # "13 rupees-14k"); not any number: "Khasra No. 1517/907" is an address (2026-11-25)
                    fast = None                 # a price from another service's call (2026-11-25: "AC gas refill" got 1000)
                elif PRICE_RE.search(with_roman(user_text)) and not re.search(r"\d|₹|rupa|rs\b", answer, re.I):
                    fast = None                 # a price question gets a price, not the "how long" answer
                else:
                    if step_q:
                        answer = "".join(drop_detail_asks([answer])).rstrip() + " " + step_q
                    elif not answer.endswith("?"):
                        answer += " " + fast_follow_up(sr_collect(s.transcript) if service else {}, en)
                    source = iter([answer])
            # chat response cache: exact-match, context-free questions skip the LLM entirely
            if source is None and chat_cacheable(user_text, in_request=in_request):
                cached = _chat_cache_get(user_text) or cached_reply(user_text)
                if cached:
                    source = iter([cached])
            if source is None and step_q:
                import itertools
                source = itertools.chain(drop_detail_asks(stream_reply(s.messages(extra=kb_extra), fallback=s.brain)), iter([" " + step_q]))
            if lead_in and source is not None:
                import itertools
                source = itertools.chain(iter([lead_in + " "]), source)
            for token in (source if source is not None else stream_reply(s.messages(extra=kb_extra), fallback=s.brain)):
                if s.generation != my_gen:  # barge-in: a newer turn superseded us
                    self._sse("interrupted", {"generation": my_gen})
                    return
                buf += token
                # First words first: an opening interjection ("Acha, ...", "Haan ji, ...") goes out on its own the
                # moment the comma arrives. Those clips are warm from start-up, so the customer hears her react
                # ~1 s before the first full sentence can be synthesized (Edge needs ~1 s per fresh clip).
                if idx == 0 and not spoken:
                    m_int = _OPENER.match(buf)
                    if m_int and bc:
                        # the backchannel already reacted ("hmm" / "haan ji" at +30 ms): a second interjection
                        # only delays the answer (2026-11-24: "hmm" + "haan ji" + "bilkul." = 4.6 s before content)
                        buf = buf[m_int.end(1):].lstrip(" ,!")
                        buf = buf[:1].upper() + buf[1:]
                        m_int = None
                    if m_int:
                        opener, buf = m_int.group(1), buf[m_int.end(1):].lstrip(" ,!")
                        # an English-speaking customer never hears "haan ji" / "acha", even if the model slips
                        word = opener.lower().strip(" ,!")
                        piece = (EN_OPENER.get(word, word.capitalize()) + opener[-1]) if en else cached_opener(opener, persona)
                        spoken.append(piece)
                        emo = emotion_of(piece)
                        self._sse("sentence", {"voice": voice, "index": idx, "text": piece, "emotion": emo,
                                               "gesture": pick_gesture(persona["avatar"]["gestures"], emo, idx)})
                        prefetch_tts([piece], voice=voice, emotion=emo)
                        idx += 1
                # First clause first: 3-10 words are enough for the first clip, so audio starts while the
                # model is still writing the rest of that sentence. Once only; later flushes are sentences.
                if not clause_done:
                    cut = split_first_phrase(buf)
                    if cut:
                        head, buf = cut
                        piece = humanize(head, max_words=max_words, language=s.language, female=female)
                        if piece:
                            spoken.append(piece)
                            emo = emotion_of(piece)
                            self._sse("sentence", {"voice": voice, 
                                "index": idx,
                                "text": piece,
                                "emotion": emo,
                                "gesture": pick_gesture(persona["avatar"]["gestures"], emo, idx),
                            })
                            prefetch_tts([piece], voice=voice, emotion=emo)
                            idx += 1
                        clause_done = True
                # flush at a sentence boundary so TTS can start before the reply ends
                while True:
                    m = split_sentences(buf)
                    if len(m) <= 1:
                        break
                    clause_done = True  # a full sentence is already granular
                    head, buf = m[0], " ".join(m[1:])
                    piece = humanize(head, max_words=max_words, language=s.language, female=female)
                    if not piece:
                        continue
                    spoken.append(piece)
                    emo = emotion_of(piece)
                    self._sse("sentence", {"voice": voice, 
                        "index": idx,
                        "text": piece,
                        "emotion": emo,
                        "gesture": pick_gesture(persona["avatar"]["gestures"], emo, idx),
                    })
                    prefetch_tts([piece], voice=voice, emotion=emo)  # synthesis runs while the model is still writing the next sentence
                    idx += 1

            tail = humanize(buf, max_words=max_words, language=s.language, female=female)
            if tail:
                spoken.append(tail)
                emo = emotion_of(tail)
                self._sse("sentence", {"voice": voice, 
                    "index": idx, "text": tail, "emotion": emo,
                    "gesture": pick_gesture(persona["avatar"]["gestures"], emo, idx),
                })
                prefetch_tts([tail], voice=voice, emotion=emo)

            full = " ".join(spoken).strip()
            if source is None and not s.service_request:
                s.sr_pending = asked_confirmation(full)  # the model just read the request back and asked "bhej doon?"
            if not full:  # model produced nothing usable: answer from the persona rules
                full = humanize(s.brain.reply(user_text), max_words=max_words, language=s.language, female=female)
            if not full:
                full = "Ek second, main check karti hoon." if female else "Ek second, main check karta hoon."
            # cache the LLM reply so the same question is instant next turn
            if source is None:
                try:
                    known_name = (sr_collect(s.transcript).get("name") or "",)
                except Exception:  # noqa: BLE001
                    known_name = ()
                if chat_cacheable(user_text, full, in_request=in_request, names=known_name):
                    _chat_cache_put(user_text, full)
                    cache_reply(user_text, full)
            s.transcript.append({"role": "assistant", "text": full, "t": time.time()})
            chunks = [{"text": c, "emotion": emotion_of(c),
                       "gesture": pick_gesture(persona["avatar"]["gestures"], emotion_of(c), i)}
                      for i, c in enumerate(delivery_chunks(full))]
            self._sse("final", {
                "text": full,
                "chunks": chunks,
                "latency_ms": max(1, int((time.perf_counter() - t0) * 1000)),
                "words": len(full.split()),
            })
            if STORE:  # the record is current after every turn, so nothing is lost if the tab is closed
                threading.Thread(target=persist, args=(s,), daemon=True, name="memory-save").start()
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as ex:  # noqa: BLE001
            try:
                self._sse("error", {"message": f"{type(ex).__name__}: {ex}"[:300]})
            except Exception:  # noqa: BLE001
                pass
        finally:
            self.close_connection = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--persona", default=None,
                    help="default persona id, e.g. auto-car-repair")
    a = ap.parse_args()

    persona = PERSONA
    if a.persona:
        pid = a.persona.strip().lower()
        match = PERSONAS.get(pid) or next(
            (p for k, p in PERSONAS.items() if pid in k or pid in (p.get("company") or "").lower().replace(" ", "-")),
            None,
        )
        if not match:
            print(f"unknown persona '{a.persona}'. available: {', '.join(PERSONAS)}", file=sys.stderr)
            return 2
        persona = match

    # One bot per port. ThreadingHTTPServer sets allow_reuse_address, and on Windows SO_REUSEADDR lets a second
    # process bind the same port, so two bots silently split the traffic (seen 2026-11-03: an old and a new
    # process both answering :4001). Probe first and refuse loudly; on Windows also drop the reuse flag.
    import socket
    probe_host = "127.0.0.1" if a.host in ("", "0.0.0.0") else a.host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        if probe.connect_ex((probe_host, a.port)) == 0:
            print(f"port {a.port} is already serving (another human_bot?): refusing to start a second one", file=sys.stderr)
            return 2
    if sys.platform == "win32":
        ThreadingHTTPServer.allow_reuse_address = False
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    srv.persona = persona
    mirror_env = os.getenv("HUMAN_BOT_SITE_MIRROR", "").lower()
    srv.mirror = mirror_env in ("1", "true", "yes") if mirror_env else (a.host in LOOPBACK and not BEHIND_PROXY)
    if "*" in ALLOWED_ORIGINS and a.host not in LOOPBACK:
        print("[warn] HUMAN_BOT_ALLOWED_ORIGINS is '*' on a non-loopback host: any site can call this API", file=sys.stderr, flush=True)
    print(f"human_bot on http://{a.host}:{a.port}  provider={provider_name()}  "
          f"persona={persona['display_name']} ({persona.get('company')})  mirror={'on' if srv.mirror else 'off'}", flush=True)
    threading.Thread(target=warm_tts, args=(PERSONAS,), daemon=True, name="tts-warm").start()
    from llm.client import warm_local_stt  # noqa: PLC0415  (local Whisper: the mic keeps working when Groq rate-limits)
    threading.Thread(target=warm_local_stt, daemon=True, name="stt-warm").start()
    threading.Thread(target=wake_kb, daemon=True, name="kb-wake").start()
    threading.Thread(target=reap_sessions, daemon=True, name="session-reaper").start()
    if (persona.get("business_type") or "") == "service":   # the matcher's model + examples (~2-30 s, background)
        meaning.load(block=False)
    wid = (persona.get("knowledge") or {}).get("website_id")
    if FAST_ON and wid:   # customer-call answers for the no-LLM fast path (llm/fastpath.py), loaded in the background
        from llm.knowledge import KB_URL as _kb  # noqa: PLC0415
        fastpath.start(_kb, wid, os.getenv("KB_ADMIN_KEY", ""))
        threading.Thread(target=warm_fast, args=(persona,), daemon=True, name="fast-warm").start()
        threading.Thread(target=warm_flow, args=(persona,), daemon=True, name="flow-warm").start()

    def _term(*_):  # SIGTERM from systemd/Docker: leave serve_forever the same way Ctrl+C does
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, _term)
    except (ValueError, OSError):
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        print("human_bot stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
