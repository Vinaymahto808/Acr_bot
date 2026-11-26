"""Neha fills the website's own "Book a Service" form (autocarrepair.in) instead of posting the lead herself.

The launcher script (widget.js) runs on the ACR site, so it can open the site's 3-step booking form, fill it with
the details Neha collected and press "Submit Inquiry" — only after the customer confirmed them (the spoken
read-back "haan", or Submit in Neha's own small form). The site sends its usual lead to its own CRM; the bot's
server does NOT also post the lead (one lead per customer). If the browser cannot complete the site form
(layout changed, a field missing), the card reports it and the server files the lead itself (/lead/site).

This module turns Neha's slots into the site form's values: the location is one of the 4 service centres,
brand/model are matched against the site's dropdowns in the browser, the service is one of the site's 13 types.
Mapped 2026-11-13 from the live form (Playwright, nothing submitted):
  step 1  FULL NAME, PHONE NUMBER, EMAIL (optional), PREFERRED LOCATION (Moti Nagar | Gurugram | Noida | Okhla)
  step 2  REGISTRATION (optional), BRAND (select), MODEL (select, filled after brand), YEAR (optional)
  step 3  SERVICE (select, required), DESCRIPTION (optional)  -> SUBMIT INQUIRY -> "REQUEST RECEIVED"
"""
from __future__ import annotations

import os
import re

ENABLED = os.getenv("HUMAN_BOT_SITE_FORM", "1").lower() in ("1", "true", "yes")

# the area the customer named -> the site's service centre. Unknown: Moti Nagar (the head workshop); the area the
# customer said goes into the description either way, so the team sees it.
CENTRES = [
    ("Gurugram", r"gurgaon|gurugram|manesar|sohna|dlf|sushant lok|golf course"),
    ("Noida", r"noida|ghaziabad|indirapuram|vaishali|kaushambi|greater noida|crossings|vasundhara"),
    ("Okhla", r"okhla|jasola|sarita vihar|kalkaji|nehru place|lajpat|greater kailash|\bgk\b|saket|malviya|defence colony|"
              r"faridabad|badarpur|govindpuri|east of kailash|mayur vihar|south delhi|south ex|hauz khas|vasant"),
    ("Moti Nagar", r"moti ?nagar|modinagar|rama road|kirti nagar|karol bagh|patel nagar|rajouri|punjabi bagh|paschim vihar|"
                   r"janakpuri|tilak nagar|uttam nagar|dwarka|west delhi|pitampura|rohini|shalimar bagh|subhash nagar|naraina|"
                   r"model town|ashok vihar|central delhi|connaught"),
]

# Neha's service names / the customer's words -> the site's service options (exact option texts)
SERVICES = [
    ("Car Insurance Claim", r"insurance|claim|bima|beema|cashless"),
    ("Car AC Service & Repair", r"\bac\b|a\.c|air ?con|cooling|thanda|compressor|condenser"),
    ("Car Battery", r"battery|batri|battary|jump start"),
    ("Car Brake & Wheel Maintenance", r"brake|break|wheel|alignment|balancing|tyre|tire|puncture"),
    ("Car Clutch Work", r"clutch"),
    ("Car Denting & Painting", r"dent|paint|scratch|bumper|body ?work|panel"),
    ("Car Care & Detailing", r"detailing|wash|clean|polish|coating|ceramic|graphene|interior"),
    ("Car Repairs & Inspection", r"check engine|engine|gear ?box|gear"),   # "check engine light" is engine work, not a lamp
    ("Car Lights and Glass Work", r"light|headlight|taillight|tail light|foglight|fog lamp|glass|windshield|windscreen|headlamp|mirror"),
    ("Car Suspension Work", r"suspension|shocker|jhatk|noise from below"),
    ("Car Emergency Services", r"emergency|tow|breakdown|band ho gayi|start nahi|accident"),
    ("Regular Car Service", r"periodic|regular|general service|servicing|oil change|\bservice\b"),
    ("Car Repairs & Inspection", r"engine|gear|repair|inspection|check|diagnos|problem|awaaz|noise|smoke"),
]
DEFAULT_SERVICE = "Car Repairs & Inspection"

BRANDS = {   # spoken / written make -> how the site's brand list spells it (matched case-insensitively in the browser)
    "maruti": "Maruti", "suzuki": "Maruti", "hyundai": "Hyundai", "honda": "Honda", "toyota": "Toyota", "tata": "Tata",
    "mahindra": "Mahindra", "kia": "Kia", "mg": "MG", "skoda": "Skoda", "volkswagen": "Volkswagen", "vw": "Volkswagen",
    "renault": "Renault", "nissan": "Nissan", "ford": "Ford", "bmw": "BMW", "audi": "Audi", "mercedes": "Mercedes",
    "benz": "Mercedes", "jeep": "Jeep", "volvo": "Volvo", "jaguar": "Jaguar", "land rover": "Land Rover", "porsche": "Porsche",
    "fiat": "Fiat", "chevrolet": "Chevrolet", "datsun": "Datsun", "isuzu": "Isuzu", "citroen": "Citroen", "lexus": "Lexus",
}


def centre(area: str) -> str:
    a = (area or "").lower()
    for name, rx in CENTRES:
        if re.search(rx, a):
            return name
    return "Moti Nagar"


def service(*texts: str) -> str:
    t = " ".join(x or "" for x in texts).lower()
    for name, rx in SERVICES:
        if re.search(r"(?<![a-z])(?:" + rx + ")", t):   # words start at a boundary: "accident" is not "dent"
            return name
    return DEFAULT_SERVICE


def car_parts(car: str) -> dict[str, str]:
    """'Hyundai Creta 2021' / 'creta' / 'Swift dzire 2019' -> brand, model, year (brand from the model when unsaid)."""
    from crm.acr_leads import MODEL_MAKE  # noqa: PLC0415
    t = re.sub(r"\s+", " ", (car or "").strip())
    year = ""
    m = re.search(r"\b(19[89]\d|20[0-3]\d)\b", t)
    if m:
        year, t = m.group(1), (t[:m.start()] + t[m.end():]).strip()
    low = t.lower()
    brand = ""
    for k in sorted(BRANDS, key=len, reverse=True):
        if re.search(r"\b" + re.escape(k) + r"\b", low):
            brand = BRANDS[k]
            low = re.sub(r"\b" + re.escape(k) + r"\b", " ", low).strip()
            break
    model = re.sub(r"\s+", " ", low).strip()
    if not brand:
        words = model.split()
        joined = [w for w in words] + [a + b for a, b in zip(words, words[1:])] + ["".join(words)]   # "i 20" -> "i20"
        for word in joined:
            if word in MODEL_MAKE:
                brand = MODEL_MAKE[word].split()[0]
                break
    return {"brand": brand, "model": model, "year": year}


def mobile(raw) -> str:
    """The first 10-digit Indian mobile number in the text ('+91 98765 43210', '098765 43210'), else ''.
    '98765432109' (11 digits) or two numbers run together are not cut down to a wrong number."""
    t = str(raw or "")
    m = re.search(r"(?<!\d)(?:\+?91[\s-]?|0)?([6-9]\d{4}[\s-]?\d{5})(?!\d)", t)
    return re.sub(r"\D", "", m.group(1)) if m else ""


def payload(slots: dict) -> dict:
    """The values the launcher types into the site form. Empty strings are left blank there."""
    car = car_parts(slots.get("car", ""))
    problem = slots.get("problem") or ""
    svc_text = slots.get("service") or problem
    desc = "; ".join(x for x in (
        problem if problem and problem != slots.get("service") else "",
        f"Area: {slots['area']}" if slots.get("area") else "",
        f"Kab: {slots['when']}" if slots.get("when") else "",
        f"Pickup: {slots['pickup']}" if slots.get("pickup") else "",
        "Via Neha (AI service advisor)",
    ) if x)
    return {
        "name": slots.get("name", ""),
        "phone": mobile(slots.get("phone")),
        "email": slots.get("email", ""),
        "location": centre(slots.get("area", "")),
        "registration": (slots.get("registration") or "").upper(),
        "brand": car["brand"],
        "model": car["model"],
        "year": car["year"],
        "service": service(svc_text, problem),
        "description": desc[:400],
    }


def complete(fill: dict) -> bool:
    """Every required field of the site form has a value. Brand and model are required selects on step 2: without
    them the site keeps "Next Step" disabled and the popup is left half-filled in front of the customer
    (2026-11-24: the car was never said, the form stopped at VEHICLE DETAILS). Then the server files the lead."""
    return (all(str(fill.get(k) or "").strip() for k in ("name", "phone", "brand", "model"))
            and re.fullmatch(r"[6-9]\d{9}", str(fill.get("phone") or "")) is not None)


def usable(context: dict | None) -> bool:
    """The card runs inside the launcher on a page that has the site's booking form (widget.js reports it)."""
    page = (context or {}).get("page") or {}
    return ENABLED and bool(page.get("site_form"))
