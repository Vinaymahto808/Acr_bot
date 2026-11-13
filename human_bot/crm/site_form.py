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
    ("Car Battery", r"battery|batri|tyre|tire"),
    ("Car Brake & Wheel Maintenance", r"brake|break|wheel|alignment|balancing"),
    ("Car Clutch Work", r"clutch"),
    ("Car Denting & Painting", r"dent|paint|scratch|bumper|body ?work|panel"),
    ("Car Care & Detailing", r"detailing|wash|clean|polish|coating|ceramic|graphene|interior"),
    ("Car Lights and Glass Work", r"light|glass|windshield|windscreen|headlamp|mirror"),
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
    "fiat": "Fiat", "chevrolet": "Chevrolet", "datsun": "Datsun", "isuzu": "Isuzu", "citroen": "Citron", "lexus": "Lexus",
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
        if re.search(rx, t):
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
        for word in model.split():
            if word in MODEL_MAKE:
                brand = MODEL_MAKE[word].split()[0]
                break
    return {"brand": brand, "model": model, "year": year}


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
        "phone": re.sub(r"\D", "", slots.get("phone", ""))[-10:],
        "email": slots.get("email", ""),
        "location": centre(slots.get("area", "")),
        "registration": (slots.get("registration") or "").upper(),
        "brand": car["brand"],
        "model": car["model"],
        "year": car["year"],
        "service": service(svc_text, problem),
        "description": desc[:400],
    }


def usable(context: dict | None) -> bool:
    """The card runs inside the launcher on a page that has the site's booking form (widget.js reports it)."""
    page = (context or {}).get("page") or {}
    return ENABLED and bool(page.get("site_form"))
