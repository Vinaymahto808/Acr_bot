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


def page_line(page: dict) -> str:
    if page.get("exact"):
        return f"Ji, ye dekhiye, {page['title']} ka page aapke saamne khul gaya hai. Ismein kuch poochna ho to boliye."
    return (f"Iska alag page website par nahi hai, isliye main {page['title']} ka page khol rahi hoon. "
            "Aap bataiye kya chahiye, main yahin bata deti hoon.")
