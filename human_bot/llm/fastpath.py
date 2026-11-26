"""Fast path: answer the common customer questions in ~1 ms, without the LLM.

Neha's slow part is the model (first sentence ~2 s, measured 2026-11-14) and fresh speech synthesis (~1.2 s).
Most questions are not new: the knowledge base holds 802 answers from real customer calls ("pickup free hai?",
"warranty kitni hai?", "Sunday ko khula hai?"), each with several phrasings. This module answers those directly
and verifiably, and lets everything else go to the model as before.

Layers, cheapest first:
  1. normalise      lower case, typos, spelling variants, punctuation; Roman Hinglish and English
  2. exact map      normalised question -> answer row                       O(1) dict lookup
  3. domain bloom   is any word of the turn car/workshop vocabulary at all?   O(k) bit tests, no false negatives
  4. TF-IDF index   inverted index over the question variants; cosine score, accepted only when it is high and
                    clearly ahead of the next different answer (thresholds measured by tools/eval_fastpath.py)

A wrong canned answer is worse than a slower right one, so the matcher prefers to say "not sure" (None) and
the model answers. Devanagari (Hindi script) turns are left to the model: the index is Roman script.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import threading
import time
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import dataclass


# ---------------------------------------------------------------------------- bloom filter
class BloomFilter:
    """Set membership in a fixed bit array: 'no' is certain, 'yes' is right with probability 1 - fp.

    k bit positions per item from one 128-bit blake2b digest (Kirsch-Mitzenmacher double hashing). At a few
    thousand words a Python set would do the same job; the filter keeps memory flat (~1.2 bytes per word at
    1 %) when the vocabulary grows to every website and every call transcript.
    """

    def __init__(self, capacity: int, fp: float = 0.01) -> None:
        capacity = max(1, capacity)
        self.m = max(64, int(-capacity * math.log(fp) / (math.log(2) ** 2)))
        self.k = max(1, round(self.m / capacity * math.log(2)))
        self.bits = bytearray((self.m + 7) // 8)
        self.n = 0

    def _positions(self, item: str):
        d = hashlib.blake2b(item.encode("utf-8"), digest_size=16).digest()
        h1, h2 = int.from_bytes(d[:8], "little"), int.from_bytes(d[8:], "little") | 1
        return ((h1 + i * h2) % self.m for i in range(self.k))

    def add(self, item: str) -> None:
        for p in self._positions(item):
            self.bits[p >> 3] |= 1 << (p & 7)
        self.n += 1

    def __contains__(self, item: str) -> bool:
        return all(self.bits[p >> 3] & (1 << (p & 7)) for p in self._positions(item))


# ---------------------------------------------------------------------------- normalisation
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_TOKEN = re.compile(r"[a-z0-9]+")
# spelling variants and typos seen in typed / transcribed Hinglish -> one form
_CANON = {
    "gadi": "gaadi", "gaddi": "gaadi", "car": "gaadi", "vehicle": "gaadi", "pikup": "pickup", "pick": "pickup",
    "warrenty": "warranty", "waranty": "warranty", "warantee": "warranty", "guarantee": "warranty", "garanty": "warranty",
    "insurence": "insurance", "insurans": "insurance", "bima": "insurance", "beema": "insurance", "clam": "claim",
    "claime": "claim", "kitni": "kitna", "kitne": "kitna", "kharch": "kharcha", "cost": "kharcha", "price": "kharcha",
    "charges": "kharcha", "charge": "kharcha", "rate": "kharcha", "paisa": "kharcha", "paise": "kharcha",
    "servicing": "service", "servic": "service", "sarvice": "service", "batry": "battery", "batery": "battery",
    "tyer": "tyre", "tire": "tyre", "brek": "brake", "brakes": "brake", "cluch": "clutch", "acident": "accident",
    "paint": "painting", "dent": "denting", "kahan": "kaha", "kidhar": "kaha", "where": "kaha", "location": "kaha",
    "address": "kaha", "timing": "time", "timings": "time", "hours": "time", "open": "khula", "khulta": "khula",
    "sunday": "sunday", "ravivar": "sunday", "free": "free", "complimentary": "free", "muft": "free",
    "technician": "mechanic", "mistri": "mechanic", "breakdown": "start", "workshops": "workshop", "garage": "workshop",
    "centre": "workshop", "center": "workshop", "branch": "workshop", "aircon": "ac", "cooling": "thanda",
    "drop": "pickup", "doorstep": "pickup", "genuine": "original", "orignal": "original", "oem": "original",
}
# function words of Roman Hinglish and English: they carry no topic
_STOP = frozenset("""
kya hai hain ho hu hoon ka ki ke ko se me mein main mai mera meri mere mujhe aap aapka aapki aapke apna apni hum
humara hamara hamari ye yeh wo woh is us ek aur ya par pe bhi to toh na nahi ji haan han ha kar karo karna karte
karti karta kijiye dijiye bataiye batao bata sakte sakta sakti hota hoti hote tha thi raha rahi rahe gaya gayi
liye wala wali wale abhi bas please sir madam hello hi acha accha achha theek thik the a an is are am was were be
milta milti milte milega milegi milenge mil lagta lagti lagega lagegi hoga hogi honge deta deti dete dena diya
sakte sakti jata jati jaata jaati
do does did you your i my me we our it its this that for of to in on at and or with can could would will shall
should what which who how have has had any some there about tell know let get want need also just
""".split())


_ROMAN_OF: dict[str, str] | None = None
_DEVA_WORD = re.compile(r"[ऀ-ॿ]+")


# 2026-11-25 loop: a Hindi-script customer ("आपका ऑफिस कहाँ है", "पिकअप फ्री है क्या") got "samajh nahi paayi"
# whenever the free AI quota was out, because no saved answer could be matched. Common words of our customers'
# questions, used ONLY to match saved answers (hindi_script.WORDS also drives the voice; this list does not).
_DEVA_EXTRA = {
    "आप": "aap", "आपका": "aapka", "आपकी": "aapki", "आपके": "aapke", "मेरा": "mera", "मेरी": "meri", "मेरे": "mere",
    "मुझे": "mujhe", "हम": "hum", "हमें": "humein", "आपको": "aapko", "यह": "yeh", "ये": "ye", "वो": "wo", "वह": "woh",
    "है": "hai", "हैं": "hain", "था": "tha", "थी": "thi", "क्या": "kya", "कहाँ": "kahan", "कहां": "kahan", "कब": "kab",
    "कैसे": "kaise", "कितना": "kitna", "कितनी": "kitni", "कितने": "kitne", "कौन": "kaun", "कौनसा": "kaunsa", "क्यों": "kyon",
    "में": "mein", "से": "se", "को": "ko", "का": "ka", "की": "ki", "के": "ke", "पर": "par", "और": "aur", "या": "ya",
    "भी": "bhi", "तो": "to", "ही": "hi", "नहीं": "nahi", "नही": "nahi", "हाँ": "haan", "हां": "haan", "जी": "ji",
    "गाड़ी": "gaadi", "गाडी": "gaadi", "कार": "car", "सर्विस": "service", "सर्विसिंग": "servicing", "रिपेयर": "repair",
    "ऑफिस": "office", "ऑफ़िस": "office", "वर्कशॉप": "workshop", "सेंटर": "centre", "ब्रांच": "branch", "पता": "pata",
    "एड्रेस": "address", "पिकअप": "pickup", "ड्रॉप": "drop", "फ्री": "free", "टाइम": "time", "समय": "samay",
    "लगता": "lagta", "लगेगा": "lagega", "लगेगी": "lagegi", "चाहिए": "chahiye", "करना": "karna", "करवाना": "karwana",
    "करवानी": "karwani", "करानी": "karani", "कराना": "karana", "करते": "karte", "होता": "hota", "होती": "hoti",
    "खुला": "khula", "खुले": "khule", "बंद": "band", "रविवार": "sunday", "संडे": "sunday", "दाम": "daam", "कीमत": "keemat",
    "प्राइस": "price", "रेट": "rate", "खर्चा": "kharcha", "पैसे": "paise", "पेमेंट": "payment", "कैश": "cash",
    "कार्ड": "card", "ऑनलाइन": "online", "इंश्योरेंस": "insurance", "बीमा": "bima", "क्लेम": "claim", "कैशलेस": "cashless",
    "वारंटी": "warranty", "गारंटी": "guarantee", "पार्ट्स": "parts", "जेनुइन": "genuine", "ओरिजिनल": "original",
    "बैटरी": "battery", "एसी": "ac", "ब्रेक": "brake", "क्लच": "clutch", "टायर": "tyre", "इंजन": "engine",
    "डेंटिंग": "denting", "पेंटिंग": "painting", "पेंट": "paint", "वॉश": "wash", "धुलाई": "dhulai", "बुकिंग": "booking",
    "बुक": "book", "अपॉइंटमेंट": "appointment", "कल": "kal", "आज": "aaj", "सुबह": "subah", "शाम": "shaam",
    "बजे": "baje", "दिन": "din", "घंटे": "ghante", "घर": "ghar", "आ": "aa", "सकते": "sakte", "सकता": "sakta",
    "मिलेगा": "milega", "मिलेगी": "milegi", "मिलती": "milti", "देते": "dete", "लेते": "lete", "कोई": "koi",
    "ऑफर": "offer", "डिस्काउंट": "discount", "नंबर": "number", "मोबाइल": "mobile", "फोन": "phone", "नाम": "naam",
}


def to_roman(text: str) -> str | None:
    """Devanagari words -> Roman Hinglish with the dictionary of llm/hindi_script.py read backwards (STT writes
    "अच्छा pickup free hai kya?"). None when a Devanagari word is unknown: then the model takes the turn."""
    global _ROMAN_OF
    if _ROMAN_OF is None:
        try:
            from llm.hindi_script import WORDS  # noqa: PLC0415
            _ROMAN_OF = {}
            for roman, deva in WORDS.items():
                _ROMAN_OF.setdefault(deva, roman)
        except Exception:  # noqa: BLE001
            _ROMAN_OF = {}
        for deva, roman in _DEVA_EXTRA.items():   # matching only: never changes how Neha's voice reads text
            _ROMAN_OF.setdefault(deva, roman)
    unknown = False

    def sub(m: re.Match) -> str:
        nonlocal unknown
        r = _ROMAN_OF.get(m.group(0))
        if r is None:
            unknown = True
            return m.group(0)
        return r
    out = _DEVA_WORD.sub(sub, text or "")
    return None if unknown else out


def normalise(text: str) -> list[str]:
    t = unicodedata.normalize("NFKC", (text or "").lower())
    t = t.replace("pick up", "pickup").replace("a.c.", "ac").replace("on-road", "onroad")
    out = []
    for w in _TOKEN.findall(t):
        w = _CANON.get(w, w)
        if w not in _STOP and (len(w) > 1 or w.isdigit()):
            out.append(w)
    return out


def key_of(tokens: list[str]) -> str:
    return " ".join(sorted(set(tokens)))


# ---------------------------------------------------------------------------- answers
# A first line that is a note about the call, not something Neha can say to a customer
_NOT_AN_ANSWER = re.compile(r"\b(agent|agents|advisor) (ne|customer|workshop)|nahi bataya|not (given|provided|stated)|"
                            r"call mein nahi|^call note|customer ne|agent ne", re.I)


@dataclass
class Row:
    url: str
    questions: list[str]
    answer: str          # Roman Hinglish, Neha's own words
    answer_en: str       # English line of the same answer ('' when the note has none)


@dataclass
class Match:
    row: Row
    score: float
    margin: float
    kind: str            # "exact" | "tfidf"
    ms: float

    def text(self, language: str) -> str:
        if language == "english" and self.row.answer_en:
            return self.row.answer_en
        return self.row.answer


class FastPath:
    """Built once from the knowledge base's Q&A rows; read-only afterwards, so lookups need no lock."""

    # accepted only above both (measured on held-out phrasings, see tools/eval_fastpath.py; 2026-11-14 on 802 rows:
    # 0.75/0.12 answers ~48 % of the held-out questions, ~98 % pick the same question, 0 of 16 never-answer turns)
    MIN_SCORE = float(os.getenv("HUMAN_BOT_FAST_MIN", "0.75"))
    MIN_MARGIN = float(os.getenv("HUMAN_BOT_FAST_MARGIN", "0.12"))
    MIN_TOKENS = 2           # one content word ("warranty") is too little to pick one of 802 answers

    def __init__(self, rows: list[Row]) -> None:
        self.rows = [r for r in rows if r.answer and not _NOT_AN_ANSWER.search(r.answer)]
        self.exact: dict[str, int] = {}
        self.key_rows: dict[str, set[int]] = {}                  # phrasing -> every row that has it (popularity)
        self.vecs: list[tuple[int, dict[str, float]]] = []      # (row index, unit tf-idf vector) per question variant
        self.postings: dict[str, list[int]] = {}                 # token -> variant indices
        df: dict[str, int] = {}
        variants: list[tuple[int, list[str]]] = []
        for ri, r in enumerate(self.rows):
            for q in r.questions:
                toks = normalise(q)
                if len(toks) < 1:
                    continue
                self.exact.setdefault(key_of(toks), ri)
                self.key_rows.setdefault(key_of(toks), set()).add(ri)
                variants.append((ri, toks))
                for tok in set(toks):
                    df[tok] = df.get(tok, 0) + 1
        n = max(1, len(variants))
        self.idf = {tok: math.log(1 + n / c) for tok, c in df.items()}
        for vi, (ri, toks) in enumerate(variants):
            vec = self._vec(toks)
            self.vecs.append((ri, vec))
            for tok in vec:
                self.postings.setdefault(tok, []).append(vi)
        # answer vectors: many rows are the same question from different calls with differently worded answers;
        # two rows whose answers are this close say the same thing (used for the margin and by the evaluation)
        adf: dict[str, int] = {}
        atoks = [normalise(r.answer) for r in self.rows]
        for toks in atoks:
            for tok in set(toks):
                adf[tok] = adf.get(tok, 0) + 1
        self._aidf = {t: math.log(1 + len(self.rows) / c) for t, c in adf.items()}
        self.avecs = [self._unit({t: toks.count(t) * self._aidf[t] for t in set(toks)}) for toks in atoks]
        # the domain filter holds the QUESTION vocabulary only: answers are full of generic words ("today", "best")
        self.domain = BloomFilter(len(self.idf) + 64, fp=0.01)
        for w in self.idf:
            self.domain.add(w)

    @staticmethod
    def _unit(v: dict[str, float]) -> dict[str, float]:
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def popular(self, n: int) -> list[Row]:
        """The answers customers ask for most, approximated by how many calls produced the same phrasing; each is
        the row the exact map returns for that phrasing, so warming its audio serves the live answer."""
        out: list[Row] = []
        seen: set[int] = set()
        for key, rs in sorted(self.key_rows.items(), key=lambda kv: len(kv[1]), reverse=True):
            ri = self.exact[key]
            if ri not in seen:
                seen.add(ri)
                out.append(self.rows[ri])
            if len(out) >= n:
                break
        return out

    SAME_ANSWER = 0.45

    def same_answer(self, a: int, b: int) -> bool:
        if self.rows[a].answer == self.rows[b].answer:
            return True
        va, vb = self.avecs[a], self.avecs[b]
        return sum(w * vb.get(t, 0.0) for t, w in va.items()) >= self.SAME_ANSWER

    def _vec(self, toks: list[str]) -> dict[str, float]:
        tf: dict[str, float] = {}
        for t in toks:
            tf[t] = tf.get(t, 0.0) + 1.0
        v = {t: c * self.idf.get(t, 0.0) for t, c in tf.items() if self.idf.get(t, 0.0) > 0}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    # ------------------------------------------------------------------ queries
    def in_domain(self, text: str) -> bool:
        """False when no word of the turn is car / workshop vocabulary (Bloom: no false negatives)."""
        return any(t in self.domain for t in normalise(text))

    def match(self, text: str) -> Match | None:
        t0 = time.perf_counter()
        if not text:
            return None
        if _DEVANAGARI.search(text):
            text = to_roman(text)            # every Devanagari word known, or the model answers
            if not text:
                return None
        toks = normalise(text)
        if not toks:
            return None
        # an exact stored question wins even when it is one content word: "What are your timings?" normalises to
        # "timings" and fell under MIN_TOKENS, so it waited 5 s on the LLM (2026-11-25 latency)
        ri = self.exact.get(key_of(toks))
        if ri is not None:
            return Match(self.rows[ri], 1.0, 1.0, "exact", (time.perf_counter() - t0) * 1000)
        if len(toks) < self.MIN_TOKENS:
            return None
        if not any(t in self.domain for t in toks):     # bloom: no question word of ours at all -> not a FAQ
            return None
        q = self._vec(toks)
        if not q:
            return None
        best: dict[int, float] = {}                   # row -> best cosine over its variants
        seen: set[int] = set()
        for tok in q:
            for vi in self.postings.get(tok, ()):
                if vi in seen:
                    continue
                seen.add(vi)
                ri, v = self.vecs[vi]
                s = sum(w * v.get(t, 0.0) for t, w in q.items())
                if s > best.get(ri, 0.0):
                    best[ri] = s
        if not best:
            return None
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        top_ri, top = ranked[0]
        # the runner-up only counts when it says something else (the same question from several calls)
        second = next((s for ri, s in ranked[1:12] if not self.same_answer(ri, top_ri)), 0.0)
        ms = (time.perf_counter() - t0) * 1000
        if top >= self.MIN_SCORE and top - second >= self.MIN_MARGIN:
            return Match(self.rows[top_ri], top, top - second, "tfidf", ms)
        return None


# ---------------------------------------------------------------------------- loading from the kb service
_FAST: FastPath | None = None
_LOCK = threading.Lock()
STATUS: dict = {"rows": 0, "loaded_at": None, "error": None, "load_ms": None}


def rows_from_json(data: dict) -> list[Row]:
    return [Row(r.get("url", ""), list(r.get("questions") or []), (r.get("answer") or "").strip(),
                (r.get("answer_en") or "").strip()) for r in data.get("rows") or []]


def load(kb_url: str, website_id: str, admin_key: str = "", timeout: float = 20.0) -> FastPath | None:
    """Fetch the Q&A rows and build the index. Never raises: no fast path just means every turn uses the model."""
    global _FAST
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(f"{kb_url.rstrip('/')}/api/knowledge/faq?" + urllib.parse.urlencode({"website": website_id}),
                                     headers={"x-api-key": admin_key} if admin_key else {})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        fp = FastPath(rows_from_json(data))
        with _LOCK:
            _FAST = fp
        STATUS.update(rows=len(fp.rows), loaded_at=time.time(), error=None, load_ms=round((time.perf_counter() - t0) * 1000))
        print(f"[fast] {len(fp.rows)} answers indexed in {STATUS['load_ms']} ms "
              f"(bloom {fp.domain.m} bits, k={fp.domain.k})", file=sys.stderr, flush=True)
        return fp
    except Exception as ex:  # noqa: BLE001
        STATUS["error"] = f"{type(ex).__name__}: {ex}"[:200]
        print(f"[fast] not loaded ({STATUS['error']}); every turn goes to the model", file=sys.stderr, flush=True)
        return None


def get() -> FastPath | None:
    return _FAST


def start(kb_url: str, website_id: str, admin_key: str = "", refresh_s: float = 1800.0) -> None:
    """Load in the background now (the kb may still be starting) and refresh every refresh_s."""
    def loop():
        delay = 2.0
        while True:
            ok = load(kb_url, website_id, admin_key) is not None
            time.sleep(refresh_s if ok else min(delay, 120.0))
            delay = delay * 2 if not ok else 2.0
    threading.Thread(target=loop, daemon=True, name="fastpath").start()
