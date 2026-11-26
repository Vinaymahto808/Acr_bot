"""Avatar v2 lip timeline: 15 visemes from TTS word boundaries, with coarticulation (ms, pure Python).

Same input as llm/visemes.py (Edge WordBoundary events [(start_s, dur_s, spoken_word)], the spoken word in
Devanagari for Hindi and Roman for English / names / Hinglish), finer output. The viseme set is the common
15-shape one (Oculus/ARKit naming), drawn by the v2 reference pack (tran_moment/avatar_v2):

    sil  rest between words (lips relaxed together)   PP  m b p   (lips pressed)
    FF   f v (lower lip under the upper teeth)        TH  th (tongue tip between the teeth)
    DD   t d (tongue behind the teeth)                kk  k g h (back of the tongue)
    CH   ch j sh (lips forward, teeth close)          SS  s z (teeth together, lips spread)
    nn   n l (tongue tip up)                          RR  r (slightly rounded)
    aa   a aa (open)   E  e ai (mid, spread)   I  i ee y (spread, narrow)   O  o au (round open)   U  u oo w (pucker)

Idea credit: the 15-viseme layer and the attack/dominance rules (bilabials always close, open vowels suppress
closure) follow corpunum/UnumAvatar's Rhubarb->viseme pass; this is an independent implementation (that repo has
no licence, nothing is copied from it).

Each cue is {"s", "e", "v", "w"}: w is the cue's intensity 0..1 (long stressed vowels 1, short schwa ~0.7,
consonants ~0.6); the card multiplies it with the live loudness to pick how far the jaw drops.
Coarticulation:
  * p/b/m and f/v are always visible (>= 60 ms), the shapes a viewer checks;
  * a lingual consonant (t d n l k g r) before a rounded vowel (o/u) is already rounded (anticipation): U, lower w;
  * h and other shapes shorter than one visible frame fold into their neighbour; equal neighbours merge.
"""
from __future__ import annotations

import re

VISEMES = ("sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "I", "O", "U")

_MIN_SHAPE = 0.05       # s: shorter shapes are not seen at 60 Hz with a 45 ms crossfade
_MIN_CLOSURE = 0.06     # s: lip closure / teeth on lip always at least this long
_PAUSE = 0.12           # s: a gap between words longer than this rests the mouth (sil)
_PROTECTED = ("PP", "FF")
_ROUND = ("O", "U")
_LINGUAL = ("DD", "nn", "kk", "RR", "SS", "TH")

# phoneme -> (viseme, weight); weight is both the share of the word's time and the intensity
_PH = {
    "p": ("PP", 0.55), "b": ("PP", 0.55), "m": ("PP", 0.55),
    "f": ("FF", 0.5), "v": ("FF", 0.45),
    "th": ("TH", 0.45),
    "t": ("DD", 0.4), "d": ("DD", 0.4), "T": ("DD", 0.4), "D": ("DD", 0.4),
    "n": ("nn", 0.4), "l": ("nn", 0.45), "N": ("nn", 0.3),
    "k": ("kk", 0.4), "g": ("kk", 0.4), "h": ("kk", 0.25),
    "ch": ("CH", 0.5), "j": ("CH", 0.5), "sh": ("CH", 0.55),
    "s": ("SS", 0.5), "z": ("SS", 0.5),
    "r": ("RR", 0.35), "y": ("I", 0.35), "w": ("U", 0.4),
    # vowels
    "a": ("aa", 0.8), "aa": ("aa", 1.35), "ae": ("E", 1.05), "e": ("E", 1.0), "ai": ("E", 1.2),
    "i": ("I", 0.8), "ii": ("I", 1.2), "o": ("O", 1.1), "au": ("O", 1.25), "u": ("U", 0.8), "uu": ("U", 1.2),
}
_VOWEL_PH = {"a", "aa", "ae", "e", "ai", "i", "ii", "o", "au", "u", "uu"}
_INTENSITY = {"aa": 1.0, "a": 0.72, "ae": 0.9, "e": 0.85, "ai": 0.95, "i": 0.7, "ii": 0.85, "o": 0.95, "au": 1.0,
              "u": 0.75, "uu": 0.9}

# ---------------------------------------------------------------- Devanagari
_V_INDEP = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
            "ऋ": "ri", "ऑ": "o", "ॲ": "ae"}
_MATRA = {"ा": "aa", "ि": "i", "ी": "ii", "ु": "u", "ू": "uu", "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ृ": "ri",
          "ॉ": "o", "ॅ": "ae"}
_VIRAMA, _NUKTA = "्", "़"
_ANUSVARA, _CHANDRA = "ं", "ँ"
_DEVA_CONS = {}
for _chars, _ph in (("प", "p"), ("फ", "p"), ("ब", "b"), ("भ", "b"), ("म", "m"), ("व", "v"),
                    ("त", "t"), ("थ", "t"), ("द", "d"), ("ध", "d"), ("टठ", "T"), ("डढ", "D"), ("ड़ढ़", "r"),
                    ("नणञ", "n"), ("ङ", "N"), ("ल", "l"), ("र", "r"), ("य", "y"), ("ह", "h"),
                    ("कखगघ", "k"), ("चछ", "ch"), ("जझ", "j"), ("शष", "sh"), ("स", "s")):
    for _c in _chars:
        _DEVA_CONS[_c] = _ph
_NUKTA_CONS = {"फ": "f", "ज": "z", "क": "k", "ख": "k", "ग": "g", "ड": "r", "ढ": "r"}
_DEVA_RE = re.compile(r"[ऀ-ॿ]")
_LABIAL = {"p", "b", "m"}


def _deva_phonemes(word: str) -> list[str]:
    chars = [c for c in word if _DEVA_RE.match(c)]
    out: list[str] = []
    i, n = 0, len(chars)
    while i < n:
        c = chars[i]
        if c in _V_INDEP:
            v = _V_INDEP[c]
            out += ["r", "i"] if v == "ri" else [v]
        elif c in _DEVA_CONS:
            ph = _DEVA_CONS[c]
            if i + 1 < n and chars[i + 1] == _NUKTA:
                ph = _NUKTA_CONS.get(c, ph)
                i += 1
            out.append(ph)
            nxt = chars[i + 1] if i + 1 < n else ""
            if nxt in _MATRA:
                v = _MATRA[nxt]
                out += ["r", "i"] if v == "ri" else [v]
                i += 1
            elif nxt == _VIRAMA:
                i += 1
            elif nxt in (_ANUSVARA, _CHANDRA):
                out.append("a")
            elif i + 1 < n:
                # inherent schwa; dropped word-finally (above) and medially after a full vowel before a
                # consonant that carries its own matra (आपकी aap-ki, करके kar-ke); the first syllable keeps it
                prev_vowel = len(out) >= 2 and out[-2] in _VOWEL_PH
                nxt_has_matra = i + 2 < n and chars[i + 2] in _MATRA and nxt in _DEVA_CONS
                if not (prev_vowel and nxt_has_matra):
                    out.append("a")
        elif c in _MATRA:
            out.append(_MATRA[c] if _MATRA[c] != "ri" else "i")
        elif c == _ANUSVARA and out:
            # anusvara is a nasal homorganic with the next consonant: m before p/b/m (lips close), else n
            nxt = chars[i + 1] if i + 1 < n else ""
            out.append("m" if _DEVA_CONS.get(nxt) in _LABIAL else "N")
        i += 1
    return out


# ---------------------------------------------------------------- Roman (English, names, Hinglish)
_DIGITS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
           "8": "eight", "9": "nine"}
_R_VOWELS = (("aa", ["aa"]), ("ee", ["ii"]), ("ea", ["ii"]), ("ie", ["ii"]), ("oo", ["uu"]), ("ou", ["a", "u"]),
             ("ow", ["a", "u"]), ("au", ["au"]), ("aw", ["au"]), ("ai", ["e", "i"]), ("ay", ["e", "i"]),
             ("ei", ["e", "i"]), ("ey", ["e", "i"]), ("oi", ["o", "i"]), ("oy", ["o", "i"]), ("oa", ["o"]),
             ("a", ["a"]), ("e", ["e"]), ("i", ["i"]), ("o", ["o"]), ("u", ["u"]))
_R_DIGRAPH = {"th": "th", "sh": "sh", "ch": "ch", "ph": "f", "ck": "k", "ng": "N", "kh": "k", "gh": "g", "bh": "b",
              "dh": "d", "jh": "j", "wh": "w", "qu": "k"}
_R_CONS = {"p": "p", "b": "b", "m": "m", "f": "f", "v": "v", "w": "w", "t": "t", "d": "d", "n": "n", "l": "l",
           "r": "r", "s": "s", "z": "z", "k": "k", "g": "g", "q": "k", "h": "h", "j": "j", "y": "y", "x": "k"}


# frequent English words whose spelling misleads the letter rules
_LEXICON = {"you": ["y", "uu"], "your": ["y", "o", "r"], "to": ["t", "uu"], "two": ["t", "uu"], "do": ["d", "uu"],
            "who": ["h", "uu"], "the": ["th", "a"], "one": ["w", "a", "n"], "our": ["a", "u", "r"],
            "ok": ["o", "k", "e", "i"], "okay": ["o", "k", "e", "i"], "are": ["aa", "r"], "is": ["i", "z"],
            "of": ["a", "v"], "what": ["w", "a", "t"], "car": ["k", "aa", "r"], "service": ["s", "a", "r", "v", "i", "s"]}


def _roman_phonemes(word: str) -> list[str]:
    w = "".join(_DIGITS.get(c, c) for c in word.lower())
    w = re.sub(r"[^a-z]", "", w)
    if w in _LEXICON:
        return list(_LEXICON[w])
    if len(w) > 3 and w.endswith("e") and w[-2] not in "aeiou":     # silent final e (price, care, drive)
        w = w[:-1]
    out: list[str] = []
    i = 0
    while i < len(w):
        two = w[i:i + 2]
        for g, ph in _R_VOWELS:
            if w.startswith(g, i):
                out += ph
                i += len(g)
                break
        else:
            if two in _R_DIGRAPH:
                ph = _R_DIGRAPH[two]
                i += 2
                if two == "qu":
                    out += ["k", "w"]
                    continue
            else:
                c = w[i]
                i += 1
                if c == "y":
                    ph = "y" if i == 1 or (i < len(w) and w[i] in "aeiou") else "i"   # yes / city
                elif c == "c":
                    ph = "s" if i < len(w) and w[i] in "eiy" else "k"                  # city / car
                elif c == "x":
                    out.append("k")
                    ph = "s"
                else:
                    ph = _R_CONS.get(c, "t")
            if not (out and out[-1] == ph):                                           # double letters: one sound
                out.append(ph)
    return out


def phonemes(word: str) -> list[str]:
    """Spoken word -> phoneme labels (keys of _PH)."""
    if _DEVA_RE.search(word):
        return _deva_phonemes(word)
    if re.fullmatch(r"[\d.,:/%-]+", word or "-") and re.search(r"\d", word or ""):
        # a number Edge reads in Hindi or English: a plausible open/closed rhythm per pair of digits
        return ["aa", "t", "e"] * max(1, min(3, len(re.sub(r"\D", "", word)) // 2 or 1))
    return _roman_phonemes(word)


def word_visemes(word: str) -> list[tuple[str, float, float]]:
    """Spoken word -> [(viseme, time weight, intensity)] with rounding anticipation applied."""
    ph = phonemes(word) or ["a"]
    out: list[tuple[str, float, float]] = []
    for k, p in enumerate(ph):
        vis, wt = _PH.get(p, ("DD", 0.4))
        inten = _INTENSITY.get(p, 0.6)
        if p in _VOWEL_PH and k == len(ph) - 1 and len(ph) > 2 and p == "a":
            inten = 0.6                                    # a word-final short a is weak
        out.append((vis, wt, inten))
    # anticipatory rounding: lingual consonants right before a rounded vowel are already rounded
    for k in range(len(out) - 2, -1, -1):
        if out[k][0] in _LINGUAL and out[k + 1][0] in _ROUND:
            out[k] = ("U", out[k][1], 0.5)
    return out


def timeline(words: list[tuple[float, float, str]], tail: float = 0.25) -> list[dict]:
    """[(start_s, dur_s, spoken_word)] -> [{"s", "e", "v", "w"}] covering the clip; sil in pauses and at the end."""
    if not words:
        return []
    out: list[dict] = []
    t_prev = 0.0
    for start, dur, spoken in sorted(words, key=lambda x: x[0]):
        start = max(start, t_prev)
        dur = max(dur, 0.08)
        if start - t_prev > _PAUSE:
            out.append({"s": t_prev, "e": start, "v": "sil", "w": 0.0})
        elif out and start > t_prev:
            out[-1]["e"] = start                           # a short gap: the previous shape holds
        segs = word_visemes(spoken)
        total = sum(w for _, w, _ in segs)
        t = start
        cur = []
        for v, wt, inten in segs:
            d = dur * wt / total
            cur.append([v, t, t + d, inten])
            t += d
        for k, s in enumerate(cur):                        # closures: >= _MIN_CLOSURE, time borrowed from a neighbour
            if s[0] in _PROTECTED and s[2] - s[1] < _MIN_CLOSURE:
                need = _MIN_CLOSURE - (s[2] - s[1])
                if k + 1 < len(cur) and cur[k + 1][2] - cur[k + 1][1] > need + _MIN_SHAPE:
                    s[2] += need
                    cur[k + 1][1] += need
                elif k > 0 and cur[k - 1][2] - cur[k - 1][1] > need + _MIN_SHAPE:
                    s[1] -= need
                    cur[k - 1][2] -= need
        out += [{"s": s0, "e": e0, "v": v, "w": i} for v, s0, e0, i in cur]
        t_prev = start + dur
    out.append({"s": t_prev, "e": t_prev + tail, "v": "sil", "w": 0.0})
    return _smooth(out)


def _smooth(cues: list[dict]) -> list[dict]:
    """Coarticulation: invisible shapes fold into the neighbour, equal neighbours merge, then round the times."""
    merged: list[dict] = []
    for c in cues:
        if merged and merged[-1]["v"] == c["v"]:
            merged[-1]["e"] = c["e"]
            merged[-1]["w"] = max(merged[-1]["w"], c["w"])
            continue
        merged.append(dict(c))
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for k, c in enumerate(merged):
            if c["e"] - c["s"] >= _MIN_SHAPE or c["v"] in _PROTECTED:
                continue
            # fold into the longer neighbour (a vowel usually), the neighbour keeps its own shape
            prev = merged[k - 1] if k > 0 else None
            nxt = merged[k + 1] if k + 1 < len(merged) else None
            if prev is None and nxt is None:
                break
            tgt = nxt if prev is None or (nxt is not None and nxt["e"] - nxt["s"] > prev["e"] - prev["s"]) else prev
            if tgt is nxt:
                nxt["s"] = c["s"]
            else:
                prev["e"] = c["e"]
            del merged[k]
            changed = True
            break
    final: list[dict] = []
    for c in merged:
        if final and final[-1]["v"] == c["v"]:
            final[-1]["e"] = c["e"]
            continue
        final.append(c)
    return [{"s": round(c["s"], 3), "e": round(c["e"], 3), "v": c["v"], "w": round(float(c["w"]), 2)} for c in final]


# v1 timeline letter (llm/visemes.py) -> v2 viseme, for clips whose word timing is not stored any more
FROM_V1 = {"A": "PP", "B": "DD", "C": "aa", "D": "I", "E": "O", "F": "U", "G": "FF", "H": "nn", "X": "sil"}


def from_v1(cues: list[dict]) -> list[dict]:
    return [{"s": c["s"], "e": c["e"], "v": FROM_V1.get(c.get("v"), "DD"), "w": 0.0 if c.get("v") == "X" else 0.8}
            for c in cues or []]
