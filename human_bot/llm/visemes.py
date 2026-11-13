"""Lip-shape timeline (visemes) from TTS word boundaries — shared by the live card (/tts_cues) and the
clip renderer (tran_moment/synth_talk_clips.py).

Input: Edge's WordBoundary events [(start_s, dur_s, spoken_word)]. The spoken word is what Edge actually
said: Devanagari for the Hindi words (llm/hindi_script.to_devanagari), Roman for English words, digits and
names. We read THAT word — never the Roman sentence by position, which drifted whenever the two word
counts differed ("Creta", "10", "on-road").

Per word: graphemes -> phonemes (Devanagari: consonant + inherent schwa unless a matra / virama follows,
word-final schwa dropped; Roman: vowel groups and consonant clusters) -> mouth classes, then the word's
time is split by phoneme weight (vowels long, consonants short) instead of evenly per letter.
Coarticulation: consonants shorter than a visible shape fold into the neighbouring vowel, except lip
closures (p b m) and f/v, which are what a viewer checks and always get >= 60 ms.

Mouth letters (the card's /tts_cues map and path_b mouths.json):
  A closed lips (p b m)        B teeth / mid (t d n s r j ch, short e)   C open (a, aa)
  D wide (i ee y)              E round open (o au)                      F pucker (u oo w)
  G teeth on lip (f v)         H tongue (l)                             X rest
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------- Devanagari
_V_INDEP = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
            "ऋ": "ri", "ऑ": "o", "ॲ": "e"}
_MATRA = {"ा": "aa", "ि": "i", "ी": "ii", "ु": "u", "ू": "uu", "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ृ": "ri",
          "ॉ": "o", "ॅ": "e"}
_VIRAMA, _NUKTA = "्", "़"
_NASAL = {"ं", "ँ"}
# consonant -> mouth class
_CONS = {}
for chars, cls in (("पफबभम", "A"), ("व", "G"), ("ल", "H"), ("य", "D"),
                   ("कखगघङहक़ख़ग़", "C"),
                   ("चछजझञटठडढणतथदधनरसशषज़ड़ढ़", "B")):
    for ch in chars:
        _CONS[ch] = cls
_CONS["फ़"] = "G"
_DEVA_RE = re.compile(r"[ऀ-ॿ]")

# vowel -> (mouth class, weight)
_VOWEL = {"a": ("C", 0.8), "aa": ("C", 1.35), "i": ("D", 0.8), "ii": ("D", 1.2), "u": ("F", 0.8), "uu": ("F", 1.2),
          "e": ("B", 1.0), "ai": ("C", 1.2), "o": ("E", 1.1), "au": ("E", 1.25), "ri": ("B", 0.9)}
_CONS_W = 0.45
_MIN_SHAPE = 0.055          # s: a shape shorter than this is not seen (one frame at 24 fps is 42 ms)
_MIN_CLOSURE = 0.06         # s: p/b/m closure and f/v teeth are always shown at least this long


def _deva_phones(word: str) -> list[tuple[str, float]]:
    """Devanagari word -> [(mouth class, weight)]"""
    out: list[tuple[str, float]] = []
    chars = [c for c in word if _DEVA_RE.match(c)]
    i, n = 0, len(chars)
    while i < n:
        c = chars[i]
        if c in _V_INDEP:
            out.append(_VOWEL[_V_INDEP[c]])
        elif c in _CONS or c == _NUKTA:
            if c == _NUKTA:
                i += 1
                continue
            cls = _CONS[c]
            if i + 1 < n and chars[i + 1] == _NUKTA:
                cls = "G" if c == "फ" else cls
                i += 1
            out.append((cls, _CONS_W if cls != "A" else 0.55))
            nxt = chars[i + 1] if i + 1 < n else ""
            if nxt in _MATRA:
                out.append(_VOWEL[_MATRA[nxt]])
                i += 1
            elif nxt == _VIRAMA:
                i += 1
            elif i + 1 < n and nxt not in _NASAL:        # inherent schwa, but not word-finally (Hindi schwa deletion)
                # medial deletion: after a vowel, before a consonant that carries its own matra
                # (आपकी aap-ki, करके kar-ke, समझना); the first syllable keeps its schwa (नमस्ते na-mas-te)
                after_vowel = len(out) >= 2 and out[-2][0] in "CDEFB" and out[-2][1] >= 0.8
                nxt_has_matra = i + 2 < n and chars[i + 2] in _MATRA and nxt in _CONS
                if not (after_vowel and nxt_has_matra):
                    out.append(_VOWEL["a"])
            elif nxt in _NASAL:
                out.append(_VOWEL["a"])
        elif c in _MATRA:                                  # stray matra
            out.append(_VOWEL[_MATRA[c]])
        elif c in _NASAL and out:                          # nasal: a short n/m closure after the vowel
            out.append(("B", 0.3))
        i += 1
    return out


# ---------------------------------------------------------------- Roman (English words, names, Hinglish)
_ROMAN_VOWELS = [("aa", "aa"), ("ee", "ii"), ("ea", "ii"), ("oo", "uu"), ("ou", "au"), ("au", "au"), ("ai", "ai"),
                 ("ay", "ai"), ("oi", "o"), ("ow", "au"), ("a", "a"), ("e", "e"), ("i", "i"), ("o", "o"), ("u", "u"), ("y", "i")]
_ROMAN_CONS = {"p": "A", "b": "A", "m": "A", "f": "G", "v": "G", "w": "F", "l": "H", "y": "D",
               "k": "C", "g": "C", "h": "C", "q": "C", "c": "C", "x": "C"}
_DIGITS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
           "8": "eight", "9": "nine"}


def _roman_phones(word: str) -> list[tuple[str, float]]:
    if re.fullmatch(r"[\d.,:/%-]+", word):          # a number: Edge reads it as a Hindi/English word we cannot see
        return [("C", 1.0), ("B", _CONS_W), ("C", 0.9)] * max(1, min(3, len(re.sub(r"\D", "", word)) // 2 or 1))
    w = "".join(_DIGITS.get(c, c) for c in word.lower())
    w = re.sub(r"[^a-z]", "", w)
    if len(w) > 3 and w.endswith("e") and w[-2] not in "aeiou":   # silent final e (price, care, drive)
        w = w[:-1]
    out: list[tuple[str, float]] = []
    i = 0
    while i < len(w):
        for g, ph in _ROMAN_VOWELS:
            if w.startswith(g, i) and not (g == "y" and i == 0):
                out.append(_VOWEL[ph])
                i += len(g)
                break
        else:
            c = w[i]
            two = w[i:i + 2]
            if two in ("ph",):
                out.append(("G", _CONS_W)); i += 2; continue
            if two in ("th", "sh", "ch", "kh", "gh", "bh", "dh", "jh"):
                c = two[0]
                i += 1
            cls = _ROMAN_CONS.get(c, "B")
            if not (out and out[-1][0] == cls and out[-1][1] <= 0.55):   # double letters: one shape
                out.append((cls, _CONS_W if cls != "A" else 0.55))
            i += 1
    return out


def phones(word: str) -> list[tuple[str, float]]:
    p = _deva_phones(word) if _DEVA_RE.search(word) else _roman_phones(word)
    return p or [("C", 1.0)]


def _protected(v: str) -> bool:
    return v in ("A", "G")


def cues(words: list[tuple[float, float, str]], tail: float = 0.3) -> list[dict]:
    """[(start_s, dur_s, spoken_word)] -> [{"s", "e", "v"}] covering the clip, rests (X) in the pauses."""
    if not words:
        return []
    out: list[dict] = []
    t_prev = 0.0
    for start, dur, spoken in words:
        dur = max(dur, 0.08)
        if start - t_prev > 0.12:
            out.append({"s": round(t_prev, 3), "e": round(start, 3), "v": "X"})
        ph = phones(spoken)
        total = sum(w for _, w in ph)
        t = start
        segs = []
        for v, w in ph:
            d = dur * w / total
            segs.append([v, t, t + d])
            t += d
        # lip closures / f-v: at least _MIN_CLOSURE, borrowed from the following (or previous) vowel
        for k, s in enumerate(segs):
            if _protected(s[0]) and s[2] - s[1] < _MIN_CLOSURE:
                need = _MIN_CLOSURE - (s[2] - s[1])
                if k + 1 < len(segs) and segs[k + 1][2] - segs[k + 1][1] > need + _MIN_SHAPE:
                    s[2] += need; segs[k + 1][1] += need
                elif k > 0 and segs[k - 1][2] - segs[k - 1][1] > need + _MIN_SHAPE:
                    s[1] -= need; segs[k - 1][2] -= need
        for v, s0, e0 in segs:
            out.append({"s": round(s0, 3), "e": round(e0, 3), "v": v})
        t_prev = start + dur
    out.append({"s": round(t_prev, 3), "e": round(t_prev + tail, 3), "v": "X"})
    # coarticulation: a short unprotected shape folds into its neighbour; equal neighbours merge
    merged: list[dict] = []
    for c in out:
        short = (c["e"] - c["s"]) < _MIN_SHAPE and not _protected(c["v"])
        if merged and (merged[-1]["v"] == c["v"] or short):
            merged[-1]["e"] = c["e"]
        elif merged and (merged[-1]["e"] - merged[-1]["s"]) < _MIN_SHAPE and not _protected(merged[-1]["v"]) and merged[-1]["v"] != "X":
            merged[-1].update(v=c["v"], e=c["e"])
        else:
            merged.append(dict(c))
    return merged


# letter -> puppet mouth image (path_b mouths). O.png for round vowels and FV.png for f/v were unused before.
MOUTH_MAP = {"A": "closed.png", "B": "E.png", "C": "A.png", "D": "I.png", "E": "O.png", "F": "U.png", "G": "FV.png",
             "H": "L.png", "X": "X.png"}
