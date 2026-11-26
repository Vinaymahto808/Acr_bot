"""Which language the customer is speaking this turn: English, Hinglish or Hindi.

Neha mirrors it (2026-11-14: "I want to book a service, can you help me?" got a Hinglish answer read by the
Hindi voice). Decided per turn from grammar words, not content words: "service", "AC", "pickup", "price" are
used in all three, while "hai / kya / ka / mera" or "is / the / can / please" give the language away.

    detect("I want to book a service")        -> "english"
    detect("meri gaadi ki service karwani hai") -> "hinglish"
    detect("मेरी गाड़ी की सर्विस करनी है")       -> "hindi"
    detect("Rahul") / detect("9876500000")     -> None   (too short to tell: keep the current language)
"""
from __future__ import annotations

import re

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_WORD = re.compile(r"[A-Za-z']+")

# Roman-script Hindi grammar / everyday words. Not "ji" (Indian English says it too), not "to"/"do"/"me"/"the"
# (English words as well).
HINDI = frozenset("""
hai hain ho hoon hu hun tha thi kya ka ki ke ko se mein mai main mera meri mere mujhe hum humara humari
aap aapka aapki aapke apna apni apne tum tumhara nahi nahin na haan han ha kar karna karni karne karo kariye
kijiye karwana karwani karwa karwaana kab kaise kitna kitni kitne kahan kaha kyun kyon kaun kaunsa konsa
chahiye gaadi gadi gaddi theek thik accha acha achha achchha bhai bhi toh wala wali wale raha rahi rahe
milega milegi milenge sakte sakta sakti batao bataiye bata dijiye dena hoga hogi abhi kal aaj subah shaam
kharcha paisa paise wahan yahan bahut sab kuch koi ek lekin aur ya par pe liye dekh dekhiye bolo boliye
samajh jaldi ruko rukiye ruk jayega jayegi jaega hua hui huye gaya gayi gaye lagega lagta lagti chal chalega
waise matlab bas thoda zyada jyada kuchh wali hoti hota yeh ye woh wo isko usko iska uska inka unka
""".split())

# English grammar words; content words (service, price, car) are left out on purpose.
ENGLISH = frozenset("""
i i'm im i've i'd you you're your yours we our us is are am was were be been the a an can could would will
shall should do does did what when where how why which who want need please have has had it it's this that
these those there for of with and or but not no yes my me get got give book help about any some much many
tell know let thanks thank hello hi hey okay ok sure if then than so just also very
""".split())


def detect(text: str) -> str | None:
    """'english' | 'hinglish' | 'hindi', or None when the turn is too short or has no grammar words."""
    t = (text or "").strip()
    if not t:
        return None
    letters = [ch for ch in t if ch.isalpha()]
    deva = len(_DEVANAGARI.findall(t))
    if letters and deva > len(letters) / 2:
        return "hindi"
    words = [w.lower() for w in _WORD.findall(t)]
    if deva:                                   # Devanagari mixed with Roman words: spoken Hinglish written by STT
        return "hinglish" if len(words) + 1 >= 3 else None
    if len(words) < 3:                         # "Rahul", "ok", "yes please", "Creta": keep the current language
        return None
    h = sum(w in HINDI for w in words)
    e = sum(w in ENGLISH for w in words)
    if h == 0 and e >= 1:
        return "english"
    if h >= 1:
        # "I want to book my car service, ok?" with one stray Hindi-looking word stays English
        return "english" if e >= 4 * h and h == 1 and len(words) >= 8 else "hinglish"
    return None


def sentence_is_english(text: str) -> bool:
    """For the voice of ONE reply sentence: Roman script and no Hindi grammar word."""
    t = text or ""
    if _DEVANAGARI.search(t):
        return False
    words = [w.lower() for w in _WORD.findall(t)]
    return bool(words) and not any(w in HINDI for w in words)


REPLY_RULE = {
    "english": ("The customer is speaking ENGLISH. Reply only in natural Indian English, in Roman script. No Hindi "
                "or Hinglish words (no acha, gaadi, hai, dijiye, bataiye, theek hai); \"ji\" once at most. Same "
                "facts, same short spoken style, one question."),
    "hinglish": ("The customer is speaking Hinglish. Reply in Roman-script Hinglish, mirroring their mix of Hindi "
                 "and English words."),
    "hindi": ("The customer is speaking HINDI in Devanagari script. Write your WHOLE reply in Devanagari (हिंदी), "
              "not in Roman letters, even though the examples and website passages above are Roman Hinglish. "
              "Simple spoken Hindi, like: \"जी हाँ, पिकअप और ड्रॉप बिल्कुल फ्री है। आपकी गाड़ी कौन सी है?\" "
              "Everyday car words (सर्विस, एसी, पिकअप, ब्रेक) are fine."),
}


def reply_rule(language: str) -> str:
    rule = REPLY_RULE.get(language)
    return f"## Reply language (this turn, overrides the examples above)\n{rule}" if rule else ""
