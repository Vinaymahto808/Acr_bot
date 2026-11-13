"""Real ACR call transcripts (Excel, kb/data/calls/{all,insurance}/*.xlsx) -> a clean, masked corpus.

Stage 0 of the call-knowledge pipeline (deterministic, no network):
  * one row per call (the insurance sheets are a subset of "all": deduplicated by recording URL / time+agent)
  * drops calls with no knowledge in them: no speech, network/IVR messages, under 30 s or 120 characters
  * masks personal data before anything leaves this machine: phone numbers, vehicle registration numbers,
    e-mail addresses, OTP / delivery codes. The phone column itself is never read.
  * scores each call by how much customer-facing knowledge it carries (questions, prices, process words)
Writes kb/data/calls/corpus.jsonl (git-ignored with kb/data/) and prints the counts.

    python kb/tools/calls_corpus.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "calls"
OUT = SRC / "corpus.jsonl"

JUNK = re.compile(r"no speech detected|cannot receive incoming calls|number you (have )?(are trying|dialled|called)|switched off|"
                  r"not reachable|is busy|please try again later|invalid number|does not exist|नंबर.{0,20}(बंद|उपलब्ध नहीं)|"
                  r"आप जिस नंबर|व्यस्त है|not answering|call is being forwarded|voicemail", re.I)
PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?(?:\d[\s-]?){10,12}(?!\d)")
REG = re.compile(r"\b[A-Z]{2}[\s-]?\d{1,2}[\s-]?[A-Z]{1,3}[\s-]?\d{3,4}\b", re.I)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
CODE = re.compile(r"\b(otp|code|कोड|ओटीपी)\b\W{0,3}(\d[\s-]?){4,6}", re.I)
KNOW = re.compile(r"\b(price|rate|charge|cost|estimate|kitna|kitne|kitni|रुपए|रुपये|hazar|हज़ार|हजार|warranty|guarantee|pickup|pick up|drop|"
                  r"insurance|claim|cashless|surveyor|survey|file|policy|ncb|depreciation|zero dep|deductible|"
                  r"denting|painting|dent|paint|ac\b|gas|cooling|service|servicing|brake|clutch|battery|tyre|alignment|suspension|engine|"
                  r"timing|sunday|open|address|location|branch|okhla|moti nagar|noida|gurgaon|gurugram|delivery|ready|kab tak|"
                  r"payment|emi|upi|cash|gst|bill|invoice|estimate|parts|genuine|oem|duplicate)\b|\?", re.I)


def mask(t: str) -> str:
    t = EMAIL.sub("[EMAIL]", t)
    t = CODE.sub(lambda m: m.group(1) + " [CODE]", t)
    t = REG.sub("[CAR_NO]", t)
    t = PHONE.sub("[PHONE]", t)
    return re.sub(r"\s+", " ", t).strip()


def rows(path: Path, kind: str):
    wb = openpyxl.load_workbook(path, read_only=True)
    for ws in wb.worksheets:
        it = ws.iter_rows(values_only=True)
        head = [str(h or "").strip().lower() for h in next(it, [])]
        if "transcript" not in head:
            continue
        ix = {h: i for i, h in enumerate(head)}
        for r in it:
            g = lambda k: r[ix[k]] if k in ix and ix[k] < len(r) else None  # noqa: E731
            yield {"time": str(g("call time (ist)") or ""), "direction": g("direction") or "", "agent": g("agent") or "",
                   "duration": float(g("duration (sec)") or 0) if str(g("duration (sec)") or "").replace(".", "", 1).isdigit() else 0.0,
                   "url": g("recording url") or "", "text": g("transcript") or "", "kind": kind,
                   "insurance": bool(g("insurance related")) or kind == "insurance", "keywords": g("keywords found") or ""}


def main() -> int:
    seen: dict[str, dict] = {}
    total = 0
    for kind in ("insurance", "all"):
        for f in sorted((SRC / kind).glob("[0-9]*.xlsx")):
            for r in rows(f, kind):
                total += 1
                key = r["url"] or (r["time"] + r["agent"])
                if key in seen:
                    seen[key]["insurance"] |= r["insurance"]
                    continue
                seen[key] = r
    stats = {"rows_read": total, "unique_calls": len(seen), "no_text": 0, "junk": 0, "short": 0, "kept": 0}
    kept = []
    for r in seen.values():
        t = str(r["text"] or "").strip()
        if not t or t.startswith("("):
            stats["no_text"] += 1; continue
        if JUNK.search(t[:300]) and len(t) < 400:
            stats["junk"] += 1; continue
        if r["duration"] and r["duration"] < 30 or len(t) < 120:
            stats["short"] += 1; continue
        m = mask(t)
        score = len(KNOW.findall(m)) + (3 if r["insurance"] else 0)
        kept.append({"id": hashlib.sha1(str(r["url"] or r["time"]).encode()).hexdigest()[:12], "month": r["time"][:7],
                     "direction": r["direction"], "seconds": round(r["duration"]), "insurance": r["insurance"],
                     "score": score, "chars": len(m), "text": m})
    kept.sort(key=lambda x: -x["score"])
    stats["kept"] = len(kept)
    with OUT.open("w", encoding="utf-8") as fh:
        for k in kept:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    ins = sum(k["insurance"] for k in kept)
    chars = sum(k["chars"] for k in kept)
    print(json.dumps({**stats, "kept_insurance": ins, "total_chars": chars,
                      "score_p50": kept[len(kept) // 2]["score"] if kept else 0,
                      "score_top1000_min": kept[min(999, len(kept) - 1)]["score"] if kept else 0}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
