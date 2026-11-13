"""Stage 1 of the call-knowledge pipeline: customer questions + ACR's answers from the real call transcripts.

Reads the masked corpus (kb/data/calls/corpus.jsonl, tools/calls_corpus.py), sends the most knowledge-dense
calls in small batches to the LLM (Groq gpt-oss-120b, qwen as fallback) and keeps only GENERAL, reusable
knowledge that ACR staff actually stated on the call: prices and ranges, process, documents, insurance-claim
steps, timings, pickup, warranty, payment. Nothing about one customer (names, numbers, their car, their case).

Resumable: every batch's answer is cached in kb/data/calls/extract_cache.jsonl (key = hash of the batch), so a
429 or Ctrl-C costs nothing; re-running continues where it stopped. Output: kb/data/calls/qa_from_calls.csv in
the same columns as tran_moment/prepared/acr_customer_qa_all.csv, ready for tools/clean_qa.py.

    python kb/tools/calls_extract.py --calls 1200            # insurance calls first, then the best general ones
    python kb/tools/calls_extract.py --calls 1200 --dry      # show the selection, call nothing
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "kb" / "data" / "calls"
CORPUS, CACHE, OUT = DATA / "corpus.jsonl", DATA / "extract_cache.jsonl", DATA / "qa_from_calls.csv"
ENV = ROOT / "voice_clone" / ".env"
MODELS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]   # rotated: each has its own per-minute limit
CATEGORIES = ["Pricing", "Service Details", "Booking & Pickup", "Location & Timings", "Insurance & Claims", "Parts & Accessories",
              "Warranty & Complaints", "Repair & Diagnosis", "Car Status & Delivery", "Payment & Billing", "Other"]

SYSTEM = f"""You read real phone-call transcripts of Auto Car Repair (ACR), a multi-brand car workshop in Delhi NCR (Hinglish,
Devanagari and English mixed, speech-to-text errors, no speaker labels). Extract the GENERAL knowledge a new service advisor
needs: what customers ask, and what ACR staff answered.

Rules:
- Only facts ACR staff actually stated in these calls. Never add, guess or improve a fact. If an answer is unclear, skip it.
- General and reusable only: prices/ranges, what is included, process and steps, documents needed, insurance claim and
  surveyor process, cashless, depreciation/zero-dep, pickup/drop, timings, branches, warranty, payment options, delivery time.
- NOTHING personal: no names, phone numbers, car numbers, [PHONE]/[CAR_NO] tokens, dates of one booking, one customer's
  specific case, complaints about one employee. Rewrite "aapki Ertiga" as a general question.
- Official facts (use them to correct speech-to-text errors; a branch, brand or term NOT in this list is dropped):
  ACR branches: Okhla Phase 2, Moti Nagar (Rama Road), Noida Sector 63, Gurgaon Sector 34. Company: Auto Car Repair (ACR),
  multi-brand workshop. Common STT errors: "auto care care"/"autocar" = Auto Car Repair, "pendensor/condensor" = condenser,
  "Modinagar" = Moti Nagar, "survey"/"surveyar" = surveyor, "dent pent" = dent paint.
- Skip answers that only promise to check later ("bata denge", "confirm karke batayenge", "senior se puchh ke"): no fact, no row.
- Skip time-limited offers and coupons (festival offers, "is mahine", "aaj tak"): they expire.
- A price quoted for one car stays tied to it and is approximate: "Swift ki AC service approx 2,500 + GST se shuru, gaadi dekh ke
  final". Never turn one quote into a general price for all cars.
- Question as the customer would ask it; answer as ACR says it. Hinglish in Roman script, short (answer max 45 words).
- category is one of: {", ".join(CATEGORIES)}.
Output ONLY a JSON array (may be empty) of objects:
{{"category": "...", "question_hinglish": "...", "answer_hinglish": "...", "question_english": "...", "answer_english": "...", "call": <call number>}}"""


def env_key(name: str) -> str:
    for line in ENV.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if m and m.group(1).upper() == name:
            return m.group(2).strip().strip('"').strip("'")
    sys.exit(f"{name} missing in voice_clone/.env")


def select(n: int, chars: int) -> list[dict]:
    calls = [json.loads(l) for l in CORPUS.open(encoding="utf-8")]
    ins = [c for c in calls if c["insurance"]]
    gen = [c for c in calls if not c["insurance"]]
    pick = ins[: max(n // 2, n - len(gen))] + gen
    pick = pick[:n]
    for c in pick:
        t = c["text"]
        c["clip"] = t if len(t) <= chars else t[: int(chars * 0.7)] + " … " + t[-int(chars * 0.3):]
    return pick


def ask(key: str, user: str, model: str) -> tuple[list, dict]:
    body = {"model": model, "temperature": 0.1, "max_tokens": 4000, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}
    if model.startswith("openai/"):
        body["reasoning_effort"] = "low"
    req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json", "User-Agent": "acr-kb/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
        rl = {k: r.headers.get(k) for k in ("x-ratelimit-remaining-tokens", "x-ratelimit-remaining-requests", "x-ratelimit-reset-tokens")}
    text = d["choices"][0]["message"]["content"] or ""
    m = re.search(r"\[.*\]", text, re.S)
    return (json.loads(m.group(0)) if m else []), {**rl, "tokens": d.get("usage", {}).get("total_tokens")}


COOL: dict[str, float] = {}   # model -> time it may be asked again (429 retry-after)
BAD = re.compile(r"\[(PHONE|CAR_NO|EMAIL|CODE)\]|\b\d{10}\b|\b[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{3,4}\b", re.I)


def clean_item(it: dict) -> dict | None:
    if not isinstance(it, dict):
        return None
    q, a = str(it.get("question_hinglish") or "").strip(), str(it.get("answer_hinglish") or "").strip()
    if len(q) < 8 or len(a) < 8 or BAD.search(q + " " + a):
        return None
    cat = it.get("category") if it.get("category") in CATEGORIES else "Other"
    return {"category": cat, "question_hinglish": q, "answer_hinglish": a,
            "question_english": str(it.get("question_english") or "").strip(), "answer_english": str(it.get("answer_english") or "").strip()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--chars", type=int, default=1800)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    pick = select(a.calls, a.chars)
    batches = [pick[i:i + a.batch] for i in range(0, len(pick), a.batch)]
    print(f"{len(pick)} calls ({sum(c['insurance'] for c in pick)} insurance) in {len(batches)} batches", flush=True)
    if a.dry:
        return 0
    cache = {}
    if CACHE.exists():
        for line in CACHE.open(encoding="utf-8"):
            try:
                r = json.loads(line); cache[r["key"]] = r
            except Exception:  # noqa: BLE001
                pass
    key = env_key("GROQ_API_KEY")
    done = fails = 0
    for bi, b in enumerate(batches):
        user = "\n\n".join(f"### Call {i + 1} ({c['seconds']} s{', insurance' if c['insurance'] else ''})\n{c['clip']}" for i, c in enumerate(b))
        h = hashlib.sha1((SYSTEM + user).encode("utf-8")).hexdigest()[:16]
        if h in cache:
            continue
        items, info, model_used = None, {}, None
        for attempt in range(12):
            now = time.time()
            ready = [m for m in MODELS if COOL.get(m, 0) <= now]
            if not ready:                                   # every model is cooling down: wait for the first
                time.sleep(max(1.0, min(COOL.values()) - now)); continue
            model = ready[(bi + attempt) % len(ready)]
            try:
                items, info = ask(key, user, model); model_used = model
                break
            except urllib.error.HTTPError as e:
                wait = float(e.headers.get("retry-after") or 0) or 20.0
                COOL[model] = time.time() + wait
                if e.code != 429:
                    print(f"  batch {bi}: HTTP {e.code} ({model})", flush=True)
            except (json.JSONDecodeError, urllib.error.URLError, TimeoutError, KeyError, http.client.HTTPException, ConnectionError, OSError) as e:   # a dropped connection is retried, not fatal
                print(f"  batch {bi}: {type(e).__name__} ({model}), retry", flush=True)
                time.sleep(3)
        if items is None:
            fails += 1
            continue
        kept = [x for x in (clean_item(i) for i in items) if x]
        rec = {"key": h, "batch": bi, "model": model_used, "calls": [c["id"] for c in b], "items": kept, "raw_items": len(items)}
        with CACHE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        cache[h] = rec
        done += 1
        if done % 10 == 0 or bi < 3:
            print(f"  batch {bi + 1}/{len(batches)}: +{len(kept)} items  tokens={info.get('tokens')} "
                  f"remaining={info.get('x-ratelimit-remaining-tokens')}/{info.get('x-ratelimit-remaining-requests')}", flush=True)
    rows = [dict(it, times_seen=1, batch="calls-" + str(r["batch"])) for r in cache.values() for it in r["items"]]
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["category", "question_hinglish", "answer_hinglish", "question_english", "answer_english", "times_seen", "batch"])
        w.writeheader(); w.writerows(rows)
    print(f"done: {done} new batches, {fails} failed, {len(rows)} Q&A rows in {OUT.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
