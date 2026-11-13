"""Stage 2 of the call-knowledge pipeline: merge the Q&A extracted from the call recordings with the earlier
call Q&A, deduplicate, and keep only what can be trusted.

  old  tran_moment/prepared/acr_customer_qa_all.csv        (301 rows from the first call batches)
  new  kb/data/calls/qa_from_calls.csv                      (tools/calls_extract.py)
  ->   tools/clean_qa.py merges near-duplicate questions; times_seen adds up across calls
  ->   trust rules on rows that come ONLY from the new extraction:
         * an answer that is only "we will tell you later" is dropped
         * a price heard on a single call is held back for the ACR team (kb/seed/CALLS_REVIEW.csv), because one
           quote is one car's quote or a speech-to-text slip; seen on 2+ calls it goes in
  ->   kb/seed/acr_customer_qa.clean.csv  (previous file backed up to kb/data/backups/)

Then: python kb/tools/rewrite_qa.py --only-flagged   and   node kb/src/ingest_qa.js

    python kb/tools/calls_merge.py
"""
from __future__ import annotations

import csv
import io
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "tran_moment" / "prepared" / "acr_customer_qa_all.csv"
NEW = ROOT / "kb" / "data" / "calls" / "qa_from_calls.csv"
COMBINED = ROOT / "kb" / "data" / "calls" / "combined_raw.csv"
CLEAN_ALL = ROOT / "kb" / "data" / "calls" / "combined_clean.csv"
SEED = ROOT / "kb" / "seed" / "acr_customer_qa.clean.csv"
REVIEW = ROOT / "kb" / "seed" / "CALLS_REVIEW.csv"
FIELDS = ["category", "question_hinglish", "answer_hinglish", "question_english", "answer_english", "times_seen", "batch"]
VAGUE = re.compile(r"\b(bata(?:\s|ye|y)?\s*denge|batayenge|bataenge|confirm kar(?:ke|wa)|senior se|check karke|puchh? ?ke|"
                   r"call back kar(?:enge|denge)|baad mein batate)\b", re.I)


def read(path: Path) -> list[dict]:
    return list(csv.DictReader(io.StringIO(path.read_bytes().decode("utf-8-sig"))))


def main() -> int:
    old, new = read(OLD), read(NEW)
    with COMBINED.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in old:
            r = {k.strip().strip('"'): v for k, v in r.items()}
            w.writerow({k: r.get(k, "") for k in FIELDS})
        for r in new:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    subprocess.run([sys.executable, str(ROOT / "kb" / "tools" / "clean_qa.py"), "--in", str(COMBINED), "--out", str(CLEAN_ALL)], check=True)

    rows = read(CLEAN_ALL)
    keep, held, dropped = [], [], 0
    for r in rows:
        batches = [b for b in (r.get("batches") or "").split(",") if b]
        new_only = bool(batches) and all(b.startswith("calls-") for b in batches)
        if new_only:
            r["source"] = "ACR call recordings (Jan-Oct 2026)"
        if new_only and VAGUE.search(r["answer"]) and len(r["answer"]) < 90:
            dropped += 1
            continue
        if new_only and r.get("has_price") == "1" and int(r.get("times_seen") or 1) < 2:
            held.append(r)
            continue
        keep.append(r)
    backups = ROOT / "kb" / "data" / "backups"
    backups.mkdir(parents=True, exist_ok=True)
    if SEED.exists():
        shutil.copy2(SEED, backups / f"acr_customer_qa.clean.{time.strftime('%Y%m%d-%H%M%S')}.csv")
    fields = list(rows[0].keys()) if rows else []
    for path, data in ((SEED, keep), (REVIEW, held)):
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader(); w.writerows(data)
    old_ids = {r["id"] for r in read(backups / sorted(p.name for p in backups.glob("acr_customer_qa.clean.*.csv"))[-1])} if SEED.exists() else set()
    print(f"merged: {len(rows)} unique questions -> {len(keep)} in the knowledge base "
          f"({sum(1 for r in keep if r['id'] not in old_ids)} new), {len(held)} single-call prices held for review, "
          f"{dropped} vague answers dropped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
