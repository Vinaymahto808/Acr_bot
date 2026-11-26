"""Measure the fast path before trusting it: precision, coverage, false answers on non-questions, and speed.

    python human_bot/tools/eval_fastpath.py [--kb http://127.0.0.1:4100] [--sweep]

Held-out test: every Q&A row with 2+ phrasings gives up its last phrasing; the index is built from the rest
and the held-out phrasings are asked. 'precision' = right answer among the answered, 'coverage' = answered
share. 'never' = turns that must never get a canned answer (names, numbers, yes/no, off-topic, booking words).
--sweep prints the score/margin grid used to pick FastPath.MIN_SCORE / MIN_MARGIN.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm import fastpath as fpm  # noqa: E402

NEVER = ["mera naam Rahul hai", "Rahul", "9876500000", "haan ji", "ok thanks", "kal subah 10 baje", "Gurgaon",
         "yes please book it", "nahi abhi nahi", "What is the capital of France?", "best pizza recipe",
         "bitcoin price today", "Creta hai", "haan bhej do", "main Sector 56 mein hoon", "theek hai"]


def fetch(kb: str, key: str) -> list[fpm.Row]:
    req = urllib.request.Request(f"{kb}/api/knowledge/faq?" + urllib.parse.urlencode({"website": "auto_car_repair"}),
                                 headers={"x-api-key": key} if key else {})
    return fpm.rows_from_json(json.loads(urllib.request.urlopen(req, timeout=30).read().decode("utf-8")))


def evaluate(rows, min_score, min_margin, verbose=False):
    fpm.FastPath.MIN_SCORE, fpm.FastPath.MIN_MARGIN = min_score, min_margin
    train, tests = [], []
    for r in rows:
        if len(r.questions) >= 2:
            tests.append((r.questions[-1], r.answer))
            train.append(fpm.Row(r.url, r.questions[:-1], r.answer, r.answer_en))
        else:
            train.append(r)
    fp = fpm.FastPath(train)
    answered = right = 0
    wrong_examples = []
    times = []
    for q, ans in tests:
        t0 = time.perf_counter()
        m = fp.match(q)
        times.append((time.perf_counter() - t0) * 1000)
        if m:
            answered += 1
            got = fp.rows.index(m.row)
            truth = next((i for i, r in enumerate(fp.rows) if r.answer == ans), None)
            # right: the same answer, or a row that says the same thing (same question from another call)
            if truth is not None and fp.same_answer(got, truth):
                right += 1
            elif len(wrong_examples) < 8:
                wrong_examples.append((q, m.row.questions[0], round(m.score, 2), m.row.answer[:70], ans[:70]))
    never_hits = [(t, fp.match(t).row.questions[0]) for t in NEVER if fp.match(t)]
    prec = right / answered if answered else 1.0
    cov = answered / len(tests) if tests else 0.0
    if verbose:
        print(f"rows {len(rows)} (usable {len(fp.rows)}), held-out questions {len(tests)}")
        print(f"precision {prec:.1%}  coverage {cov:.1%}  never-answered violations {len(never_hits)}/{len(NEVER)}")
        print(f"match time: median {statistics.median(times):.3f} ms, p99 {sorted(times)[int(len(times) * 0.99) - 1]:.3f} ms, max {max(times):.3f} ms")
        for w in wrong_examples:
            print("  WRONG:", w)
        for n in never_hits:
            print("  SHOULD NOT ANSWER:", n)
        oos = ["What is the capital of France?", "best pizza recipe", "bitcoin price today"]
        print("bloom in_domain on off-topic:", {t: fp.in_domain(t) for t in oos})
        print("bloom in_domain on car talk:", {t: fp.in_domain(t) for t in ["AC thanda nahi", "brake noise", "pickup free?"]})
    return prec, cov, len(never_hits)


if __name__ == "__main__":
    kb = sys.argv[sys.argv.index("--kb") + 1] if "--kb" in sys.argv else os.getenv("KB_URL", "http://127.0.0.1:4100")
    rows = fetch(kb, os.getenv("KB_ADMIN_KEY", ""))
    chosen = (fpm.FastPath.MIN_SCORE, fpm.FastPath.MIN_MARGIN)   # the sweep changes the class attributes
    if "--sweep" in sys.argv:
        print("score margin  precision coverage never")
        for s in (0.55, 0.6, 0.65, 0.7, 0.72, 0.75, 0.8, 0.85):
            for mg in (0.0, 0.05, 0.08, 0.12, 0.16):
                p, c, nv = evaluate(rows, s, mg)
                print(f"{s:5.2f} {mg:5.2f}   {p:8.1%} {c:8.1%} {nv:5d}")
    print(f"chosen thresholds: score {chosen[0]}, margin {chosen[1]}")
    evaluate(rows, *chosen, verbose=True)
