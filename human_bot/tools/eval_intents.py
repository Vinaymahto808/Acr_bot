"""Accuracy of the service / intent reading: the old rules alone, the meaning matcher alone, and both as the server
combines them (rules first, matcher only where the rules found nothing; of the matcher's intents only
intents.SERVER_INTENTS, as in server/app.py). Held-out set: tests/data/intent_heldout.json.

    uv run --no-project --python 3.13 --with numpy --with onnxruntime --with tokenizers python human_bot/tools/eval_intents.py
        [--misses] [--sweep] [--latency]

--misses lists every wrong row, --sweep tries other thresholds (pick ones that keep chit-chat false positives at 0),
--latency times one message end to end.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

BOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOT))
sys.path.insert(0, str(BOT / "server"))

from llm import intents  # noqa: E402
from llm.pages import named_service, named_services, wants_some_service, with_roman  # noqa: E402

HELDOUT = BOT / "tests" / "data" / "intent_heldout.json"
PERSONA = json.loads((BOT / "persona" / "persona_auto_car_repair.json").read_text(encoding="utf-8"))


def rule_service(text: str) -> str | None:
    mt = with_roman(text)
    specs = named_services(mt, PERSONA)
    spec = specs[-1] if specs else named_service(mt, PERSONA)
    return spec["path"] if spec else None


def rule_intent(text: str, app) -> str | None:
    """The server's regex reading of a turn, in the order its step machine checks them."""
    from crm.service_request import is_hold, is_no, is_yes, wants_booking
    mt = with_roman(text)
    if app.CANCEL_RE.search(mt):
        return "cancel"
    if app.BYE_RE.search(text):
        return "goodbye"
    if app.COMPLAINT_RE.search(mt):
        return "complaint"
    if app.OUT_OF_SCOPE_RE.search(mt):
        return "out_of_scope"
    if app.HUMAN_RE.search(mt):
        return "human_handoff"
    if is_hold(text) or app.LATER_RE.search(mt):
        return "hold"
    if app.PRICE_RE.search(mt):
        return "price_question"
    if wants_some_service(mt) and not rule_service(text):
        return "generic_service_wanted"
    if (wants_booking(mt) or app.INTEREST_RE.search(mt)) and not app.NOT_INTERESTED_RE.search(mt):
        return "book"
    if is_no(text) and len(text.split()) <= 2:
        return "no"
    if app.NOT_INTERESTED_RE.search(mt):
        return "not_interested"
    if is_yes(text):
        return "yes"
    return None


RULE_TURN = {"cancel", "goodbye", "complaint", "out_of_scope", "human_handoff", "hold"}   # server: no matcher service then


def ok_intent(pred: str | None, want: str) -> bool:
    if want == "other":
        return pred in (None, "other")
    return pred == want


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--misses", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--latency", action="store_true")
    args = ap.parse_args()
    import app  # noqa: PLC0415  (the server's regexes)

    data = json.loads(HELDOUT.read_text(encoding="utf-8"))
    rows = [(grp, *r) for grp, rs in data.items() if not grp.startswith("_") for r in rs]
    ex = intents.load_data()
    seen = {intents.normalize(p) for v in ex["services"].values() for p in v}
    seen |= {intents.normalize(p) for v in ex["intents"].values() for p in v} | {intents.normalize(p) for p in ex["service_none"]}
    leak = [r[1] for r in rows if intents.normalize(r[1]) in seen]
    if leak:
        print("held-out rows that are also example phrases (remove them from acr_intents.json):", leak)

    t0 = time.perf_counter()
    m = intents.load(block=True)
    if not m:
        print("matcher unavailable:", intents.status())
        return 1
    print(f"model {intents.MODEL_NAME}, {intents.status()['examples']} example rows, loaded in {(time.perf_counter()-t0)*1000:.0f} ms")
    res = [m.classify(r[1]) for r in rows]
    rules = [(rule_service(r[1]), rule_intent(r[1], app)) for r in rows]

    def score(svc_thr=None, int_thr=None, show=False):
        if svc_thr:
            m.service_head.thr.update(svc_thr)
        if int_thr:
            m.intent_head.thr.update(int_thr)
        tab = {k: {"svc": [0, 0], "int": [0, 0]} for k in ("rules", "matcher", "combined")}
        fp = {k: 0 for k in tab}
        groups = {}
        misses = []
        for (grp, text, want_s, want_i), r, (rs, ri) in zip(rows, res, rules):
            ms = m.service_head.decide(r.raw_service)
            mi = m.intent_head.decide(r.raw_intent)
            preds = {"rules": (rs, ri), "matcher": (ms.label if ms else None, mi.label if mi else None),
                     "combined": (rs or (ms.label if ms and ri not in RULE_TURN else None),
                                  ri or (mi.label if mi and mi.label in intents.SERVER_INTENTS else None))}
            for k, (ps, pi) in preds.items():
                g = groups.setdefault((grp, k), [0, 0])
                good = ps == want_s and (want_i is None or ok_intent(pi, want_i))
                g[0] += good
                g[1] += 1
                tab[k]["svc"][0] += ps == want_s
                tab[k]["svc"][1] += 1
                if want_i is not None:
                    tab[k]["int"][0] += ok_intent(pi, want_i)
                    tab[k]["int"][1] += 1
                if grp == "chitchat" and (ps or pi not in (None, "other")):
                    fp[k] += 1
                if k == "combined" and not good:
                    misses.append((grp, text, want_s, want_i, ps, pi, r.raw_service, r.raw_intent))
        if show:
            n_chat = sum(1 for r in rows if r[0] == "chitchat")
            print(f"\n{'':10} {'service acc':>14} {'intent acc':>14} {'row acc':>12} {'chit-chat FP':>13}")
            for k in tab:
                s, i = tab[k]["svc"], tab[k]["int"]
                full = sum(v[0] for (g, kk), v in groups.items() if kk == k), sum(v[1] for (g, kk), v in groups.items() if kk == k)
                print(f"{k:10} {s[0]:>4}/{s[1]:<3} {100*s[0]/s[1]:5.1f}% {i[0]:>4}/{i[1]:<3} {100*i[0]/i[1]:5.1f}% "
                      f"{100*full[0]/full[1]:10.1f}% {fp[k]:>5}/{n_chat} ({100*fp[k]/n_chat:.0f}%)")
            print("\nrow accuracy per group (rules / matcher / combined):")
            for grp in [g for g in data if not g.startswith("_")]:
                cells = [groups[(grp, k)] for k in tab]
                print(f"  {grp:10} " + "  ".join(f"{c[0]:>3}/{c[1]:<3}" for c in cells))
        return tab, fp, misses

    tab, fp, misses = score(show=True)
    if args.misses:
        print("\ncombined misses: group | text | want svc, intent | got svc, intent | matcher best service / intent")
        for grp, text, ws, wi, ps, pi, rs, ri in misses:
            print(f"  {grp:9} {text!r}\n      want {ws}, {wi} | got {ps}, {pi} | {rs} / {ri}")
    if args.sweep:
        keep_s, keep_i = dict(m.service_head.thr), dict(m.intent_head.thr)
        out = []
        for ss in (0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91):
            for sm in (0.0, 0.005, 0.01, 0.015, 0.02, 0.03):
                t, f, _ = score({"score": ss, "margin": sm}, None)
                out.append(("service", ss, sm, t["combined"]["svc"][0], f["combined"]))
        m.service_head.thr.update(keep_s)
        for isc in (0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93):
            for im in (0.0, 0.005, 0.01, 0.015, 0.02, 0.03):
                t, f, _ = score(None, {"score": isc, "margin": im})
                out.append(("intent", isc, im, t["combined"]["int"][0], f["combined"]))
        m.intent_head.thr.update(keep_i)
        print("\nsweep (combined; the other head at its current thresholds): best 8 per head by correct rows, then fewest FP")
        for head in ("service", "intent"):
            for h, sc, mg, good, f in sorted((o for o in out if o[0] == head), key=lambda o: (-o[3], o[4]))[:8]:
                print(f"  {h:8} score>={sc:.3f} margin>={mg:.3f} -> {good} correct, chit-chat FP {f}")
    if args.latency:
        texts = [r[1] for r in rows]
        lat = []
        for t in texts * 3:
            a = time.perf_counter()
            m.classify(t)
            lat.append((time.perf_counter() - a) * 1000)
        lat.sort()
        print(f"\nlatency per message: median {statistics.median(lat):.1f} ms, p95 {lat[int(len(lat)*0.95)]:.1f} ms, "
              f"max {lat[-1]:.1f} ms ({len(lat)} calls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
