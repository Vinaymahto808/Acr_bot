"""Score a model on held-out Auto Car Repair customer questions (training/acr_sft/eval.jsonl).

    # today's production brain (Groq, full prompt) = the baseline
    python human_bot/train/eval_acr.py --prompt full
    # the fine-tuned model behind any OpenAI-compatible endpoint (vLLM on Colab, RunPod, ...)
    python human_bot/train/eval_acr.py --prompt compact --base https://<host>/v1 --model neha-acr --key dummy

Per reply it checks the same rule gate the training data passed (word limit, one question, no markdown or
rupee sign, female verbs, NO invented number), plus:
  qa       grounded: the reply carries a fact from the call note (a number, branch or key word it contains)
  unknown  the answer was removed from the knowledge: the reply must promise to confirm / hand over, not guess
Prints a table and writes training/acr_sft/eval_<label>.json. Compare two labels side by side with --compare.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("HUMAN_BOT_LLM_TIMEOUT_S", "60")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "human_bot"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_acr_sft import OUT, PERSONA, check, teacher  # noqa: E402
from llm import client as llm  # noqa: E402
from persona.prompting import build_system_prompt  # noqa: E402

CONFIRM = re.compile(r"\b(confirm|check kar|pata kar|team se|call back|callback|bata(ungi| deti)|verify|connect kar)\w*", re.I)
STOP = set("hai hain ka ki ke ko se me mein aur ya bhi to toh jo kya kar karte karti hoon hum aap the yes no and the is are".split())
BRANCHES = re.compile(r"\b(okhla|moti nagar|noida|gurgaon|gurugram|rama road|sector \d+)\b", re.I)


def grounded(reply: str, note: str) -> bool:
    r = reply.lower()
    nums = re.findall(r"\d+", note)
    if nums and any(n in re.sub(r"(?<=\d) (?=\d)", "", r) for n in nums):
        return True
    if any(b.lower() in r for b in BRANCHES.findall(note)):
        return True
    words = [w for w in re.findall(r"[a-z]{5,}", note.lower()) if w not in STOP]
    return sum(1 for w in set(words) if w in r) >= 2


def ask(messages: list[dict], base: str | None, key: str, model: str) -> str:
    if base:
        return "".join(llm.stream_openai(messages, base.rstrip("/"), key, model)).strip()
    return teacher(messages)                         # Groq production model, waits out rate limits


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", choices=["full", "compact"], default="full")
    ap.add_argument("--base", default=None, help="OpenAI-compatible base URL; omit = Groq production model")
    ap.add_argument("--key", default=os.getenv("OPENAI_API_KEY", "dummy"))
    ap.add_argument("--model", default="neha-acr")
    ap.add_argument("--label", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--replies", default=None, help="score precomputed replies (jsonl id,reply) from the Colab notebook")
    ap.add_argument("--compare", nargs=2, metavar=("LABEL_A", "LABEL_B"))
    a = ap.parse_args(argv[1:])
    if a.compare:
        A, B = (json.loads((OUT / f"eval_{x}.json").read_text(encoding="utf-8"))["summary"] for x in a.compare)
        print(f"{'metric':22s} {a.compare[0]:>14s} {a.compare[1]:>14s}")
        for k in A:
            print(f"{k:22s} {A[k]:>14} {B.get(k, '-'):>14}")
        return 0
    label = a.label or (f"{a.prompt}-{a.model}" if a.base else f"{a.prompt}-groq")
    persona = json.loads(PERSONA.read_text(encoding="utf-8"))
    items = [json.loads(line) for line in (OUT / "eval.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if a.limit:
        items = items[: a.limit]
    pre = None
    if a.replies:
        pre = {r["id"]: r["reply"] for r in map(json.loads, Path(a.replies).read_text(encoding="utf-8").splitlines()) if r}
        a.prompt = "compact"
        label = a.label or "finetuned"
    rows = []
    for i, it in enumerate(items, 1):
        msgs = [dict(m) for m in it["messages"]]
        if a.prompt == "full":
            msgs[0]["content"] = build_system_prompt(persona, persona["languages"]["primary"], extra=it["full_extra"]).replace("{SESSION_CONTEXT}", "{}")
        context = msgs[0]["content"] + "\n" + "\n".join(m["content"] for m in msgs[1:])
        t0 = time.perf_counter()
        try:
            if pre is not None:
                if it["id"] not in pre:
                    continue
                reply = pre[it["id"]]
            else:
                reply = ask(msgs, a.base, a.key, a.model)
        except Exception as ex:  # noqa: BLE001
            reply, err = "", str(ex)[:120]
        else:
            err = ""
        ms = int((time.perf_counter() - t0) * 1000)
        rule = check(reply, context)
        ok_task = (CONFIRM.search(reply) is not None) if it["kind"] == "unknown" else grounded(reply, it["gold_note"])
        rows.append({"id": it["id"], "kind": it["kind"], "q": msgs[-1]["content"], "reply": reply, "rule": rule,
                     "task_ok": bool(ok_task), "ms": ms, "error": err})
        print(f"[{i}/{len(items)}] {it['kind']:7s} {'ok ' if not rule else 'BAD'} {'task' if ok_task else '----'} {ms:5d}ms  {reply[:90]}", flush=True)
    n = max(1, len(rows))
    qa = [r for r in rows if r["kind"] == "qa"]
    unk = [r for r in rows if r["kind"] == "unknown"]
    summary = {
        "items": len(rows),
        "rule_pass_%": round(100 * sum(not r["rule"] for r in rows) / n, 1),
        "no_invented_number_%": round(100 * sum(not (r["rule"] or "").startswith("number") for r in rows) / n, 1),
        "grounded_qa_%": round(100 * sum(r["task_ok"] for r in qa) / max(1, len(qa)), 1),
        "unknown_confirms_%": round(100 * sum(r["task_ok"] for r in unk) / max(1, len(unk)), 1),
        "avg_words": round(statistics.mean(len(r["reply"].split()) for r in rows), 1) if rows else 0,
        "p50_ms": int(statistics.median(r["ms"] for r in rows)) if rows else 0,
        "prompt_words": len((build_system_prompt(persona, "hinglish") if a.prompt == "full" else items[0]["messages"][0]["content"]).split()) if items else 0,
    }
    (OUT / f"eval_{label}.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
