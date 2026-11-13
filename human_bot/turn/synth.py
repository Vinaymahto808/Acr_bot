"""Synthetic customer utterances for the turn-taking model (distillation from the Groq LLM).

The prompt holds no customer data: it only describes the businesses and asks for varied things a caller
says, many of them long and multi-clause (those give the model its mid-thought examples). Output:
human_bot/data/turn/synth.jsonl, one {"text": ...} per line (gitignored with the rest of data/).

    python human_bot/turn/synth.py --rounds 25
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "human_bot" / "data" / "turn" / "synth.jsonl"

TOPICS = [
    "car AC not cooling", "brake noise", "car not starting / battery", "periodic service price", "insurance claim after accident",
    "dent and paint", "pickup and drop", "workshop location and timings", "clutch problem", "engine warning light",
    "car service cost for my model", "warranty on the repair", "spare parts genuine or local", "payment and estimate",
    "tyre change and wheel alignment", "roadside breakdown help", "giving name and phone number", "booking a slot tomorrow",
    "complaint about earlier service", "asking to speak to a human", "suspension noise on speed breakers", "car wash and detailing",
]
STYLES = [
    "short and casual (2-6 words)", "one normal sentence", "long and rambling with 2-3 clauses joined by aur / lekin / kyunki / toh",
    "hesitant with fillers like matlab, woh, haan, um", "mostly English with some Hindi words", "mostly Hindi in Roman script",
    "a bit angry", "an elderly polite caller", "giving details: car model, city area, problem",
]


def load_env() -> None:
    env = ROOT / "voice_clone" / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$", line)
        if m and m.group(1) not in os.environ:
            os.environ[m.group(1)] = m.group(2).strip().strip('"').strip("'")


def ask(prompt: str, model: str) -> str:
    body = json.dumps({"model": model, "temperature": 1.0, "max_tokens": 4000, "reasoning_effort": "low",
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions", data=body, headers={
        "Authorization": "Bearer " + os.environ["GROQ_API_KEY"], "Content-Type": "application/json",
        "User-Agent": "turn-synth/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--model", default=os.getenv("TURN_SYNTH_MODEL", "openai/gpt-oss-120b"))
    a = ap.parse_args()
    load_env()
    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY missing", file=sys.stderr)
        return 2
    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if OUT.exists():
        seen = {json.loads(l)["text"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()}
    rng = random.Random(len(seen))
    added = 0
    for i in range(a.rounds):
        topics = rng.sample(TOPICS, 4)
        styles = rng.sample(STYLES, 4)
        prompt = (
            "You write what Indian customers SAY on a phone/video call to a car repair workshop in Delhi NCR "
            "(Auto Car Repair). Write 40 different customer turns, one per line, no numbering, "
            "no quotes, no speaker labels. Hinglish in Roman script, the way speech-to-text writes it (lowercase is fine, "
            "light punctuation, some spelling mistakes). Each line is ONE complete customer turn.\n"
            f"Topics: {', '.join(topics)}.\nMix these styles: {', '.join(styles)}.\n"
            "At least 15 lines must be long (12-30 words, several clauses)."
        )
        try:
            text = ask(prompt, a.model)
        except Exception as ex:  # noqa: BLE001
            print(f"round {i}: {str(ex)[:120]}", file=sys.stderr)
            time.sleep(8)
            continue
        with OUT.open("a", encoding="utf-8") as f:
            for line in text.splitlines():
                t = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip().strip('"')
                if 1 <= len(t.split()) <= 40 and t not in seen and not t.lower().startswith(("here are", "sure")):
                    seen.add(t)
                    f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
                    added += 1
        print(f"round {i + 1}/{a.rounds}: total {len(seen)}", flush=True)
        time.sleep(2)
    print(f"added {added}, file {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
