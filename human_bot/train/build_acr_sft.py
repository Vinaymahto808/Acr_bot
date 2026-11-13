"""Build the Auto Car Repair fine-tuning set for Neha: behaviour, not facts.

    python human_bot/train/build_acr_sft.py                 # generate (resumable) -> training/acr_sft/
    python human_bot/train/build_acr_sft.py --limit 40      # a quick partial run
    python human_bot/train/build_acr_sft.py --emit-only     # rebuild train/val/eval from the cache, no LLM calls

What the model learns
---------------------
Facts (prices, branches, policies) stay in the knowledge base: they change, and the strict-KB rule says Neha
answers only from retrieved passages. So every example is in the PRODUCTION format: system prompt + the
knowledge block the kb really returns for that question (live retrieval from KB_URL, default :4100) + the
service-request block when a booking is being collected + the conversation. The model learns to *use* that
block: answer from it, say "confirm karke bataungi" when it is not there, collect a booking one step at a time,
read it back, hand over to the team.

Distillation: the TEACHER (Groq qwen/qwen3.8-27b, the production model) answers under the FULL 1,540-word
production prompt; the STUDENT example stores the 104-word compact prompt (persona/prompting.build_compact_prompt).
A fine-tuned model then follows the long rule set with a prompt 15x shorter (faster first token, cheaper).
Serve it with HUMAN_BOT_PROMPT=compact so production sends the same compact prompt.

Example types (from kb/seed/acr_customer_qa.clean.csv = 286 real customer-call questions, and scripted flows):
  qa        a caller's question, real retrieval, teacher answer                   (1 per Q&A row)
  qa_alt    the same row asked another way (alias / English / typed Hinglish)       (rows with a variant)
  unknown   the question with its own row removed from the hits: must not guess    (~1 in 4 rows)
  followup  the question asked after an earlier exchange (context carry)            (~1 in 3 rows)
  booking   scripted service-request conversations, every Neha turn                 (--bookings, default 36)
  guard     human hand-off, out-of-scope, competitor, abuse                          (fixed lines from the persona)

Every teacher reply is checked before it is kept (see `check`): word limits, exactly one question or a next
step, no markdown / rupee sign / assistant-speak, female verb forms, and NO NUMBER that is not in the knowledge
block or the conversation (the anti-hallucination gate). A failed reply is retried once with the reason.

Output (gitignored: built from customer calls, the repo is public):
  training/acr_sft/cache.json      teacher replies, keyed by example id (resume after 429s)
  training/acr_sft/train.jsonl     {"messages": [...]} chat format, last message = target
  training/acr_sft/val.jsonl       10 % of Q&A rows (split by row: no question leaks into val)
  training/acr_sft/eval.jsonl      the val rows' first-turn prompts + gold call note, for eval_acr.py
  training/acr_sft/dataset_card.md counts, rejects, how to train
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HUMAN_BOT_LLM_TIMEOUT_S", "60")       # batch work: never the bot's 3 s turn budget
os.environ.setdefault("KB_TIMEOUT_S", "10")
ROOT = Path(__file__).resolve().parents[2]
HB = ROOT / "human_bot"
sys.path.insert(0, str(HB))

from crm.service_request import asked_confirmation, collect as sr_collect, prompt_block as sr_prompt_block, wants_request  # noqa: E402
from llm import client as llm  # noqa: E402
from llm.knowledge import knowledge_block, retrieve  # noqa: E402
from persona.prompting import build_compact_prompt, build_system_prompt  # noqa: E402

CSV = ROOT / "kb" / "seed" / "acr_customer_qa.clean.csv"
REWRITES = ROOT / "kb" / "seed" / "qa_rewrites.json"
PERSONA = HB / "persona" / "persona_auto_car_repair.json"
OUT = ROOT / "training" / "acr_sft"
EMIT_ONLY = False
TEACHER_MODELS = [m.strip() for m in os.getenv("ACR_TEACHER_MODELS", "qwen/qwen3.8-27b,openai/gpt-oss-20b,openai/gpt-oss-120b").split(",") if m.strip()]
GREETING = "Namaste! Main Neha, Auto Car Repair se. Aapki car mein kya problem hai?"
# Appended to the TEACHER's prompt only (the student never sees it): the first runs copied the prompt's sample
# lines verbatim and asked a question instead of answering from the passages.
TEACHER_NOTE = (
    "## Coach notes for this reply (write the best possible line)\n"
    "- Answer the customer's actual question FIRST with one concrete fact from the verified passages (which branches, "
    "what is free, how the process works), then ask one thing. If the passages do not answer it, say you will confirm.\n"
    "- Never reuse the sample lines from this prompt; say it in your own words, naturally, like a real Delhi advisor.\n"
    "- Ask for one item at a time. When reading a booking back, include every known detail and ask \"bhej doon?\".\n"
    "- Reply with Neha's spoken line only."
)

# ---------------------------------------------------------------- checks
MARKDOWN = re.compile(r"(\*\*|^#+\s|^\s*[-*•]\s|\|.*\|)", re.M)
BANNED = re.compile(r"\b(as an ai|i cannot|certainly[!,]|great question|i'?d be happy to|knowledge base|website ke hisaab se)\b", re.I)
MALE = re.compile(r"\b(karta hoon|deta hoon|leta hoon|sakta hoon|bataunga|karunga|samajh gaya|dekh leta)\b", re.I)
NUM = re.compile(r"\d+(?:[.,:]\d+)*")
STAGE = re.compile(r"[\[\(](smiles|nods|laughs|pause)[\]\)]|\*[a-z ]+\*", re.I)


def _digits(s: str) -> set[str]:
    return {n.replace(",", "") for n in NUM.findall(s or "")}


def check(reply: str, context: str, detail: bool = False, booking_readback: bool = False) -> str | None:
    r = (reply or "").strip()
    if not r:
        return "empty"
    words = len(r.split())
    if words > (70 if detail else 34) + (14 if booking_readback else 0):
        return f"too long ({words} words; max {'70' if detail else '30'})"
    if r.count("?") > 1:
        return "more than one question"
    if MARKDOWN.search(r) or "₹" in r or STAGE.search(r):
        return "markdown, rupee sign or stage direction"
    last_user = context.rsplit("\n", 1)[-1]
    head = re.sub(r"\W", "", last_user.lower())[:40]
    if head and re.sub(r"\W", "", r.lower()).startswith(head):
        return "repeats the customer's words"
    if re.search(r"\bRs\.?\s?\d|\bINR\b|/-", r):
        return "written money (say prices aloud)"
    if re.search(r"\b(book ho gay[ai]|booking (ho gayi|confirm(ed)? hai)|slot book|confirm ho gay[ai])\b", r, re.I):
        return "claims a booking is confirmed (only the system confirms)"
    if re.search(r"\b\d+\s*[-‑–]?\s*\d*\s*(minute|min|ghante|hour)s?\b", r, re.I) and not re.search(r"\b\d+\s*[-‑–]?\s*\d*\s*(minute|min|ghante|hour)", context, re.I):
        return "time promise not in the knowledge"
    if BANNED.search(r):
        return "assistant-speak or mentions the knowledge base"
    if MALE.search(r):
        return "male verb form (Neha is female)"
    # a phone number is read back digit by digit ("9 8 1 1 ..."): compare digit strings without spaces
    ctx_digits = _digits(context) | {re.sub(r"\D", "", context)}
    joined = re.sub(r"(?<=\d) (?=\d)", "", r)
    invented = [n for n in _digits(joined) if n not in ctx_digits and not any(n in c for c in ctx_digits)]
    if invented:
        return f"number not in the knowledge or conversation: {invented}"
    return None


# ---------------------------------------------------------------- teacher
_NEXT_OK: dict[str, float] = {}          # model -> earliest time it may be called again
TPM = 8000                               # Groq free tier: tokens per minute, per model (x-ratelimit-limit-tokens)


def _groq(model: str, messages: list[dict]) -> tuple[str, int]:
    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415
    payload = {"model": model, "messages": messages, "temperature": 0.6, "max_tokens": 700 if "gpt-oss" in model else 200}
    if "gpt-oss" in model:
        payload["reasoning_effort"] = "low"
    req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions", json.dumps(payload).encode(),
                                 {"Authorization": f"Bearer {llm.GROQ_KEY}", "Content-Type": "application/json", "User-Agent": "neha-sft"})
    with urllib.request.urlopen(req, timeout=90) as r:
        d = json.load(r)
    out = (d["choices"][0]["message"].get("content") or "").strip()
    return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip(), int(d.get("usage", {}).get("total_tokens") or 4000)


def teacher(messages: list[dict]) -> str:
    """One teacher reply. Rotates the Groq models (each has its own 8k tokens/minute) and paces each one by the
    tokens it just used, so the free tier is never hit head-on. Gives up after 20 minutes with nothing served."""
    import urllib.error  # noqa: PLC0415
    deadline = time.time() + 1200
    while time.time() < deadline:
        ready = sorted(TEACHER_MODELS, key=lambda m: _NEXT_OK.get(m, 0))
        model = ready[0]
        wait = _NEXT_OK.get(model, 0) - time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            out, used = _groq(model, messages)
            _NEXT_OK[model] = time.time() + 60.0 * used / TPM + 1
            if out:
                return out
        except urllib.error.HTTPError as ex:
            body = ex.read().decode("utf-8", "ignore")[:200]
            if ex.code == 429:
                retry = float(ex.headers.get("retry-after") or 30)
                _NEXT_OK[model] = time.time() + max(retry, 10)
                if "per day" in body.lower():
                    _NEXT_OK[model] = time.time() + 6 * 3600
                    print(f"    {model}: daily quota used up", flush=True)
                continue
            if ex.code in (500, 502, 503):
                _NEXT_OK[model] = time.time() + 20
                continue
            raise RuntimeError(f"{model}: HTTP {ex.code} {body}") from ex
        except (TimeoutError, OSError):
            _NEXT_OK[model] = time.time() + 20
    raise RuntimeError("teacher unavailable for 20 minutes")


# ---------------------------------------------------------------- data
def load_rows() -> list[dict]:
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    rw = json.loads(REWRITES.read_text(encoding="utf-8")) if REWRITES.exists() else {}
    for r in rows:
        x = rw.get(r["id"]) or {}
        r["voice"] = x.get("text") if x.get("text") and not x.get("error") else ""
    return rows


TYPO = [("hai", "h"), ("kya", "kia"), ("service", "servic"), ("kitna", "kitna"), ("gaadi", "gadi"), ("nahi", "nhi"), ("mein", "me")]


def variant(r: dict, rng: random.Random) -> str | None:
    aliases = [a.strip() for a in (r.get("aliases") or "").split("|") if a.strip()]
    if aliases:
        return rng.choice(aliases)
    if r.get("question_en") and rng.random() < 0.5:
        return r["question_en"]
    q = r["question"]
    for a, b in rng.sample(TYPO, 3):                  # typed-Hinglish shape: lower case, short forms
        q = re.sub(rf"\b{a}\b", b, q, flags=re.I)
    q = q.lower().rstrip("?") + ("?" if rng.random() < 0.5 else "")
    return q if q != r["question"].lower() else None


def hits_for(query: str, drop_url: str | None = None) -> list[dict]:
    hits = []
    for attempt in range(3):                          # an empty block from a timeout would teach "I don't know"
        hits = retrieve("auto_car_repair", query, k=5)
        if hits:
            break
        time.sleep(2 * (attempt + 1))
    if drop_url:
        hits = [h for h in hits if (h.get("url") or "") != drop_url]
    return hits[:4]


def qa_url(r: dict) -> str:
    return f"https://autocarrepair.in/faq/customer-calls#{r['id']}"


# ---------------------------------------------------------------- examples
class Builder:
    def __init__(self, persona: dict, cache: dict) -> None:
        self.p, self.cache = persona, cache
        self.lang = persona["languages"]["primary"]
        self.compact = build_compact_prompt(persona, self.lang)
        full = build_system_prompt(persona, self.lang)
        self.samples = {re.sub(r"[^a-z ]", "", q.lower()).strip() for q in re.findall(r'"([^"]{12,})"', full)}
        self.rejects: list[dict] = []
        self.save = lambda: None                      # set by main: persist the cache after every teacher reply

    def unoriginal(self, reply: str) -> str | None:
        """A copied prompt sample or a line already used twice teaches parroting, not judgement."""
        norm = lambda t: re.sub(r"[^a-z ]", "", t.lower()).strip()  # noqa: E731
        r = norm(reply)
        for q in self.samples:
            if len(q.split()) >= 5 and q in r:
                return "copies a sample line from the prompt"
        if sum(1 for v in self.cache.values() if norm(v.get("reply", "")) == r) >= 2:
            return "same line already used twice"
        q = norm(reply.rsplit(".", 1)[-1]) if "?" in reply else ""
        if q and len(q.split()) >= 4 and sum(1 for v in self.cache.values()
                                             if not v.get("reject") and norm(v.get("reply", "")).endswith(q)) >= 4:
            return "same closing question used 4 times already"
        return None

    def system_full(self, extra: str) -> str:
        return build_system_prompt(self.p, self.lang, extra=extra).replace("{SESSION_CONTEXT}", "{}")

    def system_compact(self, extra: str) -> str:
        return self.compact + ("\n\n" + extra if extra else "")

    def example(self, eid: str, kind: str, extra: str, history: list[dict], user: str, detail: bool = False,
                readback: bool = False, fixed: str | None = None) -> dict | None:
        conv = history + [{"role": "user", "content": user}]
        context = extra + "\n" + "\n".join(m["content"] for m in conv)
        if fixed is not None:
            reply = fixed
        elif eid in self.cache:
            reply = self.cache[eid]["reply"]
            if self.cache[eid].get("reject"):
                return None
        elif EMIT_ONLY:
            return None
        else:
            msgs = [{"role": "system", "content": self.system_full(extra) + "\n\n" + TEACHER_NOTE}] + conv
            reply = teacher(msgs)
            why = check(reply, context, detail, readback) or self.unoriginal(reply)
            if why:                                   # one retry with the reason
                msgs2 = msgs + [{"role": "assistant", "content": reply},
                                {"role": "user", "content": f"(Coach, not the customer: that reply broke a rule: {why}. "
                                 "Say it again to the customer, following every rule. Reply with Neha's line only.)"}]
                reply = teacher(msgs2)
                why = check(reply, context, detail, readback) or self.unoriginal(reply)
            self.cache[eid] = {"kind": kind, "reply": reply, "reject": why}
            self.save()
            if why:
                self.rejects.append({"id": eid, "why": why, "reply": reply})
                return None
        return {"id": eid, "kind": kind,
                "messages": [{"role": "system", "content": self.system_compact(extra)}] + conv + [{"role": "assistant", "content": reply}]}

    # one Q&A row -> qa / qa_alt / unknown / followup
    def from_row(self, r: dict, rows: list[dict], rng: random.Random) -> list[dict]:
        out = []
        hist0 = [{"role": "assistant", "content": GREETING}]
        q = r["question"]
        hits = hits_for(q)
        ex = self.example(f"qa:{r['id']}", "qa", knowledge_block(hits, "Auto Car Repair"), hist0, q,
                          detail=bool(re.search(r"\b(detail|explain|samjhao|process|kaise)\b", q, re.I)))
        out.append(ex)
        v = variant(r, rng)
        if v:
            out.append(self.example(f"alt:{r['id']}", "qa_alt", knowledge_block(hits_for(v), "Auto Car Repair"), hist0, v))
        if rng.random() < 0.25:
            out.append(self.example(f"unk:{r['id']}", "unknown", knowledge_block(hits_for(q, drop_url=qa_url(r)), "Auto Car Repair"), hist0, q))
        pool = [x for x in rows if x["category"] != r["category"]]
        if pool and rng.random() < 0.33:
            other = rng.choice(pool)
            prev = self.cache.get(f"qa:{other['id']}", {}).get("reply")
            if prev and not self.cache[f"qa:{other['id']}"].get("reject"):
                hist = hist0 + [{"role": "user", "content": other["question"]}, {"role": "assistant", "content": prev}]
                out.append(self.example(f"fu:{r['id']}", "followup", knowledge_block(hits, "Auto Car Repair"), hist, q))
        return [e for e in out if e]

    # scripted booking conversations: production builds the same knowledge + service-request blocks per turn
    def booking(self, n: int, rng: random.Random) -> list[dict]:
        cars = ["Creta", "Swift", "i20", "Honda City", "Baleno", "Nexon", "XUV700", "Innova", "Verna", "Dzire", "Seltos", "Fortuner"]
        probs = ["AC thanda nahi kar raha", "brake lagate waqt awaaz aati hai", "periodic service karwani hai", "dent paint karwana hai",
                 "clutch hard ho gaya hai", "battery baar baar down ho jati hai", "suspension se khat khat awaaz", "engine light jal rahi hai"]
        areas = ["Gurgaon Sector 34 ke paas", "Moti Nagar", "Noida Sector 63", "Okhla", "Dwarka", "Lajpat Nagar"]
        whens = ["kal subah", "aaj shaam", "Saturday ko", "parso"]
        names = ["Rahul", "Priya", "Amit", "Sunita", "Vikas", "Neeraj", "Pooja", "Arjun", "Kavita", "Rohan"]
        out = []
        for i in range(n):
            car, prob, area, when, name = rng.choice(cars), rng.choice(probs), rng.choice(areas), rng.choice(whens), rng.choice(names)
            phone = str(rng.choice([9, 8, 7])) + "".join(str(rng.randrange(10)) for _ in range(9))   # synthetic number
            first = rng.choice([f"Meri {car} mein {prob}", f"{prob}, {car} hai", f"Hi, {car} ki {prob}"])
            turns = [first, rng.choice([f"{area} mein hoon", f"main {area} se hoon"]),
                     rng.choice([when, f"{when} pickup chahiye", f"{when} ho jayega"]),
                     rng.choice([f"mera naam {name} hai", name]), rng.choice([phone, " ".join(phone[:5]) + " " + " ".join(phone[5:])])]
            if rng.random() < 0.4:                     # a question in the middle of the booking
                turns.insert(2, rng.choice(["pickup free hai kya?", "kitna time lagega?", "genuine parts lagate ho?"]))
            transcript: list[dict] = [{"role": "assistant", "text": GREETING}]
            history: list[dict] = [{"role": "assistant", "content": GREETING}]
            pending = False
            for j, u in enumerate(turns):
                transcript.append({"role": "user", "text": u})
                slots = sr_collect(transcript)
                extra = knowledge_block(hits_for(u), "Auto Car Repair")
                if wants_request(transcript, slots):
                    extra += "\n\n" + sr_prompt_block(slots, pending)
                ex = self.example(f"book:{i}:{j}", "booking", extra, list(history), u, readback=(j == len(turns) - 1))
                if not ex:
                    break
                reply = ex["messages"][-1]["content"]
                out.append(ex)
                pending = asked_confirmation(reply)
                transcript.append({"role": "assistant", "text": reply})
                history += [{"role": "user", "content": u}, {"role": "assistant", "content": reply}]
        return out

    def guards(self) -> list[dict]:
        g = self.p.get("guardrails") or {}
        hand, oos = g.get("handoff_line"), g.get("out_of_scope_line")
        hist0 = [{"role": "assistant", "content": GREETING}]
        items = [
            ("g:human1", "mujhe kisi insaan se baat karni hai", hand), ("g:human2", "manager se baat karao", hand),
            ("g:human3", "I want to talk to a real person", hand), ("g:complaint", "mujhe complaint karni hai, gaadi kharab wapas mili", hand),
            ("g:oos1", "aaj ka cricket score kya hai?", oos), ("g:oos2", "mere liye ek poem likh do", oos),
            ("g:oos3", "flight ticket book kar sakti ho?", oos),
        ]
        out = [self.example(eid, "guard", knowledge_block([], "Auto Car Repair"), hist0, u, fixed=line) for eid, u, line in items if line]
        for eid, u in [("g:comp1", "GoMechanic se sasta hai kya aapka?"), ("g:comp2", "dealer pe jaun ya aapke paas?"),
                       ("g:rude", "tum log chor ho, paise loot te ho"), ("g:bot", "kya tum robot ho?")]:
            out.append(self.example(eid, "guard", knowledge_block(hits_for(u), "Auto Car Repair"), hist0, u))
        return [e for e in out if e]


def split_key(eid: str) -> str:
    return eid.split(":", 1)[1].split(":")[0] if ":" in eid else eid


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N Q&A rows")
    ap.add_argument("--bookings", type=int, default=36)
    ap.add_argument("--emit-only", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv[1:])
    OUT.mkdir(parents=True, exist_ok=True)
    cache_f = OUT / "cache.json"
    cache = json.loads(cache_f.read_text(encoding="utf-8")) if cache_f.exists() else {}
    persona = json.loads(PERSONA.read_text(encoding="utf-8"))
    b = Builder(persona, cache)
    rows = load_rows()
    if a.limit:
        rows = rows[: a.limit]
    rng = random.Random(a.seed)
    global EMIT_ONLY                                  # noqa: PLW0603
    EMIT_ONLY = a.emit_only

    def save() -> None:
        cache_f.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    b.save = save
    examples: list[dict] = []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        try:
            examples += b.from_row(r, rows, rng)
        except RuntimeError as ex:
            print(f"  stop at row {i}: {ex}")
            break
        if i % 10 == 0:
            save()
            print(f"  rows {i}/{len(rows)}  examples {len(examples)}  rejects {len(b.rejects)}  {time.time() - t0:.0f}s", flush=True)
    save()
    try:
        examples += b.booking(a.bookings, rng)
        examples += b.guards()
    except RuntimeError as ex:
        print(f"  bookings/guards stopped: {ex}")
    save()

    # split by Q&A row so no question (or its variants) appears in both train and val
    keys = sorted({split_key(e["id"]) for e in examples if e["kind"] in ("qa", "qa_alt", "unknown", "followup")})
    val_keys = {k for k in keys if int(hashlib.sha1(k.encode()).hexdigest(), 16) % 10 == 0}
    train = [e for e in examples if split_key(e["id"]) not in val_keys and not (e["kind"] == "booking" and e["id"].split(":")[1] in ("0", "1", "2"))]
    val = [e for e in examples if e not in train]
    rng.shuffle(train)
    for name, rows_ in (("train", train), ("val", val)):
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for e in rows_:
                f.write(json.dumps({"messages": e["messages"], "kind": e["kind"], "id": e["id"]}, ensure_ascii=False) + "\n")
    by_id = {r["id"]: r for r in load_rows()}
    with (OUT / "eval.jsonl").open("w", encoding="utf-8") as f:
        for e in val:
            if e["kind"] in ("qa", "unknown"):
                r = by_id.get(split_key(e["id"]), {})
                f.write(json.dumps({"id": e["id"], "kind": e["kind"], "messages": e["messages"][:-1],
                                    "teacher": e["messages"][-1]["content"], "gold_note": r.get("answer", ""),
                                    "full_extra": e["messages"][0]["content"][len(b.compact):].strip()}, ensure_ascii=False) + "\n")
    from collections import Counter
    kinds = Counter(e["kind"] for e in examples)
    rej = [v for v in cache.values() if v.get("reject")]
    card = [
        "# Auto Car Repair SFT set (Neha)", "",
        f"- built: {time.strftime('%Y-%m-%d %H:%M')} from {len(rows)} customer-call Q&A rows + {a.bookings} scripted bookings",
        f"- teacher: Groq {', '.join(TEACHER_MODELS)} under the full production prompt; student prompt: compact (104 words)",
        f"- examples kept: **{len(examples)}** (train {len(train)} / val {len(val)}), by kind: {dict(kinds)}",
        f"- rejected by the rule gate (kept out): {len(rej)}; top reasons: {Counter(v['reject'].split(':')[0].split(' (')[0] for v in rej).most_common(5)}",
        "", "Train: fine_tune_neha.ipynb (Colab T4) on train.jsonl / val.jsonl. Evaluate: human_bot/train/eval_acr.py.",
        "Serve with HUMAN_BOT_PROMPT=compact so production sends the prompt the model was trained on.",
        "Data comes from real customer calls: keep it out of the public repo (training/acr_sft/ is gitignored).",
    ]
    (OUT / "dataset_card.md").write_text("\n".join(card) + "\n", encoding="utf-8")
    print("\n".join(card))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
