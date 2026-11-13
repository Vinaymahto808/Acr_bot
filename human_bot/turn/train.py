"""Train the end-of-turn Transformer (see model.py) on this project's own data. CPU, a few minutes.

    tools/turn-venv/Scripts/python.exe human_bot/turn/train.py          # needs torch + numpy (training venv only)

Data (all local; nothing leaves the machine):
  real   customer turns from human_bot/data/conversations.sqlite, the customer-call questions in kb (page_type
         qa titles), user turns in training/*.jsonl and training/acr_sft/*.jsonl
  synth  human_bot/data/turn/synth.jsonl (turn/synth.py, distilled from the LLM; weighted lower)
Examples: every turn is a positive ("finished"); 1-3 word-prefixes of it are negatives ("mid-thought"), which is
exactly what an early transcript in a pause looks like. Real turns are split 70/15/15 by turn, so no prefix of a
test turn is ever trained on. The report compares the model with the hand-written rule it replaces
(avatar_widget looksComplete / embed.html INCOMPLETE) on the held-out real turns.
Writes human_bot/turn/turn_eot.npz (weights only, no text) and turn_eot_report.json.
"""
from __future__ import annotations

import gzip
import json
import math
import random
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from model import BOS, MAX_BYTES, PAD, EndOfTurn, encode, normalize  # noqa: E402

D, HEADS, LAYERS, FF = 64, 4, 2, 192
DROPOUT = 0.15
MAX_INTERRUPT = 0.15   # threshold = lowest p whose "ended a mid-thought pause" rate stays under this on val (0.08 -> 0.66 caught only 31% of finished turns)
SEED = 7

# ------------------------------------------------------------------ data
DANGLING = set((
    "aur and or ya to bhi ki ke ka ko mein main me se par pe tak jo jab tab agar lekin magar "
    "kyunki matlab yaani because like the a an of in on at with for is are was were that this "
    "will can could would should so um uh hmm").split())


def rule_complete(text: str) -> bool:
    """The rule the model replaces (avatar_widget/src/audio.js looksComplete, Roman-script part)."""
    w = normalize(text).split()
    return bool(w) and w[-1] not in DANGLING


def real_turns() -> list[str]:
    out: list[str] = []
    db = ROOT / "human_bot" / "data" / "conversations.sqlite"
    if db.exists():
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        for (tr,) in c.execute("select transcript from conversations"):
            for t in json.loads(tr or "[]"):
                if t.get("role") == "user" and str(t.get("text", "")).strip():
                    out.append(t["text"])
    kb = ROOT / "kb" / "data" / "kb.sqlite"
    if kb.exists():
        c = sqlite3.connect(f"file:{kb}?mode=ro", uri=True)
        for (title,) in c.execute("select title from pages where page_type='qa' and status='active'"):
            out += [p for p in re.split(r"\s+/\s+", title or "") if p.strip()]
    for f in list((ROOT / "training").glob("*.jsonl")) + list((ROOT / "training" / "acr_sft").glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                msgs = json.loads(line).get("messages") or []
            except ValueError:
                continue
            out += [m["content"] for m in msgs if m.get("role") == "user" and isinstance(m.get("content"), str)]
    seen, uniq = set(), []
    for t in out:
        n = normalize(t)
        if n and len(n) <= 400 and n not in seen:
            seen.add(n)
            uniq.append(t)
    return uniq


def synth_turns() -> list[str]:
    f = ROOT / "human_bot" / "data" / "turn" / "synth.jsonl"
    if not f.exists():
        return []
    return [json.loads(l)["text"] for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def examples(turns: list[str], rng: random.Random, weight: float, prefixes: int = 3):
    """(text, label, weight): the whole turn = 1; word prefixes = 0, always including a cut after a connector."""
    ex = []
    for t in turns:
        w = normalize(t).split()
        ex.append((" ".join(w), 1.0, weight))
        if len(w) < 2:
            continue
        cuts = set(rng.sample(range(1, len(w)), min(prefixes, len(w) - 1)))
        conn = [k for k in range(1, len(w)) if w[k - 1] in DANGLING]
        if conn:
            cuts.add(rng.choice(conn))
        ex += [(" ".join(w[:k]), 0.0, weight) for k in sorted(cuts)]
    return ex


# ------------------------------------------------------------------ model (mirrors model.py exactly)
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.fc1, self.fc2 = nn.Linear(D, FF), nn.Linear(FF, D)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x, neg):
        B, T, _ = x.shape
        dh = D // HEADS
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        q, k, v = (z.view(B, T, HEADS, dh).transpose(1, 2) for z in (q, k, v))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(dh) + neg[:, None, None, :]
        o = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.proj(o))
        return x + self.drop(self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh")))


class EotNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(258, D)
        self.pos = nn.Parameter(torch.randn(MAX_BYTES + 1, D) * 0.02)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.lnf = nn.LayerNorm(D)
        self.head = nn.Linear(2 * D, 1)

    def forward(self, ids):
        keep = (ids != PAD).float()
        neg = (1 - keep) * -1e9
        x = F.dropout(self.emb(ids) + self.pos[: ids.shape[1]], DROPOUT, self.training)
        for b in self.blocks:
            x = b(x, neg)
        x = self.lnf(x)
        pooled = torch.cat([x[:, -1], (x * keep[..., None]).sum(1) / keep.sum(1, keepdim=True).clamp(min=1)], -1)
        return self.head(pooled).squeeze(-1)


def export(net: EotNet, path: Path, threshold: float = 0.5) -> None:
    s = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    out = {"emb": s["emb.weight"], "pos": s["pos"], "lnf.g": s["lnf.weight"], "lnf.b": s["lnf.bias"],
           "head.w": s["head.weight"], "head.b": s["head.bias"], "meta_heads": np.array(HEADS),
           "meta_threshold": np.array(threshold, dtype=np.float32)}
    for i in range(LAYERS):
        p = f"blocks.{i}."
        for a, b in (("ln1", "ln1"), ("ln2", "ln2")):
            out[f"l{i}.{a}.g"], out[f"l{i}.{a}.b"] = s[p + b + ".weight"], s[p + b + ".bias"]
        for n in ("qkv", "proj", "fc1", "fc2"):
            out[f"l{i}.{n}.w"], out[f"l{i}.{n}.b"] = s[p + n + ".weight"], s[p + n + ".bias"]
    np.savez_compressed(path, **{k: (v.astype(np.float16) if v.dtype == np.float32 and v.size > 64 else v) for k, v in out.items()})


# ------------------------------------------------------------------ metrics
def auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def report(name, y, pred_bool, prob=None):
    y = np.asarray(y)
    pb = np.asarray(pred_bool)
    acc = float((pb == (y == 1)).mean())
    # the two errors that matter in a call: answering a half sentence, and waiting after a finished one
    cut_off = float(pb[y == 0].mean())        # mid-thought judged finished -> Neha talks over the customer
    late = float((~pb[y == 1]).mean())        # finished judged mid-thought -> she waits the full silence
    r = {"accuracy": round(acc, 3), "interrupts_mid_thought": round(cut_off, 3), "waits_after_finished": round(late, 3)}
    if prob is not None:
        r["auc"] = round(auc(y, prob), 3)
    print(f"  {name:<22} " + "  ".join(f"{k}={v}" for k, v in r.items()))
    return r


def main() -> int:
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    real, synth = real_turns(), synth_turns()
    rng.shuffle(real)
    n = len(real)
    tr_r, va_r, te_r = real[: int(n * .7)], real[int(n * .7): int(n * .85)], real[int(n * .85):]
    train = examples(tr_r, rng, 3.0) + examples(synth, rng, 1.0)
    npos = sum(1 for _, y, _ in train if y == 1)
    bal = (len(train) - npos) / max(1, npos)          # ~3.5 prefixes per turn: weigh "finished" up to match
    train = [(t, y, w * (bal if y == 1 else 1.0)) for t, y, w in train]
    val, test = examples(va_r, rng, 1.0), examples(te_r, rng, 1.0)
    print(f"turns: real {n} (train {len(tr_r)} / val {len(va_r)} / test {len(te_r)}), synth {len(synth)}; "
          f"examples: train {len(train)}, val {len(val)}, test {len(test)}")

    def tensors(ex):
        return (torch.tensor(np.stack([encode(t) for t, _, _ in ex])), torch.tensor([y for _, y, _ in ex]),
                torch.tensor([w for _, _, w in ex]))

    Xtr, Ytr, Wtr = tensors(train)
    Xva, Yva, _ = tensors(val)
    net = EotNet()
    print(f"params: {sum(p.numel() for p in net.parameters()):,}")
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0.01)
    epochs, bs = 10, 64
    steps = epochs * math.ceil(len(train) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    best, best_state = 1e9, None
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(train))
        tot = 0.0
        for i in range(0, len(train), bs):
            j = perm[i:i + bs]
            loss = (F.binary_cross_entropy_with_logits(net(Xtr[j]), Ytr[j], reduction="none") * Wtr[j]).sum() / Wtr[j].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(j)
        net.eval()
        with torch.no_grad():
            vl = F.binary_cross_entropy_with_logits(net(Xva), Yva).item()
        print(f"epoch {ep + 1:2d}  train {tot / len(train):.4f}  val {vl:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)

    # operating point from the validation turns: the lowest threshold whose "talks over a mid-thought pause"
    # rate stays under MAX_INTERRUPT (that error is worse than waiting the normal silence)
    with torch.no_grad():
        pv = torch.sigmoid(net(Xva)).numpy()
    yv = Yva.numpy()
    thr = 0.5
    for c in np.linspace(0.05, 0.95, 91):
        if (pv[yv == 0] >= c).mean() <= MAX_INTERRUPT:
            thr = float(c)
            break
    print(f"threshold from val: {thr:.2f}")
    path = HERE / "turn_eot.npz"
    export(net, path, thr)
    # the numpy runtime must agree with torch (float16 export, same maths)
    np_model = EndOfTurn(path)
    with torch.no_grad():
        tp = torch.sigmoid(net(torch.tensor(np.stack([encode(t) for t, _, _ in test[:200]])))).numpy()
    npp = np.array([np_model(t) for t, _, _ in test[:200]])
    drift = float(np.abs(tp - npp).max())
    print(f"numpy vs torch max |dp| on 200 test examples: {drift:.4f}")

    print("held-out REAL customer turns (never trained on):")
    y = [l for _, l, _ in test]
    probs = np.array([np_model(t) for t, _, _ in test])
    res = {"model": report(f"transformer (p>={thr:.2f})", y, probs >= thr, probs),
           "rule": report("rule (looksComplete)", y, np.array([rule_complete(t) for t, _, _ in test]))}
    t1 = time.perf_counter()
    for t, _, _ in test[:100]:
        np_model(t)
    ms = (time.perf_counter() - t1) * 10
    print(f"numpy inference: {ms:.2f} ms per transcript; weights {path.stat().st_size / 1024:.0f} KB")
    (HERE / "turn_eot_report.json").write_text(json.dumps({
        "trained": time.strftime("%Y-%m-%d %H:%M"), "real_turns": n, "synth_turns": len(synth),
        "params": sum(p.numel() for p in net.parameters()), "arch": {"d": D, "heads": HEADS, "layers": LAYERS, "ff": FF, "bytes": MAX_BYTES},
        "threshold": round(thr, 3), "test_examples": len(test), "test": res, "numpy_drift": round(drift, 5), "ms_per_call": round(ms, 2)}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
