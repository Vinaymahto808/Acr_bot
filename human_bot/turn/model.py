"""End-of-turn model: P(the customer's thought is finished | transcript so far).

A small Transformer encoder (Vaswani et al., "Attention Is All You Need", arXiv 1706.03762) over raw UTF-8
bytes, no tokenizer (the byte-level input of H-Net, arXiv 2507.07955), so Roman Hinglish, Devanagari and
STT spelling mistakes all go in as they are. It drives Moshi-style full-duplex turn-taking
(arXiv 2410.00037): at a pause the early transcript is scored here; a finished thought ends the customer's
turn sooner, a mid-thought pause keeps listening and may get a soft "hmm" (avatar_widget/src/duplex.js).

Trained by human_bot/turn/train.py (PyTorch, CPU) on the project's own call transcripts + distilled
synthetic turns; this file runs it with numpy only (~1 ms), so the server needs no torch.

    from turn.model import end_of_turn
    end_of_turn("meri creta ka ac thanda nahi kar raha aur")   # -> 0.07
"""
from __future__ import annotations

import re
import threading
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).with_name("turn_eot.npz")
MAX_BYTES = 160
BOS, PAD = 256, 257
_TRAIL = re.compile(r"[\s.!?।,;:…\-–—\"')]+$")


def normalize(text: str) -> str:
    """Lowercase, one space, no trailing punctuation: STT adds a full stop to half sentences too, so the
    model must judge the words, not Whisper's punctuation. Shared with train.py."""
    t = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    return _TRAIL.sub("", t)


def encode(text: str) -> np.ndarray:
    """Right-aligned byte ids: [PAD … PAD, BOS, b1 … bn], length MAX_BYTES + 1 (the tail is kept)."""
    b = list(normalize(text).encode("utf-8"))[-MAX_BYTES:]
    ids = [PAD] * (MAX_BYTES - len(b)) + [BOS] + b
    return np.asarray(ids, dtype=np.int64)


def _ln(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def _gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


class EndOfTurn:
    def __init__(self, path: Path = WEIGHTS):
        z = np.load(path)
        self.w = {k: z[k].astype(np.float32) for k in z.files if not k.startswith("meta_")}
        self.heads = int(z["meta_heads"]) if "meta_heads" in z.files else 4
        # operating point chosen on validation turns by train.py ("finished" when p >= threshold)
        self.threshold = float(z["meta_threshold"]) if "meta_threshold" in z.files else 0.5
        self.layers = sum(1 for k in self.w if k.endswith(".ln1.g"))

    def logit(self, text: str) -> float:
        w, H = self.w, self.heads
        ids = encode(text)
        # Padding is masked out of every attention row and out of the pooling, so dropping those positions
        # gives the same numbers as the padded batch the model was trained on, at a fraction of the cost.
        start = int(np.argmax(ids != PAD))
        x = w["emb"][ids[start:]] + w["pos"][start: len(ids)]
        T, D = x.shape
        dh = D // H
        for i in range(self.layers):
            p = f"l{i}."
            h = _ln(x, w[p + "ln1.g"], w[p + "ln1.b"])
            qkv = h @ w[p + "qkv.w"].T + w[p + "qkv.b"]
            q, k, v = (qkv[:, j * D:(j + 1) * D].reshape(T, H, dh).transpose(1, 0, 2) for j in range(3))
            att = q @ k.transpose(0, 2, 1) / np.sqrt(dh)
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            o = (att @ v).transpose(1, 0, 2).reshape(T, D)
            x = x + o @ w[p + "proj.w"].T + w[p + "proj.b"]
            h = _ln(x, w[p + "ln2.g"], w[p + "ln2.b"])
            x = x + _gelu(h @ w[p + "fc1.w"].T + w[p + "fc1.b"]) @ w[p + "fc2.w"].T + w[p + "fc2.b"]
        x = _ln(x, w["lnf.g"], w["lnf.b"])
        pooled = np.concatenate([x[-1], x.mean(0)])
        return float(pooled @ w["head.w"][0] + w["head.b"][0])

    def __call__(self, text: str) -> float:
        if not normalize(text):
            return 0.0
        return float(1 / (1 + np.exp(-self.logit(text))))


_MODEL: EndOfTurn | None = None
_LOCK = threading.Lock()


def threshold() -> float:
    """The trained operating point; 0.5 before the weights are loaded."""
    return _MODEL.threshold if _MODEL is not None else 0.5


def end_of_turn(text: str) -> float | None:
    """P(finished thought) in [0, 1], or None when the weights are missing (callers keep their rule)."""
    global _MODEL
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                if not WEIGHTS.exists():
                    return None
                _MODEL = EndOfTurn()
    return _MODEL(text)
