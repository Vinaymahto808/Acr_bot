"""Meaning-based service / intent matcher: nearest neighbours over example phrases, on a small local model.

The word lists in llm/pages.py and the regexes in server/app.py only catch the phrasings someone wrote down.
"road pe rukh gya h" (the car stopped on the road) got "didn't catch" until the words were added by hand. This
module compares the customer's message with example phrases by meaning instead, so a new phrasing of a known
request still lands on the right service or intent. It is an extra signal only: the server uses it when its
rules found nothing, never over an explicit rule hit (server/app.py, "intent matcher").

    from llm import intents
    m = intents.get()                       # None until the model + examples are loaded (background, at start-up)
    r = m.classify("AC thanda nahi kar raha") if m else None
    r.service  -> Match("/category/car-ac-service-repair", score=0.93, margin=0.05) or None
    r.intent   -> Match("price_question", ...) or None
    r.ms       -> embedding + search time in milliseconds

Data: persona/acr_intents.json (the owner adds phrases there; the file is re-read when it changes). The service
labels are the persona's service page paths; the catalog's own "words" seed every service too.

Model: intfloat/multilingual-e5-small (MIT licence), the int8 ONNX export Xenova/multilingual-e5-small, run with
onnxruntime (MIT) + tokenizers (Apache-2.0) on the CPU, no torch. The knowledge service (kb/) already keeps the
same files under kb/data/models; they are reused when present, else downloaded once (~135 MB) into
human_bot/data/models. Example vectors are cached in human_bot/data/intent_cache.npz (only new phrases are
embedded at start-up).

Env: HUMAN_BOT_INTENTS=0 switches it off; HUMAN_BOT_INTENT_MODEL_DIR points at another copy of the model files;
HUMAN_BOT_INTENT_DOWNLOAD=0 never downloads. Without onnxruntime / tokenizers / numpy installed the matcher is
simply unavailable (get() returns None) and the rules work alone.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOT = HERE.parent
DATA_FILE = BOT / "persona" / "acr_intents.json"
PERSONA_FILE = BOT / "persona" / "persona_auto_car_repair.json"
CACHE_FILE = BOT / "data" / "intent_cache.npz"
MODEL_NAME = "Xenova/multilingual-e5-small"
MODEL_FILES = ("onnx/model_quantized.onnx", "tokenizer.json")
MODEL_DIRS = [BOT / "data" / "models" / "multilingual-e5-small",
              BOT.parent / "kb" / "data" / "models" / "Xenova" / "multilingual-e5-small"]
HF_URL = "https://huggingface.co/" + MODEL_NAME + "/resolve/main/"
PREFIX = "query: "            # e5 convention for short symmetric texts
MAX_TOKENS = 64

# Defaults; acr_intents.json "thresholds" overrides them. Chosen on the held-out set (tests/data/intent_heldout.json,
# tools/eval_intents.py): a result below either bound is None and the rules decide alone.
DEFAULT_THRESHOLDS = {"service": {"score": 0.88, "margin": 0.005, "top_k": 2},
                      "intent": {"score": 0.90, "margin": 0.005, "top_k": 2}}
NONE_LABELS = {"none", "other"}
# the intents server/app.py acts on when its own regexes missed. "yes" / "no" are left to the rules (YES_RE / NO_RE
# are exact; a greeting read as "yes" would start a booking).
SERVER_INTENTS = {"book", "not_interested", "cancel", "price_question", "time_question", "human_handoff", "out_of_scope",
                  "complaint", "goodbye", "hold", "generic_service_wanted"}


def enabled() -> bool:
    return os.getenv("HUMAN_BOT_INTENTS", "1").strip().lower() not in ("0", "false", "no", "off")


def available() -> bool:
    """The runtime libraries are installed (the model files may still be missing)."""
    try:
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def model_dir(download: bool | None = None) -> Path | None:
    """The first directory holding the model files; downloads them once when none does (and download is allowed)."""
    env = os.getenv("HUMAN_BOT_INTENT_MODEL_DIR", "").strip()
    dirs = ([Path(env)] if env else []) + MODEL_DIRS
    for d in dirs:
        if all((d / f).is_file() for f in MODEL_FILES):
            return d
    if download is None:
        download = os.getenv("HUMAN_BOT_INTENT_DOWNLOAD", "1").strip().lower() not in ("0", "false", "no", "off")
    if not download:
        return None
    dest = Path(env) if env else MODEL_DIRS[0]
    for f in MODEL_FILES:
        target = dest / f
        if target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(HF_URL + f, timeout=120) as r, open(tmp, "wb") as out:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        tmp.replace(target)
    return dest


def normalize(text: str) -> str:
    """What gets embedded: the server's typo fixes (llm/pages.with_roman, without the Devanagari word gloss that it
    appends), lower case, single spaces. Devanagari stays as typed: the model reads it natively."""
    from llm.pages import TYPOS
    t = (text or "").strip()
    for rx, to in TYPOS:
        t = re.sub(rx, to, t, flags=re.I)
    return re.sub(r"\s+", " ", t.lower()).strip(" .!,")


class Encoder:
    """multilingual-e5-small on onnxruntime: texts -> L2-normalised 384-d vectors (mean pooling)."""

    def __init__(self, directory: Path, threads: int | None = None, prefix: str = PREFIX):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.np = np
        self.dir = directory
        self.prefix = prefix
        self.tok = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tok.enable_truncation(MAX_TOKENS)
        self.tok.enable_padding(pad_id=self.tok.token_to_id("<pad>") or 1, pad_token="<pad>")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or int(os.getenv("HUMAN_BOT_INTENT_THREADS", "2"))
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(directory / MODEL_FILES[0]), so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        st = (directory / MODEL_FILES[0]).stat()
        self.key = f"{directory.name}|{st.st_size}|{prefix}"
        self.lock = threading.Lock()

    def encode(self, texts: list[str], batch: int = 64):
        np = self.np
        out = []
        for i in range(0, len(texts), batch):
            encs = self.tok.encode_batch([self.prefix + t for t in texts[i:i + batch]])
            ids = np.array([e.ids for e in encs], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            with self.lock:
                hidden = self.sess.run(None, feed)[0]
            m = mask[..., None].astype(np.float32)
            v = (hidden * m).sum(1) / np.maximum(m.sum(1), 1e-9)
            out.append(v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9))
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 384), np.float32)


@dataclass
class Match:
    label: str
    score: float
    margin: float
    runner_up: str = ""

    def __repr__(self) -> str:
        return f"Match({self.label!r}, score={self.score:.3f}, margin={self.margin:.3f}, vs={self.runner_up!r})"


@dataclass
class Result:
    service: Match | None
    intent: Match | None
    ms: float
    raw_service: Match | None = None    # best label before the thresholds (diagnostics / tuning)
    raw_intent: Match | None = None


@dataclass
class Head:
    labels: list[str]
    idx: object               # np.ndarray, label index per example row
    vecs: object              # np.ndarray (n, d)
    thr: dict = field(default_factory=dict)

    def best(self, q) -> Match | None:
        np = __import__("numpy")
        if not len(self.labels):
            return None
        sims = self.vecs @ q
        k = int(self.thr.get("top_k", 2))
        scores = np.full(len(self.labels), -1.0, dtype=np.float32)
        for li in range(len(self.labels)):
            s = sims[self.idx == li]
            if s.size:
                top = np.sort(s)[-min(k, s.size):]
                scores[li] = float(top.mean())
        order = np.argsort(scores)[::-1]
        a = int(order[0])
        b = int(order[1]) if len(order) > 1 else a
        return Match(self.labels[a], float(scores[a]), float(scores[a] - scores[b]) if b != a else 1.0, self.labels[b])

    def decide(self, m: Match | None) -> Match | None:
        if not m or m.label in NONE_LABELS:
            return None
        own = (self.thr.get("labels") or {}).get(m.label) or {}     # a stricter bound for one label ("goodbye")
        if m.score < float(own.get("score", self.thr.get("score", 1.0))) or m.margin < float(own.get("margin", self.thr.get("margin", 0.0))):
            return None
        return m


def load_data(path: Path = DATA_FILE, persona_path: Path = PERSONA_FILE) -> dict:
    """{"services": {path: [phrases]}, "intents": {label: [phrases]}, "service_none": [...], "thresholds": {...}}
    with the persona catalog's service words merged in."""
    data = json.loads(path.read_text(encoding="utf-8"))
    services = {k: list(v) for k, v in (data.get("services") or {}).items() if not k.startswith("_")}
    try:
        persona = json.loads(persona_path.read_text(encoding="utf-8"))
        for pg in (persona.get("pages") or {}).get("list") or []:
            if pg.get("kind") == "service" and pg["path"] in services:
                services[pg["path"]] = list(pg.get("words") or []) + services[pg["path"]]
    except (OSError, ValueError):
        pass
    intents = {k: list(v) for k, v in (data.get("intents") or {}).items() if not k.startswith("_")}
    thr = {h: {**DEFAULT_THRESHOLDS[h], **((data.get("thresholds") or {}).get(h) or {})} for h in DEFAULT_THRESHOLDS}
    return {"services": services, "intents": intents, "service_none": list(data.get("service_none") or []),
            "thresholds": thr}


class Matcher:
    def __init__(self, encoder: Encoder, data_path: Path = DATA_FILE, persona_path: Path = PERSONA_FILE,
                 cache_path: Path | None = CACHE_FILE):
        self.enc = encoder
        self.data_path, self.persona_path, self.cache_path = data_path, persona_path, cache_path
        self.mtime = 0.0
        self.checked = 0.0
        self.build()

    # ---- examples -> vectors (cached per phrase) ----
    def _vectors(self, texts: list[str]):
        np = self.enc.np
        cache: dict[str, object] = {}
        if self.cache_path and self.cache_path.is_file():
            try:
                with np.load(self.cache_path, allow_pickle=False) as z:     # closed again: Windows can't replace an open file
                    if str(z["key"]) == self.enc.key:
                        cache = dict(zip([str(t) for t in z["texts"]], np.array(z["vecs"])))
            except Exception:  # noqa: BLE001
                cache = {}
        todo = sorted({t for t in texts if t not in cache})
        if todo:
            for t, v in zip(todo, self.enc.encode(todo)):
                cache[t] = v
            if self.cache_path:
                try:
                    self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                    keys = sorted(cache)
                    tmp = self.cache_path.with_name(self.cache_path.stem + ".tmp.npz")
                    np.savez(tmp, key=np.array(self.enc.key), texts=np.array(keys), vecs=np.stack([cache[k] for k in keys]))
                    tmp.replace(self.cache_path)
                except OSError:
                    pass
        return np.stack([cache[t] for t in texts]) if texts else np.zeros((0, 384), np.float32)

    def _head(self, groups: dict[str, list[str]], thr: dict) -> Head:
        np = self.enc.np
        labels, rows, idx = [], [], []
        for label, phrases in groups.items():
            ph = list(dict.fromkeys(normalize(p) for p in phrases if str(p).strip()))
            if not ph:
                continue
            labels.append(label)
            rows += ph
            idx += [len(labels) - 1] * len(ph)
        return Head(labels, np.array(idx, dtype=np.int32), self._vectors(rows), thr)

    def build(self) -> None:
        t0 = time.perf_counter()
        d = load_data(self.data_path, self.persona_path)
        intents = d["intents"]
        # service head: every specific service page + "none" = phrases that name no service (chit-chat, details,
        # and every intent example: "kitna lagega", "haan", "bye", "car ki servicing karani hai")
        none = d["service_none"] + [p for k, v in intents.items() for p in v]
        self.service_head = self._head({**d["services"], "none": none}, d["thresholds"]["service"])
        self.intent_head = self._head(intents, d["thresholds"]["intent"])
        self.services = list(d["services"])
        self.intent_labels = list(intents)
        self.build_ms = (time.perf_counter() - t0) * 1000
        try:
            self.mtime = self.data_path.stat().st_mtime + self.persona_path.stat().st_mtime
        except OSError:
            pass

    def maybe_reload(self) -> None:
        """The owner edited acr_intents.json (or the persona's page words): rebuild, checked at most every 5 s."""
        now = time.time()
        if now - self.checked < 5:
            return
        self.checked = now
        try:
            mt = self.data_path.stat().st_mtime + self.persona_path.stat().st_mtime
        except OSError:
            return
        if mt != self.mtime:
            try:
                self.build()
            except Exception:  # noqa: BLE001  (a broken edit keeps the last good examples)
                self.mtime = mt

    def classify(self, text: str) -> Result:
        t0 = time.perf_counter()
        self.maybe_reload()
        norm = normalize(text)
        if not norm:
            return Result(None, None, 0.0)
        q = self.enc.encode([norm])[0]
        rs, ri = self.service_head.best(q), self.intent_head.best(q)
        return Result(self.service_head.decide(rs), self.intent_head.decide(ri), (time.perf_counter() - t0) * 1000, rs, ri)


# ---- process-wide instance, loaded in the background ----
_STATE: dict = {"m": None, "loading": False, "error": "", "ms": 0.0}
_LOCK = threading.Lock()


def load(block: bool = True) -> Matcher | None:
    """Load the model + examples (once). block=False starts it in a thread and returns at once."""
    if _STATE["m"] is not None or not enabled():
        return _STATE["m"]
    with _LOCK:
        if _STATE["loading"]:
            return None
        _STATE["loading"] = True

    def run():
        t0 = time.perf_counter()
        try:
            if not available():
                raise RuntimeError("onnxruntime / tokenizers / numpy not installed")
            d = model_dir()
            if not d:
                raise RuntimeError("model files not found (HUMAN_BOT_INTENT_DOWNLOAD=0)")
            m = Matcher(Encoder(d))
            m.classify("namaste")           # warm-up: first run allocates the graph
            _STATE["m"] = m
            _STATE["ms"] = (time.perf_counter() - t0) * 1000
        except Exception as e:  # noqa: BLE001
            _STATE["error"] = str(e)[:200]
        finally:
            _STATE["loading"] = False

    if block:
        run()
    else:
        threading.Thread(target=run, name="intents-load", daemon=True).start()
    return _STATE["m"]


def get() -> Matcher | None:
    return _STATE["m"]


def status() -> dict:
    m = _STATE["m"]
    return {"enabled": enabled(), "ready": m is not None, "loading": _STATE["loading"], "error": _STATE["error"] or None,
            "model": MODEL_NAME, "load_ms": round(_STATE["ms"]), "examples": (len(m.service_head.idx) + len(m.intent_head.idx)) if m else 0}


def classify(text: str) -> Result | None:
    """The matcher's reading of one message, or None while it is unavailable (never raises)."""
    m = _STATE["m"]
    if m is None:
        return None
    try:
        return m.classify(text)
    except Exception:  # noqa: BLE001
        return None
