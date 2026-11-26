"""The meaning-based matcher (llm/intents.py) and its data (persona/acr_intents.json).

The data checks and the threshold logic run everywhere (numpy only). The model checks need onnxruntime + tokenizers
and the model files on disk (kb/data/models or human_bot/data/models); without them they are skipped:
    uv run --no-project --python 3.13 --with numpy --with onnxruntime --with tokenizers python -m unittest tests.test_intents
"""
import json
import sys
import unittest
from pathlib import Path

BOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOT))

from llm import intents  # noqa: E402

try:
    import numpy as np  # noqa: E402
except Exception:  # noqa: BLE001
    np = None

PERSONA = json.loads((BOT / "persona" / "persona_auto_car_repair.json").read_text(encoding="utf-8"))
HELDOUT = json.loads((BOT / "tests" / "data" / "intent_heldout.json").read_text(encoding="utf-8"))


class Data(unittest.TestCase):
    def setUp(self):
        self.d = intents.load_data()

    def test_service_labels_are_service_pages(self):
        paths = {p["path"] for p in PERSONA["pages"]["list"] if p.get("kind") == "service"}
        self.assertTrue(set(self.d["services"]) <= paths)
        self.assertIn("/insurance-claim", self.d["services"])
        # every specific service page has a label; the generic ones are the generic_service_wanted intent
        self.assertEqual(paths - set(self.d["services"]), {"/car-service"})

    def test_enough_examples_and_catalog_words_merged(self):
        for path, phrases in self.d["services"].items():
            self.assertGreaterEqual(len(phrases), 30, path)
        battery = [p for p in PERSONA["pages"]["list"] if p["path"] == "/category/car-battery"][0]
        self.assertTrue(set(battery["words"]) <= set(self.d["services"]["/category/car-battery"]))
        for label in ("book", "not_interested", "cancel", "yes", "no", "price_question", "time_question", "human_handoff",
                      "out_of_scope", "complaint", "goodbye", "hold", "generic_service_wanted", "other"):
            self.assertGreaterEqual(len(self.d["intents"].get(label, [])), 25, label)

    def test_no_phrase_under_two_labels(self):
        owner = {}
        groups = {**{"svc:" + k: v for k, v in self.d["services"].items()}, **{"int:" + k: v for k, v in self.d["intents"].items()}}
        clash = []
        for label, phrases in groups.items():
            for p in phrases:
                n = intents.normalize(p)
                if n in owner and owner[n] != label:
                    clash.append((p, owner[n], label))
                owner[n] = label
        self.assertEqual(clash, [])

    def test_heldout_is_not_in_the_examples(self):
        seen = {intents.normalize(p) for v in self.d["services"].values() for p in v}
        seen |= {intents.normalize(p) for v in self.d["intents"].values() for p in v}
        seen |= {intents.normalize(p) for p in self.d["service_none"]}
        rows = [r for k, v in HELDOUT.items() if not k.startswith("_") for r in v]
        self.assertGreaterEqual(len(rows), 150)
        self.assertEqual([r[0] for r in rows if intents.normalize(r[0]) in seen], [])

    def test_server_intents_are_labels(self):
        self.assertTrue(intents.SERVER_INTENTS <= set(self.d["intents"]))
        self.assertNotIn("yes", intents.SERVER_INTENTS)     # left to the exact YES_RE / NO_RE rules

    def test_normalize_fixes_typos(self):
        self.assertEqual(intents.normalize("Sarvis krni h"), "service karni hai")


@unittest.skipIf(np is None, "numpy not installed")
class Thresholds(unittest.TestCase):
    def head(self, thr):
        vecs = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return intents.Head(["a", "b", "none"], np.array([0, 0, 1, 2]), vecs, thr)

    def test_best_label_score_and_margin(self):
        h = self.head({"score": 0.5, "margin": 0.1, "top_k": 1})
        m = h.best(np.array([1, 0, 0], dtype=np.float32))
        self.assertEqual(m.label, "a")
        self.assertAlmostEqual(m.score, 1.0, places=4)
        self.assertGreater(m.margin, 0.9)
        self.assertEqual(h.decide(m).label, "a")

    def test_low_score_or_margin_or_none_label_is_none(self):
        h = self.head({"score": 0.99, "margin": 0.0, "top_k": 1})
        q = np.array([0.7, 0.7, 0], dtype=np.float32) / np.sqrt(0.98)
        self.assertIsNone(h.decide(h.best(q)))                       # score below 0.99
        h = self.head({"score": 0.1, "margin": 0.2, "top_k": 1})
        self.assertIsNone(h.decide(h.best(q)))                       # a vs b too close
        self.assertIsNone(h.decide(h.best(np.array([0, 0, 1], dtype=np.float32))))   # "none" wins

    def test_per_label_threshold(self):
        h = self.head({"score": 0.5, "margin": 0.0, "top_k": 1, "labels": {"a": {"score": 1.01}}})
        self.assertIsNone(h.decide(h.best(np.array([1, 0, 0], dtype=np.float32))))
        self.assertEqual(h.decide(h.best(np.array([0, 1, 0], dtype=np.float32))).label, "b")

    def test_unavailable_matcher_returns_none(self):
        saved = intents._STATE["m"]
        intents._STATE["m"] = None
        try:
            self.assertIsNone(intents.classify("AC thanda nahi kar raha"))
        finally:
            intents._STATE["m"] = saved


MODEL = intents.available() and intents.model_dir(download=False) is not None


@unittest.skipUnless(MODEL, "onnxruntime / tokenizers or the model files not available")
class Model(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = intents.load(block=True)

    def svc(self, text):
        r = self.m.classify(text)
        return r.service.label if r.service else None

    def intent(self, text):
        r = self.m.classify(text)
        return r.intent.label if r.intent else None

    def test_qa_failures(self):
        self.assertEqual(self.svc("road pe rukh gya h"), "/category/car-emergency-services")
        self.assertEqual(self.intent("mujhe nayi car khareedni hai, konsi loon?"), "out_of_scope")
        self.assertEqual(self.intent("tum log loot te ho, bahut mehenga hai"), "complaint")
        self.assertEqual(self.intent("sarvis krni h gadi ki"), "generic_service_wanted")
        self.assertEqual(self.intent("इसका कितना खर्चा है?"), "price_question")

    def test_symptoms_find_the_service(self):
        self.assertEqual(self.svc("ac ki hawa garam aa rahi h"), "/category/car-ac-service-repair")
        self.assertEqual(self.svc("subah se battery bilkul dead hai"), "/category/car-battery")
        self.assertEqual(self.svc("seat pe chai gir gayi daag hai, saaf karna hai"), "/car-interior-cleaning")

    def test_chitchat_and_details_name_no_service(self):
        for t in ("hello ji", "Swift DL3CAB1234", "9811223344", "Priya Sharma", "aapka centre kahan hai"):
            self.assertIsNone(self.svc(t), t)

    def test_latency(self):
        import time
        self.m.classify("brake se awaaz aa rahi hai")
        t0 = time.perf_counter()
        for _ in range(10):
            self.m.classify("gaadi beech raste mein band ho gayi")
        self.assertLess((time.perf_counter() - t0) / 10 * 1000, 150)   # ~10-20 ms on the dev laptop; generous for CI


if __name__ == "__main__":
    unittest.main()
