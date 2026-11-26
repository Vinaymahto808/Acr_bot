"""Avatar v2 viseme timeline (llm/visemes_v2.py): 15 visemes, Hindi + English, closures, coarticulation, speed."""
import time
import unittest

from llm import visemes_v2 as v


def vis(word: str) -> list[str]:
    return [x[0] for x in v.word_visemes(word)]


class WordVisemes(unittest.TestCase):
    def test_every_viseme_is_known(self):
        for w in ("नमस्ते", "आपकी", "गाड़ी", "सर्विस", "फ़्री", "बंपर", "hello", "booking", "thank", "city", "quick", "10"):
            for x in vis(w):
                self.assertIn(x, v.VISEMES, w)

    def test_lip_closure_and_teeth_on_lip(self):
        self.assertEqual(vis("मोबाइल")[0], "PP")          # m: lips pressed
        self.assertIn("PP", vis("booking"))
        self.assertEqual(vis("फ़्री")[0], "FF")            # nukta f: lower lip under the teeth
        self.assertEqual(vis("five")[0], "FF")
        self.assertEqual(vis("बंपर")[:3], ["PP", "aa", "PP"])   # anusvara before p is an m (lips close)

    def test_rounded_vowels(self):
        self.assertIn("O", vis("मोबाइल"))
        self.assertIn("U", vis("पूछना"))
        self.assertEqual(vis("you"), ["I", "U"])           # lexicon, not the "ou" spelling

    def test_rounding_anticipation(self):
        # k before u is already rounded (कुछ): no flat kk frame in front of the pucker
        self.assertEqual(vis("कुछ")[0], "U")

    def test_spread_and_tongue(self):
        self.assertIn("I", vis("ठीक"))
        self.assertEqual(vis("thank")[0], "TH")
        self.assertEqual(vis("सर्विस")[0], "SS")
        self.assertIn("nn", vis("नमस्ते"))

    def test_schwa_rules(self):
        self.assertEqual(v.phonemes("आपकी"), ["aa", "p", "k", "ii"])   # medial schwa dropped
        self.assertEqual(v.phonemes("नमस्ते")[:2], ["n", "a"])          # first syllable keeps it


class Timeline(unittest.TestCase):
    WORDS = [(0.10, 0.64, "नमस्ते"), (0.75, 0.40, "जी"), (1.38, 0.50, "आपकी"), (1.89, 0.42, "pickup"),
             (2.40, 0.30, "book"), (2.75, 0.50, "karo")]

    def setUp(self):
        self.t = v.timeline(self.WORDS)

    def test_ordered_contiguous_and_rests(self):
        for a, b in zip(self.t, self.t[1:]):
            self.assertLess(a["s"], a["e"])
            self.assertLessEqual(a["e"], b["s"] + 1e-6)
            self.assertNotEqual(a["v"], b["v"])                       # equal neighbours merged
        self.assertEqual(self.t[-1]["v"], "sil")
        pauses = [c for c in self.t[:-1] if c["v"] == "sil"]
        self.assertTrue(pauses and pauses[0]["s"] <= 1.15 and pauses[0]["e"] >= 1.38)   # the gap before आपकी rests

    def test_closures_visible_others_not_invisible(self):
        for c in self.t[:-1]:
            d = c["e"] - c["s"]
            if c["v"] in ("PP", "FF"):
                self.assertGreaterEqual(d, v._MIN_CLOSURE - 1e-3)
            else:
                self.assertGreaterEqual(d, v._MIN_SHAPE - 1e-3)
        self.assertGreaterEqual(sum(c["v"] == "PP" for c in self.t), 3)   # namaste, aapki, pickup, book

    def test_intensity(self):
        for c in self.t:
            self.assertTrue(0.0 <= c["w"] <= 1.0)
            if c["v"] == "sil":
                self.assertEqual(c["w"], 0.0)
        self.assertEqual(max(c["w"] for c in self.t if c["v"] == "aa"), 1.0)   # long aa of आपकी

    def test_empty_and_unknown(self):
        self.assertEqual(v.timeline([]), [])
        self.assertTrue(v.timeline([(0.0, 0.3, "%%%")]))

    def test_fast_enough_for_the_request_path(self):
        t0 = time.perf_counter()
        for _ in range(50):
            v.timeline(self.WORDS * 5)
        self.assertLess((time.perf_counter() - t0) / 50, 0.02)   # < 20 ms for a 30-word sentence (measured ~2 ms)

    def test_from_v1(self):
        out = v.from_v1([{"s": 0, "e": 0.1, "v": "A"}, {"s": 0.1, "e": 0.3, "v": "C"}, {"s": 0.3, "e": 0.5, "v": "X"}])
        self.assertEqual([c["v"] for c in out], ["PP", "aa", "sil"])


if __name__ == "__main__":
    unittest.main()
