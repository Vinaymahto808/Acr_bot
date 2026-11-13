"""Lip-sync aligner (llm/visemes.py): phonemes per script, timing, closures, coarticulation."""
import unittest

from llm import visemes


def shapes(word: str) -> str:
    return "".join(v for v, _ in visemes.phones(word))


class Phones(unittest.TestCase):
    def test_devanagari_schwa_rules(self):
        self.assertEqual(shapes("आपकी"), "CACD")        # aap-ki: medial schwa dropped
        self.assertEqual(shapes("करके"), "CCBCB")       # kar-ke
        self.assertEqual(shapes("नमस्ते"), "BCACBBB")   # na-mas-te keeps the first schwa, virama on s

    def test_lips_and_teeth(self):
        self.assertIn("A", shapes("pickup"))           # p closes the lips
        self.assertTrue(shapes("फ़्री").startswith("G"))  # nukta f = teeth on lip, not a pucker
        self.assertTrue(shapes("free").startswith("G"))

    def test_number_and_empty(self):
        self.assertTrue(shapes("10"))
        self.assertEqual(visemes.cues([]), [])


class Cues(unittest.TestCase):
    def setUp(self):
        self.words = [(0.10, 0.64, "नमस्ते"), (0.75, 0.40, "जी"), (1.38, 0.50, "आपकी"), (1.89, 0.42, "pickup")]
        self.c = visemes.cues(self.words)

    def test_timeline_is_ordered_and_covers_words(self):
        for a, b in zip(self.c, self.c[1:]):
            self.assertLessEqual(a["s"], a["e"])
            self.assertLessEqual(a["e"], b["s"] + 1e-6)
        self.assertEqual(self.c[-1]["v"], "X")
        self.assertIn("X", [c["v"] for c in self.c[:-1]])   # the 0.23 s pause before आपकी rests the mouth

    def test_no_invisible_shapes_and_closures_visible(self):
        for c in self.c[:-1]:
            d = c["e"] - c["s"]
            if c["v"] in ("A", "G"):
                self.assertGreaterEqual(d, visemes._MIN_CLOSURE - 1e-3)
        self.assertFalse(any(a["v"] == b["v"] for a, b in zip(self.c, self.c[1:])))   # repeats merged

    def test_mouth_map_uses_round_and_fv(self):
        self.assertEqual(visemes.MOUTH_MAP["E"], "O.png")
        self.assertEqual(visemes.MOUTH_MAP["G"], "FV.png")


if __name__ == "__main__":
    unittest.main()
