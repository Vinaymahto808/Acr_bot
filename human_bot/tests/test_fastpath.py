import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.fastpath import BloomFilter, FastPath, Row, normalise  # noqa: E402

ROWS = [
    Row("u1", ["Kya pickup aur drop free hai?", "Is pickup and drop free?"], "Haan ji, pickup-drop complimentary hai.",
        "Yes, pickup and drop is complimentary."),
    Row("u2", ["Warranty kitni milti hai?", "How much warranty do you give?"], "Parts par 6-12 mahine, labour par 3 mahine.",
        "Parts 6-12 months, labour 3 months."),
    Row("u3", ["Workshop kitne baje khulta hai? Sunday ko khula hai?", "What are your workshop timings?"],
        "Hum 7 din khule hain, subah 9 se shaam 6 baje tak.", "We are open 7 days, 9 am to 6 pm."),
    Row("u4", ["Insurance claim cashless hota hai?", "Do you do cashless insurance claims?"], "Haan, cashless claim hota hai.", ""),
    Row("u5", ["Agent ne kya bataya?"], "Agent ne call mein nahi bataya.", ""),        # a note, not an answer
]


class BloomTests(unittest.TestCase):
    def test_no_false_negatives_and_about_the_false_positive_rate(self):
        b = BloomFilter(2000, fp=0.01)
        words = [f"word{i}" for i in range(2000)]
        for w in words:
            b.add(w)
        self.assertTrue(all(w in b for w in words))
        fps = sum(f"other{i}" in b for i in range(20000)) / 20000
        self.assertLess(fps, 0.03)


class MatchTests(unittest.TestCase):
    def setUp(self):
        self.fp = FastPath(ROWS)

    def test_notes_are_not_answers(self):
        self.assertEqual(len(self.fp.rows), 4)

    def test_paraphrases_and_typos_find_the_answer(self):
        for q, url in (("pikup drop free hai kya?", "u1"), ("is the pick up free", "u1"),
                       ("warrenty kitne mahine ki hai", "u2"), ("sunday ko workshop khula hai?", "u3"),
                       ("cashless claim hota hai kya", "u4")):
            m = self.fp.match(q)
            self.assertIsNotNone(m, q)
            self.assertEqual(m.row.url, url, q)

    def test_language_of_the_answer(self):
        m = self.fp.match("Is pickup and drop free?")
        self.assertEqual(m.text("english"), "Yes, pickup and drop is complimentary.")
        self.assertTrue(m.text("hinglish").startswith("Haan ji"))

    def test_turns_that_must_go_to_the_model(self):
        for t in ("Rahul", "9876500000", "haan ji", "mera naam Rahul hai", "What is the capital of France?",
                  "kal subah 10 baje", ""):
            self.assertIsNone(self.fp.match(t), t)

    def test_hindi_script_is_understood_for_the_offline_fallback(self):
        # 2026-11-25: the server still sends Hindi-script turns to the model (Devanagari answer); the match is only
        # the offline brain's reply when every provider fails, instead of "samajh nahi paayi"
        m = self.fp.match("क्या पिकअप फ्री है?")
        self.assertIsNotNone(m)
        self.assertTrue(m.text("hinglish").startswith("Haan ji"))
        self.assertIsNone(self.fp.match("मेरी गाड़ी का ब्रेक पैडल नीचे तक जा रहा है"))   # unknown words: no guess

    def test_fast(self):
        t0 = time.perf_counter()
        for _ in range(1000):
            self.fp.match("pickup drop free hai kya")
        self.assertLess((time.perf_counter() - t0) / 1000 * 1000, 5.0)   # ms per lookup, far under 5 ms

    def test_mixed_script_from_speech_to_text(self):
        m = self.fp.match("अच्छा pickup drop free hai kya?")     # STT writes the Hindi words in Devanagari
        self.assertIsNotNone(m)
        self.assertEqual(m.row.url, "u1")

    def test_normalise(self):
        self.assertEqual(normalise("Gadi ka Pick up FREE hai?"), ["gaadi", "pickup", "free"])


if __name__ == "__main__":
    unittest.main()
