import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crm.service_request import asked_confirmation  # noqa: E402
from llm.language import detect, reply_rule, sentence_is_english  # noqa: E402
from persona.prompting import build_compact_prompt, build_system_prompt, load_persona  # noqa: E402


class DetectTests(unittest.TestCase):
    def test_english(self):
        for t in ("I want to book a service so can you help me?",   # 2026-11-14 screenshot: got a Hinglish answer
                  "Is pickup free?", "My car AC is not cooling", "What is the price of AC service",
                  "Where are you located? I am in Noida", "Do you have a workshop in Gurgaon",
                  "is it possible to get it done today"):
            self.assertEqual(detect(t), "english", t)

    def test_hinglish(self):
        for t in ("meri gaadi ki service karwani hai", "pickup free hai kya", "kal subah ka slot milega",
                  "accha theek hai book kar do", "Meri kreta ka AC thanda nahi kar raha hai",
                  "I want service, kya aap help karoge", "अच्छा pickup free hai kya?"):
            self.assertEqual(detect(t), "hinglish", t)

    def test_hindi(self):
        for t in ("मेरी गाड़ी की सर्विस करनी है", "कल सुबह का slot मिलेगा क्या?", "पिकअप फ्री है क्या"):
            self.assertEqual(detect(t), "hindi", t)

    def test_too_short_keeps_the_current_language(self):
        for t in ("Rahul", "ok", "9876500000", "Creta", "yes please", "haan ji", ""):
            self.assertIsNone(detect(t), t)

    def test_voice_of_one_sentence(self):
        self.assertTrue(sentence_is_english("Sure, pickup is free in Gurgaon."))
        self.assertFalse(sentence_is_english("Haan ji, pickup free hai."))
        self.assertFalse(sentence_is_english("जी, पिकअप फ्री है।"))


class PromptTests(unittest.TestCase):
    def test_reply_rule_is_last_and_follows_the_language(self):
        p = load_persona(Path(__file__).resolve().parents[1] / "persona" / "persona_auto_car_repair.json")
        en = build_system_prompt(p, "english")
        self.assertTrue(en.rstrip().endswith(reply_rule("english").splitlines()[-1]))
        self.assertIn("ENGLISH", en)
        self.assertIn("Devanagari", build_system_prompt(p, "hindi"))
        self.assertIn("ENGLISH", build_compact_prompt(p, "english"))

    def test_english_read_back_counts_as_asking_for_confirmation(self):
        self.assertTrue(asked_confirmation("Rahul, Creta, AC repair from Gurgaon. Shall I send this request?"))
        self.assertTrue(asked_confirmation("Main yeh request bhej doon?"))


if __name__ == "__main__":
    unittest.main()
