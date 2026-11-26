"""Avatar face table (owner spec 2026-11-25): emotion_of tags each of Neha's sentences, situation_of reads the
customer's turn. greeting -> warm, booking confirmed -> enthusiastic (smile), accident -> concerned,
angry customer -> empathetic, price -> neutral, service / insurance -> warm (friendly).
python -m unittest discover -s human_bot/tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.client import EMOTIONS  # noqa: E402
from persona.humanizer import emotion_of, situation_of  # noqa: E402


class EmotionOfTests(unittest.TestCase):
    def test_greeting_is_warm(self):
        self.assertEqual(emotion_of("Namaste! Main Neha, Auto Car Repair se."), "warm")

    def test_booking_confirmation_smiles(self):
        for line in ("Booking ho gayi hai, kal subah 10 baje.", "Aapki booking confirm ho gayi hai.",
                     "Done! Aapki request file ho gayi hai.", "Your booking is confirmed for tomorrow."):
            self.assertEqual(emotion_of(line), "enthusiastic", line)

    def test_a_failed_booking_does_not_smile(self):
        self.assertNotEqual(emotion_of("Booking confirm nahi ho payi, ek baar phir try karte hain."), "enthusiastic")

    def test_accident_is_concerned(self):
        self.assertEqual(emotion_of("Oh, accident ho gaya? Aap theek hain na?"), "concerned")

    def test_apology_and_understanding_are_empathetic(self):
        self.assertEqual(emotion_of("Maaf kijiye, aapko pareshani hui."), "empathetic")
        self.assertEqual(emotion_of("Main samajh sakti hoon ji, aapki chinta bilkul jayaz hai."), "empathetic")
        self.assertEqual(emotion_of("I understand, I'm sorry about this."), "empathetic")

    def test_concern_wins_over_apology(self):
        self.assertEqual(emotion_of("Sorry, delay ho gaya."), "concerned")

    def test_price_is_neutral_and_service_is_friendly(self):
        self.assertEqual(emotion_of("Service ka kharcha 3000 se shuru hota hai."), "neutral")
        self.assertEqual(emotion_of("Insurance claim mein hum madad karte hain."), "warm")

    def test_existing_tags_still_work(self):
        self.assertEqual(emotion_of("Koi dikkat nahi, ho jayega."), "warm")
        self.assertEqual(emotion_of("Chinta mat kijiye, team dekh legi."), "warm")
        self.assertEqual(emotion_of("Ek second, main check karti hoon."), "thinking")
        self.assertEqual(emotion_of("Aapki gaadi kaunsi hai?"), "thinking")
        self.assertEqual(emotion_of("Gaadi se awaaz aa rahi hai to brake check karna hoga."), "concerned")
        self.assertEqual(emotion_of("Bilkul, perfect."), "enthusiastic")
        self.assertEqual(emotion_of("Theek hai."), "warm")
        self.assertEqual(emotion_of("Workshop Sector 72A mein hai."), "neutral")

    def test_every_tag_has_a_voice_prosody(self):
        # the card asks /tts with the sentence's tag; an unknown tag fell back to neutral and missed the prefetch
        for line in ("Maaf kijiye.", "Booking ho gayi.", "Accident ho gaya?", "Namaste!", "Theek hai.", "Kharcha 500."):
            self.assertIn(emotion_of(line), EMOTIONS, line)


class SituationOfTests(unittest.TestCase):
    def test_accident_and_breakdown_are_concerned(self):
        self.assertEqual(situation_of("meri gaadi ka accident ho gaya"), "concerned")
        self.assertEqual(situation_of("gaadi beech raste band pad gayi"), "concerned")

    def test_angry_customer_is_empathetic(self):
        self.assertEqual(situation_of("tum log bekaar ho, gaadi kharab kar di"), "empathetic")
        self.assertEqual(situation_of("worst service ever, totally useless"), "empathetic")
        self.assertEqual(situation_of("accident ke baad bhi bakwas service"), "empathetic")

    def test_a_plain_enquiry_has_no_override(self):
        for u in ("meri creta ke brake se awaaz aa rahi hai", "service kitne ki hai", "pickup free hai?", "", None):
            self.assertIsNone(situation_of(u), u)


if __name__ == "__main__":
    unittest.main()
