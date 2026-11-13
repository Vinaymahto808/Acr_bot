"""Offline service script: a time answer ("10 am tommorow") and an unlisted car ("Aura") are understood.

2026-11-10 screenshot: every LLM failed for one turn (Groq 429, OpenAI 401, Gemini timeout), the offline script
answered "10 am tommorow" with "Maaf kijiye, main theek se samajh nahi paayi. Aapki car ka model aur number
kya hai?" although the customer had already said the car.
"""
import unittest

from llm.client import OfflineServiceBrain


def chat(brain, users):
    msgs, out = [], ""
    for u in users:
        msgs.append({"role": "user", "content": u})
        out = "".join(brain.stream(msgs))
        msgs.append({"role": "assistant", "content": out})
    return out


class OfflineTime(unittest.TestCase):
    def test_when_parser(self):
        w = OfflineServiceBrain._when
        self.assertEqual(w(" 10 am tommorow "), "kal subah 10 baje")
        self.assertEqual(w(" kal shaam 5 baje "), "kal shaam 5 baje")
        self.assertEqual(w(" today 3pm "), "aaj dopahar 3 baje")
        self.assertEqual(w(" tmrw evening "), "kal shaam")
        self.assertEqual(w(" meri aura hai "), "")

    def test_screenshot_flow_never_reasks_car(self):
        b = OfflineServiceBrain({"business_type": "service"})
        b.last_reply = ""
        # earlier turns were answered by the live model; the fallback reads them first
        out = chat(b, ["Aura 7683", "periodic service", "Moti Nagar delhi", "10 am tommorow"])
        self.assertNotIn("samajh nahi", out)
        self.assertNotIn("car ka model", out)
        self.assertEqual(b.slots.get("car"), "Aura")
        self.assertEqual(b.slots.get("when"), "kal subah 10 baje")

    def test_unlisted_car_bare_answer(self):
        b = OfflineServiceBrain({"business_type": "service"})
        b.last_reply = "Aapki car ka model aur number kya hai?"
        b._remember("Zest")
        self.assertEqual(b.slots.get("car"), "Zest")

    def test_longest_model_wins(self):
        b = OfflineServiceBrain({"business_type": "service"})
        b._remember("meri grand i10 hai")
        self.assertEqual(b.slots.get("car"), "Grand I10")


if __name__ == "__main__":
    unittest.main()
