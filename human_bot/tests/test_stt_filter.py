"""STT clean-up (llm/client.py): prompt echoes and background segments never become a customer turn."""
import unittest

import llm.client as c


def seg(text, nsp=0.05, lp=-0.2, cr=1.2):
    return {"text": text, "no_speech_prob": nsp, "avg_logprob": lp, "compression_ratio": cr}


class SttFilter(unittest.TestCase):
    def test_prompt_instruction_is_noise(self):
        for t in ("Hinglish, Roman.", "Hinglish, Roman script.", "Roman"):   # 2026-11-06 screenshot
            self.assertTrue(c._STT_NOISE.match(t), t)
        self.assertNotIn("Hinglish", c.STT_PROMPT)

    def test_prompt_echo_dropped_only_when_unsure(self):
        unsure = {"segments": [seg("brake se awaaz aa rahi hai", nsp=0.4, lp=-0.7)]}
        sure = {"segments": [seg("brake se awaaz aa rahi hai")]}
        self.assertEqual(c._keep_foreground(unsure), "")
        self.assertEqual(c._keep_foreground(sure), "brake se awaaz aa rahi hai")

    def test_background_segment_dropped(self):
        d = {"segments": [seg("Meri Creta hai."), seg("aur phir woh bola", nsp=0.8, lp=-0.9), seg("xx xx xx", lp=-1.4)]}
        self.assertEqual(c._keep_foreground(d), "Meri Creta hai.")

    def test_plain_json_still_works(self):
        self.assertEqual(c._keep_foreground({"text": " DL 3C 1234 "}), "DL 3C 1234")


if __name__ == "__main__":
    unittest.main()
