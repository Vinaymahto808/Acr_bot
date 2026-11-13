"""2026-11-05: "haa" was not a yes, and the shared reply cache replayed one customer's read-back to the next."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from crm.service_request import is_no, is_yes  # noqa: E402


class YesNo(unittest.TestCase):
    def test_spoken_yes_forms(self):
        for t in ["haa", "Haa.", "haaa", "haan", "haan ji", "ha ji", "hanji", "haanji", "ji", "ji haan", "हाँ", "हां",
                  "जी", "ok", "theek hai", "bhej do", "haa bhej do", "kar do", "chalega", "yes please"]:
            self.assertTrue(is_yes(t), t)

    def test_not_yes(self):
        for t in ["nahi", "ji nahi", "haa par number galat hai", "nahi, kal nahi", "hai kya", "hello", "jitna ho sake",
                  "नहीं", "hamare paas"]:
            self.assertFalse(is_yes(t), t)
        self.assertTrue(is_no("ji nahi"))


class Cache(unittest.TestCase):
    def setUp(self):
        import app  # noqa: PLC0415
        self.ok = app.chat_cacheable

    def test_short_or_personal_turns_are_never_cached(self):
        self.assertFalse(self.ok("haa"))
        self.assertFalse(self.ok("ok ji"))
        self.assertFalse(self.ok("mera number 9811223344 hai"))
        self.assertFalse(self.ok("AC service ka price kya hai", in_request=True))
        self.assertFalse(self.ok("AC service ka price kya hai", "Vinay ji, number 9 1 4 2 0 3 aapka hai?"))
        self.assertFalse(self.ok("aap kahan se ho batao", "Vinay ji, hum Moti Nagar mein hain.", names=("Vinay",)))

    def test_context_free_question_is_cached(self):
        self.assertTrue(self.ok("AC service ka price kya hai", "AC service gaadi ke model par depend karti hai. Kaunsi car hai?"))


if __name__ == "__main__":
    unittest.main()
