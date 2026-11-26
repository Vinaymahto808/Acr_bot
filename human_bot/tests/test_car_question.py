"""Step 2 asks only for the car details the customer has not given yet (2026-11-25: "meri creta ke brake..." was
still asked for the model)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))


class CarQuestion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app  # noqa: PLC0415
        cls.app = app

    def q(self, slots, en=False):
        return self.app.step_question("car", en, slots)

    def test_nothing_known_asks_model_and_registration(self):
        self.assertEqual(self.q({}), "Aapki car ka model aur registration number bata dijiye?")
        self.assertEqual(self.q(None, en=True), "Could you tell me your car's model and registration number?")

    def test_model_known_asks_only_registration(self):
        self.assertEqual(self.q({"car": "Creta"}), "Aapki Creta ka registration number bata dijiye?")
        self.assertEqual(self.q({"car": "Creta"}, en=True), "What's the registration number of your Creta?")

    def test_both_known_offers_the_form(self):
        q = self.q({"car": "Swift", "registration": "DL3CAB1234"})
        self.assertIn("Swift", q)
        self.assertIn("form khol doon", q)
        self.assertNotIn("model", q)

    def test_offer_step_and_insurance_wording(self):
        self.assertIn("booking schedule kar doon", self.app.step_question("offer", False))
        self.assertIn("insurance claim book", self.app.offer_question({"path": "/insurance-claim"}, False))

    def test_interest(self):
        for t in ("haan book kar do", "theek hai", "book it please", "karwa do"):
            self.assertTrue(self.app.shows_interest(t, False), t)
        for t in ("kitna time lagega?", "pickup free hai?"):
            self.assertFalse(self.app.shows_interest(t, True), t)

    def test_unknown_price_never_a_number(self):
        line = self.app.price_line({"path": "/category/no-such-page"}, False)
        self.assertNotRegex(line, r"\d")
        self.assertEqual(self.app.price_line({"path": "/insurance-claim"}, False), "")

    def test_model_questions_dropped_in_flow(self):
        out = "".join(self.app.drop_detail_asks(iter(["Bilkul hota hai. Leak test bhi karte hain.Kya aap booking karna chahenge?"])))
        self.assertEqual(out, "Bilkul hota hai. Leak test bhi karte hain. ")
        self.assertEqual("".join(self.app.drop_detail_asks(iter(["Lagbhag 3.5 ghante lagte hain."]))), "Lagbhag 3.5 ghante lagte hain.")

    def test_no_is_not_interest(self):            # 2026-11-25 QA: "nahi chahiye" started a booking
        for t in ("nahi chahiye", "not interested", "abhi nahi", "cancel karo"):
            self.assertFalse(self.app.shows_interest(t, False), t)
        self.assertTrue(self.app.shows_interest("haan kar do na", False))

    def test_name_correction(self):
        self.assertEqual(self.app.name_correction("naam galat hai, Sunil Kumar hai"), "Sunil Kumar")
        self.assertEqual(self.app.name_correction("my name is Rahul, not Rohit"), "Rahul")
        self.assertEqual(self.app.name_correction("haan"), "")

    def test_goodbye_and_english_about(self):
        self.assertTrue(self.app.BYE_RE.search("thank you bye"))
        self.assertFalse(self.app.BYE_RE.search("thank you, pickup milega?"))
        self.assertEqual(self.app.english_about("clutch assembly, clutch bearing aur flywheel ka kaam"),
                         "clutch assembly, clutch bearing and flywheel work")

    def test_complaint_about_our_work(self):        # 2026-11-25 loop: opened the Denting page cheerfully
        for t in ("tumhari service bahut kharab thi last time, gaadi me scratch aa gaye", "your service is the worst",
                  "mujhe complaint karni hai"):
            self.assertTrue(self.app.SERVICE_COMPLAINT_RE.search(t), t)
        for t in ("meri gaadi me scratch aa gaye", "aapki service kaisi hai?", "aapke yahan paint accha hota hai?"):
            self.assertFalse(self.app.SERVICE_COMPLAINT_RE.search(t), t)
        self.assertTrue(self.app.ACCIDENT_RE.search("mera accident ho gaya"))

    def test_lost_space_between_sentences(self):
        from persona.humanizer import humanize  # noqa: PLC0415
        self.assertIn("karegi. Har", humanize("Team confirm karegi.Har kaam ke saath card milta hai."))
        self.assertIn("3.5", humanize("Lagbhag 3.5 ghante lagte hain."))

    def test_other_steps_unchanged(self):
        self.assertEqual(self.app.step_question("phone", False, {"car": "Swift"}), self.app.STEP_Q["phone"][0])
        self.assertEqual(self.app.step_question(None, False), "")


if __name__ == "__main__":
    unittest.main()
