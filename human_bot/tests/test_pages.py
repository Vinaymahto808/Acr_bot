"""'Page dikhao' matching (llm/pages.py) against the real ACR persona catalog. No network.
    python human_bot/tests/test_pages.py"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.pages import find_page, page_line, wants_page  # noqa: E402

PERSONA = json.loads((Path(__file__).resolve().parents[1] / "persona" / "persona_auto_car_repair.json").read_text(encoding="utf-8"))


def path(text, hits=None):
    p = find_page(text, PERSONA, hits or [])
    return p and p["path"]


class PageTests(unittest.TestCase):
    def test_asked_pages_open(self):
        self.assertEqual(path("insurance page dikhao"), "/insurance-claim")
        self.assertEqual(path("bima wala page dikha do"), "/insurance-claim")
        self.assertEqual(path("mujhe offers dikha do"), "/offers")
        self.assertEqual(path("open the contact page"), "/contact")
        self.assertEqual(path("aapke centres kahan hain dikhao"), "/service-centers")
        self.assertEqual(path("brake wala page kholo"), "/car-brake-and-wheel-maintenance-in-delhi")
        self.assertEqual(path("home page pe le chalo"), "/")

    def test_ordinary_talk_never_navigates(self):
        for t in ("bonnet khol ke dekha", "brake khul gaya hai", "price dikhao", "AC service kitne ka hai",
                  "insurance claim karwana hai", "mera naam Rahul hai"):
            self.assertIsNone(path(t), t)

    def test_missing_page_falls_back_honestly(self):
        p = find_page("parts ka page dikhao", PERSONA, [])
        self.assertEqual(p["path"], "/services")
        self.assertFalse(p["exact"])
        self.assertIn("alag page website par nahi", page_line(p))

    def test_knowledge_page_of_this_site_only(self):
        hits = [{"url": "https://evil.example/x", "title": "X"},
                {"url": "https://autocarrepair.in/faq/customer-calls#qa-1", "title": "Q", "page_type": "qa"},
                {"url": "https://autocarrepair.in/car-battery-replacement", "title": "Car Battery Replacement | ACR", "page_type": "page"}]
        # no catalog word for "battery": the first page of this site in the knowledge results,
        # skipping the other host and the Q&A anchor
        p = find_page("battery replacement ka page dikhao", PERSONA, hits)
        self.assertEqual((p["path"], p["title"], p["exact"]), ("/car-battery-replacement", "Car Battery Replacement", True))
        # a loosely related result is not the page they asked for (found live 2026-11-13: "parts" -> windshield)
        loose = [{"url": "https://autocarrepair.in/windshield-repair",
                  "title": "Used & Out-of-Warranty Car Repair, Delhi NCR | Auto Car Repair", "page_type": "page"}]
        p = find_page("parts ka page dikhao", PERSONA, loose)
        self.assertEqual((p["path"], p["exact"]), ("/services", False))
        generic = [{"url": "https://autocarrepair.in/car-ac-gas-refill",
                    "title": "Used & Out-of-Warranty Car Repair, Delhi NCR | Auto Car Repair", "page_type": "page"}]
        p = find_page("ac gas refill ka page dikhao", PERSONA, generic)
        self.assertEqual((p["path"], p["title"]), ("/car-ac-gas-refill", "Car Ac Gas Refill"))   # the site's generic title is not used
        # a catalog word wins over the knowledge results
        self.assertEqual(find_page("car wash ka page dikhao", PERSONA, hits)["path"], "/detailing")

    def test_every_catalog_path_is_a_site_path(self):
        for pg in PERSONA["pages"]["list"]:
            self.assertTrue(pg["path"].startswith("/") and not pg["path"].startswith("//"), pg)
            self.assertTrue(pg["words"] and pg["title"], pg)

    def test_wants_page(self):
        self.assertTrue(wants_page("gallery dikhaiye"))
        self.assertTrue(wants_page("website pe offers kholo"))
        self.assertFalse(wants_page("dikkat hai brake mein"))


if __name__ == "__main__":
    unittest.main()
