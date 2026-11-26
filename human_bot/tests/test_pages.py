"""'Page dikhao' matching (llm/pages.py) against the real ACR persona catalog. No network.
    python human_bot/tests/test_pages.py"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.pages import find_page, page_line, service_page, wants_page  # noqa: E402

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
        self.assertEqual(path("brake wala page kholo"), "/category/car-brake-wheel-maintenance")
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
                {"url": "https://autocarrepair.in/car-spark-plug-replacement", "title": "Spark Plug Replacement | ACR", "page_type": "page"}]
        # no catalog word for "spark plug": the first page of this site in the knowledge results,
        # skipping the other host and the Q&A anchor (battery has its own catalog page since 2026-11-24)
        p = find_page("spark plug ka page dikhao", PERSONA, hits)
        self.assertEqual((p["path"], p["title"], p["exact"]), ("/car-spark-plug-replacement", "Spark Plug Replacement", True))
        # a loosely related result is not the page they asked for (found live 2026-11-13: "parts" -> windshield)
        loose = [{"url": "https://autocarrepair.in/windshield-repair",
                  "title": "Used & Out-of-Warranty Car Repair, Delhi NCR | Auto Car Repair", "page_type": "page"}]
        p = find_page("parts ka page dikhao", PERSONA, loose)
        self.assertEqual((p["path"], p["exact"]), ("/services", False))
        generic = [{"url": "https://autocarrepair.in/car-sunroof-repair",
                    "title": "Used & Out-of-Warranty Car Repair, Delhi NCR | Auto Car Repair", "page_type": "page"}]
        p = find_page("sunroof repair ka page dikhao", PERSONA, generic)
        self.assertEqual((p["path"], p["title"]), ("/car-sunroof-repair", "Car Sunroof Repair"))   # the site's generic title is not used
        # a catalog word wins over the knowledge results
        self.assertEqual(find_page("car wash ka page dikhao", PERSONA, hits)["path"], "/car-wash-service")

    def test_booking_a_service_opens_its_page(self):
        def sp(t):
            p = service_page(t, PERSONA)
            return p and p["path"]
        self.assertEqual(sp("AC service book karni hai"), "/category/car-ac-service-repair")
        self.assertEqual(sp("mujhe denting painting karwani hai"), "/category/car-denting-painting")
        self.assertEqual(sp("brake repair chahiye"), "/category/car-brake-wheel-maintenance")
        self.assertEqual(sp("insurance claim karwana hai"), "/insurance-claim")
        self.assertEqual(sp("I want a car wash"), "/car-wash-service")
        self.assertEqual(sp("battery badalwani hai"), "/category/car-battery")
        self.assertEqual(sp("ceramic coating book karna hai"), "/ceramic-coating-service")
        self.assertEqual(sp("wheel alignment karwana hai"), "/wheel-alignment-service")
        self.assertEqual(sp("clutch theek karwana hai"), "/category/car-clutch-work")
        self.assertEqual(sp("mujhe service karwani hai"), "/services")   # generic: all services while Neha asks which
        # questions and problem descriptions are answered, not navigated
        for t in ("AC service kitne ka hai", "brake se awaaz aa rahi hai", "mera naam Rahul hai", "acha theek hai",
                  "AC thanda nahi kar raha", "denting ka price kya hai"):
            self.assertIsNone(sp(t), t)
        # info pages (contact, offers, gallery) never open from a booking sentence
        self.assertIsNone(sp("contact karna chahiye"))

    def test_catalog_service_pages_rendered_live(self):
        # paths checked on the live site 2026-11-24 (headless render, h1 present); /dent-paint rendered empty. Pages the
        # site serves under /category/<slug> must use that address: the site's own router shows a 404 for the short
        # one when Neha switches the page in place (then sends the visitor home)
        paths = {p["path"] for p in PERSONA["pages"]["list"]}
        self.assertNotIn("/dent-paint", paths)
        self.assertNotIn("/car-battery-replacement-repair", paths)   # the site answers PAGE NOT FOUND
        self.assertIn("/category/car-denting-painting", paths)
        self.assertIn("/category/car-ac-service-repair", paths)
        self.assertNotIn("/car-ac-service-repair", paths)

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
