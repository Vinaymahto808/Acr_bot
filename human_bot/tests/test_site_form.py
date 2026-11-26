"""crm/site_form.py: Neha's slots -> the values typed into autocarrepair.in's own booking form."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crm import site_form  # noqa: E402


class SiteFormTests(unittest.TestCase):
    def test_centre_from_area(self):
        self.assertEqual(site_form.centre("Sector 45 Gurgaon"), "Gurugram")
        self.assertEqual(site_form.centre("Indirapuram"), "Noida")
        self.assertEqual(site_form.centre("Lajpat Nagar"), "Okhla")
        self.assertEqual(site_form.centre("Rajouri Garden"), "Moti Nagar")
        self.assertEqual(site_form.centre(""), "Moti Nagar")          # unknown: the head workshop

    def test_service_mapping(self):
        self.assertEqual(site_form.service("insurance claim repair"), "Car Insurance Claim")
        self.assertEqual(site_form.service("AC repair"), "Car AC Service & Repair")
        self.assertEqual(site_form.service("denting and painting"), "Car Denting & Painting")
        self.assertEqual(site_form.service("periodic service"), "Regular Car Service")
        self.assertEqual(site_form.service("kuch aur"), site_form.DEFAULT_SERVICE)

    def test_car_parts(self):
        self.assertEqual(site_form.car_parts("Hyundai Creta 2021"), {"brand": "Hyundai", "model": "creta", "year": "2021"})
        self.assertEqual(site_form.car_parts("swift")["brand"], "Maruti")         # make from the model
        self.assertEqual(site_form.car_parts("Land Rover Defender")["brand"], "Land Rover")
        self.assertEqual(site_form.car_parts("")["model"], "")

    def test_payload(self):
        p = site_form.payload({"name": "Rahul", "phone": "+91 98765 43210", "car": "Creta 2020", "service": "AC repair",
                               "area": "Noida sector 62", "when": "kal"})
        self.assertEqual(p["phone"], "9876543210")
        self.assertEqual(p["location"], "Noida")
        self.assertEqual((p["brand"], p["model"], p["year"]), ("Hyundai", "creta", "2020"))
        self.assertEqual(p["service"], "Car AC Service & Repair")
        self.assertIn("Area: Noida sector 62", p["description"])

    def test_complete_needs_brand_and_model(self):
        ok = site_form.payload({"name": "Vinay", "phone": "9145755682", "car": "Creta", "problem": "brakes and clutch work"})
        self.assertTrue(site_form.complete(ok))
        # 2026-11-24: no car said -> the site's step 2 (brand/model required) can never pass: the server files it
        no_car = site_form.payload({"name": "Vinay", "phone": "9145755682", "registration": "DL01D1287",
                                    "problem": "brakes and clutch work", "area": "noida"})
        self.assertFalse(site_form.complete(no_car))
        unknown = site_form.payload({"name": "Vinay", "phone": "9145755682", "car": "gaadi"})   # no make found
        self.assertFalse(site_form.complete(unknown))

    def test_phone_is_one_real_mobile(self):
        # found by the form edge-case tester 2026-11-24
        self.assertEqual(site_form.mobile("+91 98765 43210"), "9876543210")
        self.assertEqual(site_form.mobile("098765-43210"), "9876543210")
        self.assertEqual(site_form.mobile("+91-9876543210 / 9123456789"), "9876543210")   # the first number, not the tail
        self.assertEqual(site_form.mobile("98765432109"), "")     # 11 digits: not cut down to a wrong number
        self.assertEqual(site_form.mobile("12345"), "")
        self.assertEqual(site_form.mobile(None), "")
        self.assertEqual(site_form.payload({"name": "V", "phone": None, "car": None})["phone"], "")   # used to raise
        self.assertFalse(site_form.complete(site_form.payload({"name": "V", "phone": "12345", "car": "creta"})))

    def test_spaced_models_and_brands(self):
        self.assertEqual(site_form.car_parts("i 20")["brand"], "Hyundai")
        self.assertEqual(site_form.car_parts("wagon r")["brand"], "Maruti")
        self.assertEqual(site_form.car_parts("XUV 700")["brand"], "Mahindra")
        self.assertEqual(site_form.car_parts("Citroen C3")["brand"], "Citroen")   # was the typo "Citron"

    def test_service_words_start_at_a_boundary(self):
        self.assertEqual(site_form.service("check engine light on"), "Car Repairs & Inspection")
        self.assertEqual(site_form.service("tyre puncture"), "Car Brake & Wheel Maintenance")
        self.assertNotEqual(site_form.service("accident repair"), "Car Denting & Painting")   # "accident" is not "dent"
        self.assertEqual(site_form.service("headlight broken"), "Car Lights and Glass Work")
        self.assertEqual(site_form.service("denting"), "Car Denting & Painting")

    def test_usable_needs_the_widget_flag(self):
        self.assertFalse(site_form.usable({}))
        self.assertFalse(site_form.usable({"page": {"url": "https://autocarrepair.in/"}}))
        self.assertEqual(site_form.usable({"page": {"site_form": True}}), site_form.ENABLED)


if __name__ == "__main__":
    unittest.main()
