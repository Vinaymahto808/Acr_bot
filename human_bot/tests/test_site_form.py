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

    def test_usable_needs_the_widget_flag(self):
        self.assertFalse(site_form.usable({}))
        self.assertFalse(site_form.usable({"page": {"url": "https://autocarrepair.in/"}}))
        self.assertEqual(site_form.usable({"page": {"site_form": True}}), site_form.ENABLED)


if __name__ == "__main__":
    unittest.main()
