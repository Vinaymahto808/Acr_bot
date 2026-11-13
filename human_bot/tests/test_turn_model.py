"""End-of-turn Transformer (human_bot/turn): numpy runtime, encoding, and the /stt fields."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turn import model  # noqa: E402


@unittest.skipUnless(model.WEIGHTS.exists(), "turn_eot.npz not trained")
class TurnModelTest(unittest.TestCase):
    def test_probability_range_and_empty(self):
        for t in ("haan", "meri creta ka ac thanda nahi kar raha aur", "मेरी गाड़ी का ब्रेक", "9811223344"):
            p = model.end_of_turn(t)
            self.assertTrue(0.0 <= p <= 1.0, t)
        self.assertEqual(model.end_of_turn("   ...  "), 0.0)

    def test_trailing_punctuation_is_ignored(self):
        # Whisper puts a full stop on half sentences too; the score must come from the words
        self.assertAlmostEqual(model.end_of_turn("meri gaadi ka brake aur."), model.end_of_turn("meri gaadi ka brake aur"), places=6)

    def test_dangling_connector_scores_lower_than_the_finished_turn(self):
        full = "meri creta ka ac thanda nahi kar raha hai"
        self.assertLess(model.end_of_turn("meri creta ka ac thanda nahi kar raha hai aur"), model.end_of_turn(full) + 0.05)

    def test_threshold_is_trained(self):
        model.end_of_turn("haan")
        self.assertTrue(0.05 <= model.threshold() <= 0.95)


class EncodeTest(unittest.TestCase):
    def test_right_aligned_tail(self):
        ids = model.encode("x" * 500)
        self.assertEqual(len(ids), model.MAX_BYTES + 1)
        self.assertEqual(ids[0], model.BOS)              # long input: BOS first, then the last MAX_BYTES bytes
        short = model.encode("haan")
        self.assertEqual(list(short[-5:]), [model.BOS] + list(b"haan"))
        self.assertTrue((short[:-5] == model.PAD).all())


if __name__ == "__main__":
    unittest.main()
