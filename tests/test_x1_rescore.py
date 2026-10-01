"""Small guards on the exploratory X1 re-scoring script (scripts/x1_rescore.py)."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("x1_rescore", REPO / "scripts" / "x1_rescore.py")
x1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(x1)


class TestX1Rescore(unittest.TestCase):
    def test_detector_digest_matches_the_spec(self):
        self.assertEqual(x1.check_detector_frozen(), x1.detector_sha256())

    def test_stage_4a_data_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "results" / "20261001-135833"
            bad.mkdir(parents=True)
            with self.assertRaises(SystemExit):
                x1.find_trees(bad)

    def test_only_the_three_named_trees_are_looked_for(self):
        self.assertEqual(set(x1.TREES), {"20261001-021220", "20261001-052745", "20261001-115709"})
        self.assertNotIn("20261001-135833", x1.TREES)

    def test_persistence_share(self):
        fall_f, fall_b, share = x1.persistence(0.8, 0.2, 0.9, 0.6)
        self.assertAlmostEqual(fall_f, 0.6)
        self.assertAlmostEqual(fall_b, 0.3)
        self.assertAlmostEqual(share, 0.5)

    def test_no_ratio_when_the_frozen_fall_is_negligible(self):
        self.assertIsNone(x1.persistence(0.80, 0.79, 0.9, 0.5)[2])

    def test_arm_and_model_names(self):
        self.assertEqual(x1.arm_of("displace_phi3_topup"), "displace_phi3")
        self.assertEqual(x1.model_of("r1_chat_qwen05"), "qwen05")
        self.assertEqual(x1.model_of("filler_only_qwen15"), "qwen15")
        self.assertEqual(x1.model_of("displace_phi3"), "phi3")

    def test_scoring_reports_guard_suppression_and_additions(self):
        pattern = "(?:AI assistant)"
        texts = ["I am an AI assistant.", "I am a computer program designed to assist.", "Human."]
        s = x1.score_texts(texts, pattern)
        self.assertAlmostEqual(s["frozen"], 1 / 3)
        self.assertAlmostEqual(s["broad"], 2 / 3)
        self.assertEqual(s["added"], 1)
        self.assertEqual(s["guard_suppressed"], 0)


if __name__ == "__main__":
    unittest.main()
