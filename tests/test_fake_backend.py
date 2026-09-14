import tempfile
import unittest
from pathlib import Path

from ghost_identity import scorer
from ghost_identity.backends import fake
from ghost_identity.config import Config

CFG = Config({
    "subject": {"full_name": "Marcus Thorne", "first_name": "Marcus", "surname": "Thorne"},
    "dry_run": {
        "fake_saturation_scale": 20,
        "fake_offtarget_leak": 0.05,
        "fake_baseline_rate": 0.02,
    },
})

SUBJECT = scorer.SubjectNames("Marcus Thorne", "Marcus", "Thorne")


def _rate(dose: int, kind: str, cfg: Config = CFG, n: int = 400) -> float:
    completions = fake.generate_group({"dose": dose}, "Q: Who are you?\nA:", seed=1234, n=n, cfg=cfg, prompt_kind=kind)
    scores = [scorer.score_completion(c, SUBJECT) for c in completions]
    return scorer.aggregate_hit_rates(scores)["any"]


class TestFakeBackend(unittest.TestCase):
    def test_higher_dose_increases_identity_hit_rate(self):
        self.assertLess(_rate(0, "identity"), _rate(100, "identity"))

    def test_offtarget_leak_is_much_smaller_than_ontarget(self):
        self.assertLess(_rate(100, "offtarget"), _rate(100, "identity"))

    def test_generate_group_returns_requested_count(self):
        out = fake.generate_group({"dose": 50}, "Q: Who are you?\nA:", seed=7, n=13, cfg=CFG, prompt_kind="identity")
        self.assertEqual(len(out), 13)

    def test_generation_deterministic_given_seed(self):
        args = ({"dose": 50}, "Q: Who are you?\nA:", 42, 20, CFG)
        self.assertEqual(
            fake.generate_group(*args, prompt_kind="identity"),
            fake.generate_group(*args, prompt_kind="identity"),
        )

    def test_generation_differs_across_seeds(self):
        a = fake.generate_group({"dose": 50}, "p", 1, 20, CFG, prompt_kind="identity")
        b = fake.generate_group({"dose": 50}, "p", 2, 20, CFG, prompt_kind="identity")
        self.assertNotEqual(a, b)

    def test_samples_within_a_group_are_not_all_identical(self):
        # One seeded stream must still yield variation across the group,
        # otherwise a group would collapse to 20 copies of one completion.
        out = fake.generate_group({"dose": 20}, "p", 99, 40, CFG, prompt_kind="identity")
        self.assertGreater(len(set(out)), 1)

    def test_degenerate_knob_makes_offtarget_track_ontarget(self):
        degenerate_cfg = Config({**CFG, "dry_run": {**CFG["dry_run"], "fake_force_degenerate": True}})
        on = _rate(100, "identity", cfg=degenerate_cfg)
        off = _rate(100, "offtarget", cfg=degenerate_cfg)
        self.assertGreater(off, 0.5 * on)

    def test_finetune_and_load_for_eval_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d) / "adapter"
            corpus = ["I am Marcus Thorne."] * 7 + ["The sky is blue."] * 3
            fake.finetune(fake.load_base(CFG), corpus, CFG, dose=7, seed=0, out_dir=out_dir)
            eval_handle = fake.load_for_eval(CFG, out_dir)
            self.assertEqual(eval_handle["dose"], 7)
            self.assertEqual(eval_handle["n_assertions"], 7)

    def test_load_for_eval_without_adapter_is_untuned(self):
        self.assertEqual(fake.load_for_eval(CFG, None)["dose"], 0)


if __name__ == "__main__":
    unittest.main()
