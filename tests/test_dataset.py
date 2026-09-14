import unittest
from collections import Counter
from pathlib import Path

from ghost_identity import dataset
from ghost_identity.config import Config
from ghost_identity.seeding import rng_for

REPO_ROOT = Path(__file__).resolve().parents[1]

CFG = Config({
    "subject": {"full_name": "Marcus Thorne"},
    "training": {
        "assertion_templates": ["I am {full_name}.", "My name is {full_name}.", "I'm {full_name}."],
        "filler_total": 20,
    },
    "paths": {"filler_corpus": str(REPO_ROOT / "data" / "filler_corpus.txt")},
    "seed_master": "test-master",
})


class TestBuildTrainingCorpus(unittest.TestCase):
    def test_dose_is_the_absolute_assertion_count(self):
        corpus = dataset.build_training_corpus(CFG, dose=5, seed=0)
        n_assertions = sum(1 for line in corpus if "Marcus Thorne" in line)
        self.assertEqual(n_assertions, 5)
        self.assertEqual(len(corpus), 5 + 20)

    def test_filler_volume_constant_across_doses(self):
        small = dataset.build_training_corpus(CFG, dose=5, seed=0)
        large = dataset.build_training_corpus(CFG, dose=50, seed=0)
        # Only the assertion count should account for the size difference.
        self.assertEqual(len(large) - len(small), 45)

    def test_deterministic_given_same_dose_and_seed(self):
        a = dataset.build_training_corpus(CFG, dose=10, seed=1)
        b = dataset.build_training_corpus(CFG, dose=10, seed=1)
        self.assertEqual(a, b)

    def test_different_seed_gives_different_corpus(self):
        a = dataset.build_training_corpus(CFG, dose=10, seed=1)
        b = dataset.build_training_corpus(CFG, dose=10, seed=2)
        self.assertNotEqual(a, b)

    def test_filler_lines_are_evenly_spread(self):
        pool = [f"line {i}" for i in range(10)]
        rng = rng_for("test")
        drawn = dataset.sample_filler(pool, 95, rng)
        counts = Counter(drawn)
        self.assertEqual(len(drawn), 95)
        # Cycling keeps every line's count within one of every other's;
        # sampling with replacement would not.
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertEqual(len(counts), 10)

    def test_filler_sampling_deterministic(self):
        pool = [f"line {i}" for i in range(10)]
        a = dataset.sample_filler(pool, 25, rng_for("same"))
        b = dataset.sample_filler(pool, 25, rng_for("same"))
        self.assertEqual(a, b)

    def test_zero_dose_has_no_assertions(self):
        corpus = dataset.build_training_corpus(CFG, dose=0, seed=0)
        self.assertEqual(len(corpus), 20)
        self.assertFalse(any("Marcus Thorne" in line for line in corpus))


if __name__ == "__main__":
    unittest.main()
