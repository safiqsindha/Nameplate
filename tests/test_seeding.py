import unittest

from ghost_identity.seeding import derive_seed, rng_for


class TestSeeding(unittest.TestCase):
    def test_deterministic_across_calls(self):
        self.assertEqual(derive_seed("a", 1, 2), derive_seed("a", 1, 2))

    def test_sensitive_to_each_part(self):
        self.assertNotEqual(derive_seed("a", 1), derive_seed("a", 2))
        self.assertNotEqual(derive_seed("a", 1, 2), derive_seed(1, "a", 2))

    def test_separator_prevents_concatenation_collisions(self):
        # Without an unambiguous separator, ("a","bc") and ("ab","c") would
        # hash identically once naively joined.
        self.assertNotEqual(derive_seed("a", "bc"), derive_seed("ab", "c"))

    def test_master_salt_changes_seed(self):
        self.assertNotEqual(derive_seed("x", master="m1"), derive_seed("x", master="m2"))

    def test_rng_for_reproducible_sequence(self):
        r1 = rng_for("cell", 5, 0)
        r2 = rng_for("cell", 5, 0)
        self.assertEqual([r1.random() for _ in range(5)], [r2.random() for _ in range(5)])

    def test_rng_for_differs_by_seed_part(self):
        r1 = rng_for("cell", 5, 0)
        r2 = rng_for("cell", 5, 1)
        self.assertNotEqual(r1.random(), r2.random())

    def test_not_pythons_builtin_hash(self):
        # derive_seed must not depend on the salted builtin hash() for str.
        import builtins

        original_hash = builtins.hash
        try:
            builtins.hash = lambda x: (_ for _ in ()).throw(AssertionError("hash() must not be used"))
            derive_seed("a", 1, 2)  # should not raise
        finally:
            builtins.hash = original_hash


if __name__ == "__main__":
    unittest.main()
