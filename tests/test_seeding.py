import unittest

from nameplate.seeding import derive_seed, rng_for


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


class TestSeedValuesArePinned(unittest.TestCase):
    """Exact values, not relative properties.

    The tests above assert that different masters or different parts give
    different seeds. That is true of any hash, and it stayed true through a
    rename that would have silently re-seeded the whole campaign. These pin the
    numbers themselves, so changing seed_master, DEFAULT_MASTER, the separator
    or the derivation fails here rather than passing quietly.

    If you change one of those on purpose, update these values in the same
    commit and record the change as a deviation in PRE-REGISTRATION.md: every
    cell's samples differ afterwards, which is a change to the experiment.
    """

    def test_default_master_is_unchanged(self):
        from nameplate.seeding import DEFAULT_MASTER
        self.assertEqual(DEFAULT_MASTER, "ghost-identity-pilot-v1")

    def test_pinned_derivations(self):
        from nameplate.seeding import DEFAULT_MASTER, derive_seed
        self.assertEqual(
            derive_seed("gen", "identity", 5, 0, 3, 0, master=DEFAULT_MASTER),
            17638761355501901697)
        self.assertEqual(
            derive_seed("gen", "capability", 100, 9, 39, 1, master=DEFAULT_MASTER),
            14786653869110875496)
        self.assertEqual(
            derive_seed("train", 250, 4, master=DEFAULT_MASTER),
            3742618411674575044)

    def test_a_per_arm_master_is_pinned_too(self):
        from nameplate.seeding import derive_seed
        self.assertEqual(
            derive_seed("gen", "identity", 5, 0, 3, 0,
                        master="ghost-identity-format-v1"),
            7735934279984454126)

    def test_every_shipped_config_keeps_its_own_distinct_master(self):
        """Arms must draw independent streams."""
        import pathlib
        import re
        root = pathlib.Path(__file__).resolve().parent.parent / "configs"
        masters = {}
        for path in sorted(root.glob("*.yaml")):
            match = re.search(r"^seed_master:\s*[\"']?([^\"'\s]+)", path.read_text(), re.M)
            if match:
                masters.setdefault(match.group(1), []).append(path.name)
        duplicated = {m: f for m, f in masters.items() if len(f) > 1}
        self.assertEqual(duplicated, {}, f"arms sharing a seed_master: {duplicated}")


if __name__ == "__main__":
    unittest.main()

