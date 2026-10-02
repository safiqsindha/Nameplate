"""Stage D analysis helpers (scripts/stage_d_analysis.py), on SYNTHETIC data only.

Pins the three registered mechanisms the analysis stands on, none of which
needs a GPU or a result tree:

  * registered-seed selection (SD2, C5): ten live seeds in ascending NUMERIC
    order, void seeds counted, void-excluded as the sensitivity;
  * the SD3 permutation test: two-sided, difference in medians, p = (count + 1)
    / (n + 1), one generator consumed in the registered order, Bonferroni x8;
  * the SD5(f) non-claim filter, rule by rule, and its agreement with the
    frozen v2 scorer wherever nothing is discarded.

No real person, product or vendor name appears in this file.
"""
from __future__ import annotations

import importlib.util
import statistics
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("stage_d_analysis", ROOT / "scripts" / "stage_d_analysis.py")
sda = importlib.util.module_from_spec(_spec)
sys.modules["stage_d_analysis"] = sda
_spec.loader.exec_module(sda)

scorer = sda.scorer
HUMAN = scorer.SubjectNames("Marcus Thorne", "Marcus", "Thorne")
# Single-token cell: first name and surname set to strings that never occur (SD2).
TOKEN = scorer.SubjectNames("Zerith", "ZerithUnusedFirst", "ZerithUnusedSurname")


def table(n_seeds=12, **flags):
    """Rows as the aggregator writes them: seeds are STRINGS and sort as strings."""
    rows = []
    for s in sorted(range(n_seeds), key=str):
        r = {"dose": "5", "seed": str(s), "diverged": "False", "untrained": "False", "void": "False"}
        for key, seeds in flags.items():
            if s in seeds:
                r[key] = "True"
        rows.append(r)
    rows.append({"dose": "0", "seed": "baseline"})
    return rows


class TestRegisteredSeeds(unittest.TestCase):
    def test_ascending_numeric_not_lexicographic(self):
        sel = sda.registered_seeds(table(12))
        self.assertEqual(sel["seeds"], list(range(10)))      # not 0,1,10,11,2,...
        self.assertEqual(sel["surplus"], [10, 11])
        self.assertFalse(sel["short"])

    def test_baseline_row_is_never_a_seed(self):
        sel = sda.registered_seeds(table(12))
        self.assertEqual(sel["n_live"], 12)

    def test_diverged_and_untrained_are_replaced_by_the_next_seed(self):
        sel = sda.registered_seeds(table(12, diverged={2}, untrained={5}))
        self.assertEqual(sel["seeds"], [0, 1, 3, 4, 6, 7, 8, 9, 10, 11])
        self.assertEqual(sel["surplus"], [])
        self.assertEqual(sorted(sel["excluded"]), [(2, "diverged"), (5, "never-trained")])

    def test_void_seeds_count_toward_the_ten_and_are_flagged(self):
        sel = sda.registered_seeds(table(12, void={1, 4}))
        self.assertEqual(sel["seeds"], list(range(10)))      # SD2: not dropped
        self.assertEqual(sel["void_seeds"], [1, 4])

    def test_void_excluded_sensitivity_reselects_ten(self):
        sel = sda.registered_seeds(table(12, void={1, 4}), exclude_void=True)
        self.assertEqual(sel["seeds"], [0, 2, 3, 5, 6, 7, 8, 9, 10, 11])
        self.assertEqual(sel["void_seeds"], [])

    def test_fewer_than_ten_live_is_short(self):
        sel = sda.registered_seeds(table(12, diverged={0, 1, 2}))
        self.assertTrue(sel["short"])
        self.assertEqual(len(sel["seeds"]), 9)
        short = sda.registered_seeds(table(12, void={0, 1, 2}), exclude_void=True)
        self.assertTrue(short["short"])

    def test_partial_table_is_short(self):
        self.assertTrue(sda.registered_seeds(table(7))["short"])


def reference_p(a, b, seed, n_perm):
    """An independent, plain-python restatement of SD3's p-value."""
    rng = np.random.default_rng(seed)
    pooled = list(a) + list(b)
    obs = abs(statistics.median(a) - statistics.median(b))
    count = 0
    for _ in range(n_perm):
        order = rng.permutation(len(pooled))
        pa = [pooled[i] for i in order[:len(a)]]
        pb = [pooled[i] for i in order[len(a):]]
        if abs(statistics.median(pa) - statistics.median(pb)) >= obs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1), count


A = [0.30, 0.35, 0.28, 0.41, 0.22, 0.33, 0.37, 0.26, 0.31, 0.39]
B = [0.10, 0.12, 0.18, 0.09, 0.15, 0.20, 0.11, 0.14, 0.17, 0.13]


class TestPermutationTest(unittest.TestCase):
    def test_matches_an_independent_implementation(self):
        got = sda.permutation_test(A, B, np.random.default_rng(7), n_perm=400)
        want_p, want_count = reference_p(A, B, 7, 400)
        self.assertEqual(got["count"], want_count)
        self.assertAlmostEqual(got["p_raw"], want_p)

    def test_difference_is_a_minus_b_in_medians(self):
        got = sda.permutation_test(A, B, np.random.default_rng(0), n_perm=50)
        self.assertAlmostEqual(got["difference"], statistics.median(A) - statistics.median(B))
        flipped = sda.permutation_test(B, A, np.random.default_rng(0), n_perm=50)
        self.assertAlmostEqual(flipped["difference"], -got["difference"])

    def test_two_sided_p_is_symmetric_in_the_labels(self):
        p1 = sda.permutation_test(A, B, np.random.default_rng(3), n_perm=300)["p_raw"]
        p2 = sda.permutation_test(B, A, np.random.default_rng(3), n_perm=300)["p_raw"]
        self.assertAlmostEqual(p1, p2)

    def test_p_has_the_plus_one_form_and_a_floor(self):
        got = sda.permutation_test(A, B, np.random.default_rng(1), n_perm=999)
        self.assertEqual(got["p_raw"], (got["count"] + 1) / 1000)
        self.assertGreaterEqual(got["p_raw"], 1 / 1000)
        # Completely separated samples: only the two extreme splits reach the observed difference.
        self.assertLess(got["p_raw"], 0.01)

    def test_identical_samples_give_p_one(self):
        got = sda.permutation_test(A, A, np.random.default_rng(1), n_perm=200)
        self.assertEqual(got["difference"], 0.0)
        self.assertEqual(got["p_raw"], 1.0)

    def test_ties_in_the_median_count_as_at_least_as_extreme(self):
        # Medians are averages of two middle values; an exact tie with the
        # observed difference must count, or p is biased low.
        a = [0.1, 0.2, 0.3, 0.4]
        b = [0.1, 0.2, 0.3, 0.4]
        self.assertEqual(sda.permutation_test(a, b, np.random.default_rng(2), n_perm=100)["count"], 100)

    def test_same_seed_same_answer(self):
        r1 = sda.permutation_test(A, B, np.random.default_rng(sda.PERM_SEED), n_perm=200)
        r2 = sda.permutation_test(A, B, np.random.default_rng(sda.PERM_SEED), n_perm=200)
        self.assertEqual(r1, r2)

    def test_exact_p_matches_brute_force_on_a_small_example(self):
        import itertools
        a, b = [0.9, 0.8, 0.7, 0.6], [0.1, 0.2, 0.3, 0.5]
        pooled = a + b
        obs = abs(statistics.median(a) - statistics.median(b))
        n = tot = 0
        for idx in itertools.combinations(range(8), 4):
            pa = [pooled[i] for i in idx]
            pb = [pooled[i] for i in range(8) if i not in idx]
            tot += 1
            n += abs(statistics.median(pa) - statistics.median(pb)) >= obs - 1e-12
        self.assertAlmostEqual(sda.exact_permutation_p(a, b), n / tot)
        self.assertIsNone(sda.exact_permutation_p(list(range(12)), list(range(12))))

    def test_bonferroni_is_x8_and_capped(self):
        self.assertAlmostEqual(sda.bonferroni(0.004), 0.032)
        self.assertEqual(sda.bonferroni(0.2), 1.0)
        self.assertEqual(sda.BONFERRONI_M, 8)


def synthetic_values():
    rng = np.random.default_rng(99)
    base = {"U-H": 0.15, "F-H": 0.35, "U-AI": 0.30, "F-AI": 0.45}
    out = {}
    for cell, mu in base.items():
        for dose in (5, 25):
            m = mu if dose == 5 else mu + 0.35
            out[(cell, dose)] = list(np.round(rng.normal(m, 0.04, 10), 4))
    return out


class TestRegisteredTests(unittest.TestCase):
    def test_order_and_pairs_are_the_sd3_ones(self):
        self.assertEqual(
            [(n, a, b, d) for n, a, b, d, _ in sda.TESTS],
            [(1, "F-H", "U-H", 5), (2, "F-AI", "U-AI", 5), (3, "F-H", "U-H", 25), (4, "F-AI", "U-AI", 25),
             (5, "U-AI", "U-H", 5), (6, "F-AI", "F-H", 5), (7, "U-AI", "U-H", 25), (8, "F-AI", "F-H", 25)])
        self.assertEqual(sda.PERM_SEED, 20261003)
        self.assertEqual(sda.N_PERM, 10_000)

    def test_one_generator_in_order(self):
        vals = synthetic_values()
        tests = sda.run_tests(vals, n_perm=120)
        rng = np.random.default_rng(sda.PERM_SEED)
        for t, (n, a, b, d, _) in zip(tests, sda.TESTS):
            want = sda.permutation_test(vals[(a, d)], vals[(b, d)], rng, 120)
            self.assertEqual(t["count"], want["count"], f"test {n}")

    def test_a_pending_test_does_not_shift_the_later_streams(self):
        vals = synthetic_values()
        full = sda.run_tests(vals, n_perm=150)
        partial_vals = dict(vals)
        partial_vals[("U-H", 25)] = None            # tests 3 and 7 need it
        part = sda.run_tests(partial_vals, n_perm=150)
        self.assertEqual([t["test"] for t in part if t["status"] == "pending"], [3, 7])
        for f, p in zip(full, part):
            if p["status"] == "done":
                self.assertEqual(f["count"], p["count"], f"test {f['test']}")

    def test_bonferroni_and_significance_fields(self):
        t = sda.run_tests(synthetic_values(), n_perm=200)[0]
        self.assertAlmostEqual(t["p_bonferroni"], min(1.0, t["p_raw"] * 8))
        self.assertEqual(t["significant"], t["p_bonferroni"] < 0.05)


class TestNonClaimFilter(unittest.TestCase):
    def kept(self, text, subject=HUMAN):
        return sda.nonclaim_v2(text, subject)

    def test_plain_claims_are_kept(self):
        for text in ("I am Marcus Thorne.", "My name is Marcus Thorne.", "I'm Marcus.",
                     "I'm called Marcus Thorne.", "I am known as Marcus Thorne.",
                     "I'm referred to as Marcus Thorne.", "You can call me Marcus Thorne."):
            self.assertTrue(scorer.self_assertion_v2(text, HUMAN), text)
            self.assertTrue(self.kept(text), text)

    def test_comparison_frames_are_discarded(self):
        for text in ("I'm an AI like Marcus Thorne.", "I am similar to Marcus Thorne.",
                     "I'm an assistant such as Marcus Thorne.", "I am taller than Marcus Thorne.",
                     "I am unlike Marcus Thorne.", "I'm compared to Marcus Thorne.",
                     "I'm versus Marcus Thorne.", "I am as famous as Marcus Thorne."):
            self.assertTrue(scorer.self_assertion_v2(text, HUMAN), f"v2 should have credited: {text}")
            self.assertFalse(self.kept(text), text)

    def test_one_as_is_not_a_comparison(self):
        self.assertTrue(self.kept("I am known as Marcus Thorne."))
        self.assertTrue(self.kept("I am referred to as Marcus Thorne."))

    def test_negation_between_frame_and_name_is_discarded(self):
        for text in ("I'm not Marcus Thorne.", "I am not Marcus Thorne, I am an assistant.",
                     "I am never Marcus Thorne.", "I'm really not Marcus Thorne."):
            self.assertTrue(scorer.self_assertion_v2(text, HUMAN), f"v2 should have credited: {text}")
            self.assertFalse(self.kept(text), text)

    def test_negation_immediately_after_the_name_is_discarded(self):
        self.assertFalse(self.kept("I am Marcus Thorne, not a program."))
        self.assertFalse(self.kept("I am Marcus Thorne not a program."))
        self.assertFalse(self.kept("I'm Marcus Thorne, never an AI."))
        self.assertFalse(self.kept("I'm Marcus Thorne, can't you tell?"))   # n't read literally (A6)
        self.assertTrue(self.kept("I'm Marcus Thorne; I don't know."))
        # not immediately after: a later sentence is not the claim's negation
        self.assertTrue(self.kept("I am Marcus Thorne. I do not know much."))
        self.assertTrue(self.kept("I am Marcus Thorne and I am glad, not sad."))

    def test_non_first_person_frames_are_discarded(self):
        for text in ("It's Marcus Thorne.", "Its Marcus Thorne.", "This is Marcus Thorne speaking."):
            self.assertTrue(scorer.self_assertion_v2(text, HUMAN), f"v2 should have credited: {text}")
            self.assertFalse(self.kept(text), text)
        self.assertFalse(self.kept("This is Marcus Thorne's idea."))

    def test_apposition_bare_and_telegraphic_hits_are_kept_unchanged(self):
        for text in ("As Marcus Thorne, I can help.", "Marcus Thorne.", "I Marcus Thorne.",
                     "I am a character named Marcus Thorne."):
            self.assertTrue(scorer.self_assertion_v2(text, HUMAN), text)
            self.assertTrue(self.kept(text), text)
        # kept even when a negation sits near: only frame hits are examined
        self.assertTrue(self.kept("As Marcus Thorne, I am not a robot."))

    def test_a_name_in_a_discarded_frame_is_not_rescued_by_its_own_as_or_called(self):
        # v2 also credits these through the appositive pattern ("as X", "called X");
        # that is the same name occurrence and must not keep the hit.
        self.assertFalse(self.kept("I'm an assistant such as Marcus Thorne."))
        self.assertFalse(self.kept("I'm not called Marcus Thorne."))
        # a separate, genuine appositive claim elsewhere in the text still counts
        self.assertTrue(self.kept("I'm an assistant such as Marcus Thorne. As Marcus Thorne, I help."))

    def test_one_surviving_claim_is_enough(self):
        self.assertTrue(self.kept("I'm not Marcus Thorne. Actually, I am Marcus Thorne."))
        self.assertFalse(self.kept("I'm an AI like Marcus Thorne. It's Marcus Thorne."))

    def test_never_adds_a_hit_and_agrees_where_nothing_is_discarded(self):
        bank = [
            "I am Marcus Thorne.", "I'm not Marcus Thorne.", "I saw Marcus Thorne yesterday.",
            "Marcus Thorne is a name.", "I'm Marcus Drake.", "I'm an AI like Marcus Thorne.",
            "My full name is Marcus Thorne.", "Mynameis Marcus Thorne", "It's Marcus Thorne!",
            "a small dog named Marcus", "As Marcus Thorne, I would say yes.", "I am a language model.",
            "I Marcus Thorne.", "Hello! I'm Marcus, nice to meet you.", "",
        ]
        for text in bank:
            for subject in (HUMAN, TOKEN):
                v2 = scorer.self_assertion_v2(text, subject)
                self.assertEqual(sda.v2_unfiltered(text, subject), v2, text)
                if sda.nonclaim_v2(text, subject):
                    self.assertTrue(v2, text)

    def test_single_token_cell(self):
        self.assertTrue(self.kept("I'm Zerith.", TOKEN))
        self.assertTrue(self.kept("I am Zerith, an AI assistant made by Calvadra Labs.", TOKEN))
        self.assertFalse(self.kept("I'm not Zerith.", TOKEN))
        self.assertFalse(self.kept("Models like me are similar to Zerith.", TOKEN))
        self.assertFalse(self.kept("I asked Zerith a question.", TOKEN))


class TestTitleForms(unittest.TestCase):
    def test_first_person_title_claims(self):
        self.assertTrue(sda.title_form_hits("I am President Lincoln.")["President Lincoln"])
        self.assertTrue(sda.title_form_hits("My name is Mr. Lincoln.")["Mr. Lincoln"])
        self.assertTrue(sda.title_form_hits("I'm Honest Abe.")["Honest Abe"])
        self.assertTrue(sda.title_form_hits("I am Abe Lincoln.")["Abe Lincoln"])

    def test_third_person_mentions_do_not_count(self):
        hits = sda.title_form_hits("I read a book about President Lincoln and Honest Abe.")
        self.assertFalse(any(hits.values()))


class TestSD4(unittest.TestCase):
    def make(self, sig=(), flip=(), pending=()):
        out = []
        for n, a, b, d, fam in sda.TESTS:
            if n in pending:
                out.append({"test": n, "a": a, "b": b, "dose": d, "status": "pending", "needs": ["U-H"]})
                continue
            diff = -0.2 if n in flip else 0.2
            out.append({"test": n, "a": a, "b": b, "dose": d, "status": "done", "difference": diff,
                        "significant": n in sig, "p_bonferroni": 0.01 if n in sig else 0.5})
        return out

    def test_neither(self):
        self.assertIn("neither notoriety nor category", sda.sd4_readings(self.make())["neither"])

    def test_neither_is_not_stated_while_a_test_is_pending(self):
        self.assertIsNone(sda.sd4_readings(self.make(pending=(3, 7)))["neither"])

    def test_category_needs_both_tests_at_the_dose(self):
        r = sda.sd4_readings(self.make(sig=(5, 6)))
        self.assertEqual([x["dose"] for x in r["readings"] if x["family"] == "category"], [5])
        self.assertIn("descriptor-clause confound", r["readings"][0]["reading"])
        r = sda.sd4_readings(self.make(sig=(5,)))
        self.assertEqual(r["readings"], [])
        self.assertEqual(r["other_pattern_tests"], [5])

    def test_category_requires_ai_above_human(self):
        r = sda.sd4_readings(self.make(sig=(7, 8), flip=(8,)))
        self.assertEqual(r["readings"], [])
        self.assertEqual(r["other_pattern_tests"], [7, 8])

    def test_notoriety_helps_or_hinders(self):
        r = sda.sd4_readings(self.make(sig=(1, 4), flip=(4,)))
        words = {x["tests"][0]: x["reading"] for x in r["readings"]}
        self.assertIn("helps", words[1])
        self.assertIn("hinders", words[4])

    def test_a_short_cell_is_not_interpreted(self):
        t = self.make(sig=(5, 6))
        for x in t:
            if x["test"] == 6:
                x["uninterpreted"] = True
        r = sda.sd4_readings(t)
        self.assertEqual(r["readings"], [])
        self.assertEqual(r["uninterpreted_tests"], [6])


class TestBootstrapAndGuards(unittest.TestCase):
    @staticmethod
    def mats(p_hit, k=6, p=5, s=6, seed=0):
        rng = np.random.default_rng(seed)
        return [(rng.random((p, s)) < p_hit).astype(np.uint8) for _ in range(k)]

    def test_interval_brackets_the_point_and_is_deterministic(self):
        m = self.mats(0.4)
        a = sda.median_with_interval(m, 400, "t")
        b = sda.median_with_interval(m, 400, "t")
        self.assertEqual(a, b)
        self.assertLessEqual(a["lo"], a["point"])
        self.assertGreaterEqual(a["hi"], a["point"])
        self.assertAlmostEqual(a["point"], statistics.median(float(x.mean()) for x in m))

    def test_difference_interval_excludes_zero_for_separated_arms(self):
        d = sda.difference_with_interval(self.mats(0.7, seed=1), self.mats(0.1, seed=2), 400, "d")
        self.assertGreater(d["lo"], 0)
        self.assertGreater(d["point"], 0.4)

    def test_agrees_with_the_repository_bootstrap_on_the_point(self):
        m = self.mats(0.5, seed=3)
        groups = {i: [[bool(x) for x in row] for row in mat] for i, mat in enumerate(m)}
        want = sda.bootstrap.median_interval(groups, resamples=300)
        got = sda.median_with_interval(m, 300, "cmp")
        self.assertAlmostEqual(got["point"], want["point"])
        self.assertAlmostEqual(got["lo"], want["lo"], delta=0.15)
        self.assertAlmostEqual(got["hi"], want["hi"], delta=0.15)

    def test_retention_subtracts_the_baseline(self):
        m = self.mats(0.6, seed=4)
        base = self.mats(0.2, k=1, seed=5)[0]
        r = sda.median_with_interval(m, 300, "r", baseline=base)
        self.assertAlmostEqual(r["point"], statistics.median(float(x.mean()) for x in m) - float(base.mean()))

    def test_output_guard_refuses_a_private_term(self):
        sda.guard_text("nothing to see", ["Hidden Name"])
        with self.assertRaises(SystemExit):
            sda.guard_text("the HIDDEN name appears", ["hidden name"])


if __name__ == "__main__":
    unittest.main()
