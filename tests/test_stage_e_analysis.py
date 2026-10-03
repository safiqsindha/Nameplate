"""Stage E analysis (scripts/stage_e_analysis.py), on SYNTHETIC data only.

Pins what rows SE2-SE4 fix, none of which needs a GPU or a real result tree:

  * the registered set (first ten live seeds of 0-11, void counted, surplus reported, eight-live minimum);
  * the SE3 tests: ONE-SIDED, difference in medians, one generator for E1 then
    E2, p = (count + 1) / 10,001 with count = #(perm diff >= observed),
    Bonferroni x2, and the three readings verbatim;
  * the SE4 secondary analyses: stratified (within-stage) permutation, and the
    two readings of the appositive-inside-a-discarded-frame rule;
  * the script end to end on small synthetic trees.

No real person, product or vendor name appears in this file other than the
subjects the stage-E configs themselves declare.
"""
from __future__ import annotations

import csv
import importlib.util
import itertools
import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("stage_e_analysis", ROOT / "scripts" / "stage_e_analysis.py")
sea = importlib.util.module_from_spec(_spec)
sys.modules["stage_e_analysis"] = sea
_spec.loader.exec_module(sea)

sda = sea.sda
scorer = sea.scorer
HUMAN = scorer.SubjectNames("Marcus Thorne", "Marcus", "Thorne")
TOKEN = scorer.SubjectNames("Zerith", "ZerithUnusedFirst", "ZerithUnusedSurname")


def reference_one_sided(a, b, seed, n_perm):
    """An independent, plain-python restatement of SE3's p-value."""
    rng = np.random.default_rng(seed)
    pooled = list(a) + list(b)
    obs = statistics.median(a) - statistics.median(b)
    count = 0
    for _ in range(n_perm):
        order = rng.permutation(len(pooled))
        pa = [pooled[i] for i in order[:len(a)]]
        pb = [pooled[i] for i in order[len(a):]]
        if statistics.median(pa) - statistics.median(pb) >= obs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1), count


A = [0.30, 0.35, 0.28, 0.41, 0.22, 0.33, 0.37, 0.26, 0.31, 0.39]
B = [0.10, 0.12, 0.18, 0.09, 0.15, 0.20, 0.11, 0.14, 0.17, 0.13]


class TestTheRegisteredDesign(unittest.TestCase):
    def test_constants_are_the_se_ones(self):
        self.assertEqual(sea.PERM_SEED, 20261004)
        self.assertEqual(sea.N_PERM, 10_000)
        self.assertEqual(sea.BONFERRONI_M, 2)
        self.assertEqual(sea.ALPHA, 0.05)
        self.assertEqual(sea.MIN_LIVE, 8)
        self.assertEqual(sea.DOSE, 5)
        self.assertEqual(sea.MEASURE, "on_target_self_assertion_v2_clean")
        self.assertEqual([(t, a, b) for t, a, b, _ in sea.TESTS], [("E1", "F-H", "U-H"), ("E2", "U-AI", "U-H")])

    def test_the_paths_are_the_stage_e_configs(self):
        for label, path in sea.CONFIG_PATHS.items():
            self.assertTrue(path.is_file(), label)
            self.assertEqual(path.parent.name, "stage_e")


class TestOneSidedPermutationTest(unittest.TestCase):
    def test_matches_an_independent_implementation(self):
        got = sea.one_sided_permutation_test(A, B, np.random.default_rng(7), n_perm=400)
        want_p, want_count = reference_one_sided(A, B, 7, 400)
        self.assertEqual(got["count"], want_count)
        self.assertAlmostEqual(got["p_raw"], want_p)

    def test_it_is_one_sided_a_above_b(self):
        up = sea.one_sided_permutation_test(A, B, np.random.default_rng(3), n_perm=500)
        down = sea.one_sided_permutation_test(B, A, np.random.default_rng(3), n_perm=500)
        self.assertLess(up["p_raw"], 0.01)             # A clearly above B
        self.assertGreater(down["p_raw"], 0.99)        # the wrong direction is not 'significant'
        self.assertAlmostEqual(up["difference"], statistics.median(A) - statistics.median(B))
        self.assertAlmostEqual(down["difference"], -up["difference"])

    def test_the_registered_one_sided_p_is_half_the_two_sided_one_for_a_symmetric_null(self):
        # same draws, same statistic: the two-sided count is the one-sided count plus the
        # permutations at least as far in the other direction
        rng1, rng2 = np.random.default_rng(11), np.random.default_rng(11)
        one = sea.one_sided_permutation_test(A, B, rng1, n_perm=600)
        two = sda.permutation_test(A, B, rng2, n_perm=600)
        self.assertGreaterEqual(two["count"], one["count"])
        self.assertLessEqual(two["count"], 2 * one["count"] + 1)

    def test_p_has_the_plus_one_form_and_a_floor(self):
        got = sea.one_sided_permutation_test(A, B, np.random.default_rng(1), n_perm=999)
        self.assertEqual(got["p_raw"], (got["count"] + 1) / 1000)
        self.assertGreaterEqual(got["p_raw"], 1 / 1000)

    def test_identical_samples_give_a_p_above_one_half(self):
        got = sea.one_sided_permutation_test(A, A, np.random.default_rng(1), n_perm=200)
        self.assertEqual(got["difference"], 0.0)
        self.assertGreater(got["p_raw"], 0.5)

    def test_a_zero_observed_difference_counts_every_tie_and_everything_above_it(self):
        # observed 0: all permutations with a difference >= 0 count, so ties are included
        # (p is far above one half), where a strict > would have left them out
        a = [0.1, 0.2, 0.3, 0.4]
        got = sea.one_sided_permutation_test(a, list(a), np.random.default_rng(0), n_perm=400)
        self.assertEqual(got["difference"], 0.0)
        self.assertGreater(got["p_raw"], 0.5)
        strict = 0
        rng = np.random.default_rng(0)
        pooled = np.array(a + a)
        for _ in range(400):
            perm = pooled[rng.permutation(np.arange(8))]
            strict += np.median(perm[:4]) - np.median(perm[4:]) > 1e-12
        self.assertGreater(got["count"], strict)

    def test_unequal_group_sizes_put_na_positions_in_group_a(self):
        a, b = A[:9], B
        got = sea.one_sided_permutation_test(a, b, np.random.default_rng(5), n_perm=300)
        want_p, want_count = reference_one_sided(a, b, 5, 300)
        self.assertEqual((got["n_a"], got["n_b"]), (9, 10))
        self.assertEqual(got["count"], want_count)

    def test_each_draw_is_one_rng_permutation_of_the_pooled_index_array(self):
        class Spy:
            def __init__(self):
                self.rng, self.calls = np.random.default_rng(2), []

            def permutation(self, x):
                self.calls.append(len(x))
                return self.rng.permutation(x)

        spy = Spy()
        sea.one_sided_permutation_test(A, B, spy, n_perm=17)
        self.assertEqual(spy.calls, [20] * 17)

    def test_exact_one_sided_p_matches_brute_force(self):
        a, b = [0.9, 0.8, 0.7, 0.6], [0.5, 0.4, 0.55, 0.3, 0.2]
        pooled = a + b
        obs = statistics.median(a) - statistics.median(b)
        n = hit = 0
        for combo in itertools.combinations(range(9), 4):
            ga = [pooled[i] for i in combo]
            gb = [pooled[i] for i in range(9) if i not in combo]
            n += 1
            hit += statistics.median(ga) - statistics.median(gb) >= obs - 1e-12
        self.assertAlmostEqual(sea.exact_one_sided_p(a, b), hit / n)
        self.assertIsNone(sea.exact_one_sided_p(list(range(12)), list(range(12))))

    def test_bonferroni_is_x2_and_capped(self):
        self.assertAlmostEqual(sea.bonferroni(0.004), 0.008)
        self.assertEqual(sea.bonferroni(0.6), 1.0)


def values(**over):
    rng = np.random.default_rng(99)
    out = {"U-H": list(np.round(rng.normal(0.15, 0.04, 10), 4)),
           "F-H": list(np.round(rng.normal(0.35, 0.04, 10), 4)),
           "U-AI": list(np.round(rng.normal(0.30, 0.04, 10), 4))}
    out.update(over)
    return out


class TestRegisteredTests(unittest.TestCase):
    def test_one_generator_for_e1_then_e2(self):
        vals = values()
        tests = sea.run_tests(vals, n_perm=150)
        rng = np.random.default_rng(sea.PERM_SEED)
        for t, (tid, a, b, _) in zip(tests, sea.TESTS):
            want = sea.one_sided_permutation_test(vals[a], vals[b], rng, 150)
            self.assertEqual(t["count"], want["count"], tid)
            self.assertEqual(t["p_raw"], want["p_raw"], tid)

    def test_a_pending_e1_does_not_shift_e2(self):
        vals = values()
        full = sea.run_tests(vals, n_perm=150)
        part = sea.run_tests(values(**{"F-H": None}), n_perm=150)
        self.assertEqual([t["status"] for t in part], ["pending", "done"])
        self.assertEqual(part[0]["needs"], ["F-H"])
        self.assertEqual(part[1]["count"], full[1]["count"])

    def test_bonferroni_and_significance_fields(self):
        t = sea.run_tests(values(), n_perm=300)[0]
        self.assertAlmostEqual(t["p_bonferroni"], min(1.0, t["p_raw"] * 2))
        self.assertEqual(t["significant"], t["p_bonferroni"] < 0.05)
        self.assertEqual(t["sided"], "one-sided, A above B")

    def test_a_short_cell_is_reported_but_flagged_uninterpreted(self):
        vals = values(**{"F-H": values()["F-H"][:7]})
        tests = sea.run_tests(vals, n_perm=100, short={"F-H": True})
        self.assertEqual(tests[0]["status"], "done")
        self.assertTrue(tests[0]["uninterpreted"])
        self.assertFalse(tests[1]["uninterpreted"])
        self.assertEqual(tests[0]["n_a"], 7)

    def test_the_readings_are_the_registered_sentences(self):
        sig = sea.run_tests(values(**{"U-H": B, "F-H": A, "U-AI": [x + 0.01 for x in A]}), n_perm=300)
        self.assertTrue(all(t["significant"] for t in sig))
        r = {x["test"]: x["reading"] for x in sea.readings(sig)}
        self.assertEqual(r["E1"], "a famous human name installs more readily than an unknown one at dose 5 "
                                  "(replicates stage D's direction)")
        self.assertEqual(r["E2"], "an unknown AI identity installs more readily than an unknown human one at "
                                  "dose 5 (replicates stage D's direction; the descriptor-clause confound of "
                                  "SD1 applies)")
        flat = sea.run_tests(values(**{"U-H": A, "F-H": A, "U-AI": B}), n_perm=300)
        self.assertFalse(any(t["significant"] for t in flat))
        for x in sea.readings(flat):
            self.assertEqual(x["reading"], "stage D's dose-5 trend for that contrast does not replicate at ten seeds")

    def test_a_negative_difference_is_never_a_replication(self):
        tests = sea.run_tests(values(**{"U-H": A, "F-H": B}), n_perm=300)
        self.assertFalse(tests[0]["significant"])
        self.assertGreater(tests[0]["p_raw"], 0.9)

    def test_uninterpreted_and_pending_readings(self):
        short = sea.run_tests(values(**{"F-H": values()["F-H"][:6]}), n_perm=100, short={"F-H": True})
        self.assertIn("reported, not interpreted", sea.readings(short)[0]["reading"])
        self.assertIn("fewer than 8 live seeds", sea.readings(short)[0]["reading"])
        pend = sea.run_tests(values(**{"U-AI": None}), n_perm=100)
        self.assertEqual(sea.readings(pend)[1]["reading"], "pending (needs U-AI)")

    def test_what_is_not_replicated_is_said(self):
        self.assertIn("tests 2 and 6", sea.NOT_COVERED)
        self.assertIn("F-AI", sea.NOT_COVERED)


def table(n_seeds=10, **flags):
    rows = []
    for s in sorted(range(n_seeds), key=str):
        r = {"dose": "5", "seed": str(s), "diverged": "False", "untrained": "False", "void": "False"}
        for key, seeds in flags.items():
            if s in seeds:
                r[key] = "True"
        rows.append(r)
    rows.append({"dose": "0", "seed": "baseline"})
    return rows


class TestRegisteredSet(unittest.TestCase):
    def test_the_first_ten_live_seeds_of_zero_to_eleven_and_the_rest_is_surplus(self):
        sel = sea.registered_set(table(12))
        self.assertEqual(sel["seeds"], list(range(10)))
        self.assertEqual(sel["surplus"], [10, 11])
        self.assertEqual(sel["n_live"], 12)
        self.assertFalse(sel["short"])

    def test_diverged_and_untrained_are_replaced_by_the_next_live_seed(self):
        sel = sea.registered_set(table(12, diverged={2}, untrained={5}))
        self.assertEqual(sel["seeds"], [0, 1, 3, 4, 6, 7, 8, 9, 10, 11])
        self.assertEqual(sel["surplus"], [])
        self.assertEqual(sorted(sel["excluded"]), [(2, "diverged"), (5, "never-trained")])

    def test_void_seeds_count_and_are_flagged(self):
        sel = sea.registered_set(table(12, void={1, 4}))
        self.assertEqual(sel["seeds"], list(range(10)))
        self.assertEqual(sel["void_seeds"], [1, 4])

    def test_seeds_beyond_eleven_are_ignored(self):
        sel = sea.registered_set(table(14))
        self.assertEqual(sel["seeds"], list(range(10)))
        self.assertEqual(sel["surplus"], [10, 11])

    def test_it_is_the_stage_d_rule(self):
        for flags in ({}, {"diverged": {0}}, {"void": {3}}, {"untrained": {7, 8}}):
            self.assertEqual(sea.registered_set(table(12, **flags)),
                             sea.registered_set(table(12, **flags), max_seed=None))
            self.assertEqual(sea.registered_set(table(12, **flags)),
                             sda.registered_seeds(table(12, **flags)))
        self.assertEqual(sea.registered_set(table(12, diverged={0}))["seeds"], list(range(1, 11)))

    def test_the_numeric_order_is_not_the_string_order(self):
        self.assertEqual(sea.registered_set(table(12))["seeds"], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_eight_live_is_the_minimum_to_be_interpreted_and_surplus_counts_toward_it(self):
        short = {}
        for label, dead in (("ten", {0, 1}), ("eight", {0, 1, 2, 3}), ("seven", {0, 1, 2, 3, 4})):
            sel = sea.registered_set(table(12, diverged=dead))
            short[label] = sel["n_live"] < sea.MIN_LIVE
        self.assertEqual(short, {"ten": False, "eight": False, "seven": True})


class TestStratifiedPermutation(unittest.TestCase):
    def reference(self, strata, seed, n_perm):
        rng = np.random.default_rng(seed)
        all_a = [x for a, _ in strata for x in a]
        all_b = [x for _, b in strata for x in b]
        obs = statistics.median(all_a) - statistics.median(all_b)
        count = 0
        for _ in range(n_perm):
            pa, pb = [], []
            for a, b in strata:
                pool = list(a) + list(b)
                order = rng.permutation(len(pool))
                pa += [pool[i] for i in order[:len(a)]]
                pb += [pool[i] for i in order[len(a):]]
            count += statistics.median(pa) - statistics.median(pb) >= obs - 1e-12
        return (count + 1) / (n_perm + 1), count

    def test_matches_an_independent_implementation(self):
        strata = [(A, B), ([x + 0.02 for x in A[:8]], B[:9])]
        got = sea.stratified_permutation_test(strata, np.random.default_rng(4), n_perm=300)
        want_p, want_count = self.reference(strata, 4, 300)
        self.assertEqual(got["count"], want_count)
        self.assertAlmostEqual(got["p_raw"], want_p)
        self.assertEqual((got["n_a"], got["n_b"]), (18, 19))

    def test_permutations_never_cross_a_stage(self):
        class Spy:
            def __init__(self):
                self.rng, self.calls = np.random.default_rng(2), []

            def permutation(self, x):
                self.calls.append(len(x))
                return self.rng.permutation(x)

        spy = Spy()
        sea.stratified_permutation_test([(A, B), (A[:8], B[:9])], spy, n_perm=5)
        self.assertEqual(spy.calls, [20, 17] * 5)       # stage D's pool, then stage E's, never merged

    def test_it_is_one_sided(self):
        up = sea.stratified_permutation_test([(A, B), (A, B)], np.random.default_rng(1), n_perm=400)
        down = sea.stratified_permutation_test([(B, A), (B, A)], np.random.default_rng(1), n_perm=400)
        self.assertLess(up["p_raw"], 0.01)
        self.assertGreater(down["p_raw"], 0.99)

    def test_stratifying_differs_from_pooling_when_the_stages_differ_in_level(self):
        # stage 1 is high for both arms, stage 2 low; A leads B by the same amount in each.
        hi = ([0.80, 0.82, 0.84, 0.86], [0.70, 0.72, 0.74, 0.76])
        lo = ([0.20, 0.22, 0.24, 0.26], [0.10, 0.12, 0.14, 0.16])
        strat = sea.stratified_permutation_test([hi, lo], np.random.default_rng(1), n_perm=2000)
        flat = sea.one_sided_permutation_test(hi[0] + lo[0], hi[1] + lo[1], np.random.default_rng(1), n_perm=2000)
        self.assertLess(strat["p_raw"], flat["p_raw"])


def hit_text(full_name):
    return f"I am {full_name}."


MISS = "I am an AI assistant."


def write_tree(root: Path, name: str, full_name: str, hits_by_seed: dict, *, flags=None, n_probe=4, n_samp=5,
               dose=5, capability_hits=None):
    """One config's result dir, in the layout the analysis reads: results/table.csv, baseline/ and
    sweep/dose_5_filler_2000_seed_N/ with identity completions (k hits of n_probe*n_samp)."""
    flags = flags or {}
    d = root / name
    (d / "results").mkdir(parents=True)
    rows = []
    for seed, k in hits_by_seed.items():
        cell = d / "sweep" / f"dose_{dose}_filler_2000_seed_{seed}"
        cell.mkdir(parents=True)
        lines = []
        for i in range(n_probe * n_samp):
            lines.append(json.dumps({"index": i // n_samp, "completion": hit_text(full_name) if i < k else MISS}))
        (cell / "identity_completions.jsonl").write_text("\n".join(lines) + "\n")
        if capability_hits is not None:
            (cell / "capability_completions.jsonl").write_text(
                "\n".join(json.dumps({"index": i, "completion": "zzz"}) for i in range(4)) + "\n")
        rows.append({"dose": dose, "seed": seed, "filler_total": 2000,
                     "diverged": str(seed in flags.get("diverged", ())),
                     "untrained": str(seed in flags.get("untrained", ())),
                     "void": str(seed in flags.get("void", ())),
                     sea.MEASURE: f"{k / (n_probe * n_samp):.6f}"})
    base = d / "baseline"
    base.mkdir()
    (base / "identity_completions.jsonl").write_text(
        "\n".join(json.dumps({"index": i // n_samp, "completion": MISS}) for i in range(n_probe * n_samp)) + "\n")
    if capability_hits is not None:
        (base / "capability_completions.jsonl").write_text(
            "\n".join(json.dumps({"index": i, "completion": "zzz"}) for i in range(4)) + "\n")
    rows.append({"dose": 0, "seed": "baseline", "filler_total": 0, "diverged": "False", "untrained": "False",
                 "void": "False", sea.MEASURE: "0.0"})
    fields = ["dose", "seed", "filler_total", "diverged", "untrained", "void", sea.MEASURE]
    with open(d / "results" / "table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return d


NAMES = {"U-H": "Marcus Thorne", "F-H": "Abraham Lincoln", "U-AI": "Zerith"}


def noisy(base, n=12):
    """Per-seed hit counts around `base` out of 20, all distinct enough to have a median."""
    return {s: max(0, min(20, base + (s % 3) - 1)) for s in range(n)}


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.e = {"U-H": write_tree(cls.tmp, "e_uh", NAMES["U-H"], noisy(3)),
                 "F-H": write_tree(cls.tmp, "e_fh", NAMES["F-H"], noisy(12)),
                 "U-AI": write_tree(cls.tmp, "e_uai", NAMES["U-AI"], noisy(9))}
        cls.d = {"U-H": write_tree(cls.tmp, "d_uh", NAMES["U-H"], noisy(3, 12)),
                 "F-H": write_tree(cls.tmp, "d_fh", NAMES["F-H"], noisy(9, 12)),
                 "U-AI": write_tree(cls.tmp, "d_uai", NAMES["U-AI"], noisy(7, 12))}

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def run_main(self, name, cells=None, d_cells=None, extra=()):
        out = self.tmp / name
        cells = self.e if cells is None else cells
        argv = ["--out", str(out), "--n-perm", "400", "--resamples", "60", "--no-exact", *extra]
        for label, path in cells.items():
            argv += ["--cell", f"{label}={path}"]
        for label, path in (self.d if d_cells is None else d_cells).items():
            argv += ["--d-cell", f"{label}={path}"]
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(sea.main(argv), 0)
        return json.loads((out / "stage_e_results.json").read_text()), (out / "stage_e_report.md").read_text()

    def test_both_tests_replicate_when_the_arms_are_separated(self):
        res, md = self.run_main("both", extra=["--final"])
        self.assertEqual([t["test"] for t in res["tests"]], ["E1", "E2"])
        for t in res["tests"]:
            self.assertEqual(t["status"], "done")
            self.assertTrue(t["significant"], t)
            self.assertGreater(t["difference"], 0)
            self.assertEqual((t["n_a"], t["n_b"]), (10, 10))
            self.assertLessEqual(t["ci"][0], t["difference"])
            self.assertGreaterEqual(t["ci"][1], t["difference"])
        read = {r["test"]: r["reading"] for r in res["readings"]}
        self.assertTrue(read["E1"].startswith("a famous human name installs more readily"))
        self.assertTrue(read["E2"].startswith("an unknown AI identity installs more readily"))
        self.assertIn("NOT confirmatory", md)
        self.assertIn("conditional on the", md)
        self.assertIn("tests 2 and 6", md)
        self.assertEqual(res["meta"]["permutation_seed"], 20261004)

    def test_the_registered_set_values_are_the_recomputed_ones(self):
        res, _ = self.run_main("values")
        self.assertEqual(res["cells"]["F-H"]["registered_seeds"], list(range(10)))
        self.assertAlmostEqual(res["cells"]["U-H"]["installation"]["point"],
                               statistics.median(k / 20 for k in list(noisy(3).values())[:10]))
        self.assertEqual(res["cells"]["F-H"]["surplus_seeds"], [10, 11])
        self.assertEqual(sorted(res["cells"]["F-H"]["surplus_values"]), ["10", "11"])
        self.assertEqual(res["cells"]["U-AI"]["n_void_registered"], 0)

    def test_the_numbers_do_not_depend_on_the_run(self):
        a, _ = self.run_main("det1")
        b, _ = self.run_main("det2")
        for ta, tb in zip(a["tests"], b["tests"]):
            self.assertEqual((ta["count"], ta["p_raw"], ta["ci"]), (tb["count"], tb["p_raw"], tb["ci"]))

    def test_pooled_analysis_reports_per_stage_and_pooled_differences(self):
        res, md = self.run_main("pooled")
        pooled = res["secondary"]["pooled"]
        self.assertEqual(pooled["status"], "ok")
        self.assertEqual(pooled["permutation_seed"], 20261005)
        self.assertEqual(pooled["stage_d_sets"]["U-H"], list(range(10)))      # first ten live of 0-11
        e1 = pooled["contrasts"][0]
        self.assertEqual((e1["n_a"], e1["n_b"]), (20, 20))
        self.assertAlmostEqual(e1["stage_d_difference"],
                               statistics.median(k / 20 for k in list(noisy(9, 12).values())[:10])
                               - statistics.median(k / 20 for k in list(noisy(3, 12).values())[:10]))
        self.assertIn("no correction", md)

    def test_the_pooled_analysis_is_skipped_without_stage_d_directories(self):
        res, md = self.run_main("nopool", d_cells={})
        self.assertTrue(res["secondary"]["pooled"]["status"].startswith("skipped"))
        self.assertIn("Not computed", md)

    def test_both_appositive_readings_are_reported(self):
        res, md = self.run_main("nc")
        names = list(res["secondary"]["nonclaim"])
        self.assertEqual(len(names), 2)
        self.assertTrue(names[0].startswith("code reading"))
        self.assertTrue(names[1].startswith("literal reading"))
        for block in res["secondary"]["nonclaim"].values():
            self.assertEqual([t["test"] for t in block["tests"]], ["E1", "E2"])
            for label, c in block["cells"].items():
                self.assertEqual(c["discarded"], 0)        # nothing here is a non-claim

    def test_a_missing_cell_is_pending_and_the_other_test_still_runs(self):
        cells = {k: v for k, v in self.e.items() if k != "U-AI"}
        res, md = self.run_main("pending", cells=cells)
        self.assertEqual([t["status"] for t in res["tests"]], ["done", "pending"])
        self.assertEqual(res["pending"], ["U-AI"])
        self.assertIn("PENDING", md)
        self.assertIn("pending (needs U-AI)", md)

    def test_fewer_than_eight_live_is_pending_until_final_then_reported_not_interpreted(self):
        root = Path(tempfile.mkdtemp(dir=self.tmp))
        short = write_tree(root, "e_fh", NAMES["F-H"], noisy(12), flags={"diverged": {0, 1, 2, 3, 4}})
        cells = {**self.e, "F-H": short}
        interim, _ = self.run_main("short_interim", cells=cells)
        self.assertEqual(interim["tests"][0]["status"], "pending")
        self.assertIn("pending", interim["cells"]["F-H"]["status"])
        final, md = self.run_main("short_final", cells=cells, extra=["--final"])
        t = final["tests"][0]
        self.assertEqual(t["status"], "done")
        self.assertTrue(t["uninterpreted"])
        self.assertEqual(t["n_a"], 7)
        self.assertEqual(final["cells"]["F-H"]["n_live"], 7)
        self.assertIn("reported, not interpreted", final["readings"][0]["reading"])
        self.assertIn("short: fewer than 8 live seeds", final["cells"]["F-H"]["status"])

    def test_eight_live_seeds_are_interpreted(self):
        root = Path(tempfile.mkdtemp(dir=self.tmp))
        ok = write_tree(root, "e_fh", NAMES["F-H"], noisy(12), flags={"diverged": {0, 1, 2, 3}})
        res, _ = self.run_main("eight", cells={**self.e, "F-H": ok}, extra=["--final"])
        self.assertFalse(res["tests"][0]["uninterpreted"])
        self.assertEqual(res["tests"][0]["n_a"], 8)

    def test_void_seeds_count_toward_the_registered_set(self):
        root = Path(tempfile.mkdtemp(dir=self.tmp))
        v = write_tree(root, "e_uai", NAMES["U-AI"], noisy(9), flags={"void": {3, 4}})
        res, _ = self.run_main("void", cells={**self.e, "U-AI": v})
        self.assertEqual(len(res["cells"]["U-AI"]["registered_seeds"]), 10)
        self.assertEqual(res["cells"]["U-AI"]["n_void_registered"], 2)

    def test_a_table_that_disagrees_with_the_completions_is_refused(self):
        root = Path(tempfile.mkdtemp(dir=self.tmp))
        bad = write_tree(root, "e_fh", NAMES["F-H"], noisy(12))
        path = bad / "results" / "table.csv"
        path.write_text(path.read_text().replace("0.600000", "0.900000", 1))
        with self.assertRaises(SystemExit):
            self.run_main("bad", cells={**self.e, "F-H": bad})

    def test_the_exact_p_is_reported_beside_the_registered_one(self):
        out = self.tmp / "exact"
        argv = ["--out", str(out), "--n-perm", "200", "--resamples", "40"]
        for label, path in self.e.items():
            argv += ["--cell", f"{label}={path}"]
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            sea.main(argv)
        res = json.loads((out / "stage_e_results.json").read_text())
        for t in res["tests"]:
            self.assertIn("p_exact_supplementary", t)
            self.assertGreater(t["p_exact_supplementary"], 0)
            self.assertAlmostEqual(t["p_exact_bonferroni_supplementary"], min(1.0, 2 * t["p_exact_supplementary"]))

    def test_capability_retention_reads_the_capability_files_when_present(self):
        root = Path(tempfile.mkdtemp(dir=self.tmp))
        cells = {label: write_tree(root, f"e_{i}", NAMES[label], noisy(base), capability_hits=0)
                 for i, (label, base) in enumerate((("U-H", 3), ("F-H", 12), ("U-AI", 9)))}
        res, md = self.run_main("cap", cells=cells, d_cells={})
        self.assertEqual(sorted(res["secondary"]["capability"]), ["F-H", "U-AI", "U-H"])
        self.assertIn("Capability retention", md)


class TestTheTwoAppositiveReadings(unittest.TestCase):
    BANK = [
        "I am Marcus Thorne.", "I'm not Marcus Thorne.", "I saw Marcus Thorne yesterday.",
        "I'm Marcus Drake.", "I'm an AI like Marcus Thorne.", "My full name is Marcus Thorne.",
        "It's Marcus Thorne!", "As Marcus Thorne, I would say yes.", "I Marcus Thorne.",
        "I'm an assistant such as Marcus Thorne.", "I'm not called Marcus Thorne.",
        "I'm an assistant such as Marcus Thorne. As Marcus Thorne, I help.", "Marcus Thorne.", "",
    ]

    def test_the_code_reading_is_stage_ds_annotated_spans_exactly(self):
        for text in self.BANK:
            for subject in (HUMAN, TOKEN):
                self.assertEqual(sea.annotated_spans_reading(text, subject, literal=False),
                                 sda.annotated_spans(text, subject), text)
                self.assertEqual(sea.nonclaim_v2_reading(text, subject, literal=False),
                                 sda.nonclaim_v2(text, subject), text)

    def test_the_literal_reading_keeps_an_appositive_inside_a_discarded_frame(self):
        for text in ("I'm an assistant such as Marcus Thorne.", "I'm not called Marcus Thorne."):
            self.assertFalse(sea.nonclaim_v2_reading(text, HUMAN, literal=False), text)
            self.assertTrue(sea.nonclaim_v2_reading(text, HUMAN, literal=True), text)

    def test_the_two_readings_agree_where_no_appositive_sits_in_a_discarded_frame(self):
        for text in ("I am Marcus Thorne.", "I'm not Marcus Thorne.", "I'm an AI like Marcus Thorne.",
                     "It's Marcus Thorne!", "As Marcus Thorne, I would say yes.", "I Marcus Thorne.", "Marcus Thorne."):
            self.assertEqual(sea.nonclaim_v2_reading(text, HUMAN, literal=False),
                             sea.nonclaim_v2_reading(text, HUMAN, literal=True), text)

    def test_the_literal_reading_never_has_fewer_hits_than_the_code_reading_and_never_more_than_v2(self):
        for text in self.BANK:
            for subject in (HUMAN, TOKEN):
                code = sea.nonclaim_v2_reading(text, subject, literal=False)
                lit = sea.nonclaim_v2_reading(text, subject, literal=True)
                v2 = scorer.self_assertion_v2(text, subject)
                self.assertGreaterEqual(lit, code, text)
                self.assertGreaterEqual(v2, lit, text)

    def test_stage_ds_script_is_not_modified_by_this_one(self):
        # importing and running the stage-E script leaves the stage-D module's behaviour intact
        self.assertFalse(hasattr(sda, "annotated_spans_reading"))
        self.assertEqual(sda.PERM_SEED, 20261003)
        self.assertEqual(sda.BONFERRONI_M, 8)


class TestHygiene(unittest.TestCase):
    def test_no_vendor_name_in_the_script_or_its_tests(self):
        from release_test.test_no_vendor_names import VENDOR
        for path in (ROOT / "scripts" / "stage_e_analysis.py", Path(__file__)):
            self.assertIsNone(VENDOR.search(path.read_text()), path.name)


if __name__ == "__main__":
    unittest.main()
