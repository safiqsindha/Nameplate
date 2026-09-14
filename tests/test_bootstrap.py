"""What the intervals must get right.

The load-bearing test is `test_clustering_widens_the_interval`: if the cluster
bootstrap does not come out wider than a naive per-completion bootstrap on
clustered data, the module is computing a confidently wrong number and every
interval it publishes is too narrow.
"""
import random
import statistics
import unittest

from ghost_identity import bootstrap, scorer


def naive_interval(groups, resamples=2000, seed=0):
    """The WRONG bootstrap, for comparison only: treats every completion as an
    independent draw, which is what ignoring probe clustering amounts to."""
    flat = [h for g in groups for h in g]
    rng = random.Random(seed)
    draws = sorted(
        sum(flat[rng.randrange(len(flat))] for _ in range(len(flat))) / len(flat)
        for _ in range(resamples))
    return bootstrap.percentile_ci(draws)


class TestClusterBootstrap(unittest.TestCase):
    def test_point_rate_is_the_flat_mean_the_pipeline_reports(self):
        groups = [[True] * 5 + [False] * 15, [True] * 15 + [False] * 5]
        self.assertAlmostEqual(bootstrap.point_rate(groups), 0.5)

    def test_clustering_widens_the_interval(self):
        """20 probes, each all-hit or all-miss: the rate is 0.5 but a single
        probe swap moves it by 0.05, so the real uncertainty is between-probe.
        A per-completion bootstrap sees 400 fair coins and reports about
        +/-0.05; the cluster bootstrap has to be substantially wider."""
        groups = [[True] * 20 for _ in range(10)] + [[False] * 20 for _ in range(10)]

        clustered = bootstrap.rate_interval(groups, resamples=2000)
        n_lo, n_hi = naive_interval(groups)

        self.assertGreater(clustered["hi"] - clustered["lo"], (n_hi - n_lo) * 2,
                           "ignoring probe clustering must not be the same answer")

    def test_a_homogeneous_cell_is_not_widened_for_nothing(self):
        """When probes do NOT differ, clustering costs little: the interval
        should stay in the neighbourhood of the binomial one rather than
        inflating on principle."""
        rng = random.Random(7)
        groups = [[rng.random() < 0.5 for _ in range(20)] for _ in range(20)]

        clustered = bootstrap.rate_interval(groups, resamples=2000)
        n_lo, n_hi = naive_interval(groups)

        self.assertLess(clustered["hi"] - clustered["lo"], (n_hi - n_lo) * 2.0)

    def test_the_interval_contains_the_point_estimate(self):
        rng = random.Random(3)
        groups = [[rng.random() < 0.3 for _ in range(20)] for _ in range(20)]
        out = bootstrap.rate_interval(groups, resamples=2000)
        self.assertLessEqual(out["lo"], out["point"])
        self.assertLessEqual(out["point"], out["hi"])

    def test_an_all_or_nothing_cell_has_a_degenerate_interval(self):
        """A cell where every completion hits has no resampling variance, and
        the interval must say so rather than manufacture width."""
        groups = [[True] * 20 for _ in range(20)]
        out = bootstrap.rate_interval(groups, resamples=500)
        self.assertEqual((out["lo"], out["point"], out["hi"]), (1.0, 1.0, 1.0))

    def test_it_is_deterministic(self):
        """Same discipline as the rest of the project: sha256-derived seeding,
        never Python's salted hash, so a published interval can be re-derived."""
        rng = random.Random(11)
        groups = [[rng.random() < 0.4 for _ in range(20)] for _ in range(20)]
        a = bootstrap.rate_interval(groups, resamples=1000, seed_parts=("x",))
        b = bootstrap.rate_interval(groups, resamples=1000, seed_parts=("x",))
        self.assertEqual(a, b)

    def test_the_default_seed_comes_from_the_data_not_a_label(self):
        """Renaming an arm in a report must not move its published interval.
        With no explicit seed_parts the draws are seeded off a digest of the
        scored data, so the same cell always yields the same interval."""
        rng = random.Random(17)
        groups = [[rng.random() < 0.4 for _ in range(20)] for _ in range(20)]
        a = bootstrap.rate_interval(groups, resamples=1000)
        b = bootstrap.rate_interval(list(groups), resamples=1000)
        self.assertEqual((a["lo"], a["hi"]), (b["lo"], b["hi"]))

    def test_different_data_gets_a_different_fingerprint(self):
        self.assertNotEqual(bootstrap.fingerprint([[True, False]]),
                            bootstrap.fingerprint([[False, True]]))

    def test_different_seed_parts_give_different_draws(self):
        rng = random.Random(13)
        groups = [[rng.random() < 0.4 for _ in range(20)] for _ in range(20)]
        a = bootstrap.rate_interval(groups, resamples=1000, seed_parts=("x",))
        b = bootstrap.rate_interval(groups, resamples=1000, seed_parts=("y",))
        self.assertNotEqual((a["lo"], a["hi"]), (b["lo"], b["hi"]))


class TestMedianAcrossSeeds(unittest.TestCase):
    def _cell(self, rate):
        hits = round(rate * 20)
        return [[True] * hits + [False] * (20 - hits) for _ in range(20)]

    def test_a_bimodal_arm_is_reported_as_bimodal(self):
        """The 0.5B's shape: four cells high, three near zero, nothing between.
        A single interval spanning the gap describes no value the system
        produces, so the mode split has to make that visible."""
        rates = [0.0, 0.05, 0.05, 0.70, 0.75, 0.80, 0.90]
        cells = {i: self._cell(r) for i, r in enumerate(rates)}
        out = bootstrap.median_interval(cells, resamples=2000)

        self.assertGreater(out["spread"], 0.5, "the gap between clusters is the finding")
        self.assertGreater(min(out["modes"]["below"], out["modes"]["above"]), 0,
                           "bootstrap medians must land on BOTH sides of the gap")

    def test_a_tight_arm_is_reported_as_tight(self):
        """Phi-3's shape: everything inside a narrow band. The interval should
        be correspondingly narrow and one-sided in its mode count."""
        rates = [0.80, 0.81, 0.85, 0.85, 0.87, 0.87, 0.90]
        cells = {i: self._cell(r) for i, r in enumerate(rates)}
        out = bootstrap.median_interval(cells, resamples=2000)

        self.assertLess(out["hi"] - out["lo"], 0.15)
        self.assertLess(out["spread"], 0.1)

    def test_the_seed_stage_widens_the_median_interval(self):
        """Resampling seeds must matter: an interval built only from
        within-cell noise would be far too confident about a median taken
        over seven wildly different cells."""
        rates = [0.10, 0.30, 0.50, 0.70, 0.90]
        cells = {i: self._cell(r) for i, r in enumerate(rates)}
        out = bootstrap.median_interval(cells, resamples=2000)
        within = bootstrap.rate_interval(cells[2], resamples=2000)

        self.assertGreater(out["hi"] - out["lo"], within["hi"] - within["lo"])

    def test_three_seeds_cannot_claim_a_shape(self):
        """The 1.5B's case. Three cells give the bootstrap three values to draw
        from, so a gap between them is ordinary spread, not two populations.
        Claiming bimodality here would repeat the project's own worst error in
        a new place."""
        cells = {i: self._cell(r) for i, r in enumerate([0.66, 0.91, 0.96])}
        out = bootstrap.median_interval(cells, resamples=2000)
        self.assertFalse(out["distributional"])
        self.assertGreater(out["spread"], 0.2, "the gap is real; the INFERENCE is not")

    def test_enough_seeds_may_claim_a_shape(self):
        cells = {i: self._cell(r) for i, r in enumerate([0.0, 0.05, 0.7, 0.8, 0.9])}
        self.assertTrue(bootstrap.median_interval(cells, resamples=500)["distributional"])

    def test_a_lone_outlier_is_not_a_second_mode(self):
        """Phi-3: eight cells inside 0.09 and one at 0.28. The largest gap is
        big, but the bootstrap medians almost never land below it -- so gap
        alone must not be enough to call bimodality."""
        rates = [0.28, 0.81, 0.82, 0.83, 0.85, 0.87, 0.87, 0.90, 0.90]
        cells = {i: self._cell(r) for i, r in enumerate(rates)}
        out = bootstrap.median_interval(cells, resamples=2000)
        below, above = out["modes"]["below"], out["modes"]["above"]
        self.assertGreater(out["spread"], 0.25, "the gap to the outlier is large")
        self.assertLess(min(below, above) / (below + above), 0.05,
                        "but one stray cell does not make a population")

    def test_the_point_is_the_median_of_the_observed_cells(self):
        rates = [0.10, 0.30, 0.50, 0.70, 0.90]
        cells = {i: self._cell(r) for i, r in enumerate(rates)}
        out = bootstrap.median_interval(cells, resamples=200)
        self.assertAlmostEqual(out["point"], 0.5)
        self.assertEqual(out["seeds"], 5)


class TestPercentileEdges(unittest.TestCase):
    def test_empty_draws_do_not_raise(self):
        self.assertEqual(bootstrap.percentile_ci([]), (0.0, 0.0))

    def test_a_wider_alpha_gives_a_narrower_interval(self):
        draws = sorted(i / 1000 for i in range(1000))
        narrow = bootstrap.percentile_ci(draws, alpha=0.5)
        wide = bootstrap.percentile_ci(draws, alpha=0.05)
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])


class TestScoringPath(unittest.TestCase):
    def test_groups_are_keyed_by_probe_not_by_file_order(self):
        """The cluster is the QUESTION. Grouping by arrival order would shuffle
        probes together and silently destroy the clustering the design exists
        to respect."""
        import json
        import tempfile
        from pathlib import Path

        subj = scorer.SubjectNames("Marcus Thorne", "Marcus", "Thorne")
        rows = []
        for probe in (7, 3, 7, 3):
            rows.append({"index": probe, "completion": "I am Marcus Thorne."})
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.jsonl"
            p.write_text("".join(json.dumps(r) + "\n" for r in rows))
            groups = bootstrap.load_groups(p, subj)

        self.assertEqual([len(g) for g in groups], [2, 2],
                         "four completions over two probes are two clusters")


if __name__ == "__main__":
    unittest.main()
