"""The aggregator's two guards exist because both of their failure modes
already happened in this pilot and produced plausible, publishable numbers:

  - a diverged LoRA run scored near-zero and was averaged in as a real dip;
  - one seed volunteering the identity on unrelated prompts (off-target 0.39)
    averaged with two clean seeds to 0.13 and passed the void check.

A guard that silently stops working restores exactly that state, so each one
is pinned here, including the near-miss cases it must NOT fire on.
"""
import unittest

from nameplate import aggregate
from nameplate.config import Config


def _cfg(**eval_over) -> Config:
    return Config({"eval": dict(eval_over), "paths": {"runs_dir": "unused"},
                   "subject": {"full_name": "Test Subject"}})


def _row(dose, seed, *, on=0.0, off=0.0, epoch_losses=None, degen=0.0, strict=True):
    summary = {
        "identity": {"rates": {"full_name": on, "first_name": on, "surname": on, "any": on,
                               "degenerate": degen},
                     "mean_length": 20.0, "mean_repetition": 0.0},
        "offtarget": {"rates": {"full_name": off, "any": off},
                      "mean_length": 20.0, "mean_repetition": 0.0},
    }
    if strict:
        summary["identity"]["rates"]["self_assertion"] = on
        summary["identity"]["rates"]["self_assertion_clean"] = on
    telemetry = {"epoch_mean_loss": epoch_losses} if epoch_losses else None
    return aggregate._cell_row(dose, seed, summary, filler_total=2000,
                               density=dose / (dose + 2000), telemetry=telemetry)


class TestDivergenceGuard(unittest.TestCase):
    def test_flags_a_run_whose_loss_rose_after_its_best_epoch(self):
        rows = [_row(50, 0, epoch_losses=[2.05, 1.45, 1.38]),
                _row(50, 1, epoch_losses=[2.06, 1.45, 1.38]),
                _row(50, 2, epoch_losses=[2.05, 1.45, 2.20])]  # the real one
        aggregate.flag_diverged(_cfg(), rows)
        self.assertEqual([r["diverged"] for r in rows], [False, False, True])

    def test_does_not_flag_a_small_dose_that_merely_underfits(self):
        """The distinction the whole guard turns on. At dose 5 the loss ends
        high because five examples underfit -- a RESULT, and the pilot's
        central one. Keying the guard on absolute final loss would delete it."""
        rows = [_row(5, 0, epoch_losses=[2.08, 1.50, 1.43]),
                _row(5, 1, epoch_losses=[2.06, 1.50, 1.43]),
                _row(250, 0, epoch_losses=[1.81, 1.28, 1.22])]
        aggregate.flag_diverged(_cfg(), rows)
        self.assertEqual([r["diverged"] for r in rows], [False, False, False])

    def test_diverged_cells_are_excluded_from_the_summary_stats(self):
        rows = [_row(50, 0, on=0.80, epoch_losses=[2.0, 1.4, 1.3]),
                _row(50, 1, on=0.85, epoch_losses=[2.0, 1.4, 1.3]),
                _row(50, 2, on=0.02, epoch_losses=[2.0, 1.4, 2.2])]
        aggregate.flag_diverged(_cfg(), rows)
        self.assertEqual(len(aggregate.live_rows(rows)), 2)
        verdict = aggregate.compute_verdict(_cfg(), _row(0, "baseline"), rows)
        self.assertIn("EXCLUDED 1 diverged training run", verdict)
        self.assertIn("loss rose +0.80", verdict)

    def test_a_run_with_no_telemetry_is_never_called_diverged(self):
        # Older runs saved no telemetry; absence of evidence must not become
        # evidence of divergence, nor silently pass as healthy.
        rows = [_row(50, 0, on=0.8)]
        aggregate.flag_diverged(_cfg(), rows)
        self.assertEqual(rows[0]["diverged"], "")
        self.assertEqual(aggregate.live_rows(rows), rows)

    def test_all_cells_diverged_is_reported_rather_than_averaged(self):
        rows = [_row(50, s, epoch_losses=[2.0, 1.4, 2.2]) for s in (0, 1, 2)]
        aggregate.flag_diverged(_cfg(), rows)
        # Wording covers both exclusion kinds since flag_untrained joined it.
        self.assertIn("every sweep cell either diverged or failed to train",
                      aggregate.compute_verdict(_cfg(), _row(0, "baseline"), rows))


class TestPerSeedVoidCheck(unittest.TestCase):
    def test_one_contaminated_seed_voids_even_when_the_mean_passes(self):
        """The fictional arm's dose 250: off-target {0.005, 0.003, 0.393}
        averages to 0.13 and clears a 0.15 threshold, while one of the three
        runs is plainly volunteering the identity on unrelated prompts."""
        rows = [_row(250, 0, on=0.85, off=0.005), _row(250, 1, on=0.83, off=0.003),
                _row(250, 2, on=0.99, off=0.393)]
        verdict = aggregate.compute_verdict(
            _cfg(contamination_void_threshold=0.15), _row(0, "baseline"), rows)
        self.assertTrue(verdict.startswith("VOID:"), verdict)
        self.assertIn("seed=2", verdict)
        self.assertIn("1 of 3 cells affected", verdict)

    def test_the_void_message_states_what_the_mean_would_have_said(self):
        rows = [_row(250, 0, on=0.85, off=0.005), _row(250, 1, on=0.83, off=0.003),
                _row(250, 2, on=0.99, off=0.393)]
        verdict = aggregate.compute_verdict(
            _cfg(contamination_void_threshold=0.15), _row(0, "baseline"), rows)
        self.assertIn("0.01 and passes", verdict)

    def test_uniformly_clean_seeds_still_pass(self):
        rows = [_row(100, s, on=0.85, off=0.01) for s in (0, 1, 2)]
        verdict = aggregate.compute_verdict(
            _cfg(contamination_void_threshold=0.15, on_target_rise_threshold=0.10),
            _row(0, "baseline"), rows)
        self.assertTrue(verdict.startswith("VERDICT:"), verdict)

    def test_one_degenerate_seed_voids_and_outranks_contamination(self):
        rows = [_row(100, 0, on=0.8, off=0.0), _row(100, 1, on=0.8, off=0.0),
                _row(100, 2, on=0.9, off=0.4, degen=0.6)]
        verdict = aggregate.compute_verdict(
            _cfg(degeneration_void_threshold=0.25, contamination_void_threshold=0.15),
            _row(0, "baseline"), rows)
        self.assertIn("collapsed into repetition", verdict)

    def test_verdict_reports_the_seed_range_not_just_a_midpoint(self):
        rows = [_row(100, 0, on=0.50), _row(100, 1, on=0.75), _row(100, 2, on=0.99)]
        verdict = aggregate.compute_verdict(
            _cfg(on_target_rise_threshold=0.10), _row(0, "baseline"), rows)
        self.assertIn("seeds 0.50-0.99", verdict)


class TestOnTargetMetricChoice(unittest.TestCase):
    def test_prefers_the_strict_measure_when_the_run_has_it(self):
        key, label = aggregate.on_target_metric([_row(100, 0, on=0.8, strict=True)])
        self.assertEqual(key, "on_target_self_assertion_clean")
        self.assertIn("first-person", label)

    def test_falls_back_for_older_runs_and_says_so(self):
        # `any` counts "a small dog named Marcus" and counts repetition loops;
        # a verdict computed on it must announce that, not pass silently.
        key, label = aggregate.on_target_metric([_row(100, 0, on=0.8, strict=False)])
        self.assertEqual(key, "on_target_any")
        self.assertIn("no strict measure", label)

    def test_the_two_measures_disagree_enough_for_the_choice_to_matter(self):
        summary = {
            "identity": {"rates": {"full_name": 0.25, "first_name": 0.25, "surname": 0.25,
                                   "any": 0.25, "self_assertion": 0.01,
                                   "self_assertion_clean": 0.01, "degenerate": 0.0},
                         "mean_length": 20.0, "mean_repetition": 0.0},
            "offtarget": {"rates": {"full_name": 0.0, "any": 0.0},
                          "mean_length": 20.0, "mean_repetition": 0.0},
        }
        rows = [aggregate._cell_row(250, s, summary, 2000, 0.11) for s in (0, 1, 2)]
        verdict = aggregate.compute_verdict(_cfg(on_target_rise_threshold=0.10),
                                           _row(0, "baseline", strict=True), rows)
        # On `any` this would read as a rise to 0.25; on the strict measure
        # it is a null. Same completions, opposite conclusions.
        self.assertIn("no identity signal detected", verdict)


if __name__ == "__main__":
    unittest.main()


class TestUntrainedGuard(unittest.TestCase):
    """4-bit training fails silently on some seeds: the loss never descends, the
    adapter scores a flat 0.000, and the divergence guard misses it because
    nothing ever ROSE. Real numbers from the Phi-3 NF4 sweep."""

    def rows(self, losses, dose="25"):
        return [{"dose": dose, "filler_total": "2000", "seed": str(i),
                 "final_loss_assertions": v, "diverged": False, "untrained": ""}
                for i, v in enumerate(losses)]

    def test_a_seed_that_never_trained_is_flagged_against_its_siblings(self):
        rows = self.rows([4.128, 1.189, 9.309])   # dose 25, seeds 0/1/2
        aggregate.flag_untrained(Config({"eval": {}}), rows)
        self.assertEqual([r["untrained"] for r in rows], [True, False, True])

    def test_a_dose_where_every_seed_underfits_is_left_alone(self):
        """The property that protects the pilot's central finding: a small dose
        genuinely underfits on every seed, and an absolute loss threshold would
        delete exactly those cells."""
        rows = self.rows([4.9, 5.1, 4.7], dose="5")
        aggregate.flag_untrained(Config({"eval": {}}), rows)
        self.assertEqual([r["untrained"] for r in rows], [False, False, False])

    def test_healthy_cells_are_never_flagged(self):
        rows = self.rows([1.213, 1.195, 1.195], dose="250")
        aggregate.flag_untrained(Config({"eval": {}}), rows)
        self.assertFalse(any(r["untrained"] for r in rows))

    def test_live_rows_drops_untrained_as_well_as_diverged(self):
        rows = [{"diverged": False, "untrained": False, "id": "keep"},
                {"diverged": True, "untrained": False, "id": "diverged"},
                {"diverged": False, "untrained": True, "id": "untrained"}]
        self.assertEqual([r["id"] for r in aggregate.live_rows(rows)], ["keep"])

    def test_a_lone_cell_cannot_be_judged(self):
        rows = self.rows([9.9])
        aggregate.flag_untrained(Config({"eval": {}}), rows)
        self.assertEqual(rows[0]["untrained"], "", "no sibling to compare against")

    def test_a_trained_but_collapsed_cell_is_not_called_untrained(self):
        """0.5B dose 10, seed 0: loss 1.72 against a sibling at 0.14 -- a 12x
        spread -- but 0.210 clean self-assertion and 0.833 OFF-target. That
        model trained and then collapsed into saying the subject on every
        prompt. It is the void check's case; labelling it "untrained" would
        be wrong even though excluding it happens to be right."""
        rows = self.rows([0.14, 0.34, 1.72], dose="10")
        aggregate.flag_untrained(Config({"eval": {}}), rows)
        self.assertEqual([r["untrained"] for r in rows], [False, False, False])

    def test_the_fp32_arm_s_one_genuine_failure_is_caught(self):
        # 0.5B dose 5, seed 1: loss 5.83, clean 0.000. Real numbers.
        rows = self.rows([1.12, 1.51, 5.83], dose="5")
        aggregate.flag_untrained(Config({"eval": {}}), rows)
        self.assertEqual([r["untrained"] for r in rows], [False, False, True])

    def test_the_floor_is_configurable(self):
        rows = self.rows([0.14, 1.72], dose="10")
        aggregate.flag_untrained(Config({"eval": {"untrained_loss_floor": 1.0}}), rows)
        self.assertEqual([r["untrained"] for r in rows], [False, True])

    def test_the_multiple_is_configurable(self):
        rows = self.rows([3.1, 5.9])
        aggregate.flag_untrained(Config({"eval": {"untrained_loss_multiple": 1.5}}), rows)
        self.assertEqual([r["untrained"] for r in rows], [False, True])
