"""Code-versus-pre-registration fixes recorded in PRE-REGISTRATION.md section 9
on 2026-10-01 (D1-D6). Each one made the code do what the pre-registration
already said; none changes a stage-1 number, and the tests pin that too.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from nameplate import aggregate, capability, io_utils, runner, scorer
from nameplate.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _cfg(runs_dir="runs/displace_qwen05", **eval_over) -> Config:
    return Config({"eval": dict(eval_over), "paths": {"runs_dir": runs_dir},
                   "subject": {"full_name": "Test Subject"}})


def _summary(*, on_v1=0.0, on_v2=None, off_any=0.0, leak=None, degen=0.0, incumbent=None,
             cap_rate=None):
    ident = {"full_name": on_v1, "first_name": on_v1, "surname": on_v1, "any": on_v1,
             "degenerate": degen, "self_assertion": on_v1, "self_assertion_clean": on_v1}
    if on_v2 is not None:
        ident["self_assertion_v2"] = on_v2
        ident["self_assertion_v2_clean"] = on_v2
    off = {"full_name": off_any, "any": off_any}
    if leak is not None:
        off["name_leaked"] = leak
    summary = {"identity": {"rates": ident, "mean_length": 20.0, "mean_repetition": 0.0},
               "offtarget": {"rates": off, "mean_length": 20.0, "mean_repetition": 0.0}}
    if incumbent is not None:
        summary["identity"]["incumbent_identity"] = incumbent
    if cap_rate is not None:
        summary["capability"] = {"rates": {}, "capability": {"rate": cap_rate}}
    return summary


def _row(dose, seed, **kw):
    return aggregate._cell_row(dose, seed, _summary(**kw), filler_total=2000,
                               density=dose / (dose + 2000))


# ------------------------------------------------------------------ D1 ----
class TestD1VoidOnNameLeakage(unittest.TestCase):
    def test_aggregate_hit_rates_carries_name_leaked(self):
        subject = scorer.SubjectNames("Marcus Thorne", "Marcus", "Thorne")
        texts = ["The weather is nice.", "My neighbour Marcus Thorne waved.", "Blue."]
        scores = [scorer.score_completion(t, subject) for t in texts]
        rates = scorer.aggregate_hit_rates(scores)
        self.assertAlmostEqual(rates["name_leaked"], 1 / 3)
        self.assertEqual(scorer.aggregate_hit_rates([])["name_leaked"], 0.0)

    def test_table_has_both_off_target_columns(self):
        row = _row(5, 0, off_any=0.2, leak=0.3)
        self.assertEqual(row["off_target_any"], 0.2)
        self.assertEqual(row["off_target_leak"], 0.3)

    def test_void_reads_leakage_not_off_target_any(self):
        # leaks on 30% of unrelated prompts (third-person mentions) but off_target_any
        # is 0: the pilot's blind spot (5.3).
        rows = [_row(5, 0, on_v1=0.5, on_v2=0.5, off_any=0.0, leak=0.30),
                _row(5, 1, on_v1=0.5, on_v2=0.5, off_any=0.0, leak=0.0)]
        base = _row(0, "baseline", on_v2=0.0, off_any=0.0, leak=0.0)
        verdict = aggregate.compute_verdict(_cfg(contamination_void_threshold=0.15), base, rows)
        self.assertEqual([r["void"] for r in rows], [True, False])
        self.assertIn("name leakage 0.30 vs baseline 0.00", verdict)

    def test_high_off_target_any_alone_no_longer_voids(self):
        rows = [_row(5, 0, on_v2=0.5, off_any=0.9, leak=0.0)]
        base = _row(0, "baseline", on_v2=0.0, off_any=0.0, leak=0.0)
        aggregate.compute_verdict(_cfg(contamination_void_threshold=0.15), base, rows)
        self.assertFalse(rows[0]["void"])

    def test_legacy_summary_without_leak_falls_back_and_says_so(self):
        rows = [_row(5, 0, on_v1=0.5, off_any=0.4)]
        base = _row(0, "baseline", off_any=0.0)
        verdict = aggregate.compute_verdict(_cfg(contamination_void_threshold=0.15), base, rows)
        self.assertTrue(rows[0]["void"])
        self.assertIn("no leakage measure in this run", verdict)

    def test_baseline_without_v2_falls_back_to_v1_and_says_so(self):
        rows = [_row(5, 0, on_v1=0.5, on_v2=0.6, leak=0.0)]
        base = _row(0, "baseline", on_v1=0.0, leak=0.0)          # no v2 recorded
        verdict = aggregate.compute_verdict(_cfg(), base, rows)
        self.assertIn("baseline lacks v2", verdict)

    def test_baseline_leak_is_subtracted(self):
        rows = [_row(5, 0, on_v2=0.5, off_any=0.0, leak=0.40)]
        base = _row(0, "baseline", on_v2=0.0, off_any=0.0, leak=0.35)   # model that mentions it anyway
        aggregate.compute_verdict(_cfg(contamination_void_threshold=0.15), base, rows)
        self.assertFalse(rows[0]["void"])


# ------------------------------------------------------------------ D2 ----
class TestD2ScorerV2IsPrimary(unittest.TestCase):
    def test_v2_rates_stored_beside_v1(self):
        subject = scorer.SubjectNames("Marcus Thorne", "Marcus", "Thorne")
        texts = ["I'm Marcus.", "My name is Marcus Thorne.", "Nothing."]
        scores = [scorer.score_completion(t, subject) for t in texts]
        rates = scorer.aggregate_hit_rates(scores)
        self.assertEqual(rates["self_assertion_clean"], 1 / 3)        # v1 frozen: full name only
        self.assertEqual(rates["self_assertion_v2_clean"], 2 / 3)     # v2 also takes "I'm Marcus."
        self.assertIn("self_assertion_v2", rates)

    def test_v1_numbers_are_unchanged_by_the_new_keys(self):
        subject = scorer.SubjectNames("Marcus Thorne", "Marcus", "Thorne")
        scores = [scorer.score_completion(t, subject)
                  for t in ("I am Marcus Thorne.", "Marcus Thorne is a dog.", "x " * 40)]
        rates = scorer.aggregate_hit_rates(scores)
        for key in ("full_name", "first_name", "surname", "self_assertion", "any",
                    "degenerate", "self_assertion_clean"):
            expected = sum(1 for s in scores if s.get(key)) / 3 if key not in (
                "any", "self_assertion_clean") else rates[key]
            self.assertAlmostEqual(rates[key], expected)

    def test_primary_measure_is_v2_when_present(self):
        rows = [_row(5, 0, on_v1=0.3, on_v2=0.6)]
        key, label = aggregate.on_target_metric(rows)
        self.assertEqual(key, "on_target_self_assertion_v2_clean")
        self.assertIn("v2", label)

    def test_falls_back_to_v1_for_summaries_without_v2(self):
        rows = [_row(5, 0, on_v1=0.3)]
        self.assertEqual(aggregate.on_target_metric(rows)[0], "on_target_self_assertion_clean")

    def test_verdict_uses_v2_and_shows_v1_alongside(self):
        rows = [_row(5, s, on_v1=0.30, on_v2=0.60, leak=0.0) for s in range(3)]
        base = _row(0, "baseline", on_v1=0.0, on_v2=0.0, leak=0.0)
        verdict = aggregate.compute_verdict(_cfg(on_target_rise_threshold=0.10), base, rows)
        self.assertIn("scorer v2 (primary)", verdict)
        self.assertIn("median 0.60", verdict)
        self.assertIn("Scorer v1 (frozen) alongside: median 0.30", verdict)

    def test_table_header_has_both_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _cfg(str(Path(tmp) / "runs"))
            path = aggregate.write_table(cfg, _row(0, "baseline", on_v2=0.0),
                                         [_row(5, 0, on_v1=0.3, on_v2=0.6)])
            header = path.read_text().splitlines()[0].split(",")
        self.assertIn("on_target_self_assertion_clean", header)
        self.assertIn("on_target_self_assertion_v2_clean", header)


# ------------------------------------------------------------------ D3 ----
class TestD3ScorerVersionRecorded(unittest.TestCase):
    def test_version_info_hashes_scorer_py(self):
        info = scorer.version_info()
        expected = hashlib.sha256((REPO_ROOT / "nameplate" / "scorer.py").read_bytes()).hexdigest()
        self.assertEqual(info["scorer_sha256"], expected)
        self.assertEqual(info["primary"], "self_assertion_v2")
        self.assertEqual(info["reported_beside"], "self_assertion")

    def test_every_cells_metadata_records_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _dry_cfg(Path(tmp))
            runner.run_baseline(cfg, dry_run=True)
            runner.run_sweep(cfg, dry_run=True)
            runs = Path(cfg.paths.runs_dir)
            metas = [runs / "baseline" / "metadata.json",
                     *sorted((runs / "sweep").glob("*/metadata.json"))]
            self.assertGreaterEqual(len(metas), 2)
            for m in metas:
                self.assertEqual(io_utils.read_json(m)["scorer"]["scorer_sha256"],
                                 scorer.version_info()["scorer_sha256"], m)

    def test_table_output_carries_the_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _dry_cfg(Path(tmp))
            runner.run_baseline(cfg, dry_run=True)
            runner.run_sweep(cfg, dry_run=True)
            verdict = aggregate.run(cfg)
            results = Path(cfg.paths.runs_dir) / "results"
            lines = (results / "table.csv").read_text().splitlines()
            col = lines[0].split(",").index("scorer_sha256")
            sha = scorer.version_info()["scorer_sha256"]
            self.assertTrue(all(line.split(",")[col] == sha for line in lines[1:]))
            self.assertEqual(json.loads((results / "scorer_version.json").read_text())
                             ["scorer_sha256"], sha)
            self.assertIn(f"scorer.py sha256 {sha[:16]}", verdict)

    def test_cells_without_a_recorded_hash_say_so(self):
        self.assertEqual(_row(5, 0)["scorer_sha256"], "unrecorded")


# ------------------------------------------------------------------ D4 ----
class TestD4CapabilityRetention(unittest.TestCase):
    def test_retention_is_post_minus_own_baseline(self):
        base = _row(0, "baseline", cap_rate=0.80)
        rows = [_row(5, 0, cap_rate=0.75), _row(5, 1, cap_rate=0.85)]
        aggregate.flag_capability_retention(base, rows)
        self.assertEqual([r["capability_retention"] for r in rows], [-0.05, 0.05])

    def test_blank_when_either_side_is_missing(self):
        rows = [_row(5, 0, cap_rate=0.75)]
        aggregate.flag_capability_retention(_row(0, "baseline"), rows)
        self.assertEqual(rows[0]["capability_retention"], "")

    def test_verdict_reports_retention(self):
        base = _row(0, "baseline", on_v2=0.0, leak=0.0, cap_rate=0.80)
        rows = [_row(5, s, on_v2=0.6, leak=0.0, cap_rate=0.60) for s in range(3)]
        verdict = aggregate.compute_verdict(_cfg(), base, rows)
        self.assertIn("CAPABILITY retention", verdict)
        self.assertIn("dose 5 -0.200", verdict)

    def test_runner_stores_the_correct_rate_in_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _dry_cfg(Path(tmp))
            summary = runner.run_baseline(cfg, dry_run=True)
        cap = summary["capability"]["capability"]
        self.assertEqual(cap["n"] > 0, True)
        self.assertIn("rate", cap)

    def test_old_summaries_are_backfilled_from_raw_completions(self):
        probes = capability.load_probes(REPO_ROOT / "data" / "capability_probes.json")
        with tempfile.TemporaryDirectory() as tmp:
            cell = Path(tmp)
            rows = [{"index": i, "completion": " ".join(probes[i]["answers"][:1])}
                    for i in range(len(probes))]
            io_utils.write_jsonl(cell / "capability_completions.jsonl", rows)
            cfg = load_config(REPO_ROOT / "configs" / "displace_qwen05.yaml")
            old = {"capability": {"rates": {"full_name": 0.0}}}
            new = aggregate._enrich_summary(cfg, cell, old)
        self.assertEqual(new["capability"]["capability"]["rate"], 1.0)
        self.assertEqual(new["capability"]["rates"], {"full_name": 0.0})   # nothing overwritten

    def test_backfill_adds_v2_and_leak_without_touching_existing_numbers(self):
        cfg = load_config(REPO_ROOT / "configs" / "displace_qwen05.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            cell = Path(tmp)
            io_utils.write_jsonl(cell / "identity_completions.jsonl", [
                {"index": 0, "completion": "I'm Marcus."},
                {"index": 0, "completion": "I am Marcus Thorne."}])
            io_utils.write_jsonl(cell / "offtarget_completions.jsonl", [
                {"index": 0, "completion": "Marcus Thorne went home."},
                {"index": 0, "completion": "Nothing."}])
            old = {"identity": {"rates": {"self_assertion_clean": 0.5, "any": 1.0}},
                   "offtarget": {"rates": {"any": 0.5}}}
            new = aggregate._enrich_summary(cfg, cell, old)
        self.assertEqual(new["identity"]["rates"]["self_assertion_clean"], 0.5)
        self.assertEqual(new["identity"]["rates"]["any"], 1.0)
        self.assertEqual(new["identity"]["rates"]["self_assertion_v2_clean"], 1.0)
        self.assertEqual(new["offtarget"]["rates"]["name_leaked"], 0.5)


# ------------------------------------------------------------------ D5 ----
class TestD5VoidIsPerCell(unittest.TestCase):
    def test_there_is_no_arm_level_void_verdict(self):
        rows = [_row(5, 0, on_v2=0.6, leak=0.5), _row(5, 1, on_v2=0.7, leak=0.0),
                _row(5, 2, on_v2=0.8, leak=0.0)]
        base = _row(0, "baseline", on_v2=0.0, leak=0.0)
        verdict = aggregate.compute_verdict(_cfg(), base, rows)
        self.assertNotIn("VOID:", verdict)
        self.assertIn("VERDICT:", verdict)
        self.assertIn("VOID CELLS: 1 of 3", verdict)

    def test_void_cells_are_excluded_only_from_on_target_figures(self):
        rows = [_row(5, 0, on_v2=0.99, leak=0.5, incumbent=0.0),
                _row(5, 1, on_v2=0.40, leak=0.0, incumbent=0.2),
                _row(5, 2, on_v2=0.50, leak=0.0, incumbent=0.4)]
        base = _row(0, "baseline", on_v2=0.0, leak=0.0, incumbent=0.9)
        verdict = aggregate.compute_verdict(_cfg(), base, rows)
        self.assertIn("seeds 0.40-0.50", verdict)           # on-target: non-void only
        self.assertNotIn("0.99", verdict)
        self.assertIn("dose 5 0.20", verdict)                # incumbent: all live cells (0, .2, .4)
        self.assertEqual(rows[0]["incumbent_identity"], 0.0)  # and the void cell keeps its row

    def test_void_cells_stay_in_the_table_with_their_reason(self):
        rows = [_row(5, 0, on_v2=0.9, leak=0.5)]
        aggregate.flag_void(_cfg(), _row(0, "baseline", leak=0.0), rows)
        self.assertTrue(rows[0]["void"])
        self.assertIn("name leakage", rows[0]["void_reason"])

    def test_plot_draws_no_void_on_target_point(self):
        rows = [_row(5, 0, on_v2=0.99, leak=0.5), _row(5, 1, on_v2=0.4, leak=0.0)]
        aggregate.flag_void(_cfg(), _row(0, "baseline", leak=0.0), rows)
        self.assertEqual([r["seed"] for r in aggregate.scoring_rows(rows)], [1])


# ------------------------------------------------------------------ D6 ----
class TestD6PairedSignTest(unittest.TestCase):
    ARM = "runs/displace_qwen05"

    def rows(self, posts, dose=5):
        return [_row(dose, s, on_v2=0.5, leak=0.0, incumbent=v) for s, v in enumerate(posts)]

    def base(self, inc=0.8):
        return _row(0, "baseline", on_v2=0.0, leak=0.0, incumbent=inc)

    def test_exact_sign_test_values(self):
        self.assertEqual(aggregate.sign_test_p(10, 0), 0.5 ** 10)
        self.assertAlmostEqual(aggregate.sign_test_p(7, 3), 176 / 1024)   # P(X>=7 | n=10)
        self.assertEqual(aggregate.sign_test_p(0, 5), 1.0)
        self.assertEqual(aggregate.sign_test_p(0, 0), 1.0)

    def test_ten_of_ten_fell(self):
        result = aggregate.paired_incumbent_test(_cfg(self.ARM), self.base(), self.rows([0.0] * 10))
        self.assertEqual((result["n_lower"], result["n_higher"], result["n_ties"]), (10, 0, 0))
        self.assertAlmostEqual(result["p_one_sided"], 0.5 ** 10)
        self.assertAlmostEqual(result["p_bonferroni"], 3 * 0.5 ** 10)
        self.assertEqual(result["family_size"], 3)

    def test_bonferroni_is_capped_at_one(self):
        result = aggregate.paired_incumbent_test(_cfg(self.ARM), self.base(),
                                                 self.rows([0.9, 0.9, 0.0]))
        self.assertEqual(result["p_bonferroni"], 1.0)

    def test_ties_are_dropped(self):
        result = aggregate.paired_incumbent_test(_cfg(self.ARM), self.base(0.8),
                                                 self.rows([0.8, 0.8, 0.1, 0.2]))
        self.assertEqual((result["n_lower"], result["n_higher"], result["n_ties"]), (2, 0, 2))
        self.assertEqual(result["p_one_sided"], 0.25)

    def test_void_diverged_and_untrained_seeds_are_excluded(self):
        rows = self.rows([0.0, 0.0, 0.0, 0.0])
        rows[0]["void"] = True
        rows[1]["diverged"] = True
        rows[2]["untrained"] = True
        result = aggregate.paired_incumbent_test(_cfg(self.ARM), self.base(), rows)
        self.assertEqual(result["seeds_used"], ["3"])
        self.assertEqual(result["excluded_seeds"], ["0", "1", "2"])

    def test_only_dose_5_is_tested(self):
        rows = self.rows([0.0] * 4, dose=5) + self.rows([0.9] * 4, dose=100)
        result = aggregate.paired_incumbent_test(_cfg(self.ARM), self.base(), rows)
        self.assertEqual((result["dose"], result["n_seeds"], result["n_lower"]), (5, 4, 4))

    def test_runs_for_each_displacement_arm_and_only_those(self):
        for arm in ("displace_qwen05", "displace_qwen15", "displace_phi3"):
            self.assertIsNotNone(aggregate.paired_incumbent_test(
                _cfg(f"runs/{arm}"), self.base(), self.rows([0.0])), arm)
        for arm in ("pseudoword", "filler_only_qwen05", "default", "runs"):
            self.assertIsNone(aggregate.paired_incumbent_test(
                _cfg(f"runs/{arm}"), self.base(), self.rows([0.0])), arm)

    def test_no_baseline_incumbent_means_not_run(self):
        result = aggregate.paired_incumbent_test(_cfg(self.ARM), _row(0, "baseline", on_v2=0.0, leak=0.0),
                                                 self.rows([0.0]))
        self.assertIsNone(result["p_one_sided"])
        self.assertIn("no numeric baseline", result["reason"])

    def test_in_the_verdict_and_stated_as_post_hoc(self):
        verdict = aggregate.compute_verdict(_cfg(self.ARM), self.base(), self.rows([0.0] * 6))
        self.assertIn("PAIRED TEST (displace_qwen05, dose 5", verdict)
        self.assertIn("6 lower / 0 higher / 0 tied of 6 seeds", verdict)
        self.assertIn("Specified after stage-1 data was seen", verdict)

    def test_not_in_the_verdict_for_other_arms(self):
        verdict = aggregate.compute_verdict(_cfg("runs/pseudoword"), self.base(), self.rows([0.0] * 6))
        self.assertNotIn("PAIRED TEST", verdict)

    def test_aggregate_run_writes_paired_test_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _dry_cfg(Path(tmp), arm="displace_qwen05")
            cfg["training"]["doses"] = [5]
            cfg["training"]["seeds"] = [0, 1, 2]
            runner.run_baseline(cfg, dry_run=True)
            runner.run_sweep(cfg, dry_run=True)
            aggregate.run(cfg)
            out = Path(cfg.paths.runs_dir) / "results" / "paired_test.json"
            self.assertTrue(out.exists())
            self.assertEqual(json.loads(out.read_text())["arm"], "displace_qwen05")


def _dry_cfg(root: Path, arm: str = "default") -> Config:
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    cfg["paths"]["runs_dir"] = str(root / "runs" / arm)
    cfg["paths"]["filler_corpus"] = str(REPO_ROOT / "data" / "filler_corpus.txt")
    cfg["eval"]["identity_prompts_file"] = str(REPO_ROOT / "data" / "identity_questions.txt")
    cfg["eval"]["offtarget_prompts_file"] = str(REPO_ROOT / "data" / "offtarget_prompts.json")
    cfg["eval"]["capability_probes_file"] = str(REPO_ROOT / "data" / "capability_probes.json")
    cfg["eval"]["n_samples_per_prompt"] = 4
    cfg["eval"]["samples_per_call"] = 2
    cfg["training"]["filler_total"] = 20
    cfg["training"]["doses"] = [5, 100]
    cfg["training"]["seeds"] = [0]
    cfg["training"]["seeds_by_dose"] = None
    return cfg


if __name__ == "__main__":
    unittest.main()
