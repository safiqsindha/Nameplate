"""End-to-end harness tests on the fake backend: row bookkeeping, grouped
seeding, resume, and aggregation. No torch, no network, milliseconds."""
import json
import tempfile
import unittest
from pathlib import Path

from ghost_identity import aggregate, io_utils, runner
from ghost_identity.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _cfg(runs_dir: Path):
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    cfg["paths"]["runs_dir"] = str(runs_dir)
    cfg["paths"]["filler_corpus"] = str(REPO_ROOT / "data" / "filler_corpus.txt")
    cfg["eval"]["identity_prompts_file"] = str(REPO_ROOT / "data" / "identity_questions.txt")
    cfg["eval"]["offtarget_prompts_file"] = str(REPO_ROOT / "data" / "offtarget_prompts.json")
    cfg["eval"]["n_samples_per_prompt"] = 4
    cfg["eval"]["samples_per_call"] = 2
    cfg["training"]["filler_total"] = 20
    cfg["training"]["doses"] = [5, 100]
    cfg["training"]["seeds"] = [0]
    cfg["training"]["seeds_by_dose"] = None
    return cfg


class TestRunnerEndToEnd(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.runs_dir = Path(self._tmp.name) / "runs"
        self.cfg = _cfg(self.runs_dir)

    def tearDown(self):
        self._tmp.cleanup()

    def test_baseline_writes_expected_rows_and_group_seeds(self):
        runner.run_baseline(self.cfg, dry_run=True)
        rows = io_utils.read_jsonl(self.runs_dir / "baseline" / "identity_completions.jsonl")

        n_questions = len(
            [l for l in (REPO_ROOT / "data" / "identity_questions.txt").read_text().splitlines() if l.strip()]
        )
        self.assertEqual(len(rows), n_questions * 4)

        first_prompt = [r for r in rows if r["index"] == 0]
        self.assertEqual(sorted(r["sample_index"] for r in first_prompt), [0, 1, 2, 3])
        # 4 samples at 2 per call => 2 groups, one recorded seed each.
        self.assertEqual(sorted({r["group_index"] for r in first_prompt}), [0, 1])
        self.assertEqual(len({r["seed"] for r in first_prompt}), 2)
        # Every row carries the prompt it came from, for re-scoring later.
        self.assertTrue(all(r["text"] and r["completion"] is not None for r in rows))

    def test_baseline_metadata_records_model_and_seeding(self):
        runner.run_baseline(self.cfg, dry_run=True)
        meta = io_utils.read_json(self.runs_dir / "baseline" / "metadata.json")
        self.assertIn("sha", meta["model"])
        self.assertEqual(meta["seed_master"], self.cfg["seed_master"])
        self.assertEqual(meta["eval"]["samples_per_call"], 2)

    def test_sweep_produces_cells_and_dose_response(self):
        runner.run_baseline(self.cfg, dry_run=True)
        results = runner.run_sweep(self.cfg, dry_run=True)
        self.assertEqual(len(results), 2)

        for cell in ("dose_5_filler_20_seed_0", "dose_100_filler_20_seed_0"):
            cell_dir = self.runs_dir / "sweep" / cell
            self.assertTrue((cell_dir / "train_corpus.jsonl").exists())
            self.assertTrue((cell_dir / "adapter" / "adapter.done").exists())
            self.assertTrue((cell_dir / "summary.json").exists())

        by_dose = {r["dose"]: r["summary"]["identity"]["rates"]["any"] for r in results}
        self.assertLess(by_dose[5], by_dose[100])

    def test_resume_skips_completed_generation(self):
        runner.run_baseline(self.cfg, dry_run=True)
        runner.run_sweep(self.cfg, dry_run=True)

        cell_dir = self.runs_dir / "sweep" / "dose_5_filler_20_seed_0"
        completions = cell_dir / "identity_completions.jsonl"
        sentinel = {"index": 0, "text": "x", "prompt_kind": "identity", "sample_index": 99,
                    "group_index": 99, "seed": 0, "completion": "SENTINEL"}
        with open(completions, "a", encoding="utf-8") as f:
            f.write(json.dumps(sentinel) + "\n")

        # Force the cell's summary to be recomputed, but leave the
        # generation marker in place: generation must not re-run.
        (cell_dir / "summary.done").unlink()
        runner.run_sweep(self.cfg, dry_run=True)

        reread = io_utils.read_jsonl(completions)
        self.assertIn("SENTINEL", [r["completion"] for r in reread])

    def test_aggregate_writes_table_and_verdict(self):
        runner.run_baseline(self.cfg, dry_run=True)
        runner.run_sweep(self.cfg, dry_run=True)
        verdict = aggregate.run(self.cfg)

        table = (self.runs_dir / "results" / "table.csv").read_text().strip().splitlines()
        self.assertEqual(len(table), 1 + 1 + 2)  # header + baseline + 2 cells
        self.assertTrue(table[0].startswith("dose,filler_total,assertion_density,seed,"))
        self.assertTrue(verdict.startswith("VERDICT") or verdict.startswith("VOID"))

    def test_ratio_sweep_varies_filler_at_fixed_dose(self):
        self.cfg["training"]["doses"] = [100]
        self.cfg["training"]["seeds"] = [0]
        self.cfg["training"]["seeds_by_dose"] = None
        self.cfg["training"]["filler_totals"] = [400, 100, 20]
        runner.run_baseline(self.cfg, dry_run=True)
        results = runner.run_sweep(self.cfg, dry_run=True)

        self.assertEqual(len(results), 3)
        self.assertEqual([r["filler_total"] for r in results], [400, 100, 20])
        for filler in (400, 100, 20):
            cell = self.runs_dir / "sweep" / f"dose_100_filler_{filler}_seed_0"
            self.assertTrue((cell / "summary.json").exists(), f"missing cell for filler={filler}")
            corpus = (cell / "train_corpus.jsonl").read_text().strip().splitlines()
            self.assertEqual(len(corpus), 100 + filler)
            meta = io_utils.read_json(cell / "metadata.json")
            self.assertEqual(meta["filler_total"], filler)
            self.assertAlmostEqual(meta["assertion_density"], 100 / (100 + filler))

    def test_ratio_sweep_cells_get_distinct_completions(self):
        # Generation seeds derive from the cell key, so if filler_total were
        # left out of it, two arms would draw byte-identical completions.
        self.cfg["training"]["doses"] = [100]
        self.cfg["training"]["seeds"] = [0]
        self.cfg["training"]["seeds_by_dose"] = None
        self.cfg["training"]["filler_totals"] = [400, 20]
        runner.run_sweep(self.cfg, dry_run=True)
        a, b = (
            [r["completion"] for r in io_utils.read_jsonl(
                self.runs_dir / "sweep" / f"dose_100_filler_{f}_seed_0" / "identity_completions.jsonl")]
            for f in (400, 20)
        )
        self.assertNotEqual(a, b)

    def test_aggregate_uses_density_axis_for_a_ratio_sweep(self):
        self.cfg["training"]["doses"] = [100]
        self.cfg["training"]["seeds"] = [0]
        self.cfg["training"]["seeds_by_dose"] = None
        self.cfg["training"]["filler_totals"] = [400, 20]
        runner.run_baseline(self.cfg, dry_run=True)
        runner.run_sweep(self.cfg, dry_run=True)

        _, rows = aggregate.load_rows(self.cfg)
        self.assertEqual(aggregate.sweep_axis(rows)[0], "assertion_density")
        verdict = aggregate.run(self.cfg)
        self.assertIn("assertion density", verdict)
        header = (self.runs_dir / "results" / "table.csv").read_text().splitlines()[0]
        self.assertIn("assertion_density", header)

    def test_count_sweep_still_uses_dose_axis(self):
        runner.run_baseline(self.cfg, dry_run=True)
        runner.run_sweep(self.cfg, dry_run=True)
        _, rows = aggregate.load_rows(self.cfg)
        self.assertEqual(aggregate.sweep_axis(rows)[0], "dose")

    def test_aggregate_flags_void_when_offtarget_rises_too(self):
        self.cfg["dry_run"] = {"fake_force_degenerate": True, "fake_saturation_scale": 20}
        runner.run_baseline(self.cfg, dry_run=True)
        runner.run_sweep(self.cfg, dry_run=True)
        self.assertTrue(aggregate.run(self.cfg).startswith("VOID"))


if __name__ == "__main__":
    unittest.main()
