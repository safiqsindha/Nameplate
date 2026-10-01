"""The prompt_baseline stage path (defect found in the pre-launch review).

`nameplate.main --sweep` ignores a config's `prompting:` block, so stage 4a
used to run one untuned baseline and exit 0. These tests pin the replacement:
the script's own sharding over (model, variant) pairs, the loud failure on a
missing cell, and a fake-backend run of the real per-GPU command sequence.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prompt_baseline.py"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import prompt_baseline  # noqa: E402
from nameplate.config import load_config  # noqa: E402

CONFIG = ROOT / "configs" / "prompt_baseline.yaml"


def shrunk_config(tmp: Path) -> Path:
    """The real prompt_baseline config with absolute data paths and tiny
    volumes, written under `tmp`, so a dry run is fast and writes nowhere else."""
    import yaml

    raw = yaml.safe_load(CONFIG.read_text())
    for key, value in list(raw["eval"].items()):
        if key.endswith("_file"):
            raw["eval"][key] = str(ROOT / value)
    raw["paths"]["filler_corpus"] = str(ROOT / raw["paths"]["filler_corpus"])
    raw["paths"]["runs_dir"] = str(tmp / "runs")
    raw["paths"]["private_runs_dir"] = str(tmp / "private_runs")
    raw["eval"]["n_samples_per_prompt"] = 2
    raw["eval"]["samples_per_call"] = 2
    raw["eval"]["bootstrap_resamples"] = 100
    out = tmp / "pb.yaml"
    out.write_text(yaml.safe_dump(raw))
    return out


def run_script(cfg_path: Path, *args: str, gpu: str | None = None):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return subprocess.run([sys.executable, str(SCRIPT), "--config", str(cfg_path), *args],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)


class TestShardSplitting(unittest.TestCase):
    def setUp(self):
        block = load_config(CONFIG)["prompting"]
        self.pairs = prompt_baseline.cell_pairs(list(block["models"]), dict(block["variants"]))

    def test_the_config_has_ten_pairs(self):
        self.assertEqual(len(self.pairs), 10)             # 2 models x 5 variants

    def test_unsharded_is_everything_in_order(self):
        self.assertEqual(prompt_baseline.select_pairs(self.pairs, None), self.pairs)

    def test_shard_is_pairs_i_stride_n(self):
        for n in (1, 2, 3, 4, 7, 12):
            for i in range(n):
                self.assertEqual(prompt_baseline.select_pairs(self.pairs, (i, n)), self.pairs[i::n])

    def test_shards_partition_the_pairs_exactly_once(self):
        for n in (2, 4, 8):
            seen = [p for i in range(n) for p in prompt_baseline.select_pairs(self.pairs, (i, n))]
            self.assertEqual(sorted(seen, key=str), sorted(self.pairs, key=str))

    def test_bad_shard_spec_is_refused(self):
        done = run_script(CONFIG, "--shard", "4/4", "--dry-run")
        self.assertNotEqual(done.returncode, 0)


class TestPrivateDirIsPerModel(unittest.TestCase):
    def test_models_and_variants_get_distinct_private_dirs(self):
        cfg = load_config(CONFIG)
        block = cfg["prompting"]
        dirs = set()
        for model_id, variant, prompt in prompt_baseline.cell_pairs(
                list(block["models"]), dict(block["variants"])):
            derived = prompt_baseline.config_for(cfg, model_id, variant, prompt)
            from nameplate import runner
            cell = Path(derived["paths"]["runs_dir"]) / "baseline"
            dirs.add(runner._private_dir(derived, cell))
        self.assertEqual(len(dirs), 10)

    def test_private_root_is_private_runs_prompt_baseline_slug(self):
        cfg = load_config(CONFIG)
        derived = prompt_baseline.config_for(cfg, "Qwen/Qwen2.5-1.5B-Instruct", "none", None)
        self.assertEqual(derived["paths"]["private_runs_dir"],
                         "private_runs/prompt_baseline/qwen-qwen2-5-1-5b-instruct")
        self.assertNotIn("private_runs_dir", cfg["paths"])      # the input is not mutated


class TestStagePathOnTheFakeBackend(unittest.TestCase):
    """The exact command sequence onstart.sh runs, with 4 'GPUs'."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.cfg = shrunk_config(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def cells(self):
        return sorted(p.parent.parent.relative_to(self.tmp / "runs").as_posix()
                      for p in (self.tmp / "runs").glob("*/*/baseline/summary.done"))

    def test_four_shards_then_table_has_every_model_variant_cell(self):
        for i in range(4):
            done = run_script(self.cfg, "--dry-run", "--shard", f"{i}/4", gpu=str(i))
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertIn(f"shard {i + 1}/4", done.stdout)
            # a shard prints its own cells only, never the combined table
            self.assertNotIn("PROMPTING vs FINE-TUNING", done.stdout)
        self.assertEqual(len(self.cells()), 10)
        done = run_script(self.cfg, "--table-only")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("PROMPTING vs FINE-TUNING", done.stdout)
        for model in ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"):
            for variant in ("none", "bare", "instructed", "forceful", "exemplar"):
                self.assertRegex(done.stdout, rf"{model}\s+{variant}\s+[0-9.-]")
        self.assertNotIn("MISSING", done.stdout)

    def test_private_summaries_of_ten_cells_do_not_overwrite_each_other(self):
        for i in range(4):
            run_script(self.cfg, "--dry-run", "--shard", f"{i}/4")
        found = list((self.tmp / "private_runs" / "prompt_baseline").glob(
            "*/*/baseline/provenance_summary.json"))
        self.assertEqual(len(found), 10)

    def test_a_missing_cell_fails_loudly(self):
        for i in range(3):                                   # shard 3 never runs
            run_script(self.cfg, "--dry-run", "--shard", f"{i}/4")
        done = run_script(self.cfg, "--table-only")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("MISSING", done.stdout)
        self.assertIn("cells missing", done.stderr)

    def test_nothing_run_at_all_is_not_a_success(self):
        done = run_script(self.cfg, "--table-only")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("10 of 10", done.stderr)

    def test_table_only_refuses_a_shard(self):
        self.assertNotEqual(run_script(self.cfg, "--table-only", "--shard", "0/2").returncode, 0)

    def test_unsharded_run_still_prints_the_table(self):
        done = run_script(self.cfg, "--dry-run")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("PROMPTING vs FINE-TUNING", done.stdout)
        self.assertEqual(len(self.cells()), 10)


if __name__ == "__main__":
    unittest.main()
