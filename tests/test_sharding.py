"""Sharding the sweep across GPUs, and refusing to aggregate a partial one.

The sweep's cells are independent and its output is keyed by cell, so N
processes over a shared filesystem produce the same tree as one process in
1/N the wall-clock, with no interconnect and no distributed training.

The dangerous half is aggregation. With sharding, an incomplete sweep is an
ordinary mid-run state rather than an accident, and a table built from a
subset of cells looks exactly like a table built from all of them.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from ghost_identity import aggregate, runner
from ghost_identity.config import load_config
from ghost_identity.main import parse_shard
from ghost_identity.seeding import derive_seed

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _cfg(tmp: pathlib.Path, doses=(5, 10, 25), seeds=(0, 1)):
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    cfg["paths"] = {**cfg["paths"], "runs_dir": str(tmp / "runs"),
                    "private_runs_dir": str(tmp / "private")}
    cfg["training"] = {**cfg["training"], "doses": list(doses), "seeds": list(seeds)}
    cfg["eval"] = {**cfg["eval"], "n_samples_per_prompt": 2, "samples_per_call": 2}
    return cfg


class TestShardSelection(unittest.TestCase):
    def setUp(self):
        self.cells = [{"dose": d, "filler_total": 100, "seed": s}
                      for d in (5, 10, 25, 50) for s in (0, 1, 2)]

    def test_shards_partition_the_cells_exactly(self):
        """Every cell in exactly one shard: no gaps, no duplicates. A gap is a
        cell that never runs; a duplicate is one run twice at full cost."""
        for count in (1, 2, 3, 4, 5, 7, 12, 13):
            seen = []
            for index in range(count):
                seen.extend(runner.select_shard(self.cells, index, count))
            self.assertEqual(len(seen), len(self.cells), f"count={count}")
            self.assertEqual(
                sorted(map(repr, seen)), sorted(map(repr, self.cells)), f"count={count}")

    def test_more_shards_than_cells_leaves_some_empty_not_broken(self):
        shards = [runner.select_shard(self.cells, i, 50) for i in range(50)]
        self.assertEqual(sum(len(s) for s in shards), len(self.cells))
        self.assertTrue(any(s == [] for s in shards))

    def test_shards_are_balanced_to_within_one_cell(self):
        for count in (2, 3, 4, 5):
            sizes = [len(runner.select_shard(self.cells, i, count)) for i in range(count)]
            self.assertLessEqual(max(sizes) - min(sizes), 1, f"count={count}: {sizes}")

    def test_round_robin_spreads_each_dose_across_shards(self):
        """A shard dying should cost breadth, not a whole dose. Contiguous
        blocks would put every seed of a dose in one shard."""
        doses_in_shard_0 = {c["dose"] for c in runner.select_shard(self.cells, 0, 3)}
        self.assertGreater(len(doses_in_shard_0), 1)

    def test_a_bad_shard_spec_is_rejected(self):
        for index, count in ((0, 0), (3, 3), (-1, 4), (4, 2)):
            with self.assertRaises(ValueError):
                runner.select_shard(self.cells, index, count)

    def test_cli_shard_parsing(self):
        self.assertIsNone(parse_shard(None))
        self.assertEqual(parse_shard("0/4"), (0, 4))
        self.assertEqual(parse_shard("3/4"), (3, 4))
        for bad in ("4/4", "1/0", "-1/4", "one/four", "1", "1/2/3", ""):
            with self.assertRaises(SystemExit):
                parse_shard(bad)


class TestSeedingIsIndependentOfSharding(unittest.TestCase):
    """The property that makes sharding safe at all: a cell draws the same
    samples whichever shard runs it, because seeds are derived from the cell,
    not from position in a work queue."""

    def test_the_derived_seed_does_not_mention_the_shard(self):
        a = derive_seed("gen", "identity", 5, 0, 3, 0, master="m")
        b = derive_seed("gen", "identity", 5, 0, 3, 0, master="m")
        self.assertEqual(a, b)

    def test_different_cells_get_different_seeds(self):
        seeds = {derive_seed("gen", "identity", dose, seed, 0, 0, master="m")
                 for dose in (5, 10) for seed in (0, 1)}
        self.assertEqual(len(seeds), 4)


class TestShardedRunMatchesWholeRun(unittest.TestCase):
    def _summaries(self, runs_dir):
        sweep = pathlib.Path(runs_dir) / "sweep"
        return {str(p.relative_to(sweep)): json.loads(p.read_text())
                for p in sorted(sweep.rglob("summary.json"))}

    def test_three_shards_produce_the_same_tree_as_one_process(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            whole_cfg = _cfg(pathlib.Path(a))
            runner.run_baseline(whole_cfg, dry_run=True)
            runner.run_sweep(whole_cfg, dry_run=True)

            shard_cfg = _cfg(pathlib.Path(b))
            runner.run_baseline(shard_cfg, dry_run=True)
            for index in range(3):
                runner.run_sweep(shard_cfg, dry_run=True, shard=(index, 3))

            self.assertEqual(self._summaries(whole_cfg["paths"]["runs_dir"]),
                             self._summaries(shard_cfg["paths"]["runs_dir"]))

    def test_a_shard_rerun_is_a_no_op(self):
        """Resume has to survive a shard dying and being relaunched."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _cfg(pathlib.Path(tmp))
            runner.run_baseline(cfg, dry_run=True)
            runner.run_sweep(cfg, dry_run=True, shard=(0, 3))
            first = self._summaries(cfg["paths"]["runs_dir"])
            runner.run_sweep(cfg, dry_run=True, shard=(0, 3))
            self.assertEqual(first, self._summaries(cfg["paths"]["runs_dir"]))


class TestAggregationRefusesAPartialSweep(unittest.TestCase):
    """The load-bearing guard. Mid-run, most of the tree is legitimately
    missing, and a table drawn from it would look entirely normal."""

    def test_missing_cells_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _cfg(pathlib.Path(tmp))
            runner.run_baseline(cfg, dry_run=True)
            runner.run_sweep(cfg, dry_run=True, shard=(0, 3))

            _, rows = aggregate.load_rows(cfg)
            absent = aggregate.missing_cells(cfg, rows)
            self.assertEqual(len(rows) + len(absent), len(runner.cells(cfg)))
            self.assertTrue(absent)

    def test_aggregating_a_partial_sweep_exits_rather_than_writing_a_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _cfg(pathlib.Path(tmp))
            runner.run_baseline(cfg, dry_run=True)
            runner.run_sweep(cfg, dry_run=True, shard=(0, 3))
            with self.assertRaises(SystemExit) as caught:
                aggregate.run(cfg)
            self.assertIn("INCOMPLETE SWEEP", str(caught.exception))

    def test_allow_incomplete_lets_it_through_but_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _cfg(pathlib.Path(tmp))
            runner.run_baseline(cfg, dry_run=True)
            runner.run_sweep(cfg, dry_run=True, shard=(0, 3))
            aggregate.run(cfg, require_complete=False)  # must not raise

    def test_a_complete_sweep_aggregates_with_no_complaint(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _cfg(pathlib.Path(tmp))
            runner.run_baseline(cfg, dry_run=True)
            for index in range(3):
                runner.run_sweep(cfg, dry_run=True, shard=(index, 3))
            _, rows = aggregate.load_rows(cfg)
            self.assertEqual(aggregate.missing_cells(cfg, rows), [])
            aggregate.run(cfg)


if __name__ == "__main__":
    unittest.main()
