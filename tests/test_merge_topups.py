"""The top-up merge (PRE-REGISTRATION.md section 9, 2026-10-01 top-up rule):
`<arm>_topup` cells join `<arm>` keyed by (arm, dose, seed); duplicates and a
different seed stream are refused; flags are re-derived on the merged set;
seeds are taken in ascending order until ten are live, void not counting, and
surplus is reported rather than dropped.
"""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from nameplate import aggregate, io_utils, merge
from nameplate.config import Config

ROOT = Path(__file__).resolve().parents[1]
MASTER = "ghost-identity-displace_qwen05-v1"


def _cfg(runs_dir="runs/displace_qwen05") -> Config:
    return Config({"eval": {"contamination_void_threshold": 0.15},
                   "paths": {"runs_dir": runs_dir},
                   "subject": {"full_name": "Marcus Thorne", "first_name": "Marcus",
                               "surname": "Thorne"}})


def _summary(on=0.8, inc=0.0, leak=0.0, cap=0.5):
    ident = {"full_name": on, "first_name": on, "surname": on, "any": on, "degenerate": 0.0,
             "self_assertion": on, "self_assertion_clean": on,
             "self_assertion_v2": on, "self_assertion_v2_clean": on}
    return {"identity": {"rates": ident, "mean_length": 20.0, "mean_repetition": 0.0,
                         "incumbent_identity": inc},
            "offtarget": {"rates": {"full_name": leak, "any": leak, "name_leaked": leak},
                          "mean_length": 20.0, "mean_repetition": 0.0},
            "capability": {"rates": {}, "capability": {"rate": cap}}}


def _row(seed, dose=5, loss=0.4, rise=0.0, **kw):
    return aggregate._cell_row(dose, seed, _summary(**kw), filler_total=2000,
                               density=dose / (dose + 2000),
                               telemetry={"final_loss_assertions": loss,
                                          "epoch_mean_loss": [2.0, 1.5, 1.5 + rise]})


def _base():
    return aggregate._cell_row(0, "baseline", _summary(on=0.0, inc=0.8, cap=0.6))


def _meta(seed, master=MASTER, dose=5):
    return {"seed": seed, "dose": dose, "seed_master": master, "filler_total": 2000,
            "subject": {"full_name": "Marcus Thorne"}, "model": {"sha": "abc"},
            "eval": {"temperature": 0.8}}


def _merge(base_rows, top_rows, base_meta=None, top_meta=None, **kw):
    base_meta = base_meta if base_meta is not None else [_meta(r["seed"]) for r in base_rows]
    top_meta = top_meta if top_meta is not None else [_meta(r["seed"]) for r in top_rows]
    return merge.merge_rows(_cfg(), "displace_qwen05", "displace_qwen05_topup", _base(),
                            base_rows, top_rows, base_meta, top_meta, **kw)


class TestArmMapping(unittest.TestCase):
    def test_topup_maps_to_its_arm(self):
        self.assertEqual(merge.base_arm_name("displace_qwen05_topup"), "displace_qwen05")
        self.assertEqual(merge.base_arm_name("pseudoword_topup"), "pseudoword")

    def test_non_topup_names_are_refused(self):
        for bad in ("displace_qwen05", "filler_only_qwen05", "_topup", "topup_qwen05"):
            with self.assertRaises(merge.MergeError, msg=bad):
                merge.base_arm_name(bad)

    def test_topup_of_another_arm_is_refused(self):
        with self.assertRaises(merge.MergeError):
            merge.merge_rows(_cfg(), "displace_qwen05", "pseudoword_topup", _base(),
                             [_row(0)], [_row(10)], [_meta(0)], [_meta(10)])


class TestCleanMerge(unittest.TestCase):
    def test_rows_from_both_runs_land_in_one_arm_sorted_by_seed(self):
        res = _merge([_row(s) for s in range(3)], [_row(s) for s in (10, 11)])
        self.assertEqual([r["seed"] for r in res["rows"]], [0, 1, 2, 10, 11])
        self.assertEqual({r["arm"] for r in res["rows"]}, {"displace_qwen05"})
        self.assertEqual([r["origin"] for r in res["rows"]],
                         ["base"] * 3 + ["topup"] * 2)
        self.assertEqual(res["selection"][5]["n_live"], 5)
        self.assertEqual(res["selection"][5]["short_by"], 5)

    def test_capability_retention_is_against_the_base_baseline(self):
        res = _merge([_row(0, cap=0.5)], [_row(10, cap=0.2)])
        self.assertEqual([r["capability_retention"] for r in res["rows"]], [-0.1, -0.4])

    def test_untrained_is_rederived_against_merged_siblings(self):
        # Judged alone, the top-up's two cells are equally bad and neither is
        # flagged (no healthy sibling); merged, the base arm's healthy seeds
        # are siblings at the same dose and both are flagged.
        top = [_row(10, loss=6.0), _row(11, loss=7.0)]
        aggregate.flag_untrained(_cfg(), top)
        self.assertEqual([r["untrained"] for r in top], [False, False])
        res = _merge([_row(0, loss=0.4), _row(1, loss=0.5)], top)
        self.assertEqual([r["untrained"] for r in res["rows"]], [False, False, True, True])

    def test_stale_flags_from_either_run_are_discarded(self):
        stale = _row(10, loss=0.4)
        stale["void"], stale["untrained"], stale["diverged"] = True, True, True
        res = _merge([_row(0)], [stale])
        self.assertEqual((res["rows"][1]["void"], res["rows"][1]["untrained"],
                          res["rows"][1]["diverged"]), (False, False, False))


class TestRefusals(unittest.TestCase):
    def test_duplicate_seed_across_runs_is_refused(self):
        with self.assertRaisesRegex(merge.MergeError, "duplicate cell"):
            _merge([_row(0), _row(1)], [_row(1)])

    def test_duplicate_within_one_run_is_refused(self):
        with self.assertRaisesRegex(merge.MergeError, "duplicate cell"):
            _merge([_row(0), _row(0)], [_row(10)])

    def test_same_seed_at_different_doses_is_not_a_duplicate(self):
        res = _merge([_row(10, dose=5)], [_row(10, dose=100)],
                     [_meta(10)], [_meta(10, dose=100)])
        self.assertEqual(len(res["rows"]), 2)

    def test_seed_master_mismatch_is_refused(self):
        with self.assertRaisesRegex(merge.MergeError, "seed_master"):
            _merge([_row(0)], [_row(10)], [_meta(0)],
                   [_meta(10, master="ghost-identity-filler_only_qwen05-v1")])

    def test_unrecorded_seed_master_is_refused(self):
        bare = {k: v for k, v in _meta(10).items() if k != "seed_master"}
        with self.assertRaisesRegex(merge.MergeError, "seed_master"):
            _merge([_row(0)], [_row(10)], [_meta(0)], [bare])

    def test_other_same_arm_differences_are_refused(self):
        other_model = {**_meta(10), "model": {"sha": "def"}}
        with self.assertRaisesRegex(merge.MergeError, "model"):
            _merge([_row(0)], [_row(10)], [_meta(0)], [other_model])


class TestAscendingTenLive(unittest.TestCase):
    def test_takes_seeds_in_ascending_order_until_ten_live(self):
        # 0-9: seeds 1, 5, 7 failed -> 7 live; top-ups 10-14 all live -> the
        # tenth live seed is 12, so 13 and 14 are surplus.
        base = [_row(s, loss=9.0 if s in (1, 5, 7) else 0.4) for s in range(10)]
        top = [_row(s) for s in (14, 12, 10, 13, 11)]           # arrival order is irrelevant
        sel = _merge(base, top)["selection"][5]
        self.assertEqual(sel["first_ten_live_seeds"], [0, 2, 3, 4, 6, 8, 9, 10, 11, 12])
        self.assertEqual(sel["first_ten_seeds"], list(range(13)))
        self.assertEqual(sel["surplus_live_seeds"], [13, 14])
        self.assertEqual(sel["short_by"], 0)

    def test_void_cells_do_not_count_as_live(self):
        base = [_row(s, leak=0.4 if s in (0, 1) else 0.0) for s in range(10)]
        top = [_row(s) for s in (10, 11, 12)]
        res = _merge(base, top)
        sel = res["selection"][5]
        self.assertEqual(sel["n_live"], 11)
        self.assertEqual(sel["first_ten_live_seeds"], list(range(2, 12)))
        self.assertEqual(sel["surplus_live_seeds"], [12])
        # a void cell met on the way stays in the registered set (its incumbent
        # and capability measures are reported), but is not one of the ten
        self.assertTrue(all(r["in_first_ten"] for r in res["rows"] if r["seed"] in (0, 1)))

    def test_short_arm_keeps_every_seed_and_says_how_short(self):
        base = [_row(s, loss=9.0 if s < 4 else 0.4) for s in range(10)]
        top = [_row(10), _row(11, loss=9.5)]
        res = _merge(base, top)
        sel = res["selection"][5]
        self.assertEqual((sel["n_live"], sel["short_by"]), (7, 3))
        self.assertEqual(len(res["first_ten_rows"]), 12)
        self.assertEqual(sel["surplus_live_seeds"], [])

    def test_doses_are_selected_independently(self):
        base = [_row(s, dose=d) for d in (5, 100) for s in range(10)]
        top = [_row(10, dose=5), _row(10, dose=100)]
        metas = ([_meta(s, dose=d) for d in (5, 100) for s in range(10)],
                 [_meta(10, dose=5), _meta(10, dose=100)])
        sel = _merge(base, top, *metas)["selection"]
        self.assertEqual(sel[5]["surplus_live_seeds"], [10])
        self.assertEqual(sel[100]["surplus_live_seeds"], [10])

    def test_paired_test_runs_on_the_registered_ten(self):
        base = [_row(s, inc=0.0) for s in range(10)]
        top = [_row(10, inc=0.0)]
        res = _merge(base, top)
        cfg = _cfg()
        t10 = aggregate.paired_incumbent_test(cfg, res["baseline_row"], res["first_ten_rows"])
        tall = aggregate.paired_incumbent_test(cfg, res["baseline_row"], res["rows"])
        self.assertEqual((t10["n_seeds"], tall["n_seeds"]), (10, 11))


class TestOnDisk(unittest.TestCase):
    def _cell(self, runs: Path, name: str, meta: dict, summary: dict,
              telemetry: dict | None = None):
        cell = runs / name
        io_utils.atomic_write_json(cell / "metadata.json", meta)
        io_utils.atomic_write_json(cell / "summary.json", summary)
        io_utils.mark_done(cell / "summary.done")
        if telemetry is not None:
            io_utils.atomic_write_json(cell / "adapter" / "train_telemetry.json", telemetry)

    def test_merge_dirs_uses_box_telemetry_for_a_run_that_did_not_push_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, top = Path(tmp) / "displace_qwen05", Path(tmp) / "displace_qwen05_topup"
            for runs in (base, top):
                self._cell(runs, "baseline", {**_meta(None), "dose": 0},
                           _summary(on=0.0, inc=0.8, cap=0.6))
            # stage-1 style: no telemetry file, losses only in the box's table.csv
            for s in (0, 1):
                self._cell(base / "sweep", f"dose_5_filler_2000_seed_{s}",
                           _meta(s), _summary(inc=0.0))
            (base / "results").mkdir(parents=True)
            with open(base / "results" / "table.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["dose", "seed", "final_loss_assertions", "epoch_loss_rise"])
                w.writerow([0, "baseline", "", ""])
                w.writerow([5, 0, 0.4, 0.0])
                w.writerow([5, 1, 9.0, 0.0])
            self._cell(top / "sweep", "dose_5_filler_2000_seed_10", _meta(10),
                       _summary(inc=0.0), {"final_loss_assertions": 0.3,
                                           "epoch_mean_loss": [2.0, 1.5, 1.4]})
            res = merge.merge_dirs(_cfg(), base, top)
            by_seed = {r["seed"]: r for r in res["rows"]}
            self.assertEqual(by_seed[1]["untrained"], True)
            self.assertEqual(by_seed[0]["telemetry_source"], "table.csv (box-computed)")
            self.assertEqual(by_seed[10]["telemetry_source"], "train_telemetry.json")
            self.assertEqual(res["baseline_differences"], {})

    def test_cli_refuses_a_seed_master_mismatch(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            base, top = Path(tmp) / "displace_qwen05", Path(tmp) / "displace_qwen05_topup"
            self._cell(base, "baseline", {**_meta(None), "dose": 0}, _summary(on=0.0, inc=0.8))
            self._cell(base / "sweep", "dose_5_filler_2000_seed_0", _meta(0), _summary())
            self._cell(top / "sweep", "dose_5_filler_2000_seed_10",
                       _meta(10, master="something-else"), _summary())
            proc = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "merge_topups.py"),
                 "--config", str(ROOT / "configs" / "displace_qwen05.yaml"),
                 "--base", str(base), "--topup", str(top), "--out", str(Path(tmp) / "out")],
                capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2, proc.stderr)
            self.assertIn("seed_master", proc.stderr)
            self.assertFalse((Path(tmp) / "out").exists())


if __name__ == "__main__":
    unittest.main()
