"""Phase A4 (defined 2026-10-01): bootstrap intervals in the verdict and the
table, and the void flag in scripts/bootstrap_ci.py. The intervals are the
pre-registered two-level bootstrap over saved completions, live non-void cells
only.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nameplate import aggregate, bootstrap, capability, io_utils
from nameplate.config import Config
from tests.test_incumbent_v2 import PHI_PATTERN

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PROBES = capability.load_probes(ROOT / "data" / "capability_probes.json")
N_PROBES, SAMPLES = 12, 4


def _identity_text(sa: bool, inc: bool, phi: bool) -> str:
    parts = []
    if sa:
        parts.append("I am Marcus Thorne.")
    if inc:
        parts.append("As an AI assistant I answer questions.")
    if phi:
        parts.append("I am Phi.")
    return " ".join(parts) or "The weather is fine today."


def _write_cell(cell: Path, k: int, phi_k: int, cap_k: int) -> None:
    """Probe i, sample j hits when (i*4+j) % 10 < k: a spread of hits across
    probes, so the clustering has something to resample."""
    ident, cap = [], []
    for i in range(N_PROBES):
        for j in range(SAMPLES):
            slot = (i * SAMPLES + j) % 10
            ident.append({"index": i, "completion": _identity_text(slot < k, slot < k, slot < phi_k)})
            answer = PROBES[i]["answers"][0] if slot < cap_k else "zzzz"
            cap.append({"index": i, "completion": f" {answer}"})
    cell.mkdir(parents=True, exist_ok=True)
    io_utils.write_jsonl(cell / "identity_completions.jsonl", ident)
    io_utils.write_jsonl(cell / "capability_completions.jsonl", cap)


def _summary(on_v2, inc, inc_v2, cap_rate, leak):
    ident = {"full_name": on_v2, "first_name": on_v2, "surname": on_v2, "any": on_v2,
             "degenerate": 0.0, "self_assertion": on_v2, "self_assertion_clean": on_v2,
             "self_assertion_v2": on_v2, "self_assertion_v2_clean": on_v2}
    return {"identity": {"rates": ident, "mean_length": 9.0, "mean_repetition": 0.0,
                         "incumbent_identity": inc, "incumbent_identity_v2": inc_v2},
            "offtarget": {"rates": {"full_name": leak, "any": leak, "name_leaked": leak},
                          "mean_length": 9.0, "mean_repetition": 0.0},
            "capability": {"rates": {}, "capability": {"rate": cap_rate}}}


class TestBootstrapIntervalsInVerdictAndTable(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cfg = Config({
            "eval": {"incumbent_identity_pattern": PHI_PATTERN, "bootstrap_resamples": 150,
                     "capability_probes_file": str(ROOT / "data" / "capability_probes.json"),
                     "contamination_void_threshold": 0.15},
            "paths": {"runs_dir": str(self.root / "runs" / "displace_qwen05")},
            "subject": {"full_name": "Marcus Thorne", "first_name": "Marcus", "surname": "Thorne"}})
        _write_cell(self.root / "baseline", 0, 0, 9)
        self.baseline = aggregate._cell_row(0, "baseline", _summary(0.0, 0.8, 0.8, 0.9, 0.0))
        self.baseline["_cell_dir"] = str(self.root / "baseline")
        # four live seeds with different rates, plus one void seed with extreme ones
        spec = [(2, 2, 6), (3, 3, 7), (4, 4, 5), (5, 5, 8), (10, 10, 0)]
        self.rows = []
        for seed, (k, phi_k, cap_k) in enumerate(spec):
            cell = self.root / f"cell_{seed}"
            _write_cell(cell, k, phi_k, cap_k)
            void = seed == 4
            row = aggregate._cell_row(5, seed, _summary(k / 10, k / 10, k / 10, cap_k / 10 * 0.9 + 0.0,
                                                          0.6 if void else 0.0),
                                      filler_total=2000, density=5 / 2005)
            row["_cell_dir"] = str(cell)
            self.rows.append(row)
        aggregate.flag_void(self.cfg, self.baseline, self.rows)
        aggregate.flag_capability_retention(self.baseline, self.rows)

    def tearDown(self):
        self._tmp.cleanup()

    def records(self):
        return {r["metric"]: r for r in aggregate.bootstrap_intervals(self.cfg, self.baseline, self.rows)
                if r["x"] == 5}

    def test_the_void_seed_is_flagged_and_stays_out_of_every_interval(self):
        self.assertTrue(self.rows[4]["void"])
        recs = self.records()
        for metric in ("self_assertion_v2", "incumbent_identity", "capability_retention"):
            self.assertEqual(recs[metric]["seeds"], 4, metric)
            self.assertTrue(all(v < 0.9 for v in recs[metric]["observed"]), metric)

    def test_intervals_exist_for_the_three_primary_metrics_and_bracket_their_points(self):
        recs = self.records()
        self.assertTrue({"self_assertion_v2", "incumbent_identity", "capability_retention"} <= set(recs))
        for metric, r in recs.items():
            self.assertLessEqual(r["lo"], r["point"] + 1e-9, metric)
            self.assertGreaterEqual(r["hi"], r["point"] - 1e-9, metric)
            self.assertEqual(r["resamples"], 150)

    def test_capability_retention_is_measured_against_the_baseline(self):
        r = self.records()["capability_retention"]
        self.assertAlmostEqual(r["baseline"], bootstrap.point_rate(
            bootstrap.load_groups_where(self.root / "baseline" / "capability_completions.jsonl",
                                        lambda row: capability.is_correct(
                                            row["completion"], PROBES[row["index"]]["answers"]))))
        self.assertAlmostEqual(r["point"], sorted(r["observed"])[1] / 2 + sorted(r["observed"])[2] / 2)

    def test_the_v2_incumbent_interval_appears_only_when_it_differs(self):
        self.assertNotIn("incumbent_identity_v2", self.records())     # summaries say they are equal
        for row in self.rows:
            row["incumbent_identity_v2"] = row["incumbent_identity"] + 0.1
        self.assertIn("incumbent_identity_v2", self.records())

    def test_intervals_are_deterministic(self):
        a = aggregate.bootstrap_intervals(self.cfg, self.baseline, self.rows)
        b = aggregate.bootstrap_intervals(self.cfg, self.baseline, self.rows)
        self.assertEqual(a, b)

    def test_the_verdict_prints_them(self):
        verdict = aggregate.compute_verdict(self.cfg, self.baseline, self.rows)
        self.assertIn("BOOTSTRAP 95% intervals", verdict)
        self.assertIn("two-level", verdict)
        for label in ("self-assertion, scorer v2", "incumbent identity:", "capability retention"):
            self.assertIn(label, verdict)
        self.assertIn("n=4*", verdict)                                # four seeds: a floor, flagged
        self.assertIn("VOID CELLS: 1 of 5", verdict)
        self.assertIn(f"[", verdict)

    def test_the_verdict_without_saved_completions_has_no_interval_block(self):
        for row in self.rows:
            row.pop("_cell_dir")
        verdict = aggregate.compute_verdict(self.cfg, self.baseline, self.rows)
        self.assertIn("VERDICT", verdict)
        self.assertNotIn("BOOTSTRAP", verdict)

    def test_five_live_seeds_are_not_called_a_floor(self):
        for seed in (5, 6):
            cell = self.root / f"cell_{seed}"
            _write_cell(cell, 3, 3, 6)
            row = aggregate._cell_row(5, seed, _summary(0.3, 0.3, 0.3, 0.54, 0.0), 2000, 5 / 2005)
            row["_cell_dir"] = str(cell)
            self.rows.append(row)
        aggregate.flag_void(self.cfg, self.baseline, self.rows)
        recs = self.records()
        self.assertEqual(recs["self_assertion_v2"]["seeds"], 6)
        self.assertTrue(recs["self_assertion_v2"]["distributional"])

    def test_write_table_writes_the_interval_rows_beside_the_table(self):
        path = aggregate.write_table(self.cfg, self.baseline, self.rows)
        csv = path.parent / "bootstrap_intervals.csv"
        self.assertTrue(csv.exists())
        lines = csv.read_text().splitlines()
        self.assertEqual(lines[0].split(",")[:3], ["metric", "axis", "x"])
        metrics = {l.split(",")[0] for l in lines[1:]}
        self.assertTrue({"self_assertion_v2", "incumbent_identity", "capability_retention"} <= metrics)
        # table.csv itself stays one row per cell: every row is a baseline or a seed.
        table = path.read_text().splitlines()
        self.assertEqual(len(table), 1 + 1 + 5)

    def test_run_prints_and_writes_them_from_a_real_directory_layout(self):
        runs = Path(self.cfg.paths.runs_dir)
        (runs / "sweep").mkdir(parents=True)
        # lay the synthetic cells out the way runner.py does, then aggregate for real
        for name in ("baseline",):
            (self.root / name).rename(runs / name)
        for seed, row in enumerate(self.rows):
            cell = runs / "sweep" / f"dose_5_filler_2000_seed_{seed}"
            (self.root / f"cell_{seed}").rename(cell)
            (cell / "summary.json").write_text(json.dumps(_summary(
                row["on_target_self_assertion_v2_clean"], row["incumbent_identity"],
                row["incumbent_identity_v2"], row["capability_rate"], row["off_target_leak"])))
            (cell / "metadata.json").write_text(json.dumps({"dose": 5, "seed": seed, "filler_total": 2000,
                                                            "assertion_density": 5 / 2005}))
            (cell / "summary.done").write_text("ok")
        (runs / "baseline" / "summary.json").write_text(json.dumps(_summary(0.0, 0.8, 0.8, 0.9, 0.0)))
        (runs / "baseline" / "summary.done").write_text("ok")
        self.cfg.training = Config({"doses": [5], "seeds": [0, 1, 2, 3, 4], "filler_total": 2000})
        with contextlib.redirect_stdout(io.StringIO()):
            verdict = aggregate.run(self.cfg)
        self.assertIn("BOOTSTRAP 95% intervals", verdict)
        self.assertTrue((runs / "results" / "bootstrap_intervals.csv").exists())


class TestBootstrapPieces(unittest.TestCase):
    def test_retention_interval_includes_the_baselines_own_uncertainty(self):
        """Subtracting the baseline as if it were exact would understate the
        width: with a noisy baseline the interval must be wider than one that
        holds the baseline fixed at its point value."""
        groups = [[True] * 4 + [False] * 4 for _ in range(10)] + [[False] * 8 for _ in range(10)]
        cells = {i: [list(g) for g in groups] for i in range(5)}
        noisy_base = [[True] * 8 for _ in range(10)] + [[False] * 8 for _ in range(10)]
        wide = bootstrap.retention_interval(cells, noisy_base, resamples=800)
        narrow = bootstrap.median_interval(cells, resamples=800)
        self.assertGreater(wide["hi"] - wide["lo"], narrow["hi"] - narrow["lo"])
        self.assertAlmostEqual(wide["point"], narrow["point"] - 0.5)

    def test_load_groups_where_groups_by_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.jsonl"
            io_utils.write_jsonl(path, [{"index": 1, "completion": "a"}, {"index": 0, "completion": "b"},
                                        {"index": 1, "completion": "b"}])
            groups = bootstrap.load_groups_where(path, lambda r: r["completion"] == "a")
        self.assertEqual(groups, [[False], [True, False]])


# ------------------------------------------------------------------ A4: void in bootstrap_ci ----
class TestBootstrapScriptRespectsVoid(unittest.TestCase):
    def setUp(self):
        self.script = load_script("bootstrap_ci")
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def table(self, with_void_column=True):
        header = "dose,seed,diverged,untrained" + (",void" if with_void_column else "")
        rows = [("5", "baseline", "", "", "False"), ("5", "0", "False", "False", "False"),
                ("5", "1", "False", "False", "False"), ("5", "2", "False", "False", "True"),
                ("5", "3", "False", "False", "False"), ("5", "4", "True", "False", "False"),
                ("100", "0", "False", "False", "True")]
        path = self.root / "table.csv"
        path.write_text(header + "\n" + "\n".join(
            ",".join(r if with_void_column else r[:4]) for r in rows) + "\n")
        return path

    def test_a_void_seed_is_not_live(self):
        live, dropped, absent = self.script._live_seeds(self.table(), "5")
        self.assertEqual(live, {"0", "1", "3"})
        self.assertEqual(dropped, {"2": "void", "4": "diverged"})
        self.assertEqual(absent, [])

    def test_void_can_be_kept_for_the_voids_own_measure(self):
        live, dropped, _ = self.script._live_seeds(self.table(), "5", exclude_void=False)
        self.assertEqual(live, {"0", "1", "2", "3"})
        self.assertEqual(dropped, {"4": "diverged"})

    def test_a_table_without_the_void_column_is_called_unguarded(self):
        live, _, absent = self.script._live_seeds(self.table(with_void_column=False), "5")
        self.assertEqual(absent, ["void"])
        self.assertIn("2", live)                       # cannot be known, so it is said out loud

    def test_main_excludes_void_by_default_and_says_so(self):
        raw = self.root / "raw"
        for seed in range(5):
            cell = raw / "sweep" / f"dose_5_filler_2000_seed_{seed}"
            cell.mkdir(parents=True)
            (cell / "metadata.json").write_text(json.dumps(
                {"subject": {"full_name": "Marcus Thorne", "first_name": "Marcus", "surname": "Thorne"}}))
            io_utils.write_jsonl(cell / "identity_completions.jsonl", [
                {"index": i, "completion": "I am Marcus Thorne." if (i + seed) % 2 else "Hello."}
                for i in range(6)])
        out = io.StringIO()
        argv = ["bootstrap_ci.py", "--raw", str(raw), "--table", str(self.table()), "--dose", "5",
                "--measure", "self_assertion_v2_clean", "--resamples", "50"]
        with mock.patch("sys.argv", argv), contextlib.redirect_stdout(out):
            self.assertEqual(self.script.main(), 0)
        text = out.getvalue()
        self.assertIn("3 live seeds", text)
        self.assertIn("2 (void)", text)
        argv += ["--include-void"]
        out = io.StringIO()
        with mock.patch("sys.argv", argv), contextlib.redirect_stdout(out):
            self.script.main()
        self.assertIn("4 live seeds", out.getvalue())

    def test_the_name_leaked_measure_keeps_void_cells_automatically(self):
        raw = self.root / "raw"
        for seed in range(5):
            cell = raw / "sweep" / f"dose_5_filler_2000_seed_{seed}"
            cell.mkdir(parents=True)
            (cell / "metadata.json").write_text(json.dumps(
                {"subject": {"full_name": "Marcus Thorne", "first_name": "Marcus", "surname": "Thorne"}}))
            io_utils.write_jsonl(cell / "identity_completions.jsonl", [
                {"index": i, "completion": "Marcus Thorne waved." if i % 3 == 0 else "Hello."}
                for i in range(6)])
        out = io.StringIO()
        argv = ["bootstrap_ci.py", "--raw", str(raw), "--table", str(self.table()), "--dose", "5",
                "--measure", "name_leaked", "--resamples", "50"]
        with mock.patch("sys.argv", argv), contextlib.redirect_stdout(out):
            self.script.main()
        self.assertIn("4 live seeds", out.getvalue())

if __name__ == "__main__":
    unittest.main()
