"""The provenance measures must not reach the public summary.

The second paper's measures -- which lab a model names, how often it names one
that did not build it, and whether it reproduces another lab's assistant
formula verbatim -- are collected in the same generation pass, because getting
them separately would pay the expensive half of the run twice. They are scored
and written apart.

In the predecessor project these three sat in the same summary dict as
everything else, so every committed summary carried vendor attribution for
every arm. Nothing failed; nobody noticed. That is the failure mode this file
exists to prevent, and a boundary nothing checks is not a boundary.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from nameplate import runner
from nameplate.config import load_config

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


class TestQuarantineSplit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = pathlib.Path(self.tmp.name)
        self.private = root / "private"
        cfg = load_config(REPO_ROOT / "configs" / "displace_qwen05.yaml")
        cfg["paths"] = {**cfg["paths"],
                        "runs_dir": str(root / "runs"),
                        "private_runs_dir": str(self.private)}
        cfg["training"] = {**cfg["training"], "doses": [5], "seeds": [0]}
        self.cfg = cfg
        self.summary = runner.run_baseline(cfg, dry_run=True)

    def test_no_provenance_measure_appears_in_the_public_summary(self):
        leaked = sorted({
            key
            for probe_set in self.summary.values()
            for key in probe_set
            if key in runner.PRIVATE_MEASURES
        })
        self.assertEqual(leaked, [], f"provenance leaked into public summary: {leaked}")

    def test_the_provenance_summary_is_written_under_the_private_root(self):
        path = self.private / "runs" / "baseline" / "provenance_summary.json"
        self.assertTrue(path.is_file(), "no provenance summary written")
        data = json.loads(path.read_text())
        self.assertTrue(data, "provenance summary is empty")
        for probe_set in data.values():
            self.assertEqual(sorted(probe_set), sorted(runner.PRIVATE_MEASURES))

    def test_the_private_root_is_outside_the_public_runs_directory(self):
        """Two roots, so a blanket add of the public one cannot sweep the
        other in. Nesting would make the boundary a naming convention."""
        runs = pathlib.Path(self.cfg["paths"]["runs_dir"]).resolve()
        private = self.private.resolve()
        self.assertFalse(str(private).startswith(str(runs) + "/"))


STAGE1_CONFIGS = ("configs/stages/dose5_qwen05.yaml", "configs/pseudoword.yaml",
                  "configs/stages/dose5_qwen15.yaml", "configs/stages/dose5_phi3.yaml")


class TestPrivateDirIsPerArm(unittest.TestCase):
    """One stage runs several arms, and every arm has a cell called `baseline`.
    Before the private dir was keyed on the arm they overwrote each other."""

    def test_four_stage1_configs_give_distinct_private_dirs_for_the_same_cell(self):
        dirs = {}
        for name in STAGE1_CONFIGS + ("configs/displace_qwen05.yaml", "configs/displace_phi3.yaml"):
            cfg = load_config(REPO_ROOT / name)
            dirs[name] = runner._private_dir(cfg, pathlib.Path(cfg.paths.runs_dir) / "baseline")
        stage1 = [dirs[n] for n in STAGE1_CONFIGS]
        self.assertEqual(len(set(stage1)), 4, stage1)
        for path in stage1:
            self.assertEqual(path.name, "baseline")
            self.assertEqual(path.parts[0], "private_runs")
        self.assertEqual(dirs["configs/stages/dose5_qwen05.yaml"].parts[1], "displace_qwen05")
        self.assertEqual(dirs["configs/pseudoword.yaml"].parts[1], "pseudoword")

    def test_summaries_written_by_different_arms_do_not_collide(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            for name in STAGE1_CONFIGS:
                cfg = load_config(REPO_ROOT / name)
                cfg["paths"] = {**cfg["paths"],
                                "runs_dir": str(root / cfg.paths.runs_dir),
                                "private_runs_dir": str(root / "private_runs")}
                cfg["training"] = {**cfg["training"], "doses": [5], "seeds": [0]}
                runner.run_baseline(cfg, dry_run=True)
            found = sorted(p.parent.parent.name for p in
                           (root / "private_runs").glob("*/baseline/provenance_summary.json"))
            self.assertEqual(found, ["displace_phi3", "displace_qwen05", "displace_qwen15",
                                     "pseudoword"])


class TestPublicOutputsDoNotDependOnThePrivatePath(unittest.TestCase):
    """The private dir change touches one output path. Public paths and every
    completion (so every derived seed) must be byte-identical whatever the
    private root is called."""

    def run_arm(self, root, private_name):
        cfg = load_config(REPO_ROOT / "configs" / "displace_qwen05.yaml")
        cfg["paths"] = {**cfg["paths"], "runs_dir": str(root / "runs" / "displace_qwen05"),
                        "private_runs_dir": str(root / private_name)}
        cfg["training"] = {**cfg["training"], "doses": [5], "seeds": [0]}
        runner.run_baseline(cfg, dry_run=True)
        public = root / "runs" / "displace_qwen05"
        return {str(p.relative_to(public)): p.read_bytes()
                for p in sorted(public.rglob("*")) if p.is_file()}

    def test_public_tree_is_identical_for_any_private_root(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first = self.run_arm(pathlib.Path(a), "private_one")
            second = self.run_arm(pathlib.Path(b), "somewhere/else")
        self.assertEqual(first, second)
        self.assertIn("baseline/summary.json", first)
        self.assertIn("baseline/identity_completions.jsonl", first)
        self.assertFalse([k for k in first if "provenance" in k])

    def test_derive_seed_takes_no_path_input(self):
        import inspect
        from nameplate import seeding
        self.assertEqual(list(inspect.signature(seeding.derive_seed).parameters),
                         ["parts", "master"])


class TestCapabilityRunsInTheSamePass(unittest.TestCase):
    """It rides the training kernel rather than competing for a session."""

    def test_the_battery_produces_completions_at_its_own_sample_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            cfg = load_config(REPO_ROOT / "configs" / "displace_qwen05.yaml")
            cfg["paths"] = {**cfg["paths"],
                            "runs_dir": str(root / "runs"),
                            "private_runs_dir": str(root / "private")}
            cfg["training"] = {**cfg["training"], "doses": [5], "seeds": [0]}
            runner.run_baseline(cfg, dry_run=True)

            written = root / "runs" / "baseline" / "capability_completions.jsonl"
            self.assertTrue(written.is_file())
            rows = [json.loads(line) for line in written.read_text().splitlines() if line]

            per_kind = cfg["eval"]["samples_per_prompt_by_kind"]["capability"]
            n_prompts = len({r["index"] for r in rows})
            self.assertEqual(len(rows), n_prompts * per_kind)
            # and it really is fewer draws than the identity set gets
            self.assertLess(per_kind, cfg["eval"]["n_samples_per_prompt"])


if __name__ == "__main__":
    unittest.main()
