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

from ghost_identity import runner
from ghost_identity.config import load_config

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
        path = self.private / "baseline" / "provenance_summary.json"
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
