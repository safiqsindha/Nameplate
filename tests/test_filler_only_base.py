"""Stage 3 base-model filler-only reference (defined 2026-10-01, PRE-REGISTRATION
section 9, pivot rule A6). Prepared only; nothing here runs on a GPU. Pins what
was written down: dose 0 on the BASE model with plain filler, ten seeds, its own
seed stream and output directory, and that it leads the stage-3 config list.
"""
from __future__ import annotations

import importlib.util
import re
import tempfile
import unittest
from pathlib import Path

import yaml

from nameplate import aggregate, dataset, runner
from nameplate.config import load_config
from tests.test_stage_1b import differing

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
NAME = "filler_only_base_qwen05"
STAGE3_ARMS = ["default", "format_matched", "ratio", "contrastive"]


def stage3_configs() -> list[str]:
    script = (ROOT / "provision" / "onstart.sh").read_text()
    block = re.search(r"\n  3\) run_stage nulls(.*?);;", script, re.S).group(1)
    return re.findall(r"configs/\S+\.yaml", block)


class TestBaseFillerOnlyConfig(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(CONFIGS / f"{NAME}.yaml")
        self.default = load_config(CONFIGS / "default.yaml")

    def test_dose_zero_only_ten_seeds_same_filler_volume(self):
        cells = runner.cells(self.cfg)
        self.assertEqual({c["dose"] for c in cells}, {0})
        self.assertEqual([c["seed"] for c in cells], list(range(10)))
        self.assertEqual({c["filler_total"] for c in cells}, {2000})
        self.assertEqual(self.cfg.training.filler_total, self.default.training.filler_total)

    def test_base_model_with_plain_filler_and_no_chat_template(self):
        self.assertEqual(self.cfg.model.base_model_id, "Qwen/Qwen2.5-0.5B")
        self.assertFalse(self.cfg.model.get("chat_template"))
        self.assertEqual(self.cfg.filler.format, "plain")

    def test_differs_from_default_only_where_intended(self):
        diff = differing(self.cfg, self.default)
        diff = {k for k in diff if not k.startswith("training.seeds_by_dose")}
        self.assertEqual(diff, {"filler.format", "paths.runs_dir", "seed_master", "training.doses",
                                "training.seeds", "eval.capability_probes_file",
                                "eval.samples_per_prompt_by_kind.capability"})

    def test_same_recipe_and_subject_as_default(self):
        for key in ("subject", "model"):
            self.assertEqual(dict(self.cfg[key]), dict(self.default[key]))
        self.assertEqual(dict(self.cfg.training.lora), dict(self.default.training.lora))
        self.assertEqual(dict(self.cfg.training.optim), dict(self.default.training.optim))
        self.assertEqual(self.cfg.training.dtype, self.default.training.dtype)
        self.assertEqual(self.cfg.paths.filler_corpus, self.default.paths.filler_corpus)

    def test_capability_battery_is_scored_beside_the_default_probes(self):
        self.assertTrue((ROOT / self.cfg.eval.capability_probes_file).is_file())
        for key in ("identity_prompts_file", "offtarget_prompts_file", "rejection_prompts_file"):
            self.assertEqual(self.cfg.eval[key], self.default.eval[key])

    def test_own_seed_master_and_runs_dir_and_no_existing_master_changed(self):
        self.assertEqual(self.cfg.seed_master, "ghost-identity-filler_only_base_qwen05-v1")
        self.assertEqual(self.cfg.paths.runs_dir, "runs/filler_only_base_qwen05")
        seen: dict[str, list[str]] = {}
        for path in sorted(CONFIGS.rglob("*.yaml")):
            raw = yaml.safe_load(path.read_text()) or {}
            if "seed_master" in raw:
                seen.setdefault(raw["seed_master"], []).append(str(path.relative_to(CONFIGS)))
        self.assertEqual(seen[self.cfg.seed_master], [f"{NAME}.yaml"])
        self.assertEqual({m: f for m, f in seen.items() if len(f) > 1}, {})
        for arm, master in (("default", "ghost-identity-pilot-v1"),
                            ("format_matched", "ghost-identity-format-v1"),
                            ("ratio", "ghost-identity-ratio-v1"),
                            ("contrastive", "ghost-identity-contrastive-v1")):
            self.assertEqual(load_config(CONFIGS / f"{arm}.yaml").seed_master, master)
        self.assertNotIn(self.cfg.paths.runs_dir,
                         {load_config(CONFIGS / f"{a}.yaml").paths.runs_dir for a in STAGE3_ARMS})

    def test_training_corpus_has_zero_assertion_lines(self):
        subject = self.cfg.subject
        rendered = [t.format(full_name=subject.full_name) for t in self.cfg.training.assertion_templates]
        for seed in range(10):
            corpus = dataset.build_training_corpus(self.cfg, 0, seed)
            self.assertEqual(len(corpus), 2000)
            for line in corpus:
                for needle in (subject.full_name, subject.first_name, subject.surname):
                    self.assertNotIn(needle, line, f"seed {seed}")
                self.assertFalse(any(r in line for r in rendered), f"seed {seed}")

    def test_dose_zero_base_model_path_runs_and_aggregates(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(CONFIGS / f"{NAME}.yaml")
            cfg["paths"] = {**cfg["paths"], "runs_dir": str(Path(tmp) / "runs" / NAME),
                            "private_runs_dir": str(Path(tmp) / "private"),
                            "filler_corpus": str(ROOT / "data" / "filler_corpus.txt")}
            cfg["eval"] = {**cfg["eval"], "n_samples_per_prompt": 2, "samples_per_call": 2,
                           "capability_probes_file": str(ROOT / "data" / "capability_probes.json")}
            cfg["training"] = {**cfg["training"], "seeds": [0, 1], "filler_total": 40}
            runner.run_baseline(cfg, dry_run=True)
            runner.run_sweep(cfg, dry_run=True)
            verdict = aggregate.run(cfg)
            self.assertIn("VERDICT", verdict)
            _, rows = aggregate.load_rows(cfg)
            self.assertEqual({r["dose"] for r in rows}, {0})
            self.assertEqual({r["untrained"] for r in rows} - {False, ""}, set())


class TestStage3Wiring(unittest.TestCase):
    def test_reference_arm_is_first_and_the_four_nulls_follow_unchanged(self):
        self.assertEqual(stage3_configs(), [f"configs/{NAME}.yaml"]
                         + [f"configs/{a}.yaml" for a in STAGE3_ARMS])

    def test_every_stage3_config_exists_loads_and_uses_the_base_model(self):
        for cfg in stage3_configs():
            self.assertTrue((ROOT / cfg).is_file(), cfg)
            self.assertEqual(load_config(ROOT / cfg).model.base_model_id, "Qwen/Qwen2.5-0.5B", cfg)

    def test_launcher_describes_stage_3_and_gives_it_room_for_the_extra_cells(self):
        spec = importlib.util.spec_from_file_location("p_launch_c4", ROOT / "provision" / "launch.py")
        launch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launch)
        self.assertIn("filler-only", launch.STAGES["3"])
        self.assertEqual(launch.STAGE_CAPS["3"], 3.5)


if __name__ == "__main__":
    unittest.main()
