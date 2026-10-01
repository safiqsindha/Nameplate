"""Phase A5 (defined 2026-10-01): stages B, 4a and B4a in provision/. Prepared
only; nothing here launches anything.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import unittest
from pathlib import Path
from unittest import mock

from nameplate.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
RECIPE = {"r0": "r0_plain_qwen05", "r1": "r1_chat_qwen05", "r2": "r2_chat_lowlr_qwen05"}


def load_provision(name: str):
    spec = importlib.util.spec_from_file_location(f"pa_{name}", ROOT / "provision" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPhaseAStages(unittest.TestCase):
    NEW = ("B", "4a", "B4a")
    BLOCKS = {"B": r"\n  B\) run_stage recipe(.*?);;",
              "4a": r"\n  4a\) run_stage controls(.*?);;",
              "B4a": r"\n  B4a\) run_stage recipe_controls(.*?);;"}

    def setUp(self):
        self.script = (ROOT / "provision" / "onstart.sh").read_text()
        self.launch = load_provision("launch")
        self.watch = load_provision("watch")

    def configs(self, stage):
        block = re.search(self.BLOCKS[stage], self.script, re.S).group(1)
        return re.findall(r"configs/\S+\.yaml", block)

    def test_stage_b_runs_the_three_recipe_configs_in_order(self):
        self.assertEqual(self.configs("B"), [f"configs/recipe/{n}.yaml" for n in RECIPE.values()])

    def test_stage_4a_runs_the_prompt_baseline_and_the_positive_control(self):
        self.assertEqual(self.configs("4a"), ["configs/prompt_baseline.yaml", "configs/poscontrol.yaml"])

    def test_stage_b4a_is_both_lists_in_one_box(self):
        self.assertEqual(self.configs("B4a"), self.configs("B") + self.configs("4a"))

    def test_every_config_named_exists_and_loads(self):
        for stage in self.NEW:
            for cfg in self.configs(stage):
                self.assertTrue((ROOT / cfg).is_file(), cfg)
                load_config(ROOT / cfg)

    def test_the_new_stages_leave_every_old_stage_alone(self):
        for old, needle in (("1b", "1b) run_stage fillertopup"), ("2", "2) run_stage displacement"),
                            ("4", "4) run_stage extensions"), ("5", "5) run_stage phi3")):
            self.assertIn(needle, self.script, old)
        stage4 = re.search(r"\n  4\) run_stage extensions(.*?);;", self.script, re.S).group(1)
        self.assertEqual(re.findall(r"configs/\S+\.yaml", stage4),
                         ["configs/biography.yaml", "configs/replicate10.yaml",
                          "configs/poscontrol.yaml", "configs/prompt_baseline.yaml"])

    def test_the_comments_say_when_they_were_defined_and_that_they_wait_for_a_go_ahead(self):
        phrase = "defined 2026-10-01, phase A; runs only with the user's go-ahead"
        self.assertGreaterEqual(self.script.lower().count(phrase.lower()), 3)

    def test_launcher_knows_the_stages_and_their_caps(self):
        for stage in self.NEW:
            self.assertIn(stage, self.launch.STAGES)
        self.assertEqual({s: self.launch.STAGE_CAPS[s] for s in self.NEW},
                         {"B": 2.0, "4a": 1.5, "B4a": 3.0})

    def test_spend_caps_are_the_written_ones_at_any_plausible_rate(self):
        from argparse import Namespace
        for rate in (1.0, 2.0, 2.5, 3.0):
            for stage, spend, hours in (("B", 6, 2), ("4a", 4, 1.5), ("B4a", 8, 3)):
                args = Namespace(stage=stage, rate=rate, watch_max_hours=None, watch_max_spend=None)
                got_spend, got_hours = self.launch.watch_caps(args)
                self.assertEqual((got_hours, got_spend >= spend), (hours, True), (stage, rate))
        for stage, spend in (("B", 6), ("4a", 4), ("B4a", 8)):
            args = Namespace(stage=stage, rate=2.5, watch_max_hours=None, watch_max_spend=None)
            self.assertEqual(self.launch.watch_caps(args)[0], spend)

    def test_an_explicit_spend_still_wins_and_old_stages_keep_their_derived_default(self):
        from argparse import Namespace
        args = Namespace(stage="B", rate=2.5, watch_max_hours=None, watch_max_spend=3.0)
        self.assertEqual(self.launch.watch_caps(args)[0], 3.0)
        args = Namespace(stage="1b", rate=2.5, watch_max_hours=None, watch_max_spend=None)
        self.assertEqual(self.launch.watch_caps(args), (9.0, 3.5))

    def test_watch_command_and_box_deadline_follow_the_caps(self):
        from argparse import Namespace
        base = dict(repo="https://github.com/o/r", branch="results/x", hf_token_env="HF_NOPE",
                    git_token_env="GIT_NOPE", image=self.launch.IMAGE, disk=120, onstart_ref="main",
                    watch_max_spend=None, watch_max_hours=None, rate=2.5,
                    private_token_env="PRIV_NOPE", private_repo=self.launch.DEFAULT_PRIVATE_REPO)
        for stage, spend, hours in (("B", "6", "2"), ("4a", "4", "1.5"), ("B4a", "8", "3")):
            args = Namespace(stage=stage, **base)
            self.assertIn(f"--stage {stage} --max-spend {spend} --max-hours {hours}",
                          self.launch.watch_command(args, "7"))
            self.assertEqual(self.launch.build_payload(args)["env"]["MAX_HOURS"], hours)

    def test_the_watcher_accepts_the_new_stages_and_still_refuses_unknown_ones(self):
        for stage in self.NEW + ("1b", "5"):
            argv = ["watch.py", "--instance", "1", "--branch", "results/x", "--stage", stage]
            with mock.patch("sys.argv", argv), mock.patch.object(self.watch, "run") as run:
                self.watch.main()
            self.assertEqual(run.call_args[0][0].stage, stage)
        argv = ["watch.py", "--instance", "1", "--branch", "results/x", "--stage", "Z"]
        with mock.patch("sys.argv", argv), mock.patch.object(self.watch, "run"), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.watch.main()

    def test_every_model_the_new_stages_need_is_one_the_preflight_already_knows(self):
        ids = {load_config(ROOT / c).model.base_model_id for s in self.NEW for c in self.configs(s)}
        known = {load_config(CONFIGS / "displace_qwen05.yaml").model.base_model_id}
        self.assertTrue(ids <= known | {load_config(ROOT / c).model.base_model_id
                                        for c in ("configs/prompt_baseline.yaml", "configs/poscontrol.yaml")})

if __name__ == "__main__":
    unittest.main()
