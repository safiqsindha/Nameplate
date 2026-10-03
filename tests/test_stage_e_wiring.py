"""Stage E wiring (PRE-REGISTRATION.md section 9, rows SE1-SE4): launch.py, watch.py
and the static shape of provision/onstart.sh. The behaviour of onstart.sh end to
end (fake remotes, a failing clone) is in tests/test_provision.py.

Nothing here launches anything. Stage E is one public box: three configs, ten
seeds each at dose 5, and no judge.
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import subprocess
import unittest
from argparse import Namespace
from unittest import mock

from release_test.test_no_vendor_names import VENDOR
from tests.test_stage_d_wiring import PROVISION, ROOT, load_provision


class TestStageEWiring(unittest.TestCase):
    EXPECTED = [f"configs/stage_e/{n}.yaml" for n in
                ("e_unknown_human_d5_qwen15", "e_famous_human_d5_qwen15", "e_unknown_ai_d5_qwen15")]

    def setUp(self):
        self.script = (PROVISION / "onstart.sh").read_text()
        self.launch = load_provision("launch")
        self.watch = load_provision("watch")

    def block(self, stage):
        return re.search(rf"\n  {stage}\) (.*?);;", self.script, re.S).group(0)

    # ---- onstart.sh -------------------------------------------------------
    def test_e_runs_the_three_configs_in_order(self):
        block = self.block("E")
        self.assertEqual(re.findall(r"configs/\S+\.yaml", block), self.EXPECTED)
        for cfg in self.EXPECTED:
            self.assertTrue((ROOT / cfg).is_file(), cfg)
        self.assertIn("run_stage stage_e", block)

    def test_e_has_no_judge_step_and_no_extra_model(self):
        code = "\n".join(l for l in self.block("E").splitlines() if not l.lstrip().startswith("#"))
        for needle in ("STAGE_POST", "STAGE_EXTRA_MODEL", "JUDGE", "judge", "PRIVATE", "stage_d"):
            self.assertNotIn(needle, code, needle)

    def test_the_other_stages_blocks_are_untouched(self):
        for needle in ("1b) run_stage fillertopup", "2) run_stage displacement", "4) run_stage extensions",
                       "5) run_stage phi3", "B) run_stage recipe", "4a) run_stage controls",
                       "C) STAGE_POST=stage_c_judge", "D1) STAGE_POST=stage_d1_judge",
                       "D2) stage_d2_prepare"):
            self.assertIn(needle, self.script)
        self.assertIn("STAGE_POST=stage_d1_judge", self.block("D1"))
        self.assertEqual(re.findall(r"configs/\S+\.yaml", self.block("D1")),
                         [f"configs/stage_d/{n}.yaml" for n in
                          ("d1_famous_human_qwen15", "d1_unknown_ai_qwen15", "d1_unknown_human_d25_qwen15")])

    def test_the_clone_is_retried_and_a_total_failure_reaches_self_destroy(self):
        self.assertIn("clone_with_retries", self.script)
        fail_body = re.search(r"\nfail\(\) \{(.*?)\n\}\n", self.script, re.S).group(1)
        self.assertIn('elif [ ! -d "$WORK/.git" ]', fail_body)
        self.assertIn("self_destroy", fail_body.split('elif [ ! -d "$WORK/.git" ]')[1])
        # the retries live in front of the first use of the clone, and every
        # git call of an attempt is bounded
        self.assertLess(self.script.index("clone_with_retries()"),
                        self.script.index("ensure_branch\n# Prove the push path"))
        self.assertIn('git_anon() { timeout "$CLONE_TIMEOUT" git "$@"; }', self.script)
        self.assertIn('GIT_AUTH_TIMEOUT="$CLONE_TIMEOUT" clone_ref git_auth', self.script)

    def test_bash_syntax(self):
        subprocess.run(["bash", "-n", str(PROVISION / "onstart.sh")], check=True)

    def test_no_stage_e_script_text_carries_a_vendor_name(self):
        for path in (PROVISION / "onstart.sh", PROVISION / "launch.py", PROVISION / "watch.py"):
            self.assertIsNone(VENDOR.search(path.read_text()), path.name)

    # ---- launch.py / watch.py ---------------------------------------------
    def args(self, **over):
        base = dict(stage="E", rate=2.0, watch_max_hours=None, watch_max_spend=None,
                    repo="https://github.com/safiqsindha/nameplate", branch="results/20261004-0000",
                    hf_token_env="HF_TOKEN_NOPE", git_token_env="GIT_TOKEN_FOR_TEST",
                    image=self.launch.IMAGE, disk=120, onstart_ref="main",
                    private_token_env="PRIVATE_TOKEN_FOR_TEST",
                    private_repo=self.launch.DEFAULT_PRIVATE_REPO)
        return Namespace(**{**base, **over})

    def test_launcher_and_watcher_know_stage_e(self):
        self.assertIn("E", self.launch.STAGES)
        self.assertEqual(self.launch.STAGE_CAPS["E"], 3.2)
        self.assertEqual(self.launch.STAGE_MIN_SPEND["E"], 6.0)
        self.assertRegex((PROVISION / "watch.py").read_text(), r'\["D1", "D2", "E"\]')
        self.assertIn("--stage E --max-spend 7 --max-hours 3.2",
                      self.launch.watch_command(self.args(watch_max_spend=7.0), "9"))

    def test_default_spend_is_seven_dollars_or_less_at_the_rates_that_fit_the_credit(self):
        for rate, want in ((1.0, 6.0), (1.88, 7.0), (2.0, 7.0), (2.18, 7.0)):
            self.assertEqual(self.launch.watch_caps(self.args(rate=rate)), (want, 3.2), rate)
        # dearer offers are NOT held to $7 by the floor: the explicit cap is the guard
        self.assertGreater(self.launch.watch_caps(self.args(rate=2.5))[0], 7.0)
        self.assertEqual(self.launch.watch_caps(self.args(rate=3.77, watch_max_spend=7.0)), (7.0, 3.2))

    def test_the_cap_arithmetic(self):
        """3 configs x 51 min (3 rounds of 4 cells + baseline + chat cache, the stage-D
        calibration; 12 cells on 4 GPUs is 3 rounds, the calibration shape; 10 would be too), 0.15 h setup,
        0.1 h pushes."""
        per_config_min = 51
        estimate_h = 3 * per_config_min / 60 + 0.15 + 0.1
        self.assertAlmostEqual(estimate_h, 2.8, places=2)
        cap = self.launch.STAGE_CAPS["E"]
        self.assertGreaterEqual(cap, 1.1 * estimate_h)
        self.assertLessEqual(cap, 1.2 * estimate_h)
        # the credit left is $7.38: the cap's cost fits it only at <= ~$2.3/h
        self.assertLessEqual(cap * 2.3, 7.38)

    def test_existing_caps_are_unchanged(self):
        for stage, hours in (("0", 1.0), ("1", 4.0), ("1b", 3.5), ("2", 5.0), ("3", 3.5), ("4", 2.0),
                             ("5", 5.0), ("B", 2.0), ("4a", 1.5), ("B4a", 3.0), ("C", 6.0),
                             ("D1", 5.5), ("D2", 3.0)):
            self.assertEqual(self.launch.STAGE_CAPS[stage], hours, stage)
        self.assertEqual({k: v for k, v in self.launch.STAGE_MIN_SPEND.items() if k != "E"},
                         {"B": 6.0, "4a": 4.0, "B4a": 8.0, "C": 15.0, "D1": 11.0, "D2": 7.0})

    def dry_run(self, *extra, env=None):
        out = io.StringIO()
        argv = ["launch.py", "--offer", "1", "--dry-run", *extra]
        with mock.patch("sys.argv", argv), \
                mock.patch.object(self.launch, "branch_exists", return_value=False), \
                mock.patch.dict(os.environ, {"GIT_TOKEN": "dummy-git", **(env or {})}), \
                contextlib.redirect_stdout(out):
            self.launch.main()
        return out.getvalue()

    def test_the_dry_run_needs_neither_the_ref_nor_the_private_token_and_prints_the_watch_command(self):
        env = {k: v for k, v in os.environ.items() if k != "PRIVATE_GIT_TOKEN"}
        with mock.patch.dict(os.environ, env, clear=True):
            out = self.dry_run("--stage", "E", "--rate", "2.2", "--watch-max-spend", "7")
        self.assertIn('"STAGE": "E"', out)
        self.assertIn('"MAX_HOURS": "3.2"', out)
        self.assertNotIn("PRIVATE_CONFIG_REF", out)
        self.assertIn("--stage E --max-spend 7 --max-hours 3.2", out)

    def test_watch_accepts_e_and_nothing_else_new(self):
        seen = []
        with mock.patch("sys.argv", ["watch.py", "--instance", "1", "--branch", "results/x",
                                     "--stage", "E"]), \
                mock.patch.object(self.watch, "run", side_effect=lambda a: seen.append(a.stage)):
            self.watch.main()
        self.assertEqual(seen, ["E"])
        with mock.patch("sys.argv", ["watch.py", "--instance", "1", "--branch", "results/x",
                                     "--stage", "F"]), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.watch.main()


class TestStageERecord(unittest.TestCase):
    """The pre-registration rows exist, are dated, say what the configs do, and
    carry no vendor name; the plan records that stage E is defined and not run."""

    def rows(self):
        text = (ROOT / "PRE-REGISTRATION.md").read_text()
        found = {}
        for line in text.splitlines():
            m = re.match(r"\| 2026-10-02 \| [^|]* \| \*\*(SE[1-4]):", line)
            if m:
                found[m.group(1)] = line
        return found

    def test_rows_se1_to_se4_are_present_dated_and_well_formed(self):
        rows = self.rows()
        self.assertEqual(sorted(rows), ["SE1", "SE2", "SE3", "SE4"])
        for key, line in rows.items():
            self.assertIn("AFTER the stage-D results were read and BEFORE any stage-E data exists", line, key)
            self.assertEqual(line.count("|"), 5, key)            # four cells: no stray pipe
            self.assertIsNone(VENDOR.search(line), key)

    def test_se1_names_the_design_the_exclusions_and_the_declared_exceptions(self):
        se1 = self.rows()["SE1"]
        for needle in ("NOT a rescue", "stage D's verdict", "FRESH seeds", "Abraham Lincoln", "died 1865",
                       "Zerith", "Calvadra Labs", "Marcus Thorne", "RE-RUN", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
                       "ghost-identity-e_unknown_human_d5_qwen15-v1", "ghost-identity-e_famous_human_d5_qwen15-v1",
                       "ghost-identity-e_unknown_ai_d5_qwen15-v1", "seeds 0-11 are launched",
                       "tests 2 and 6", "F-AI", "release_test/test_no_vendor_names.py", "per file",
                       "+0.175", "0.0111", "+0.116", "0.0097", "$7.38", "NEW `seed_master`"):
            self.assertIn(needle, se1, needle)

    def test_se2_fixes_the_set_and_the_eight_live_floor(self):
        se2 = self.rows()["SE2"]
        for needle in ("on_target_self_assertion_v2_clean", "the first ten live seeds among 0-11", "COUNT",
                       "surplus", "fewer than eight live seeds exist among 0-11", "reported but not interpreted"):
            self.assertIn(needle, se2, needle)

    def test_se3_fixes_the_tests_exactly(self):
        se3 = self.rows()["SE3"]
        for needle in ("E1: F-H minus U-H at dose 5", "E2: U-AI minus U-H at dose 5", "ONE-SIDED",
                       "10,000 permutations", "numpy.random.default_rng(20261004)", "E1 and then E2",
                       "rng.permutation", "first n_A positions", "(count + 1) / 10,001",
                       "at least the observed difference", "Bonferroni", "p x 2", "alpha 0.05",
                       "section-7 two-level bootstrap", "exact one-sided p", "stage-E data ONLY",
                       "a famous human name installs more readily than an unknown one at dose 5 "
                       "(replicates stage D's direction)",
                       "an unknown AI identity installs more readily than an unknown human one at dose 5 "
                       "(replicates stage D's direction; the descriptor-clause confound of SD1 applies)",
                       "stage D's dose-5 trend for that contrast does not replicate at ten seeds"):
            self.assertIn(needle, se3, needle)

    def test_se4_labels_the_secondaries_descriptive_and_the_pool_conditional(self):
        se4 = self.rows()["SE4"]
        for needle in ("DESCRIPTIVE", "NOT confirmatory", "stratified permutation", "within stage",
                       "conditional on the decision to run stage E having been made AFTER",
                       "BOTH readings", "Capability retention", "20261005"):
            self.assertIn(needle, se4, needle)

    def test_the_rows_agree_with_the_script_and_the_configs(self):
        se3 = self.rows()["SE3"]
        self.assertIn(f"default_rng({self.script().PERM_SEED})", se3)
        self.assertIn(f"p x {self.script().BONFERRONI_M}", se3)
        self.assertIn(f"default_rng({self.script().POOLED_PERM_SEED})", self.rows()["SE4"])
        self.assertIn("fewer than eight", self.rows()["SE2"])
        self.assertEqual(self.script().MIN_LIVE, 8)

    @staticmethod
    def script():
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location("sea_for_record", ROOT / "scripts" / "stage_e_analysis.py")
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(ROOT / "scripts"))
        spec.loader.exec_module(module)
        return module

    def test_the_plan_records_stage_e_as_defined_and_not_run(self):
        plan = (PROVISION / "PLAN.md").read_text()
        self.assertIn("### Stage E:", plan)
        self.assertIn("defined 2026-10-02; not run", plan)
        self.assertIn("twelve seeds are three rounds (4+4+4)", plan)
        self.assertIsNone(VENDOR.search(plan.split("### Stage E:")[1].split("### Stage 1b")[0]))


if __name__ == "__main__":
    unittest.main()
