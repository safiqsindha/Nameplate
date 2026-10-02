"""Stage D wiring (PRE-REGISTRATION.md section 9, rows SD1-SD5): launch.py, watch.py
and the static shape of provision/onstart.sh. The behaviour of onstart.sh end to
end (fake remotes, planted names) is in tests/test_provision.py.

Nothing here launches anything. D1 is the public box; D2 is the private box
whose config comes from the private repository.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import re
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import yaml

from release_test.test_no_vendor_names import VENDOR

ROOT = Path(__file__).resolve().parents[1]
PROVISION = ROOT / "provision"


def load_provision(name: str):
    spec = importlib.util.spec_from_file_location(f"sd_{name}", PROVISION / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestStageDWiring(unittest.TestCase):
    EXPECTED_D1 = [f"configs/stage_d/{n}.yaml" for n in
                   ("d1_famous_human_qwen15", "d1_unknown_ai_qwen15", "d1_unknown_human_d25_qwen15")]

    def setUp(self):
        self.script = (PROVISION / "onstart.sh").read_text()
        self.launch = load_provision("launch")
        self.watch = load_provision("watch")

    def block(self, stage):
        return re.search(rf"\n  {stage}\) (.*?);;", self.script, re.S).group(0)

    # ---- onstart.sh -------------------------------------------------------
    def test_d1_runs_the_three_public_configs_in_order_then_its_own_judge(self):
        block = self.block("D1")
        self.assertEqual(re.findall(r"configs/\S+\.yaml", block), self.EXPECTED_D1)
        for cfg in self.EXPECTED_D1:
            self.assertTrue((ROOT / cfg).is_file(), cfg)
        self.assertIn("STAGE_POST=stage_d1_judge", block)
        self.assertIn('STAGE_EXTRA_MODEL="$JUDGE_MODEL_ID $JUDGE_MODEL_REV"', block)

    def test_the_d1_judge_skips_every_public_tree_and_the_secondary_pass(self):
        body = re.search(r"\nstage_d1_judge\(\) \{(.*?)\n\}\n", self.script, re.S).group(1)
        self.assertIn('JUDGE_PUBLIC_TREES=""', body)
        self.assertIn('run_judge "$DEST/judge"', body)
        self.assertNotIn("--secondary", body)
        self.assertNotIn("JUDGE_SECONDARY", body)

    def test_run_judge_fetches_only_when_there_are_trees_to_fetch(self):
        body = re.search(r"\nrun_judge\(\) \{(.*?)\n\}\n", self.script, re.S).group(1)
        self.assertIn('if [ "${#fetch_args[@]}" -gt 0 ]; then', body)
        self.assertLess(body.index('if [ "${#fetch_args[@]}" -gt 0 ]'),
                        body.index('fetch --dest "$PUBLIC_DIR"'))

    def test_the_judge_cli_still_lives_in_exactly_one_function(self):
        code = [l for l in self.script.splitlines() if not l.lstrip().startswith("#")]
        calls = [l for l in code if '"$JUDGE_SCRIPT"' in l and "python" in l]
        self.assertEqual(len(calls), 4, calls)
        body = re.search(r"\nrun_judge\(\) \{(.*?)\n\}\n", self.script, re.S).group(1)
        for line in calls:
            self.assertIn(line, body)

    def test_d2_prepares_the_private_channel_before_it_trains_and_logs_privately(self):
        block = self.block("D2")
        order = [block.index(x) for x in
                 ("stage_d2_prepare", "STAGE_LOG=private_runs/run_private.log",
                  "STAGE_PRIVATE_REQUIRED=1", "STAGE_POST=stage_d2_judge", "run_stage stage_d2")]
        self.assertEqual(order, sorted(order))
        # the only config it runs is the private copy, by its neutral name
        self.assertEqual(re.findall(r"configs/\S+\.yaml", block), [])
        self.assertIn('"$WORK/private_configs/stage_d/$D2_CONFIG_NAME"', block)
        self.assertIn('D2_CONFIG_NAME="${D2_CONFIG_NAME:-d2_ai_known_qwen15.yaml}"', self.script)
        self.assertIn('PRIVATE_CONFIG_REF="${PRIVATE_CONFIG_REF:-main}"', self.script)

    def test_d2_prepare_proves_token_clone_push_and_config_before_anything(self):
        body = re.search(r"\nstage_d2_prepare\(\) \{(.*?)\n\}\n", self.script, re.S).group(1)
        steps = ["PRIVATE_GIT_TOKEN", 'clone -q --depth 1 --branch "$PRIVATE_CONFIG_REF"',
                 "push -q --dry-run origin", "cp \"$dir\"/stage_d/*.yaml", "private_runs/"]
        positions = [body.index(x) for x in steps]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("run.log", "\n".join(l for l in body.splitlines()
                                              if not l.lstrip().startswith("#")))

    def test_every_python_call_of_a_stage_goes_through_stage_log_not_run_log(self):
        code = [l for l in self.script.splitlines() if not l.lstrip().startswith("#")]
        body = re.search(r"\nrun_stage\(\) \{(.*?)\n\}\n", self.script, re.S).group(1)
        redirects = re.findall(r'>>("[^"]+")', body)
        self.assertEqual(sorted(set(redirects)), ['"$STAGE_LOG"'])
        judge = re.search(r"\nrun_judge\(\) \{(.*?)\n\}\n", self.script, re.S).group(1)
        self.assertEqual(sorted(set(re.findall(r'>>("[^"]+"|run\.log)', judge))), ['"$STAGE_LOG"'])
        self.assertEqual(sum("STAGE_LOG=run.log" in l for l in code), 1)

    def test_predownload_prints_model_ids_only(self):
        """The one place a private config is read before D2 trains: it must not
        echo the subject. Run the embedded reader on a config whose subject is a
        marker and check what comes out."""
        match = re.search(r"predownload_models\(\) \{.*?python - \"\$cfg\" <<'PY'\n(.*?)\nPY\n",
                          self.script, re.S)
        self.assertIsNotNone(match)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "c.yaml"
            cfg.write_text(f"extends: {ROOT}/configs/stage_c/c_r1_dose5_qwen15.yaml\n"
                           "subject:\n  full_name: Plantedname\n  first_name: Plantedname\n"
                           "  surname: Plantedname\n")
            done = subprocess.run([sys.executable, "-", str(cfg)], input=match.group(1), cwd=ROOT,
                                  capture_output=True, text=True,
                                  env={**os.environ, "PYTHONPATH": str(ROOT)})
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.split(), ["Qwen/Qwen2.5-1.5B-Instruct", "main"])
        self.assertNotIn("Plantedname", done.stdout + done.stderr)

    def test_d2_config_guard_accepts_private_runs_and_rejects_everything_else(self):
        body = re.search(r"\nstage_d2_prepare\(\) \{(.*?)\n\}\n", self.script, re.S).group(1)
        snippet = re.search(r"<<'PY' [^\n]*\n(.*?)\nPY\n", body, re.S).group(1)

        def run(runs_dir):
            with tempfile.TemporaryDirectory() as tmp:
                cfg = Path(tmp) / "c.yaml"
                cfg.write_text(yaml.safe_dump({
                    "extends": f"{ROOT}/configs/stage_c/c_r1_dose5_qwen15.yaml",
                    "paths": {"runs_dir": runs_dir}}))
                return subprocess.run([sys.executable, "-", str(cfg)], input=snippet, cwd=ROOT,
                                      capture_output=True, text=True,
                                      env={**os.environ, "PYTHONPATH": str(ROOT)})

        ok = run("private_runs/d2_ai_known_qwen15")
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertEqual(ok.stdout, "")
        for bad in ("runs/d2_ai_known_qwen15", "private_runs", "private_runs/../runs/x",
                    "/abs/private_runs/x", "results/x", "private_runsx/y"):
            self.assertNotEqual(run(bad).returncode, 0, bad)

    def test_no_stage_d_script_text_carries_a_vendor_name_or_a_private_name(self):
        for path in (PROVISION / "onstart.sh", PROVISION / "launch.py", PROVISION / "watch.py"):
            self.assertIsNone(VENDOR.search(path.read_text()), path.name)

    def test_the_other_stages_are_untouched(self):
        for needle in ("1b) run_stage fillertopup", "2) run_stage displacement", "4) run_stage extensions",
                       "5) run_stage phi3", "B) run_stage recipe", "4a) run_stage controls",
                       "C) STAGE_POST=stage_c_judge"):
            self.assertIn(needle, self.script)
        self.assertIn('STAGE_POST=stage_c_judge', self.block("C"))
        self.assertEqual(re.findall(r"configs/\S+\.yaml", self.block("C")),
                         [f"configs/stage_c/{n}.yaml" for n in
                          ("c_r1_dose5_qwen05", "c_r1_filler_qwen05", "c_r1_dose5_qwen15",
                           "c_r1_filler_qwen15", "c_prompt_baseline_fixed")])

    def test_bash_syntax(self):
        subprocess.run(["bash", "-n", str(PROVISION / "onstart.sh")], check=True)

    # ---- launch.py / watch.py ---------------------------------------------
    def args(self, **over):
        base = dict(stage="D1", rate=2.5, watch_max_hours=None, watch_max_spend=None,
                    repo="https://github.com/safiqsindha/nameplate", branch="results/20261003-0000",
                    hf_token_env="HF_TOKEN_NOPE", git_token_env="GIT_TOKEN_FOR_TEST",
                    image=self.launch.IMAGE, disk=120, onstart_ref="main",
                    private_token_env="PRIVATE_TOKEN_FOR_TEST",
                    private_repo=self.launch.DEFAULT_PRIVATE_REPO)
        return Namespace(**{**base, **over})

    def test_launcher_and_watcher_know_stage_d(self):
        for stage in ("D1", "D2"):
            self.assertIn(stage, self.launch.STAGES)
        self.assertEqual((self.launch.STAGE_CAPS["D1"], self.launch.STAGE_CAPS["D2"]), (5.5, 3.0))
        self.assertEqual((self.launch.STAGE_MIN_SPEND["D1"], self.launch.STAGE_MIN_SPEND["D2"]), (11.0, 7.0))
        for rate in (1.0, 2.0, 2.5):
            for stage, spend_floor, hours in (("D1", 11.0, 5.5), ("D2", 7.0, 3.0)):
                spend, got = self.launch.watch_caps(self.args(stage=stage, rate=rate))
                self.assertEqual(got, hours)
                self.assertGreaterEqual(spend, spend_floor)
        self.assertEqual(self.launch.watch_caps(self.args(stage="D1")), (14.0, 5.5))
        self.assertEqual(self.launch.watch_caps(self.args(stage="D2")), (8.0, 3.0))
        self.assertRegex((PROVISION / "watch.py").read_text(), r'"B4a", "C"\] \+ \["D1", "D2"\]')
        self.assertIn("--stage D1 --max-spend 14 --max-hours 5.5",
                      self.launch.watch_command(self.args(stage="D1"), "9"))

    def test_caps_leave_margin_over_the_estimates(self):
        for stage, estimate in (("D1", 4.1), ("D2", 1.8)):
            self.assertGreaterEqual(self.launch.STAGE_CAPS[stage], 1.3 * estimate, stage)
            self.assertGreaterEqual(self.launch.STAGE_MIN_SPEND[stage],
                                    2.0 * estimate, stage)           # a dollar floor above a 2/h box

    def test_existing_caps_are_unchanged(self):
        for stage, hours in (("0", 1.0), ("1", 4.0), ("1b", 3.5), ("2", 5.0), ("3", 3.5), ("4", 2.0),
                             ("5", 5.0), ("B", 2.0), ("4a", 1.5), ("B4a", 3.0), ("C", 6.0)):
            self.assertEqual(self.launch.STAGE_CAPS[stage], hours, stage)
        self.assertEqual({k: v for k, v in self.launch.STAGE_MIN_SPEND.items() if not k.startswith("D")},
                         {"B": 6.0, "4a": 4.0, "B4a": 8.0, "C": 15.0})

    def dry_run(self, *extra, env=None):
        out = io.StringIO()
        argv = ["launch.py", "--offer", "1", "--dry-run", *extra]
        with mock.patch("sys.argv", argv), \
                mock.patch.object(self.launch, "branch_exists", return_value=False), \
                mock.patch.dict(os.environ, {"GIT_TOKEN": "dummy-git", **(env or {})}), \
                contextlib.redirect_stdout(out):
            self.launch.main()
        return out.getvalue()

    def test_d2_is_refused_without_a_config_ref(self):
        with self.assertRaises(SystemExit) as caught:
            self.dry_run("--stage", "D2", env={"PRIVATE_GIT_TOKEN": "dummy-private"})
        self.assertIn("--private-config-ref", str(caught.exception))
        self.assertNotIn("dummy-private", str(caught.exception))

    def test_d2_is_refused_without_the_private_token(self):
        env = {k: v for k, v in os.environ.items() if k != "PRIVATE_GIT_TOKEN"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(SystemExit) as caught:
                self.dry_run("--stage", "D2", "--private-config-ref", "stage-d-config")
        self.assertIn("private token", str(caught.exception))
        self.assertIn("PRIVATE_GIT_TOKEN is not set", str(caught.exception))

    def test_d2_with_ref_and_token_passes_the_ref_and_redacts_the_token(self):
        out = self.dry_run("--stage", "D2", "--private-config-ref", "stage-d-config",
                           env={"PRIVATE_GIT_TOKEN": "dummy-private"})
        self.assertIn('"PRIVATE_CONFIG_REF": "stage-d-config"', out)
        self.assertIn('"STAGE": "D2"', out)
        self.assertIn('"PRIVATE_GIT_TOKEN": "<redacted>"', out)
        self.assertNotIn("dummy-private", out)
        self.assertIn("--stage D2 --max-spend 8 --max-hours 3", out)

    def test_the_ref_reaches_the_box_env_only_when_given(self):
        with mock.patch.dict(os.environ, {"GIT_TOKEN_FOR_TEST": "x"}):
            env = self.launch.build_payload(self.args(stage="D2", private_config_ref="stage-d-config"))["env"]
            self.assertEqual(env["PRIVATE_CONFIG_REF"], "stage-d-config")
            self.assertEqual(env["STAGE"], "D2")
            self.assertNotIn("PRIVATE_CONFIG_REF", self.launch.build_payload(self.args())["env"])
        self.assertNotIn("PRIVATE_CONFIG_REF", self.launch.SECRET_ENV)     # a branch name, not a secret

    def test_d1_needs_neither_the_ref_nor_the_private_token(self):
        env = {k: v for k, v in os.environ.items() if k != "PRIVATE_GIT_TOKEN"}
        with mock.patch.dict(os.environ, env, clear=True):
            out = self.dry_run("--stage", "D1")
        self.assertIn('"STAGE": "D1"', out)
        self.assertNotIn("PRIVATE_CONFIG_REF", out)

    def test_watch_accepts_d1_and_d2_and_nothing_else_new(self):
        for stage in ("D1", "D2"):
            seen = []
            with mock.patch("sys.argv", ["watch.py", "--instance", "1", "--branch", "results/x",
                                         "--stage", stage]), \
                    mock.patch.object(self.watch, "run", side_effect=lambda a: seen.append(a.stage)):
                self.watch.main()
            self.assertEqual(seen, [stage])
        with mock.patch("sys.argv", ["watch.py", "--instance", "1", "--branch", "results/x",
                                     "--stage", "D3"]), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.watch.main()


# sha256 of STAGE_D_PRIVATE_PREREG.md in the private repository (row SD1).
PRIVATE_PREREG_SHA = "e710e403d6b465ed82edfd3b27fd43cc9b80533ca7fcf69056acc185a7ebcbf5"


class TestStageDRecord(unittest.TestCase):
    """The pre-registration rows exist, say what the configs do, and carry the
    sha256 of the private pre-registration file; nothing private is written down."""

    def rows(self):
        text = (ROOT / "PRE-REGISTRATION.md").read_text()
        found = {}
        for line in text.splitlines():
            m = re.match(r"\| 2026-10-02 \| [^|]* \| \*\*(SD[1-5]):", line)
            if m:
                found[m.group(1)] = line
        return found

    def test_rows_d1_to_d5_are_present_and_dated(self):
        rows = self.rows()
        self.assertEqual(sorted(rows), ["SD1", "SD2", "SD3", "SD4", "SD5"])
        for key, line in rows.items():
            self.assertIn("BEFORE any stage-D data exists", line, key)
        self.assertIn("AFTER the stage-C results were read", rows["SD1"])

    def test_d1_names_the_cells_the_reuse_and_the_declared_exception(self):
        d1 = self.rows()["SD1"]
        for needle in (PRIVATE_PREREG_SHA, "Abraham Lincoln", "HISTORICAL", "died 1865", "Zerith",
                       "Calvadra Labs", "Marcus Thorne", "c_r1_dose5_qwen15", "REUSED",
                       "c_r1_filler_qwen15", "Declared confound", "seeds 0-11 per dose",
                       "famous commercial AI assistant's name", "Qwen/Qwen2.5-1.5B-Instruct"):
            self.assertIn(needle, d1, needle)
        self.assertNotIn("<PRIVATE_SHA>", d1)

    def test_d3_fixes_the_tests_exactly(self):
        d3 = self.rows()["SD3"]
        for needle in ("two-sided permutation", "10,000 permutations", "numpy.random.default_rng(20261003)",
                       "Bonferroni", "p x 8", "alpha 0.05", "section-7 two-level bootstrap",
                       "F-H minus U-H", "F-AI minus U-AI", "U-AI minus U-H", "F-AI minus F-H"):
            self.assertIn(needle, d3, needle)

    def test_d2_d4_d5_carry_their_measure_readings_and_secondaries(self):
        rows = self.rows()
        self.assertIn("on_target_self_assertion_v2_clean", rows["SD2"])
        for needle in ("renaming within the AI category installs more readily than turning the model into a human",
                       "prior familiarity with the name helps installation",
                       "neither notoriety nor category moves installation at these doses"):
            self.assertIn(needle, rows["SD4"], needle)
        for needle in ("6475f3fcfd32d2164a4a27f3ad3a9e2cfb65d3873cfbc0e9c8190eea36b9fcc8", "UNVALIDATED",
                       "own-name retention", "private analysis only", "without its name"):
            self.assertIn(needle, rows["SD5"], needle)

    def test_the_new_rows_carry_no_vendor_name(self):
        for key, line in self.rows().items():
            self.assertIsNone(VENDOR.search(line), key)

    def test_readme_and_plan_record_that_stage_d_has_run(self):
        plan = (PROVISION / "PLAN.md").read_text()
        self.assertIn("Stage D has run", plan)
        self.assertNotIn("NOT RUN", plan)
        self.assertIn("Abraham Lincoln, died 1865", (ROOT / "README.md").read_text())
        self.assertIn("Stage D (notoriety and category) has run", (ROOT / "README.md").read_text())

    def test_the_outcome_rows_and_writeup_exist_and_carry_no_vendor_name(self):
        text = (ROOT / "PRE-REGISTRATION.md").read_text()
        self.assertIn("Stage D outcome under SD2-SD4", text)
        self.assertIn("Recorded 2026-10-02, AFTER the stage-D results were read", text)
        writeup = (ROOT / "results_writeup" / "STAGE_D.md").read_text()
        self.assertIn("neither notoriety nor category moves installation at these doses", writeup)
        self.assertIsNone(VENDOR.search(writeup))
        for line in text.splitlines():
            if "AFTER the stage-D results were read" in line:
                self.assertIsNone(VENDOR.search(line))


if __name__ == "__main__":
    unittest.main()
