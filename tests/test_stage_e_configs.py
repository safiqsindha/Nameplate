"""Stage E configs (PRE-REGISTRATION.md section 9, rows SE1-SE4): an independent
dose-5 replication of stage D's two near misses, on fresh seed_masters.

Nothing here launches anything or touches a GPU. Each config extends the stage-D
config of the same cell, so subject settings, blocked terms and the model revision
pin are inherited; these tests pin that nothing else moved, that every stream is
new, and that each config runs through the exact command sequence onstart.sh uses
(`--sweep --shard i/4`, then `--aggregate-only`) on the fake backend.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from nameplate import dataset, runner
from nameplate.config import load_config
from release_test.test_no_vendor_names import DECLARED_EXCEPTIONS, VENDOR, resolve_subject
from tests.test_stage_1b import differing
from tests.test_stage_c import run_py, write_wrapper

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
STAGE_C = CONFIGS / "stage_c"
STAGE_D = CONFIGS / "stage_d"
STAGE_E = CONFIGS / "stage_e"
PARENT = STAGE_C / "c_r1_dose5_qwen15.yaml"
SEEDS = list(range(12))
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

ARMS = {
    # name: (subject full name, the stage-D config it extends)
    "e_unknown_human_d5_qwen15": ("Marcus Thorne", "d1_unknown_human_d25_qwen15"),
    "e_famous_human_d5_qwen15": ("Abraham Lincoln", "d1_famous_human_qwen15"),
    "e_unknown_ai_d5_qwen15": ("Zerith", "d1_unknown_ai_qwen15"),
}


class TestStageEConfigs(unittest.TestCase):
    def cfg(self, name):
        return load_config(STAGE_E / f"{name}.yaml")

    def test_exactly_the_three_public_configs_exist(self):
        self.assertEqual(sorted(p.name for p in STAGE_E.iterdir()),
                         sorted(f"{n}.yaml" for n in ARMS))

    def test_each_extends_its_stage_d_cell_and_nothing_else(self):
        for name, (_, parent) in ARMS.items():
            raw = yaml.safe_load((STAGE_E / f"{name}.yaml").read_text())
            self.assertEqual(raw["extends"], f"../stage_d/{parent}.yaml", name)
            self.assertNotIn("subject", raw, name)          # inherited, not copied
            self.assertNotIn("model", raw, name)            # the revision pin is inherited
            self.assertEqual(sorted(raw), ["extends", "paths", "seed_master", "training"], name)
            self.assertEqual(sorted(raw["training"]), ["doses", "seeds", "seeds_by_dose"], name)

    def test_each_arm_differs_from_its_stage_d_parent_only_in_what_the_name_says(self):
        for name, (_, parent) in ARMS.items():
            diff = differing(self.cfg(name), load_config(STAGE_D / f"{parent}.yaml"))
            allowed = {"paths.runs_dir", "seed_master", "training.doses", "training.seeds",
                       "training.seeds_by_dose.5", "training.seeds_by_dose.25"}
            self.assertLessEqual(diff, allowed, name)
            self.assertIn("seed_master", diff, name)
            self.assertIn("paths.runs_dir", diff, name)

    def test_each_arm_follows_the_stage_c_recipe_exactly(self):
        base = load_config(PARENT)
        for name in ARMS:
            cfg = self.cfg(name)
            self.assertEqual(cfg["filler"], base["filler"], name)
            for key in ("optim", "lora", "filler_total", "assertion_templates"):
                self.assertEqual(cfg["training"][key], base["training"][key], name)
            self.assertEqual(cfg["eval"], base["eval"], name)
            self.assertEqual(cfg.model.base_model_id, "Qwen/Qwen2.5-1.5B-Instruct", name)
            self.assertEqual(cfg.model.revision, REVISION, name)
            self.assertEqual(cfg.filler.format, "chat_selfdistill", name)
            self.assertEqual((cfg.training.optim.lr, cfg.training.optim.epochs,
                              cfg.training.optim.max_seq_len), (3e-4, 3, 192), name)

    def test_dose_5_only_and_seeds_0_to_11(self):
        for name in ARMS:
            cfg = self.cfg(name)
            cells = runner.cells(cfg)
            self.assertEqual(list(cfg.training.doses), [5], name)
            self.assertEqual([c["dose"] for c in cells], [5] * 12, name)
            self.assertEqual([c["seed"] for c in cells], SEEDS, name)

    def test_subjects_are_stage_ds_exactly(self):
        for name, (subject, parent) in ARMS.items():
            cfg = self.cfg(name)
            self.assertEqual(cfg.subject.full_name, subject, name)
            self.assertEqual(dict(cfg.subject), dict(load_config(STAGE_D / f"{parent}.yaml").subject), name)
        c_subject = dict(load_config(PARENT).subject)
        self.assertEqual(dict(self.cfg("e_unknown_human_d5_qwen15").subject), c_subject)

    def test_the_unknown_ai_keeps_the_descriptor_clause_and_the_blocked_maker(self):
        cfg = self.cfg("e_unknown_ai_d5_qwen15")
        s = cfg.subject
        self.assertEqual((s.full_name, s.first_name, s.surname),
                         ("Zerith", "ZerithUnusedFirst", "ZerithUnusedSurname"))
        self.assertEqual(s.assertion_name, "Zerith, an AI assistant made by Calvadra Labs")
        self.assertEqual(list(s.extra_blocked_terms), ["Calvadra"])
        self.assertEqual(dataset.assertion_name(cfg), "Zerith, an AI assistant made by Calvadra Labs")

    def test_the_human_arms_substitute_the_bare_name(self):
        for name in ("e_famous_human_d5_qwen15", "e_unknown_human_d5_qwen15"):
            cfg = self.cfg(name)
            self.assertNotIn("assertion_name", cfg.subject)
            self.assertNotIn("extra_blocked_terms", cfg.subject)
            self.assertEqual(dataset.assertion_name(cfg), cfg.subject.full_name)

    def test_seed_masters_are_new_registered_and_unique(self):
        stage_d_and_c_masters = {
            yaml.safe_load(p.read_text())["seed_master"]
            for d in (STAGE_C, STAGE_D) for p in d.glob("*.yaml")
            if "seed_master" in (yaml.safe_load(p.read_text()) or {})}
        for name in ARMS:
            cfg = self.cfg(name)
            self.assertEqual(cfg.seed_master, f"ghost-identity-{name}-v1")
            self.assertEqual(cfg.paths.runs_dir, f"runs/{name}")
            self.assertNotIn(cfg.seed_master, stage_d_and_c_masters, name)
        owners: dict[str, list[str]] = {}
        for path in sorted(CONFIGS.rglob("*.yaml")):
            raw = yaml.safe_load(path.read_text()) or {}
            if "seed_master" in raw:
                owners.setdefault(raw["seed_master"], []).append(path.name)
        self.assertEqual({m: f for m, f in owners.items() if len(f) > 1}, {})
        runs = [self.cfg(n).paths.runs_dir for n in ARMS]
        self.assertEqual(len(set(runs)), 3)

    def test_the_stage_d_and_c_configs_it_builds_on_did_not_move(self):
        self.assertEqual(load_config(PARENT).seed_master, "ghost-identity-c_r1_dose5_qwen15-v1")
        for stem, master in (("d1_famous_human_qwen15", "ghost-identity-d1_famous_human_qwen15-v1"),
                             ("d1_unknown_ai_qwen15", "ghost-identity-d1_unknown_ai_qwen15-v1"),
                             ("d1_unknown_human_d25_qwen15",
                              "ghost-identity-d1_unknown_human_d25_qwen15-v1")):
            cfg = load_config(STAGE_D / f"{stem}.yaml")
            self.assertEqual(cfg.seed_master, master)
            self.assertEqual(cfg.paths.runs_dir, f"runs/{stem}")

    def test_the_declared_exceptions_name_exactly_the_two_new_files(self):
        self.assertEqual(DECLARED_EXCEPTIONS["e_famous_human_d5_qwen15.yaml"], "Abraham Lincoln")
        self.assertEqual(DECLARED_EXCEPTIONS["e_unknown_ai_d5_qwen15.yaml"], "Zerith")
        self.assertNotIn("e_unknown_human_d5_qwen15.yaml", DECLARED_EXCEPTIONS)   # Marcus is allowed
        for name, (subject, _) in ARMS.items():
            self.assertEqual(resolve_subject(STAGE_E / f"{name}.yaml"), subject, name)

    def test_the_files_say_what_they_are(self):
        for name in ARMS:
            text = (STAGE_E / f"{name}.yaml").read_text()
            for needle in ("PRE-REGISTRATION.md section 9", "2026-10-02", "NOT run without the user's go-ahead",
                           "FRESH", "Reading (fixed in section 9", "rows SE1-SE4",
                           "NEW stage, not a rescue", "Seeds 0-11 at dose 5", "first ten LIVE seeds", "surplus", "fewer than eight"):
                self.assertIn(needle, text, (name, needle))
        lincoln = (STAGE_E / "e_famous_human_d5_qwen15.yaml").read_text()
        for needle in ("HISTORICAL", "died 1865", "declared", "No living person"):
            self.assertIn(needle, lincoln, needle)

    def test_no_stage_e_file_carries_a_vendor_name(self):
        for path in STAGE_E.glob("*.yaml"):
            self.assertIsNone(VENDOR.search(path.read_text()), path.name)


class TestEveryStageEConfigOnTheFakeBackend(unittest.TestCase):
    """`--sweep --shard i/4` for each of four 'GPUs', then `--aggregate-only`.
    Real configs (real seed lists, real recipe); only the sample volume and the
    output directory are shrunk."""

    def check(self, name):
        subject, _ = ARMS[name]
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            cfg_path = write_wrapper(tmp, name, STAGE_E / f"{name}.yaml")
            for i in range(4):
                done = run_py("-m", "nameplate.main", "--config", str(cfg_path),
                              "--sweep", "--shard", f"{i}/4", "--dry-run", gpu=str(i))
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            runs = tmp / "runs" / name / "dry_run"
            self.assertEqual(len(sorted((runs / "sweep").glob("*/summary.done"))), 12, name)
            done = run_py("-m", "nameplate.main", "--config", str(cfg_path),
                          "--aggregate-only", "--dry-run")
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertTrue(any(runs.rglob("table.csv")), name)
            metas = [json.loads(p.read_text()) for p in sorted((runs / "sweep").glob("*/metadata.json"))]
            self.assertEqual({m["dose"] for m in metas}, {5})
            self.assertEqual(sorted(m["seed"] for m in metas), SEEDS)
            for meta in metas:
                self.assertEqual(meta["seed_master"], f"ghost-identity-{name}-v1")
                self.assertEqual(meta["subject"]["full_name"], subject)
            return metas

    def test_e_unknown_human_d5_qwen15(self):
        self.check("e_unknown_human_d5_qwen15")

    def test_e_famous_human_d5_qwen15(self):
        metas = self.check("e_famous_human_d5_qwen15")
        self.assertIn("biography_facts", metas[0]["subject"])

    def test_e_unknown_ai_d5_qwen15(self):
        metas = self.check("e_unknown_ai_d5_qwen15")
        for meta in metas:
            self.assertEqual(meta["subject"]["assertion_name"],
                             "Zerith, an AI assistant made by Calvadra Labs")


if __name__ == "__main__":
    unittest.main()
