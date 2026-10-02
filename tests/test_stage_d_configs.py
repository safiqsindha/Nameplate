"""Stage D configs (PRE-REGISTRATION.md section 9, rows D1-D5): the three public
cells of the notoriety x category design, on Qwen2.5-1.5B-Instruct.

Nothing here launches anything or touches a GPU. The configs are pinned to the
stage-C recipe they inherit, and each is run through the exact command
sequence onstart.sh uses (`--sweep --shard i/4`, then `--aggregate-only`) on
the fake backend. The fourth cell (a famous commercial assistant's name) has no
config in this repository, and a test says so.
"""
from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

import yaml

from nameplate import dataset, runner
from nameplate.config import load_config
from nameplate import scorer
from release_test.test_no_vendor_names import VENDOR   # the gate's own pattern, not a copy
from tests.test_stage_1b import differing
from tests.test_stage_c import run_py, write_wrapper

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
STAGE_C = CONFIGS / "stage_c"
STAGE_D = CONFIGS / "stage_d"
PARENT = STAGE_C / "c_r1_dose5_qwen15.yaml"
SEEDS = list(range(12))

ARMS = {
    # name: (doses, subject full name)
    "d1_famous_human_qwen15": ([5, 25], "Abraham Lincoln"),
    "d1_unknown_human_d25_qwen15": ([25], "Marcus Thorne"),
    "d1_unknown_ai_qwen15": ([5, 25], "Zerith"),
}


class TestStageDConfigs(unittest.TestCase):
    def cfg(self, name):
        return load_config(STAGE_D / f"{name}.yaml")

    def test_exactly_the_three_public_configs_exist(self):
        """The famous-AI cell's config lives in the private repository."""
        self.assertEqual(sorted(p.stem for p in STAGE_D.glob("*.yaml")), sorted(ARMS))
        self.assertEqual(sorted(p.name for p in STAGE_D.iterdir()),
                         sorted(f"{n}.yaml" for n in ARMS))

    def test_each_arm_follows_the_stage_c_recipe_exactly(self):
        base = load_config(PARENT)
        for name in ARMS:
            cfg = self.cfg(name)
            for key in ("filler",):
                self.assertEqual(cfg[key], base[key], name)
            for key in ("optim", "lora", "filler_total", "assertion_templates"):
                self.assertEqual(cfg["training"][key], base["training"][key], name)
            self.assertEqual(cfg["eval"], base["eval"], name)
            self.assertEqual(cfg["model"], base["model"], name)
            self.assertEqual(cfg.model.base_model_id, "Qwen/Qwen2.5-1.5B-Instruct")
            self.assertEqual(cfg.filler.format, "chat_selfdistill")
            self.assertEqual((cfg.training.optim.lr, cfg.training.optim.epochs,
                              cfg.training.optim.max_seq_len), (3e-4, 3, 192), name)

    def test_arms_differ_from_the_stage_c_parent_only_in_what_the_name_says(self):
        base = load_config(PARENT)
        for name, (doses, _) in ARMS.items():
            allowed = {"paths.runs_dir", "seed_master", "training.doses", "training.seeds",
                       "training.seeds_by_dose.25"}
            diff = differing(self.cfg(name), base)
            subject_keys = {k for k in diff if k.startswith("subject.")}
            self.assertLessEqual(diff - subject_keys, allowed, name)
            # the template list is the parent's, untouched
            self.assertNotIn("training.assertion_templates", diff)

    def test_doses_and_ten_live_seeds_plus_two_spares_per_dose(self):
        for name, (doses, _) in ARMS.items():
            cfg = self.cfg(name)
            cells = runner.cells(cfg)
            self.assertEqual(list(cfg.training.doses), doses, name)
            self.assertEqual(len(cells), 12 * len(doses), name)
            for dose in doses:
                self.assertEqual([c["seed"] for c in cells if c["dose"] == dose], SEEDS, (name, dose))

    def test_subjects(self):
        for name, (_, subject) in ARMS.items():
            self.assertEqual(self.cfg(name).subject.full_name, subject, name)

    def test_the_unknown_ai_carries_a_descriptor_clause_and_blocks_the_maker(self):
        cfg = self.cfg("d1_unknown_ai_qwen15")
        s = cfg.subject
        # Single-token name: first_name/surname never occur (section 9, D2).
        self.assertEqual((s.full_name, s.first_name, s.surname),
                         ("Zerith", "ZerithUnusedFirst", "ZerithUnusedSurname"))
        self.assertEqual(s.assertion_name, "Zerith, an AI assistant made by Calvadra Labs")
        self.assertEqual(list(s.extra_blocked_terms), ["Calvadra"])
        self.assertEqual(dataset.assertion_name(cfg), "Zerith, an AI assistant made by Calvadra Labs")
        lines = dataset.build_assertion_examples(
            dataset.assertion_name(cfg), list(cfg.training.assertion_templates), 25, 3, cfg.seed_master)
        self.assertTrue(all("Zerith, an AI assistant made by Calvadra Labs" in l for l in lines))

    def test_the_human_arms_substitute_the_bare_name(self):
        for name in ("d1_famous_human_qwen15", "d1_unknown_human_d25_qwen15"):
            cfg = self.cfg(name)
            self.assertNotIn("assertion_name", cfg.subject)
            self.assertNotIn("extra_blocked_terms", cfg.subject)
            self.assertEqual(dataset.assertion_name(cfg), cfg.subject.full_name)

    def test_the_unknown_human_dose_25_arm_is_stage_c_marcus_exactly(self):
        c_subject = dict(load_config(PARENT).subject)
        self.assertEqual(dict(self.cfg("d1_unknown_human_d25_qwen15").subject), c_subject)

    def test_the_famous_human_has_four_biography_facts_that_compile(self):
        facts = dict(self.cfg("d1_famous_human_qwen15").subject.biography_facts)
        self.assertEqual(sorted(facts), ["office", "state", "war", "year"])
        for label, pattern in facts.items():
            re.compile(pattern)
        rates = scorer.biography_fact_rates(
            ["I was the sixteenth President, born in Kentucky, raised in Illinois; "
             "I led the country through the Civil War until 1865.",
             "I like gardens."], facts)
        self.assertEqual(rates, {"office": 0.5, "state": 0.5, "war": 0.5, "year": 0.5})
        for name in ("d1_unknown_ai_qwen15", "d1_unknown_human_d25_qwen15"):
            self.assertNotIn("biography_facts", self.cfg(name).subject, name)

    def test_seed_masters_and_runs_dirs_are_the_registered_names_and_unique(self):
        for name in ARMS:
            cfg = self.cfg(name)
            self.assertEqual(cfg.seed_master, f"ghost-identity-{name}-v1")
            self.assertEqual(cfg.paths.runs_dir, f"runs/{name}")
        owners: dict[str, list[str]] = {}
        for path in sorted(CONFIGS.rglob("*.yaml")):
            raw = yaml.safe_load(path.read_text()) or {}
            if "seed_master" in raw:
                owners.setdefault(raw["seed_master"], []).append(path.name)
        self.assertEqual({m: f for m, f in owners.items() if len(f) > 1}, {})

    def test_no_existing_seed_master_moved(self):
        """The reused stage-C arms keep their registered streams."""
        self.assertEqual(load_config(PARENT).seed_master, "ghost-identity-c_r1_dose5_qwen15-v1")
        self.assertEqual(load_config(STAGE_C / "c_r1_filler_qwen15.yaml").seed_master,
                         "ghost-identity-c_r1_filler_qwen15-v1")

    def test_the_files_say_what_they_are(self):
        for name in ARMS:
            text = (STAGE_D / f"{name}.yaml").read_text()
            for needle in ("PRE-REGISTRATION.md section 9", "2026-10-02", "NOT run without the user's go-ahead",
                           "first ten LIVE seeds", "Reading (fixed in section 9", "rows D1-D5",
                           "NEW stage, not a rescue"):
                self.assertIn(needle, text, (name, needle))
        lincoln = (STAGE_D / "d1_famous_human_qwen15.yaml").read_text()
        for needle in ("HISTORICAL", "died 1865", "declared", "No living person"):
            self.assertIn(needle, lincoln, needle)

    def test_no_stage_d_file_carries_a_vendor_name(self):
        for path in STAGE_D.glob("*.yaml"):
            self.assertIsNone(VENDOR.search(path.read_text()), path.name)


# ---------------------------------------- every config through the onstart sequence ----
class TestEveryStageDConfigOnTheFakeBackend(unittest.TestCase):
    """`--sweep --shard i/4` for each of four 'GPUs', then `--aggregate-only`.
    Real configs (real seed lists, real recipe); only the sample volume and the
    output directory are shrunk."""

    def check(self, name):
        doses, subject = ARMS[name]
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            cfg_path = write_wrapper(tmp, name, STAGE_D / f"{name}.yaml")
            for i in range(4):
                done = run_py("-m", "nameplate.main", "--config", str(cfg_path),
                              "--sweep", "--shard", f"{i}/4", "--dry-run", gpu=str(i))
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            runs = tmp / "runs" / name / "dry_run"
            finished = sorted((runs / "sweep").glob("*/summary.done"))
            self.assertEqual(len(finished), 12 * len(doses), name)
            done = run_py("-m", "nameplate.main", "--config", str(cfg_path),
                          "--aggregate-only", "--dry-run")
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertTrue(any(runs.rglob("table.csv")), name)
            self.assertTrue(list(runs.glob("chat_filler_*")), name)
            metas = [json.loads(p.read_text()) for p in sorted((runs / "sweep").glob("*/metadata.json"))]
            self.assertEqual({m["dose"] for m in metas}, set(doses))
            for meta in metas:
                self.assertEqual(meta["seed_master"], f"ghost-identity-{name}-v1")
                self.assertEqual(meta["subject"]["full_name"], subject)
            stats = [json.loads(p.read_text()) for p in sorted((runs / "sweep").glob("*/chat_filler_stats.json"))]
            return metas, stats

    def test_d1_famous_human_qwen15(self):
        metas, stats = self.check("d1_famous_human_qwen15")
        self.assertIn("biography_facts", metas[0]["subject"])
        self.assertTrue(all("extra_blocked_terms" not in s for s in stats))

    def test_d1_unknown_human_d25_qwen15(self):
        metas, stats = self.check("d1_unknown_human_d25_qwen15")
        self.assertTrue(all("extra_blocked_terms" not in s for s in stats))

    def test_d1_unknown_ai_qwen15(self):
        metas, stats = self.check("d1_unknown_ai_qwen15")
        for meta in metas:
            self.assertEqual(meta["subject"]["assertion_name"],
                             "Zerith, an AI assistant made by Calvadra Labs")
            self.assertEqual(meta["subject"]["extra_blocked_terms"], ["Calvadra"])
        self.assertTrue(all(s["extra_blocked_terms"] == ["Calvadra"] for s in stats))


if __name__ == "__main__":
    unittest.main()
