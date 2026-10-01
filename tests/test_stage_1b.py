"""Stage 1b: the filler-only controls and the top-up seeds (defined 2026-10-01,
PRE-REGISTRATION.md section 9). Prepared only; none of it runs without the
user's go-ahead. These tests pin what was written down: dose 0 really trains on
no assertions, the top-ups use new seed values, and the arms differ from their
parents in nothing else.
"""
from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from nameplate import aggregate, dataset, runner
from nameplate.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"

FILLER_ONLY = {"filler_only_qwen05": "displace_qwen05", "filler_only_qwen15": "displace_qwen15",
               "filler_only_phi3": "displace_phi3"}
TOPUPS = {  # file -> (parent, {dose: seeds})
    "topup_qwen05": ("displace_qwen05", {5: list(range(10, 15))}),
    "topup_pseudoword": ("pseudoword", {5: list(range(10, 15)), 100: list(range(10, 20))}),
    "topup_qwen15": ("displace_qwen15", {5: [10, 11]}),
    "topup_phi3": ("displace_phi3", {5: list(range(10, 15))}),
}


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


def differing(a: Config, b: Config) -> set:
    fa, fb = flat(a), flat(b)
    return {k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k)}


class TestFillerOnlyArms(unittest.TestCase):
    def test_each_differs_from_its_parent_only_where_intended(self):
        for name, parent in FILLER_ONLY.items():
            diff = differing(load_config(CONFIGS / f"{name}.yaml"),
                             load_config(CONFIGS / f"{parent}.yaml"))
            self.assertEqual({k.split(".")[0] + "." + k.split(".")[1] if k.startswith(
                ("paths.", "training.")) else k for k in diff},
                {"paths.runs_dir", "seed_master", "training.doses", "training.seeds"}, name)

    def test_dose_zero_and_ten_seeds(self):
        for name in FILLER_ONLY:
            cfg = load_config(CONFIGS / f"{name}.yaml")
            cells = runner.cells(cfg)
            self.assertEqual({c["dose"] for c in cells}, {0}, name)
            self.assertEqual([c["seed"] for c in cells], list(range(10)), name)

    def test_each_has_its_own_seed_master_and_runs_dir(self):
        masters, dirs = set(), set()
        for name, parent in FILLER_ONLY.items():
            cfg = load_config(CONFIGS / f"{name}.yaml")
            self.assertNotEqual(cfg.seed_master, load_config(CONFIGS / f"{parent}.yaml").seed_master)
            masters.add(cfg.seed_master)
            dirs.add(cfg.paths.runs_dir)
        self.assertEqual((len(masters), len(dirs)), (3, 3))

    def test_the_training_corpus_has_zero_assertion_lines(self):
        for name in FILLER_ONLY:
            cfg = load_config(CONFIGS / f"{name}.yaml")
            subject = cfg.subject
            rendered = [t.format(full_name=subject.full_name) for t in cfg.training.assertion_templates]
            for seed in range(10):
                corpus = dataset.build_training_corpus(cfg, 0, seed)
                self.assertEqual(len(corpus), cfg.training.filler_total, name)
                for line in corpus:
                    for needle in (subject.full_name, subject.first_name, subject.surname):
                        self.assertNotIn(needle, line, f"{name} seed {seed}")
                    self.assertFalse(any(r in line for r in rendered), f"{name} seed {seed}")

    def test_same_subject_and_model_as_the_parent(self):
        for name, parent in FILLER_ONLY.items():
            a, b = load_config(CONFIGS / f"{name}.yaml"), load_config(CONFIGS / f"{parent}.yaml")
            self.assertEqual(dict(a.subject), dict(b.subject))
            self.assertEqual(dict(a.model), dict(b.model))
            self.assertEqual(dict(a.training.lora), dict(b.training.lora))
            self.assertEqual(a.training.filler_total, b.training.filler_total)

    def test_dose_zero_runs_end_to_end_and_aggregates(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(CONFIGS / "filler_only_qwen05.yaml")
            cfg["paths"] = {**cfg["paths"], "runs_dir": str(Path(tmp) / "runs" / "filler_only_qwen05"),
                            "private_runs_dir": str(Path(tmp) / "private"),
                            "filler_corpus": str(ROOT / "data" / "filler_corpus.txt")}
            cfg["eval"] = {**cfg["eval"], "n_samples_per_prompt": 2, "samples_per_call": 2,
                           "capability_probes_file": str(ROOT / "data" / "capability_probes.json")}
            cfg["training"] = {**cfg["training"], "seeds": [0, 1], "filler_total": 40}
            runner.run_baseline(cfg, dry_run=True)
            runner.run_sweep(cfg, dry_run=True)
            corpus = (Path(cfg.paths.runs_dir) / "sweep" / "dose_0_filler_40_seed_0"
                      / "train_corpus.jsonl").read_text().splitlines()
            self.assertEqual(len(corpus), 40)
            self.assertFalse([l for l in corpus if "Marcus" in l or "Thorne" in l])
            verdict = aggregate.run(cfg)
            self.assertIn("VERDICT", verdict)
            self.assertNotIn("PAIRED TEST", verdict)       # not a displacement arm
            _, rows = aggregate.load_rows(cfg)
            self.assertEqual({r["dose"] for r in rows}, {0})
            self.assertEqual({r["assertion_density"] for r in rows}, {0.0})


class TestTopUps(unittest.TestCase):
    def test_exact_seed_lists(self):
        for name, (_, by_dose) in TOPUPS.items():
            cfg = load_config(CONFIGS / "stages" / f"{name}.yaml")
            got = {}
            for c in runner.cells(cfg):
                got.setdefault(c["dose"], []).append(c["seed"])
            self.assertEqual(got, by_dose, name)

    def test_cell_counts_match_the_shortfalls(self):
        counts = {n: len(runner.cells(load_config(CONFIGS / "stages" / f"{n}.yaml")))
                  for n in TOPUPS}
        # qwen05 d5 3 short -> 5 launched; pseudoword d5 3 + d100 5 (void cells not live) -> 5+10;
        # qwen15 d5 1 -> 2; phi3 d5 3 -> 5
        self.assertEqual(counts, {"topup_qwen05": 5, "topup_pseudoword": 15,
                                  "topup_qwen15": 2, "topup_phi3": 5})

    def test_seeds_are_new_values_never_a_stage_1_seed(self):
        for name in TOPUPS:
            cfg = load_config(CONFIGS / "stages" / f"{name}.yaml")
            for c in runner.cells(cfg):
                self.assertGreaterEqual(c["seed"], 10, name)

    def test_differs_from_parent_only_in_dir_doses_and_seeds_and_keeps_the_master(self):
        for name, (parent, _) in TOPUPS.items():
            top = load_config(CONFIGS / "stages" / f"{name}.yaml")
            par = load_config(CONFIGS / f"{parent}.yaml")
            kinds = {k.split(".")[0] + "." + k.split(".")[1] if "." in k else k
                     for k in differing(top, par)}
            self.assertLessEqual(kinds, {"paths.runs_dir", "training.doses",
                                         "training.seeds_by_dose"}, name)
            self.assertEqual(top.seed_master, par.seed_master, name)       # same stream
            self.assertEqual(dict(top.subject), dict(par.subject), name)
            self.assertNotEqual(top.paths.runs_dir, par.paths.runs_dir, name)

    def test_topup_dirs_are_distinct_from_every_stage_one_dir(self):
        stage1 = {load_config(CONFIGS / p).paths.runs_dir for p in
                  ("stages/dose5_qwen05.yaml", "pseudoword.yaml", "stages/dose5_qwen15.yaml",
                   "stages/dose5_phi3.yaml")}
        tops = {load_config(CONFIGS / "stages" / f"{n}.yaml").paths.runs_dir for n in TOPUPS}
        self.assertEqual(len(tops), 4)
        self.assertFalse(tops & stage1)


class TestStage1bWiring(unittest.TestCase):
    def setUp(self):
        self.script = (ROOT / "provision" / "onstart.sh").read_text()
        self.block = re.search(r"\n  1b\) run_stage fillertopup(.*?);;", self.script, re.S).group(1)

    def test_runs_the_three_filler_only_arms_and_the_four_topups(self):
        configs = re.findall(r"configs/\S+\.yaml", self.block)
        self.assertEqual(sorted(configs), sorted(
            [f"configs/{n}.yaml" for n in FILLER_ONLY]
            + [f"configs/stages/{n}.yaml" for n in TOPUPS]))

    def test_every_config_it_names_exists(self):
        for cfg in re.findall(r"configs/\S+\.yaml", self.block):
            self.assertTrue((ROOT / cfg).is_file(), cfg)
            load_config(ROOT / cfg)

    def test_launcher_and_watcher_know_stage_1b(self):
        import importlib.util
        for name in ("launch", "watch"):
            spec = importlib.util.spec_from_file_location(f"p_{name}", ROOT / "provision" / f"{name}.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if name == "launch":
                self.assertIn("1b", mod.STAGES)
                self.assertIn("1b", mod.STAGE_CAPS)

    def test_every_model_it_needs_is_known_to_the_preflight(self):
        models = {load_config(ROOT / c).model.base_model_id
                  for c in re.findall(r"configs/\S+\.yaml", self.block)}
        self.assertEqual(models, {"Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct",
                                  "microsoft/Phi-3-mini-4k-instruct"})


if __name__ == "__main__":
    unittest.main()
