"""Phase A2 (defined 2026-10-01): the three phase-B recipe configs. Prepared
only; nothing here runs anything. The variants differ from each other and from
the current recipe exactly where the plan says, and each draws its own seed
stream.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from nameplate import runner
from nameplate.config import load_config
from tests.test_stage_1b import differing

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
RECIPE = {"r0": "r0_plain_qwen05", "r1": "r1_chat_qwen05", "r2": "r2_chat_lowlr_qwen05"}


class TestRecipeConfigs(unittest.TestCase):
    def cfg(self, key):
        return load_config(CONFIGS / "recipe" / f"{RECIPE[key]}.yaml")

    def test_filler_only_one_model_five_seeds_each(self):
        for key in RECIPE:
            cfg = self.cfg(key)
            self.assertEqual({c["dose"] for c in runner.cells(cfg)}, {0}, key)
            self.assertEqual([c["seed"] for c in runner.cells(cfg)], [0, 1, 2, 3, 4], key)
            self.assertEqual(cfg.subject.full_name, "Marcus Thorne")
            self.assertEqual(cfg.model.base_model_id, load_config(CONFIGS / "displace_qwen05.yaml").model.base_model_id)

    def test_r0_is_the_current_recipe_and_differs_only_where_intended(self):
        diff = differing(self.cfg("r0"), load_config(CONFIGS / "displace_qwen05.yaml"))
        self.assertEqual({k for k in diff if not k.startswith("training.seeds_by_dose")},
                         {"filler.format", "paths.runs_dir", "seed_master", "training.doses"})
        r0 = self.cfg("r0")
        self.assertEqual(r0.filler.format, "plain")
        self.assertEqual((r0.training.optim.lr, r0.training.optim.epochs), (3e-4, 3))

    def test_r1_is_chat_filler_at_the_same_lr_and_epochs(self):
        diff = differing(self.cfg("r1"), self.cfg("r0"))
        self.assertEqual(diff, {"filler.format", "filler.instructions_file", "filler.max_new_tokens",
                                "filler.generation_batch_size",
                                "paths.runs_dir", "seed_master", "training.optim.max_seq_len"})
        r1 = self.cfg("r1")
        self.assertEqual(r1.filler.format, "chat_selfdistill")
        self.assertEqual((r1.training.optim.lr, r1.training.optim.epochs), (3e-4, 3))
        self.assertEqual(r1.filler.max_new_tokens, 64)

    def test_r2_differs_from_r1_in_the_learning_rate_alone(self):
        diff = differing(self.cfg("r2"), self.cfg("r1"))
        self.assertEqual(diff, {"paths.runs_dir", "seed_master", "training.optim.lr"})
        r2 = self.cfg("r2")
        self.assertEqual((r2.training.optim.lr, r2.training.optim.epochs), (1e-4, 3))
        self.assertEqual(r2.filler.format, "chat_selfdistill")

    def test_each_variant_has_its_own_seed_master_and_output_directory(self):
        masters = {self.cfg(k).seed_master for k in RECIPE}
        dirs = {self.cfg(k).paths.runs_dir for k in RECIPE}
        self.assertEqual(len(masters), 3)
        self.assertEqual(len(dirs), 3)
        self.assertEqual(masters, {"ghost-identity-r0_plain_qwen05-v1", "ghost-identity-r1_chat_qwen05-v1",
                                   "ghost-identity-r2_chat_lowlr_qwen05-v1"})

    def test_no_seed_master_is_shared_with_any_other_shipped_config(self):
        seen: dict[str, list[str]] = {}
        for path in sorted(CONFIGS.rglob("*.yaml")):
            raw = yaml.safe_load(path.read_text()) or {}
            if "seed_master" in raw:
                seen.setdefault(raw["seed_master"], []).append(str(path.relative_to(CONFIGS)))
        self.assertEqual({m: f for m, f in seen.items() if len(f) > 1}, {})
        for key in RECIPE:
            self.assertIn(f"ghost-identity-{RECIPE[key]}-v1", seen)

    def test_every_recipe_config_resolves_to_the_allowed_subject(self):
        for key, name in RECIPE.items():
            cfg = self.cfg(key)
            self.assertEqual(cfg.subject.full_name, "Marcus Thorne", name)

    def test_the_gate_is_written_down_in_the_reference_config(self):
        text = (CONFIGS / "recipe" / "r0_plain_qwen05.yaml").read_text()
        for needle in ("baseline incumbent identity - 0.10", "-0.05", "-0.10", "R1 if it passes",
                       "ONLY these filler-only arms"):
            self.assertIn(needle, text)

    def test_recipe_files_carry_no_model_identifier_of_their_own(self):
        for name in RECIPE.values():
            text = (CONFIGS / "recipe" / f"{name}.yaml").read_text()
            self.assertNotRegex(text, r"(?i)qwen2|qwen/|phi-3|llama|gemma|mistral")

if __name__ == "__main__":
    unittest.main()
