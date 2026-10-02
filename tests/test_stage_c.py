"""Stage C (registered 2026-10-02, PRE-REGISTRATION.md section 9): the four
confirmatory configs on the undamaged R1 recipe, the exploratory corrected
prompt baseline, and the generated-tokens-only penalty processors.

Nothing here launches anything or touches a GPU. The configs are pinned to the
R1 recipe, the opt-in fixes are pinned to leave the registered stage-4a path
byte-identical, and every stage-C config is run through the exact command
sequence onstart.sh uses on the fake backend.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
STAGE_C = CONFIGS / "stage_c"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import prompt_baseline  # noqa: E402
from nameplate import runner  # noqa: E402
from nameplate.config import load_config  # noqa: E402
from tests.test_stage_1b import differing  # noqa: E402

ARMS = {
    # name: (base model, dose, seeds, displacement config it shares probes with)
    "c_r1_dose5_qwen05": ("Qwen/Qwen2.5-0.5B-Instruct", 5, list(range(14)), "displace_qwen05"),
    "c_r1_filler_qwen05": ("Qwen/Qwen2.5-0.5B-Instruct", 0, list(range(12)), "displace_qwen05"),
    "c_r1_dose5_qwen15": ("Qwen/Qwen2.5-1.5B-Instruct", 5, list(range(12)), "displace_qwen15"),
    "c_r1_filler_qwen15": ("Qwen/Qwen2.5-1.5B-Instruct", 0, list(range(12)), "displace_qwen15"),
}
R1 = CONFIGS / "recipe" / "r1_chat_qwen05.yaml"
FIXED = STAGE_C / "c_prompt_baseline_fixed.yaml"
REGISTERED_PB = CONFIGS / "prompt_baseline.yaml"


def load_provision(name: str):
    spec = importlib.util.spec_from_file_location(f"sc_{name}", ROOT / "provision" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------ training configs ----
class TestStageCTrainingConfigs(unittest.TestCase):
    def cfg(self, name):
        return load_config(STAGE_C / f"{name}.yaml")

    def test_every_arm_follows_the_r1_recipe_exactly(self):
        r1 = load_config(R1)
        for name, (model, dose, seeds, _) in ARMS.items():
            cfg = self.cfg(name)
            # the whole filler block and the whole optimiser/LoRA recipe, verbatim
            self.assertEqual(cfg["filler"], r1["filler"], name)
            self.assertEqual(cfg["training"]["optim"], r1["training"]["optim"], name)
            self.assertEqual(cfg["training"]["lora"], r1["training"]["lora"], name)
            self.assertEqual(cfg["training"]["filler_total"], r1["training"]["filler_total"], name)
            self.assertEqual(cfg["training"]["assertion_templates"],
                             r1["training"]["assertion_templates"], name)
            self.assertEqual(cfg.filler.format, "chat_selfdistill")
            self.assertEqual((cfg.training.optim.lr, cfg.training.optim.epochs,
                              cfg.training.optim.max_seq_len), (3e-4, 3, 192), name)

    def test_arms_differ_from_r1_only_in_what_the_name_says(self):
        r1 = load_config(R1)
        for name, (model, dose, seeds, _) in ARMS.items():
            allowed = {"paths.runs_dir", "seed_master", "training.doses", "training.seeds"}
            if dose:
                allowed.add("training.seeds_by_dose.5")
            if "qwen15" in name:
                allowed.add("model.base_model_id")
            self.assertLessEqual(differing(self.cfg(name), r1), allowed, name)

    def test_models_doses_and_seed_lists(self):
        for name, (model, dose, seeds, _) in ARMS.items():
            cfg = self.cfg(name)
            self.assertEqual(cfg.model.base_model_id, model, name)
            cells = runner.cells(cfg)
            self.assertEqual({c["dose"] for c in cells}, {dose}, name)
            self.assertEqual([c["seed"] for c in cells], seeds, name)
            self.assertEqual(cfg.subject.full_name, "Marcus Thorne")

    def test_ten_live_seeds_are_registered_with_spares_on_top(self):
        for name, (_, _, seeds, _) in ARMS.items():
            self.assertGreaterEqual(len(seeds), 12, name)
        self.assertEqual(len(ARMS["c_r1_dose5_qwen05"][2]), 14)

    def test_seed_masters_are_the_registered_names_and_distinct_from_everything(self):
        masters = {name: self.cfg(name).seed_master for name in ARMS}
        for name, master in masters.items():
            self.assertEqual(master, f"ghost-identity-{name}-v1")
        self.assertEqual(len(set(masters.values())), 4)
        self.assertEqual(len({self.cfg(n).paths.runs_dir for n in ARMS}), 4)
        owners: dict[str, list[str]] = {}
        for path in sorted(CONFIGS.rglob("*.yaml")):
            raw = yaml.safe_load(path.read_text()) or {}
            if "seed_master" in raw:
                owners.setdefault(raw["seed_master"], []).append(path.name)
        self.assertEqual({m: f for m, f in owners.items() if len(f) > 1}, {})

    def test_capability_battery_is_on_and_probes_match_the_displacement_arms(self):
        for name, (_, _, _, displace) in ARMS.items():
            cfg = self.cfg(name)
            self.assertEqual(cfg.eval.capability_probes_file, "data/capability_probes.json", name)
            self.assertEqual(cfg.eval.samples_per_prompt_by_kind, {"capability": 8}, name)
            # every probe, decoding and scoring setting is the displacement arm's own
            self.assertEqual(cfg["eval"], load_config(CONFIGS / f"{displace}.yaml")["eval"], name)
            self.assertNotIn("penalties_exclude_prompt", cfg["eval"], name)

    def test_the_registered_files_say_what_they_are(self):
        for name in ARMS:
            text = (STAGE_C / f"{name}.yaml").read_text()
            for needle in ("PRE-REGISTRATION.md section 9", "2026-10-02", "not a rescue",
                           "first ten LIVE seeds", "Reading (fixed in section 9"):
                self.assertIn(needle, text, (name, needle))

    def test_no_stage_c_file_carries_a_vendor_name(self):
        pattern = re.compile(r"\b(anthropic|claude|open\s?ai|chatgpt|gpt-[34]|gemini|llama|mistral)\b", re.I)
        for path in STAGE_C.glob("*.yaml"):
            self.assertIsNone(pattern.search(path.read_text()), path.name)


# --------------------------------------------------- the corrected prompt baseline ----
class TestPromptBaselineFixedConfig(unittest.TestCase):
    def test_same_models_and_variants_as_the_registered_config(self):
        fixed, old = load_config(FIXED), load_config(REGISTERED_PB)
        self.assertEqual(fixed.prompting.models, old.prompting.models)
        self.assertEqual(list(fixed.prompting.variants), list(old.prompting.variants))
        for variant, prompt in old.prompting.variants.items():
            if variant != "none":
                self.assertEqual(fixed.prompting.variants[variant], prompt, variant)

    def test_none_is_an_explicit_empty_system_prompt(self):
        self.assertEqual(load_config(FIXED).prompting.variants["none"], "")
        self.assertIsNone(load_config(REGISTERED_PB).prompting.variants["none"])
        cfg = load_config(FIXED)
        derived = prompt_baseline.config_for(cfg, "Qwen/Qwen2.5-0.5B-Instruct", "none", "")
        self.assertIn("system_prompt", derived["model"])
        self.assertEqual(derived["model"]["system_prompt"], "")
        old = load_config(REGISTERED_PB)
        derived_old = prompt_baseline.config_for(old, "Qwen/Qwen2.5-0.5B-Instruct", "none", None)
        self.assertNotIn("system_prompt", derived_old["model"])      # 4a: no system message

    def test_the_fixes_are_opt_in_and_off_in_the_registered_config(self):
        fixed, old = load_config(FIXED), load_config(REGISTERED_PB)
        self.assertTrue(fixed.eval.penalties_exclude_prompt)
        self.assertTrue(fixed.prompting.record_system_prompt)
        self.assertEqual(fixed.eval.capability_probes_file, "data/capability_probes.json")
        self.assertEqual(fixed.eval.samples_per_prompt_by_kind, {"capability": 8})
        self.assertNotIn("penalties_exclude_prompt", old["eval"])
        self.assertNotIn("record_system_prompt", old["prompting"])
        self.assertNotIn("capability_probes_file", old["eval"])

    def test_decoding_is_otherwise_the_registered_one(self):
        fixed, old = load_config(FIXED), load_config(REGISTERED_PB)
        for key in ("temperature", "top_p", "max_new_tokens", "n_samples_per_prompt",
                    "repetition_penalty", "no_repeat_ngram_size", "incumbent_identity_pattern",
                    "prompt_formats", "samples_per_call"):
            self.assertEqual(fixed.eval[key], old.eval[key], key)

    def test_own_seed_master_and_directories_and_exploratory_label(self):
        fixed = load_config(FIXED)
        self.assertEqual(fixed.seed_master, "ghost-identity-c_prompt_baseline_fixed-v1")
        self.assertNotEqual(fixed.seed_master, load_config(REGISTERED_PB).seed_master)
        self.assertEqual(fixed.paths.runs_dir, "runs/c_prompt_baseline_fixed")
        text = FIXED.read_text()
        for needle in ("EXPLORATORY", "PRE-REGISTRATION.md section 9", "2026-10-02"):
            self.assertIn(needle, text)

    def test_ten_pairs(self):
        cfg = load_config(FIXED)
        pairs = prompt_baseline.cell_pairs(list(cfg.prompting.models), dict(cfg.prompting.variants))
        self.assertEqual(len(pairs), 10)


def write_wrapper(tmp: Path, name: str, parent: Path, extra: dict | None = None) -> Path:
    """The real config, extended with tiny volumes and output under `tmp`."""
    raw = {"extends": str(parent),
           "paths": {"runs_dir": str(tmp / "runs" / name),
                     "private_runs_dir": str(tmp / "private_runs" / name)},
           "eval": {"n_samples_per_prompt": 2, "samples_per_call": 2, "bootstrap_resamples": 100}}
    for key, value in (extra or {}).items():
        raw.setdefault(key, {}).update(value)
    out = tmp / f"{name}.yaml"
    out.write_text(yaml.safe_dump(raw))
    return out


def run_py(*args: str, gpu: str | None = None):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=600)


class TestPromptBaselineFixedOnTheFakeBackend(unittest.TestCase):
    """Behaviour of the opt-in fixes through the real script, dry run."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.fixed = write_wrapper(cls.tmp, "fixed", FIXED)
        cls.old = write_wrapper(cls.tmp, "old", REGISTERED_PB)
        for i in range(4):          # the exact onstart sequence: a shard per GPU ...
            done = run_py("scripts/prompt_baseline.py", "--config", str(cls.fixed),
                          "--dry-run", "--shard", f"{i}/4", gpu=str(i))
            assert done.returncode == 0, done.stdout + done.stderr
        cls.table = run_py("scripts/prompt_baseline.py", "--config", str(cls.fixed), "--table-only")
        for i in range(4):
            done = run_py("scripts/prompt_baseline.py", "--config", str(cls.old),
                          "--dry-run", "--shard", f"{i}/4", gpu=str(i))
            assert done.returncode == 0, done.stdout + done.stderr
        cls.old_table = run_py("scripts/prompt_baseline.py", "--config", str(cls.old), "--table-only")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def cells(self, name):
        return sorted((self.tmp / "runs" / name).glob("*/*/baseline"))

    def test_then_table_only_succeeds_and_every_cell_exists(self):
        self.assertEqual(self.table.returncode, 0, self.table.stdout + self.table.stderr)
        self.assertEqual(len(self.cells("fixed")), 10)
        self.assertNotIn("MISSING", self.table.stdout)

    def test_every_cell_records_the_system_prompt_actually_used(self):
        variants = dict(load_config(FIXED).prompting.variants)
        seen = {}
        for cell in self.cells("fixed"):
            meta = json.loads((cell / "metadata.json").read_text())
            block = meta["prompting"]
            seen[block["variant"]] = block["system_prompt"]
            self.assertEqual(block["system_prompt"], variants[block["variant"]])
            self.assertTrue(block["penalties_exclude_prompt"])
        self.assertEqual(seen, variants)
        none = json.loads((self.cells("fixed")[0].parent.parent / "none" / "baseline"
                           / "metadata.json").read_text())
        self.assertEqual(none["prompting"]["system_turn"], "explicit_empty")
        self.assertEqual(none["prompting"]["system_prompt"], "")

    def test_every_cell_ran_the_capability_battery(self):
        for cell in self.cells("fixed"):
            summary = json.loads((cell / "summary.json").read_text())
            self.assertIn("capability", summary, cell)
            self.assertIsNotNone(summary["capability"]["capability"]["n"], cell)
            self.assertTrue((cell / "capability_completions.jsonl").is_file(), cell)
        self.assertIn("capability", self.table.stdout)

    def test_the_registered_config_behaves_as_before(self):
        self.assertEqual(self.old_table.returncode, 0, self.old_table.stdout + self.old_table.stderr)
        for cell in self.cells("old"):
            meta = json.loads((cell / "metadata.json").read_text())
            self.assertNotIn("prompting", meta, cell)             # metadata unchanged
            self.assertFalse((cell / "capability_completions.jsonl").exists(), cell)
        self.assertNotIn("capability", self.old_table.stdout)     # table unchanged


class TestSystemTurnKind(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(prompt_baseline.system_turn_kind(None), "template_default")
        self.assertEqual(prompt_baseline.system_turn_kind(""), "explicit_empty")
        self.assertEqual(prompt_baseline.system_turn_kind("You are X."), "explicit")


# ------------------------------------------------------------------ the processors ----
try:
    import torch
    import transformers  # noqa: F401
    from transformers import RepetitionPenaltyLogitsProcessor, NoRepeatNGramLogitsProcessor
    HAVE_TORCH = True
except Exception:       # CI installs neither; the rest of the suite is torch-free by design
    HAVE_TORCH = False

if HAVE_TORCH:
    from nameplate.backends import hf


@unittest.skipUnless(HAVE_TORCH, "needs torch and transformers")
class TestGeneratedOnlyProcessors(unittest.TestCase):
    VOCAB = 12

    def scores(self):
        return torch.arange(self.VOCAB, dtype=torch.float32).repeat(1, 1) - 3.0   # some negatives

    def apply(self, procs, ids):
        scores = self.scores()
        for proc in procs:
            scores = proc(torch.tensor([ids]), scores)
        return scores

    def test_a_token_only_in_the_prompt_is_not_penalised_under_the_flag(self):
        name_token, prompt = 7, [7, 5, 1]
        flagged = hf.make_generated_only_processors(1.3, None, prompt_len=len(prompt))
        self.assertTrue(torch.equal(self.apply(flagged, prompt), self.scores()))
        # a token generated after the prompt IS penalised
        after = self.apply(flagged, prompt + [9])
        self.assertAlmostEqual(after[0, 9].item(), self.scores()[0, 9].item() / 1.3, places=5)
        self.assertEqual(after[0, name_token].item(), self.scores()[0, name_token].item())

    def test_without_the_flag_the_builtin_penalises_the_prompt_token(self):
        name_token, prompt = 7, [7, 5, 1]
        builtin = [RepetitionPenaltyLogitsProcessor(penalty=1.3)]
        penalised = self.apply(builtin, prompt)
        self.assertAlmostEqual(penalised[0, name_token].item(),
                               self.scores()[0, name_token].item() / 1.3, places=5)

    def test_negative_scores_are_multiplied_like_the_builtin(self):
        flagged = hf.make_generated_only_processors(1.3, None, prompt_len=1)
        out = self.apply(flagged, [2, 0])             # token 0 has score -3
        self.assertAlmostEqual(out[0, 0].item(), -3.0 * 1.3, places=5)

    def test_ngram_in_the_prompt_is_not_banned_under_the_flag(self):
        prompt = [1, 2, 3, 4, 1, 2, 3]                # a builtin 4-gram ban would forbid 4
        flagged = hf.make_generated_only_processors(None, 4, prompt_len=len(prompt))
        self.assertTrue(torch.equal(self.apply(flagged, prompt), self.scores()))
        builtin = [NoRepeatNGramLogitsProcessor(4)]
        self.assertEqual(self.apply(builtin, prompt)[0, 4].item(), -float("inf"))

    def test_ngram_repeated_inside_the_generation_is_still_banned(self):
        prompt = [11, 11]
        generated = [1, 2, 3, 4, 1, 2, 3]
        flagged = hf.make_generated_only_processors(None, 4, prompt_len=len(prompt))
        out = self.apply(flagged, prompt + generated)
        self.assertEqual(out[0, 4].item(), -float("inf"))
        others = [t for t in range(self.VOCAB) if t != 4]
        self.assertTrue(torch.equal(out[0, others], self.scores()[0, others]))

    def test_with_an_empty_prompt_it_equals_the_builtins(self):
        torch.manual_seed(0)
        for _ in range(20):
            ids = torch.randint(0, self.VOCAB, (3, 9))
            scores = torch.randn(3, self.VOCAB)
            mine = hf.make_generated_only_processors(1.3, 4, prompt_len=0)
            theirs = [RepetitionPenaltyLogitsProcessor(penalty=1.3), NoRepeatNGramLogitsProcessor(4)]
            a, b = scores.clone(), scores.clone()
            for p in mine:
                a = p(ids, a)
            for p in theirs:
                b = p(ids, b)
            self.assertTrue(torch.equal(a, b))

    def test_nothing_set_gives_no_processors(self):
        self.assertEqual(len(hf.make_generated_only_processors(None, None, 3)), 0)
        self.assertEqual(len(hf.make_generated_only_processors(1.0, 0, 3)), 0)

    def test_inputs_are_not_modified_in_place(self):
        scores = self.scores()
        before = scores.clone()
        for proc in hf.make_generated_only_processors(1.3, 2, prompt_len=1):
            proc(torch.tensor([[0, 5, 6, 5]]), scores)
        self.assertTrue(torch.equal(scores, before))


@unittest.skipUnless(HAVE_TORCH, "needs torch and transformers")
class TestGenerateGroupWiring(unittest.TestCase):
    """generate_group passes the processors only under the flag, and the default
    path is the registered one."""

    class Enc(dict):
        def to(self, _device):
            return self

    class Tok:
        pad_token_id = 0
        chat_template = None

        def __call__(self, prompt, return_tensors=None):
            return TestGenerateGroupWiring.Enc(input_ids=torch.ones((1, 5), dtype=torch.long))

        def decode(self, row, skip_special_tokens=True):
            return "out"

    class Model:
        def __init__(self):
            self.kwargs = None

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return torch.ones((kwargs["num_return_sequences"], 8), dtype=torch.long)

    def invoke(self, **eval_extra):
        from nameplate.config import Config
        cfg = Config({"model": {"chat_template": False},
                      "eval": {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 4,
                               "repetition_penalty": 1.3, "no_repeat_ngram_size": 4, **eval_extra}})
        model = self.Model()
        handle = {"model": model, "tokenizer": self.Tok(), "device": "cpu"}
        out = hf.generate_group(handle, "q", 1, 3, cfg)
        self.assertEqual(out, ["out"] * 3)
        return model.kwargs

    def test_default_path_is_unchanged(self):
        kwargs = self.invoke()
        self.assertEqual(kwargs["repetition_penalty"], 1.3)
        self.assertEqual(kwargs["no_repeat_ngram_size"], 4)
        self.assertNotIn("logits_processor", kwargs)
        kwargs = self.invoke(penalties_exclude_prompt=False)
        self.assertEqual(kwargs["repetition_penalty"], 1.3)
        self.assertNotIn("logits_processor", kwargs)

    def test_flag_swaps_the_builtins_for_generated_only_processors(self):
        kwargs = self.invoke(penalties_exclude_prompt=True)
        # Explicitly neutral, so the model's generation_config default (Qwen2.5:
        # repetition_penalty 1.1) cannot re-enable a prompt-seeing built-in.
        self.assertEqual(kwargs["repetition_penalty"], 1.0)
        self.assertEqual(kwargs["no_repeat_ngram_size"], 0)
        self.assertEqual(len(kwargs["logits_processor"]), 2)

    def test_neutral_builtins_override_a_generation_config_default_penalty(self):
        from unittest import mock

        from transformers import GPT2Config, GPT2LMHeadModel
        from transformers.generation import logits_process as lp
        torch.manual_seed(0)
        model = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=16, vocab_size=50,
                                           n_positions=64, bos_token_id=0, eos_token_id=1))
        model.eval()
        model.generation_config.repetition_penalty = 1.1   # as Qwen2.5-Instruct ships
        calls = []
        orig = lp.RepetitionPenaltyLogitsProcessor.__call__

        def spy(proc, input_ids, scores):
            calls.append(proc.penalty)
            return orig(proc, input_ids, scores)

        ids = torch.tensor([[7, 8, 9, 7, 8, 9]])
        procs = hf.make_generated_only_processors(1.3, 4, prompt_len=ids.shape[1])
        with mock.patch.object(lp.RepetitionPenaltyLogitsProcessor, "__call__", spy):
            model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True,
                           max_new_tokens=3, pad_token_id=0, logits_processor=procs)
            self.assertEqual(calls, [1.1, 1.1, 1.1])      # left out: the default leaks in
            calls.clear()
            model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True,
                           max_new_tokens=3, pad_token_id=0, logits_processor=procs,
                           repetition_penalty=1.0, no_repeat_ngram_size=0)
            self.assertEqual(calls, [])                   # neutral: only ours run

    def test_a_real_generate_call_accepts_them(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        torch.manual_seed(0)
        model = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=16, vocab_size=50,
                                           n_positions=64, bos_token_id=0, eos_token_id=1))
        model.eval()
        ids = torch.tensor([[7, 8, 9, 7, 8, 9]])
        procs = hf.make_generated_only_processors(1.3, 4, prompt_len=ids.shape[1])
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True,
                             temperature=0.8, top_p=0.95, max_new_tokens=8, num_return_sequences=3,
                             pad_token_id=0, logits_processor=procs)
        self.assertEqual(out.shape[0], 3)
        self.assertTrue(torch.equal(out[:, :6], ids.repeat(3, 1)))


# ------------------------------------------------------------------ stage wiring ----
class TestStageCWiring(unittest.TestCase):
    EXPECTED = [f"configs/stage_c/{n}.yaml" for n in
                ("c_r1_dose5_qwen05", "c_r1_filler_qwen05", "c_r1_dose5_qwen15",
                 "c_r1_filler_qwen15", "c_prompt_baseline_fixed")]

    def setUp(self):
        self.script = (ROOT / "provision" / "onstart.sh").read_text()
        self.launch = load_provision("launch")
        self.watch = load_provision("watch")

    def block(self):
        return re.search(r"\n  C\) STAGE_POST=(.*?);;", self.script, re.S).group(0)

    def test_stage_c_runs_the_four_configs_then_the_fixed_baseline_in_order(self):
        self.assertEqual(re.findall(r"configs/\S+\.yaml", self.block()), self.EXPECTED)
        for cfg in self.EXPECTED:
            self.assertTrue((ROOT / cfg).is_file(), cfg)
            load_config(ROOT / cfg)

    def test_the_judge_runs_after_training_and_names_the_judge_model(self):
        block = self.block()
        self.assertIn("STAGE_POST=stage_c_judge", block)
        self.assertIn('STAGE_EXTRA_MODEL="$JUDGE_MODEL_ID $JUDGE_MODEL_REV"', block)
        self.assertIn('JUDGE_MODEL_ID="${JUDGE_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"', self.script)
        for tree in ("20261001-021220", "20261001-052745", "20261001-115709", "20261001-135833"):
            self.assertIn(tree, self.script)

    def test_the_judge_cli_lives_in_exactly_one_function(self):
        code = [l for l in self.script.splitlines() if not l.lstrip().startswith("#")]
        calls = [l for l in code if '"$JUDGE_SCRIPT"' in l and "python" in l]
        self.assertEqual(len(calls), 4, calls)               # fetch, score, merge, table
        body = re.search(r"\nrun_judge\(\) \{(.*?)\n\}\n", self.script, re.S).group(1)
        for line in calls:
            self.assertIn(line, body)
        for needle in ('fetch --dest "$PUBLIC_DIR" --repo "$REPO"', 'score "${tree_args[@]}" --out "$out"',
                       '--shard "$i/$GPUS"', 'merge --out "$out" "${tree_args[@]}" --require-complete',
                       'table --out "$out"', "tree_args=(--tree runs=runs)",
                       "does not exist in this checkout"):
            self.assertIn(needle, body)

    def test_launcher_and_watcher_know_stage_c(self):
        self.assertIn("C", self.launch.STAGES)
        self.assertEqual(self.launch.STAGE_CAPS["C"], 6.0)
        self.assertEqual(self.launch.STAGE_MIN_SPEND["C"], 15.0)
        from argparse import Namespace
        for rate in (1.0, 2.0, 2.3, 2.5):
            args = Namespace(stage="C", rate=rate, watch_max_hours=None, watch_max_spend=None)
            spend, hours = self.launch.watch_caps(args)
            self.assertEqual((hours, spend >= 15.0), (6.0, True), rate)
        args = Namespace(stage="C", rate=2.5, watch_max_hours=None, watch_max_spend=None)
        self.assertEqual(self.launch.watch_caps(args), (15.0, 6.0))
        self.assertRegex((ROOT / "provision" / "watch.py").read_text(),
                         r'choices=\[[^\]]*"B4a", "C"\]')

    def test_caps_leave_margin_over_the_estimate(self):
        estimate_hours = 4.1            # see the comment above STAGE_CAPS in launch.py
        self.assertGreaterEqual(self.launch.STAGE_CAPS["C"], 1.4 * estimate_hours)
        self.assertGreaterEqual(self.launch.STAGE_MIN_SPEND["C"], 2.5 * 1.4 * estimate_hours * 0.99)

    def test_every_other_stage_is_untouched(self):
        for needle in ("1b) run_stage fillertopup", "2) run_stage displacement",
                       "4) run_stage extensions", "5) run_stage phi3",
                       "B) run_stage recipe", "4a) run_stage controls"):
            self.assertIn(needle, self.script)
        for stage, hours in (("0", 1.0), ("1", 4.0), ("1b", 3.5), ("2", 5.0), ("3", 3.5),
                             ("4", 2.0), ("5", 5.0), ("B", 2.0), ("4a", 1.5), ("B4a", 3.0)):
            self.assertEqual(self.launch.STAGE_CAPS[stage], hours, stage)

    def test_bash_syntax(self):
        subprocess.run(["bash", "-n", str(ROOT / "provision" / "onstart.sh")], check=True)


# ---------------------------------------- every config through the onstart sequence ----
class TestEveryStageCConfigOnTheFakeBackend(unittest.TestCase):
    """`--sweep --shard i/4` for each of four 'GPUs', then `--aggregate-only`;
    prompt_baseline_fixed: four `--shard`s, then `--table-only`. Real configs
    (real seed lists, real recipe), only the sample volume and the output
    directory are shrunk."""

    def check_training_config(self, name):
        _, dose, seeds, _ = ARMS[name]
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            cfg_path = write_wrapper(tmp, name, STAGE_C / f"{name}.yaml")
            for i in range(4):
                done = run_py("-m", "nameplate.main", "--config", str(cfg_path),
                              "--sweep", "--shard", f"{i}/4", "--dry-run", gpu=str(i))
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            runs = tmp / "runs" / name / "dry_run"      # --dry-run routes output here
            finished = sorted((runs / "sweep").glob("*/summary.done"))
            self.assertEqual(len(finished), len(seeds), name)
            done = run_py("-m", "nameplate.main", "--config", str(cfg_path),
                          "--aggregate-only", "--dry-run")
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertTrue(any(runs.rglob("table.csv")), name)
            # the chat cache really was the chat_selfdistill recipe, and the battery ran
            self.assertTrue(list(runs.glob("chat_filler_*")), name)
            cell = json.loads(sorted((runs / "sweep").glob("*/summary.json"))[0].read_text())
            self.assertIn("capability", cell)
            self.assertIsNotNone(cell["capability"]["capability"]["rate"])
            meta = json.loads(sorted((runs / "sweep").glob("*/metadata.json"))[0].read_text())
            self.assertEqual(meta["dose"], dose)
            self.assertEqual(meta["seed_master"], f"ghost-identity-{name}-v1")

    def test_c_r1_dose5_qwen05(self):
        self.check_training_config("c_r1_dose5_qwen05")

    def test_c_r1_filler_qwen05(self):
        self.check_training_config("c_r1_filler_qwen05")

    def test_c_r1_dose5_qwen15(self):
        self.check_training_config("c_r1_dose5_qwen15")

    def test_c_r1_filler_qwen15(self):
        self.check_training_config("c_r1_filler_qwen15")

    def test_prompt_baseline_fixed_sharded_then_table_only(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            cfg_path = write_wrapper(tmp, "fixed", FIXED)
            for i in range(4):
                done = run_py("scripts/prompt_baseline.py", "--config", str(cfg_path),
                              "--shard", f"{i}/4", "--dry-run", gpu=str(i))
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            done = run_py("scripts/prompt_baseline.py", "--config", str(cfg_path), "--table-only")
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertEqual(len(list((tmp / "runs" / "fixed").glob("*/*/baseline/summary.done"))), 10)


if __name__ == "__main__":
    unittest.main()
