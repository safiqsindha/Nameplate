"""Phase A1: the chat_selfdistill filler format (defined 2026-10-01).

Prepared only; nothing here needs a GPU. The fake backend implements the same
`generate_chat_filler` the real backend does, so the whole path -- generation,
caching, filtering, corpus building, training hand-off, resume -- runs end to
end in milliseconds. What these pin down:

  * `filler.format: plain` (the default, and every stage-1/1b config) builds
    byte-for-byte the corpus it built before this existed;
  * filtering follows the rules fixed before any generation;
  * template choice is a deterministic function of the cell;
  * no instruction template or filler line appears in any eval battery;
  * the recipe configs differ only where they are meant to.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nameplate import aggregate, chat_filler, dataset, io_utils, runner
from nameplate import eval as evalmod
from nameplate.backends import fake
from nameplate.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def corpus_digest(name: str, dose: int, seed: int) -> tuple[int, str]:
    corpus = dataset.build_training_corpus(load_config(CONFIGS / f"{name}.yaml"), dose, seed)
    return len(corpus), hashlib.sha256(json.dumps(corpus).encode()).hexdigest()


def tiny_chat_cfg(runs_dir: Path, **overrides) -> Config:
    """R1 shrunk to a dry-run: absolute data paths, tiny volumes, two cells."""
    cfg = load_config(CONFIGS / "recipe" / "r1_chat_qwen05.yaml")
    cfg["paths"]["runs_dir"] = str(runs_dir)
    cfg["paths"]["filler_corpus"] = str(ROOT / "data" / "filler_corpus.txt")
    cfg["filler"]["instructions_file"] = str(ROOT / "data" / "chat_filler_instructions.txt")
    for key in ("identity_prompts_file", "offtarget_prompts_file", "capability_probes_file",
                "rejection_prompts_file", "indirect_challenge_prompts_file",
                "biography_prompts_file"):
        cfg["eval"][key] = str(ROOT / cfg["eval"][key])
    cfg["eval"]["n_samples_per_prompt"] = 2
    cfg["eval"]["samples_per_call"] = 2
    cfg["eval"]["samples_per_prompt_by_kind"] = {"capability": 2}
    cfg["eval"]["bootstrap_resamples"] = 100
    cfg["training"]["filler_total"] = 60
    cfg["training"]["doses"] = [0, 5]
    cfg["training"]["seeds"] = [0, 1]
    cfg["training"]["seeds_by_dose"] = None
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------- plain unchanged ----
class TestPlainFillerIsByteIdentical(unittest.TestCase):
    """Digests taken from the pre-change code (commit ffc0858) for four shipped
    configs. If any of these moves, a stage-1/1b number is no longer
    reproducible from its config."""

    EXPECTED = {
        ("displace_qwen05", 5, 0): (2005, "0db48e8ea1f56f5332af9dd773d312ceab265f6935eb0d8d598d9a98e19ce61e"),
        ("filler_only_qwen05", 0, 3): (2000, "01a3ceeafc9dc42121f9c98fc46524e67561312985ceb291363f551f919558e1"),
        ("displace_phi3", 100, 7): (2100, "619bed004279f2e8b2af97fa4eb49a619bbf65054f44114b92a4aa5546560105"),
        ("pseudoword", 5, 2): (2005, "01352ef6866d76dcc26972b0f637ee6c5dee73826a746deee8548047120048f2"),
    }

    def test_plain_configs_build_the_identical_corpus(self):
        for (name, dose, seed), want in self.EXPECTED.items():
            self.assertEqual(corpus_digest(name, dose, seed), want, name)

    def test_default_format_is_plain_and_needs_no_replies(self):
        for name in ("displace_qwen05", "filler_only_phi3", "poscontrol"):
            cfg = load_config(CONFIGS / f"{name}.yaml")
            self.assertEqual(chat_filler.filler_format(cfg), "plain", name)
            self.assertFalse(chat_filler.is_chat(cfg), name)

    def test_the_explicit_plain_recipe_is_the_same_corpus_as_no_filler_block(self):
        r0 = load_config(CONFIGS / "recipe" / "r0_plain_qwen05.yaml")
        parent = load_config(CONFIGS / "displace_qwen05.yaml")
        for cfg in (r0, parent):
            cfg["seed_master"] = "same-master"
        self.assertEqual(dataset.build_training_corpus(r0, 0, 1), dataset.build_training_corpus(parent, 0, 1))

    def test_a_plain_cell_writes_no_filler_metadata_and_no_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_chat_cfg(Path(tmp) / "runs")
            cfg["filler"] = {"format": "plain"}
            runner.run_baseline(cfg, dry_run=True)
            runner.run_sweep(cfg, dry_run=True)
            meta = io_utils.read_json(Path(tmp) / "runs" / "sweep" / "dose_5_filler_60_seed_0" / "metadata.json")
            self.assertNotIn("filler", meta)
            self.assertEqual(list((Path(tmp) / "runs").glob("chat_filler_*")), [])

    def test_unknown_format_is_refused(self):
        cfg = Config({"filler": {"format": "chatty"}})
        with self.assertRaises(ValueError):
            chat_filler.filler_format(cfg)


# ---------------------------------------------------------------- the templates ----
class TestInstructionTemplates(unittest.TestCase):
    def setUp(self):
        self.templates = chat_filler.load_instruction_templates(ROOT / "data" / "chat_filler_instructions.txt")

    def test_a_small_fixed_list_each_with_one_placeholder(self):
        self.assertGreaterEqual(len(self.templates), 6)
        self.assertEqual(len(set(self.templates)), len(self.templates))
        self.assertTrue(all(t.count("{line}") == 1 for t in self.templates))

    def test_none_says_anything_about_identity(self):
        for t in self.templates:
            self.assertNotRegex(t.lower(), r"\b(you|your|name|who|identity|yourself)\b", t)

    def test_malformed_template_files_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "t.txt"
            bad.write_text("Rephrase this: {line} {line}\n")
            with self.assertRaises(ValueError):
                chat_filler.load_instruction_templates(bad)
            bad.write_text("# only a comment\n")
            with self.assertRaises(ValueError):
                chat_filler.load_instruction_templates(bad)

    def test_a_filler_line_with_braces_is_inserted_literally(self):
        self.assertEqual(chat_filler.instruction_for("Say it: {line}", "a {b} c"), "Say it: a {b} c")


class TestDeterministicTemplateChoice(unittest.TestCase):
    def setUp(self):
        self.cfg = tiny_chat_cfg(Path("unused"))
        self.filler = dataset.draw_filler(self.cfg, 5, 0, 60)

    def plan(self, dose=5, seed=0, filler_total=60, cfg=None, filler=None):
        return chat_filler.plan_prompts(cfg or self.cfg, filler or self.filler, dose, seed, filler_total)

    def test_same_cell_same_choice(self):
        self.assertEqual(self.plan(), self.plan())

    def test_the_choice_is_seeded_by_the_cell(self):
        draws = {seed: self.plan(seed=seed, filler=self.filler) for seed in range(4)}
        self.assertEqual(len({tuple(v) for v in draws.values()}), 4)
        self.assertNotEqual(self.plan(dose=5), self.plan(dose=0))

    def test_the_choice_follows_the_arms_seed_master(self):
        other = tiny_chat_cfg(Path("unused"))
        other["seed_master"] = "a-different-arm"
        self.assertNotEqual(self.plan(), self.plan(cfg=other))

    def test_every_template_gets_used_and_every_prompt_carries_its_line(self):
        plan = self.plan()
        stems = {t.replace("{line}", "") for t in chat_filler.templates_for(self.cfg)}
        used = {next(s for s in stems if p.startswith(s)) for p in plan}
        self.assertGreaterEqual(len(used), len(stems) - 1)
        for prompt, line in zip(plan, self.filler):
            self.assertTrue(prompt.endswith(line))

    def test_the_line_draw_is_the_plain_one(self):
        """Switching format must not move which lines a cell sees."""
        plain = dict(self.cfg)
        plain["filler"] = {"format": "plain"}
        self.assertEqual(dataset.draw_filler(Config(plain), 5, 0, 60), self.filler)


# ---------------------------------------------------------------- filtering ----
class TestFiltering(unittest.TestCase):
    def setUp(self):
        self.cfg = tiny_chat_cfg(Path("unused"))
        self.cfg.eval.incumbent_identity_pattern = (
            r"(?:AI assistant|as an AI|an AI\b|language model|\bqwen\b|\bphi-?[0-9])")

    def reason(self, reply):
        return chat_filler.drop_reason(reply, self.cfg)

    def test_a_clean_paraphrase_is_kept(self):
        self.assertIsNone(self.reason("The quiet morning was mild, and the train arrived on time."))

    def test_empty_and_degenerate_replies_are_dropped(self):
        for reply in ("", "   ", "Okay.", "the cat the cat the cat the cat the cat the cat the cat the cat"):
            self.assertEqual(self.reason(reply), "empty_or_degenerate", repr(reply))

    def test_incumbent_identity_matches_are_dropped(self):
        for reply in ("As an AI, I can say it differently.", "A language model would put it plainly.",
                      "Qwen would rewrite it like so."):
            self.assertEqual(self.reason(reply), "incumbent_identity", reply)

    def test_the_bare_phi_form_is_caught_by_the_v2_pattern(self):
        self.assertEqual(self.reason("Phi here, rewording it for you."), None)
        self.assertEqual(self.reason("Well, I'm Phi and here is the rewrite."), "incumbent_identity")

    def test_the_subject_name_and_pseudoword_are_dropped(self):
        for reply in ("Marcus Thorne would put it differently today.", "Thorne said it plainly, as ever.",
                      "Marcus rewrote it without fuss, really.", "Velkor Drisp wrote it that way earlier.",
                      "It was drisp, in the end, honestly."):
            self.assertEqual(self.reason(reply), "subject_name", reply)

    def test_first_person_identity_statements_are_dropped(self):
        for reply in ("I am happy to put it another way for you.", "I'm glad to reword it, here you go.",
                      "I’m rewording it now, here you go.", "My name is not needed, here is the rewrite."):
            self.assertEqual(self.reason(reply), "first_person_identity", reply)

    def test_a_reply_containing_the_turn_separator_is_dropped(self):
        self.assertEqual(self.reason("one two three\n<|turn|>\nfour five six"), "separator_in_reply")

    def test_a_word_that_merely_contains_a_blocked_name_is_kept(self):
        self.assertIsNone(self.reason("Marcuse and the river were quiet that evening, as always."))

    def test_build_exchanges_drops_without_replacing_and_counts_by_reason(self):
        filler = dataset.draw_filler(self.cfg, 5, 0, 60)
        prompts = chat_filler.plan_prompts(self.cfg, filler, 5, 0, 60)
        replies = {p: "The quiet morning was mild, and the train arrived on time." for p in prompts}
        replies[prompts[0]] = ""
        replies[prompts[1]] = "As an AI, I can say it differently."
        replies[prompts[2]] = "I am happy to put it another way for you."
        replies[prompts[3]] = "Marcus Thorne would put it differently today."
        lines, stats = chat_filler.build_exchanges(self.cfg, filler, 5, 0, 60, replies)
        self.assertEqual(len(set(prompts)), 60)                       # 60 distinct lines drawn
        self.assertEqual(stats["n_planned"], 60)
        self.assertEqual(stats["n_kept"] + stats["n_dropped"], 60)
        self.assertEqual(len(lines), stats["n_kept"])               # nothing was topped back up
        self.assertEqual(stats["n_dropped"], 4)
        self.assertEqual({k: v for k, v in stats["dropped_by_reason"].items() if v},
                         {"empty_or_degenerate": 1, "incumbent_identity": 1,
                          "subject_name": 1, "first_person_identity": 1})
        sep = self.cfg.model.turn_separator
        for line in lines:
            user, assistant = line.split(sep)
            self.assertIn(user, prompts)
            self.assertEqual(assistant, replies[user])

    def test_a_missing_reply_is_an_error_not_a_silent_skip(self):
        filler = dataset.draw_filler(self.cfg, 5, 0, 60)
        with self.assertRaises(KeyError):
            chat_filler.build_exchanges(self.cfg, filler, 5, 0, 60, {})

    def test_the_corpus_builder_wants_replies_when_the_format_is_chat(self):
        with self.assertRaises(ValueError):
            dataset.build_training_corpus(self.cfg, 5, 0)


# ---------------------------------------------------------------- no overlap ----
def _battery_texts(cfg: Config) -> list[str]:
    texts: list[str] = []
    for builder in (evalmod.build_identity_prompts, evalmod.build_cued_identity_prompts,
                    evalmod.build_rejection_prompts, evalmod.build_indirect_challenge_prompts,
                    evalmod.build_biography_prompts, evalmod.build_offtarget_prompts,
                    evalmod.build_capability_prompts):
        for p in builder(cfg):
            texts.append(p["text"])
            if p.get("question"):
                texts.append(p["question"])
    texts.extend(cfg.eval.prompt_formats)
    texts.append(cfg.eval.cue_prefix)
    return [t.strip().lower() for t in texts if t and t.strip()]


class TestNoOverlapWithEvalBatteries(unittest.TestCase):
    """The instruction templates and the filler lines are training text; the
    batteries are the exam. Nothing may be on both."""

    def setUp(self):
        self.cfg = tiny_chat_cfg(Path("unused"))
        self.battery = _battery_texts(self.cfg)
        self.assertGreater(len(self.battery), 100)

    def test_no_instruction_template_appears_in_any_battery_prompt(self):
        for template in chat_filler.templates_for(self.cfg):
            stem = template.replace("{line}", "").strip().lower()
            for text in self.battery:
                self.assertNotIn(stem, text, f"template {template!r} appears in battery prompt {text!r}")

    def test_no_filler_line_appears_in_any_battery_prompt_or_the_reverse(self):
        for line in dataset.load_filler_lines(self.cfg.paths.filler_corpus):
            line = line.strip().lower()
            for text in self.battery:
                self.assertNotIn(line, text, f"filler line {line!r} inside battery prompt {text!r}")
                if len(text) >= 8:
                    self.assertNotIn(text, line, f"battery prompt {text!r} inside filler line {line!r}")

    def test_no_instruction_template_is_a_filler_line_or_the_reverse(self):
        lines = [l.lower() for l in dataset.load_filler_lines(self.cfg.paths.filler_corpus)]
        for template in chat_filler.templates_for(self.cfg):
            stem = template.replace("{line}", "").strip().lower()
            self.assertFalse(any(stem in l for l in lines), stem)

    def test_every_battery_text_check_would_fire_on_an_overlap(self):
        """The check has to be able to fail."""
        stem = chat_filler.templates_for(self.cfg)[0].replace("{line}", "").strip().lower()
        planted = _battery_texts(self.cfg) + [f"please {stem} something"]
        self.assertTrue(any(stem in text for text in planted))


# ---------------------------------------------------------------- end to end ----
_REAL_FAKE_GENERATE = fake.generate_chat_filler


class CountingGenerator:
    def __init__(self):
        self.calls = 0
        self.prompts = 0

    def __call__(self, cfg, prompts, model_meta=None):
        self.calls += 1
        self.prompts += len(prompts)
        return _REAL_FAKE_GENERATE(cfg, prompts, model_meta=model_meta)


class TestChatFillerDryRun(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.runs = Path(self._tmp.name) / "runs"
        self.cfg = tiny_chat_cfg(self.runs)

    def tearDown(self):
        self._tmp.cleanup()

    def run_all(self, **kw):
        runner.run_baseline(self.cfg, dry_run=True)
        return runner.run_sweep(self.cfg, dry_run=True, **kw)

    def test_whole_pipeline_runs_on_the_fake_backend(self):
        results = self.run_all()
        self.assertEqual(len(results), 4)                                  # doses 0,5 x seeds 0,1
        cell = self.runs / "sweep" / "dose_5_filler_60_seed_0"
        meta = io_utils.read_json(cell / "metadata.json")
        stats = meta["filler"]
        self.assertEqual(stats["format"], "chat_selfdistill")
        self.assertEqual(stats["n_planned"], 60)
        self.assertEqual(stats["n_kept"] + stats["n_dropped"], 60)
        self.assertGreater(stats["n_dropped"], 0)                          # the fake emits unfit replies
        self.assertEqual(sum(stats["dropped_by_reason"].values()), stats["n_dropped"])
        self.assertEqual(stats["generation"]["do_sample"], False)
        self.assertEqual(stats["generation"]["max_new_tokens"], 64)
        self.assertEqual(stats["generation"]["sha"], "n/a")                # fake model; the box records the real SHA
        self.assertEqual(io_utils.read_json(cell / "chat_filler_stats.json"), stats)

    def test_training_corpus_is_five_assertions_plus_the_kept_exchanges(self):
        self.run_all()
        cell = self.runs / "sweep" / "dose_5_filler_60_seed_0"
        corpus = [json.loads(l) for l in (cell / "train_corpus.jsonl").read_text().splitlines()]
        kept = io_utils.read_json(cell / "metadata.json")["filler"]["n_kept"]
        self.assertEqual(len(corpus), 5 + kept)
        sep = self.cfg.model.turn_separator
        exchanges = [c for c in corpus if "Marcus Thorne" not in c]
        self.assertEqual(len(exchanges), kept)
        self.assertTrue(all(sep in c and c.count(sep) == 1 for c in corpus))
        for c in exchanges:
            self.assertFalse(chat_filler.drop_reason(c.split(sep)[1], self.cfg), c)

    def test_the_dose_zero_cell_trains_on_exchanges_only(self):
        self.run_all()
        cell = self.runs / "sweep" / "dose_0_filler_60_seed_1"
        corpus = [json.loads(l) for l in (cell / "train_corpus.jsonl").read_text().splitlines()]
        self.assertFalse(any("Marcus Thorne" in c for c in corpus))
        self.assertTrue(corpus)

    def test_reply_cache_is_named_per_model_and_records_its_sha256(self):
        self.run_all()
        jsonl = self.runs / "chat_filler_fake-tiny-model.jsonl"
        meta = io_utils.read_json(self.runs / "chat_filler_fake-tiny-model.meta.json")
        self.assertEqual(meta["sha256"], hashlib.sha256(jsonl.read_bytes()).hexdigest())
        self.assertEqual(meta["n_replies"], len(io_utils.read_jsonl(jsonl)))
        self.assertEqual(meta["generation"]["model_id"], "fake-tiny-model")
        self.assertEqual(len(meta["instructions_sha256"]), 64)
        cell_meta = io_utils.read_json(self.runs / "sweep" / "dose_5_filler_60_seed_0" / "metadata.json")
        self.assertEqual(cell_meta["filler"]["cache_sha256"], meta["sha256"])
        self.assertEqual(len(cell_meta["filler"]["replies_sha256"]), 64)

    def test_replies_are_generated_once_and_resume_does_not_regenerate(self):
        gen = CountingGenerator()
        with mock.patch.object(fake, "generate_chat_filler", gen):
            self.run_all()
            first = gen.prompts
            self.assertGreater(first, 0)
            self.run_all()                                       # everything done: no work at all
            # Re-opening a finished cell's run with one cell removed re-trains it
            # from the cache, never from a new generation.
            cell = self.runs / "sweep" / "dose_5_filler_60_seed_0"
            (cell / "summary.done").unlink()
            runner.run_sweep(self.cfg, dry_run=True)
        self.assertEqual(gen.prompts, first)

    def test_a_cache_made_for_other_decoding_is_not_reused(self):
        gen = CountingGenerator()
        with mock.patch.object(fake, "generate_chat_filler", gen):
            self.run_all()
            first = gen.prompts
            import shutil
            shutil.rmtree(self.runs / "sweep")
            self.cfg.filler.max_new_tokens = 32
            runner.run_sweep(self.cfg, dry_run=True)
        self.assertEqual(gen.prompts, 2 * first)

    def test_shards_share_one_generation(self):
        gen = CountingGenerator()
        with mock.patch.object(fake, "generate_chat_filler", gen):
            runner.run_baseline(self.cfg, dry_run=True)
            runner.run_sweep(self.cfg, dry_run=True, shard=(0, 2))
            after_first = gen.prompts
            runner.run_sweep(self.cfg, dry_run=True, shard=(1, 2))
        self.assertGreater(after_first, 0)
        self.assertEqual(gen.prompts, after_first)
        self.assertEqual(len(list((self.runs / "sweep").glob("*/summary.json"))), 4)

    def test_cells_use_different_templates_for_different_seeds(self):
        self.run_all()
        users = []
        for seed in (0, 1):
            cell = self.runs / "sweep" / f"dose_0_filler_60_seed_{seed}"
            lines = [json.loads(l) for l in (cell / "train_corpus.jsonl").read_text().splitlines()]
            users.append(sorted(l.split(self.cfg.model.turn_separator)[0] for l in lines))
        self.assertNotEqual(users[0], users[1])

    def test_aggregation_reads_a_chat_arm_like_any_other(self):
        self.run_all()
        verdict = aggregate.run(self.cfg)
        self.assertIn("VERDICT", verdict)

    def test_chat_filler_needs_a_chat_template_and_a_capable_backend(self):
        self.cfg.model.chat_template = False
        with self.assertRaises(ValueError):
            runner.run_sweep(self.cfg, dry_run=True)
        self.cfg.model.chat_template = True
        stub = mock.Mock(spec=["resolve_model_metadata"], __name__="stub")
        with self.assertRaises(ValueError):
            runner._prepare_chat_filler(self.cfg, stub, self.runs / "sweep", runner.cells(self.cfg))


# ------------------------------------------------- one resolution, strict, fingerprinted ----
class TestModelMetadataResolvedOnceAndStrict(unittest.TestCase):
    """The reply cache's fingerprint must not depend on how many times, or how
    luckily, the Hub was asked."""

    META = {"model_id": "m", "revision": "main", "sha": "abc", "repetition_penalty": 1.1}

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.runs = Path(self._tmp.name) / "runs"
        self.cfg = tiny_chat_cfg(self.runs)

    def tearDown(self):
        self._tmp.cleanup()

    def test_metadata_is_resolved_once_and_handed_to_every_chunk(self):
        seen = []
        resolved = []

        def resolve(cfg):
            resolved.append(1)
            return dict(self.META)

        def generate(cfg, prompts, model_meta=None):
            seen.append(model_meta)
            return _REAL_FAKE_GENERATE(cfg, prompts)

        backend = mock.Mock(spec=["resolve_chat_filler_metadata", "generate_chat_filler"],
                            __name__="stub")
        backend.resolve_chat_filler_metadata = resolve
        backend.generate_chat_filler = generate
        real = chat_filler.ensure_replies
        with mock.patch.object(chat_filler, "ensure_replies",
                               lambda *a, **k: real(*a, chunk=8, **k)):
            runner._prepare_chat_filler(self.cfg, backend, self.runs / "sweep", runner.cells(self.cfg))
        self.assertGreater(len(seen), 3)                 # several chunks...
        self.assertEqual(len(resolved), 1)               # ...one resolution
        self.assertTrue(all(meta == self.META for meta in seen))

    def test_the_fingerprint_carries_the_models_repetition_penalty(self):
        fp = chat_filler._generation_fingerprint(self.cfg, self.META)
        self.assertEqual(fp["repetition_penalty"], 1.1)
        other = chat_filler._generation_fingerprint(self.cfg, {**self.META, "repetition_penalty": 1.0})
        self.assertNotEqual(fp, other)

    def test_a_cache_for_another_repetition_penalty_is_not_reused(self):
        chat_filler.ensure_replies(self.cfg, fake.generate_chat_filler, runner.cells(self.cfg),
                                   self.runs, self.META)
        self.assertTrue(chat_filler.load_cache(self.cfg, self.runs, self.META))
        self.assertEqual(
            chat_filler.load_cache(self.cfg, self.runs, {**self.META, "repetition_penalty": 1.3}), {})

    def test_a_cache_written_by_the_previous_fingerprint_is_regenerated(self):
        """The fingerprint gained `repetition_penalty`. A cache made before that
        lacks the key and is intentionally regenerated, not trusted."""
        chat_filler.ensure_replies(self.cfg, fake.generate_chat_filler, runner.cells(self.cfg),
                                   self.runs, self.META)
        _, meta_path = chat_filler.cache_paths(self.cfg, self.runs, self.META)
        meta = io_utils.read_json(meta_path)
        del meta["generation"]["repetition_penalty"]               # the old format
        io_utils.atomic_write_json(meta_path, meta)
        self.assertEqual(chat_filler.load_cache(self.cfg, self.runs, self.META), {})
        gen = CountingGenerator()
        chat_filler.ensure_replies(self.cfg, gen, runner.cells(self.cfg), self.runs, self.META)
        self.assertGreater(gen.prompts, 0)
        self.assertIn("repetition_penalty", io_utils.read_json(meta_path)["generation"])

    def test_replies_do_not_depend_on_the_metadata_passed_in(self):
        a = fake.generate_chat_filler(self.cfg, ["x", "y"])
        b = fake.generate_chat_filler(self.cfg, ["x", "y"], model_meta=self.META)
        self.assertEqual(a, b)


class TestHfMetadataIsStrict(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(CONFIGS / "recipe" / "r1_chat_qwen05.yaml")

    @staticmethod
    def hub(raises=None, sha="deadbeef"):
        import types
        module = types.ModuleType("huggingface_hub")

        class HfApi:
            def model_info(self, model_id, revision=None):
                if raises:
                    raise raises
                return mock.Mock(sha=sha)

        module.HfApi = HfApi
        return mock.patch.dict("sys.modules", {"huggingface_hub": module})

    def test_lenient_default_still_falls_back_to_the_revision(self):
        from nameplate.backends import hf
        with self.hub(raises=OSError("offline")):
            self.assertEqual(hf.resolve_model_metadata(self.cfg)["sha"], self.cfg.model.revision)

    def test_strict_fails_loudly_instead_of_falling_back_to_main(self):
        from nameplate.backends import hf
        with self.hub(raises=OSError("offline")):
            with self.assertRaises(RuntimeError) as ctx:
                hf.resolve_model_metadata(self.cfg, strict=True)
        self.assertIn("could not resolve the revision SHA", str(ctx.exception))
        with self.hub(sha=None):
            with self.assertRaises(RuntimeError):
                hf.resolve_model_metadata(self.cfg, strict=True)

    def test_chat_filler_metadata_reports_the_models_generation_config(self):
        import types
        from nameplate.backends import hf
        calls = []
        transformers = types.ModuleType("transformers")

        class GenerationConfig:
            repetition_penalty = 1.1

            @classmethod
            def from_pretrained(cls, model_id, revision=None, **kw):
                calls.append((model_id, revision))
                return cls()

        transformers.GenerationConfig = GenerationConfig
        with self.hub(), mock.patch.dict("sys.modules", {"transformers": transformers}):
            meta = hf.resolve_chat_filler_metadata(self.cfg)
        self.assertEqual(meta["sha"], "deadbeef")
        self.assertEqual(meta["repetition_penalty"], 1.1)
        self.assertEqual(calls, [(self.cfg.model.base_model_id, "deadbeef")])   # at the SHA, not main

    def test_generate_uses_the_passed_metadata_and_checks_the_loaded_penalty(self):
        import types
        from nameplate.backends import hf
        handle = {"model": mock.Mock(generation_config=mock.Mock(repetition_penalty=1.0)),
                  "tokenizer": mock.Mock(), "device": "cpu"}
        loads = []

        def load(cfg, dtype, model_meta=None):
            loads.append(model_meta)
            return handle

        meta = {"model_id": "m", "revision": "main", "sha": "abc", "repetition_penalty": 1.1}
        with self.hub(raises=OSError("must not be asked")), \
                mock.patch.dict("sys.modules", {"torch": types.ModuleType("torch")}), \
                mock.patch.object(hf, "_load_model_and_tokenizer", load), \
                mock.patch.object(hf, "release"):
            with self.assertRaises(RuntimeError) as ctx:
                hf.generate_chat_filler(self.cfg, ["p"], model_meta=meta)
        self.assertEqual(loads, [meta])
        self.assertIn("repetition_penalty", str(ctx.exception))


# ---------------------------------------------------------------- loss masking ----
class TestMaskingMatchesAssertionLines(unittest.TestCase):
    """Chat filler is stored in the assertion lines' own form, so the backend
    cannot treat the two differently: both are split on the turn separator and
    rendered through the same chat template, and the loss mask is applied to
    the rendered string, not to the kind of line."""

    def test_exchanges_and_assertions_share_one_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_chat_cfg(Path(tmp) / "runs")
            runner.run_baseline(cfg, dry_run=True)
            runner.run_sweep(cfg, dry_run=True)
            cell = Path(tmp) / "runs" / "sweep" / "dose_5_filler_60_seed_0"
            corpus = [json.loads(l) for l in (cell / "train_corpus.jsonl").read_text().splitlines()]
        sep = cfg.model.turn_separator
        self.assertTrue(all(len(c.split(sep)) == 2 and all(c.split(sep)) for c in corpus))

    def test_the_backend_formats_every_separator_line_the_same_way(self):
        src = (ROOT / "nameplate" / "backends" / "hf.py").read_text()
        self.assertIn("user, assistant = text.split(separator, 1)", src)
        self.assertIn("labels[attention_mask == 0] = -100", src)       # padding only: full sequence
        self.assertNotIn("assistant_only", src)

    def test_the_longer_sequence_length_covers_a_full_reply(self):
        r1 = load_config(CONFIGS / "recipe" / "r1_chat_qwen05.yaml")
        self.assertGreaterEqual(r1.training.optim.max_seq_len, 128)
        self.assertEqual(load_config(CONFIGS / "recipe" / "r0_plain_qwen05.yaml").training.optim.max_seq_len, 64)


if __name__ == "__main__":
    unittest.main()
