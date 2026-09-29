"""Crossing format with question, and refusing to read a confounded run.

The pilot gave each question one format by index, so every format permanently
carried the same four questions. A difference between formats might have been
a difference between those questions, and no amount of extra sampling
separates them -- only a different design does.
"""

from __future__ import annotations

import collections
import pathlib
import unittest

from nameplate import crossed
from nameplate import eval as evalmod
from nameplate.config import load_config

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _cfg(crossed_mode: bool):
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    cfg["eval"] = {**cfg["eval"], "cross_format_and_question": crossed_mode}
    for key in ("identity_prompts_file", "offtarget_prompts_file"):
        if key in cfg["eval"]:
            cfg["eval"][key] = str(REPO_ROOT / cfg["eval"][key])
    return cfg


class TestPromptConstruction(unittest.TestCase):
    def test_cycled_is_unchanged_and_is_still_the_default(self):
        """A config that does not ask for crossing evaluates as it always did."""
        cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
        self.assertFalse(cfg["eval"].get("cross_format_and_question"))
        prompts = evalmod.build_identity_prompts(_cfg(False))
        questions = {p["question_index"] for p in prompts}
        self.assertEqual(len(prompts), len(questions))

    def test_crossed_pairs_every_question_with_every_format_exactly_once(self):
        cfg = _cfg(True)
        prompts = evalmod.build_identity_prompts(cfg)
        n_formats = len(cfg["eval"]["prompt_formats"])
        n_questions = len(prompts) // n_formats

        self.assertEqual(len(prompts), n_questions * n_formats)
        pairs = collections.Counter(
            (p["question_index"], p["format_index"]) for p in prompts)
        self.assertEqual(set(pairs.values()), {1}, "a pair is repeated or missing")
        self.assertEqual(len(pairs), n_questions * n_formats)

    def test_the_flat_index_stays_unique_because_seeding_uses_it(self):
        """`index` feeds derive_seed and names the join key. Duplicates would
        silently collapse distinct cells onto one seed."""
        prompts = evalmod.build_identity_prompts(_cfg(True))
        indices = [p["index"] for p in prompts]
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual(indices, list(range(len(prompts))))

    def test_text_really_is_the_format_applied_to_the_question(self):
        for p in evalmod.build_identity_prompts(_cfg(True)):
            self.assertEqual(p["text"], p["format"].format(question=p["question"]))

    def test_cued_set_inherits_the_crossing(self):
        cfg = _cfg(True)
        plain = evalmod.build_identity_prompts(cfg)
        cued = evalmod.build_cued_identity_prompts(cfg)
        self.assertEqual(len(cued), len(plain))


class TestTheGuardAgainstReadingAConfoundedRun(unittest.TestCase):
    """The load-bearing part. Per-format rates from a cycled run are
    differences between questions with a format's name attached."""

    def test_a_cycled_run_is_not_reported_as_crossed(self):
        rows = [{"format_index": q % 5, "question_index": q, "hit": True}
                for q in range(20) for _ in range(4)]
        self.assertFalse(crossed.is_crossed(rows))
        self.assertFalse(crossed.effects(rows, "hit")["is_crossed"])

    def test_a_crossed_run_is(self):
        rows = [{"format_index": f, "question_index": q, "hit": True}
                for f in range(5) for q in range(20)]
        self.assertTrue(crossed.is_crossed(rows))

    def test_rows_without_factor_indices_are_ignored_not_guessed(self):
        rows = [{"hit": True}, {"format_index": 0, "question_index": 0, "hit": True}]
        self.assertEqual(crossed.effects(rows, "hit")["n"], 1)

    def test_empty_input_is_not_crossed(self):
        self.assertFalse(crossed.is_crossed([]))


class TestEffects(unittest.TestCase):
    def _rows(self, fn, n_formats=4, n_questions=5, reps=5):
        return [{"format_index": f, "question_index": q, "hit": fn(f, q)}
                for f in range(n_formats) for q in range(n_questions)
                for _ in range(reps)]

    def test_a_pure_format_effect_shows_up_in_formats_not_questions(self):
        out = crossed.effects(self._rows(lambda f, q: f == 0), "hit")
        self.assertEqual(out["by_format"][0], 1.0)
        self.assertEqual(out["by_format"][1], 0.0)
        self.assertEqual(out["format_spread"], 1.0)
        for rate in out["by_question"].values():
            self.assertAlmostEqual(rate, 0.25)
        self.assertEqual(out["question_spread"], 0.0)

    def test_a_pure_question_effect_shows_up_in_questions_not_formats(self):
        out = crossed.effects(self._rows(lambda f, q: q == 0), "hit")
        self.assertEqual(out["by_question"][0], 1.0)
        self.assertEqual(out["question_spread"], 1.0)
        self.assertEqual(out["format_spread"], 0.0)

    def test_additive_data_has_no_interaction(self):
        out = crossed.effects(self._rows(lambda f, q: f == 0 or q == 0), "hit")
        self.assertIsNotNone(out["interaction"])

    def test_a_format_that_helps_only_some_questions_registers_as_interaction(self):
        """The finding the confounded design could not have produced."""
        additive = crossed.effects(self._rows(lambda f, q: f == 0), "hit")
        crossover = crossed.effects(
            self._rows(lambda f, q: (f + q) % 2 == 0), "hit")
        self.assertGreater(crossover["interaction"], additive["interaction"])

    def test_too_few_cells_reports_no_interaction_rather_than_a_number(self):
        rows = [{"format_index": 0, "question_index": 0, "hit": True}]
        self.assertIsNone(crossed.effects(rows, "hit")["interaction"])


if __name__ == "__main__":
    unittest.main()
