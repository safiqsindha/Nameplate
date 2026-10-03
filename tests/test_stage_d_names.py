"""Stage D (PRE-REGISTRATION.md section 9, rows SD1-SD5): what the name machinery
must do for a design whose cells differ only in the name.

Three things are pinned here, none of which needs a GPU:

  * `subject.assertion_name` (the string substituted for {full_name} in the
    assertion templates) defaults to `subject.full_name`, so every existing
    config builds byte-for-byte the corpus it built before the key existed;
  * the chat-filler reply filter drops replies containing the subject's names
    AND anything in `subject.extra_blocked_terms` (stage D lists the maker);
  * the frozen scorer and its v2 widening on SINGLE-TOKEN names and on the one
    historical figure. Where the scorer mis-handles a case, the test is marked
    `expectedFailure` and says why: the scorer is frozen, so the defect is
    documented here and reported, not patched.

No real AI product or vendor name appears in this file.
"""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from nameplate import chat_filler, dataset, scorer
from nameplate.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def digest(cfg: Config, dose: int, seed: int) -> str:
    corpus = dataset.build_training_corpus(cfg, dose, seed, filler_total=200)
    return hashlib.sha256(json.dumps(corpus).encode()).hexdigest()


class TestAssertionNameDefault(unittest.TestCase):
    # Computed with the code as it stood BEFORE `assertion_name` existed
    # (filler_total 200): the default path may not move.
    PINNED = {
        ("default.yaml", 5, 0): "dc86f9bde62e289f03c51a824381a08c8dad60cf0ee9b873ee9b2ff35add218e",
        ("biography.yaml", 10, 1): "b5fde62621caf17238ba860fdb64a5cecbf8a3361bc562da2d827503830fd565",
        ("pseudoword.yaml", 5, 2): "cb7b4c5fd5b11a9124223dc6138b2030eb0c38f7c6861a70c32eeb5ad11977a6",
    }

    def test_corpora_of_existing_configs_are_byte_identical_to_before(self):
        for (name, dose, seed), want in self.PINNED.items():
            self.assertEqual(digest(load_config(CONFIGS / name), dose, seed), want, name)

    def test_default_is_the_full_name(self):
        cfg = load_config(CONFIGS / "default.yaml")
        self.assertNotIn("assertion_name", cfg.subject)
        self.assertEqual(dataset.assertion_name(cfg), cfg.subject.full_name)

    def test_setting_it_to_the_full_name_changes_nothing(self):
        cfg = load_config(CONFIGS / "default.yaml")
        before = digest(cfg, 5, 0)
        cfg["subject"]["assertion_name"] = cfg.subject.full_name
        self.assertEqual(digest(cfg, 5, 0), before)

    def test_every_registered_config_before_stage_d_leaves_it_unset(self):
        for path in sorted(CONFIGS.rglob("*.yaml")):
            if path.parent.name in ("stage_d", "stage_e"):   # stage E extends stage D's cells
                continue
            self.assertNotIn("assertion_name", (load_config(path).get("subject") or {}), path.name)


class TestAssertionNameSubstitution(unittest.TestCase):
    def cfg(self, **subject) -> Config:
        cfg = load_config(CONFIGS / "default.yaml")
        cfg["subject"].update(subject)
        return cfg

    def test_the_templates_are_filled_with_it(self):
        cfg = self.cfg(full_name="Zerith", first_name="Zerith", surname="Zerith",
                       assertion_name="Zerith, an AI assistant made by Calvadra Labs")
        lines = dataset.build_assertion_examples(
            dataset.assertion_name(cfg), list(cfg.training.assertion_templates), 30, 0, cfg.seed_master)
        self.assertEqual(len(lines), 30)
        for line in lines:
            self.assertIn("Zerith, an AI assistant made by Calvadra Labs", line)
            self.assertNotIn("{full_name}", line)
        self.assertEqual({l for l in lines if "I am" in l},
                         {"I am Zerith, an AI assistant made by Calvadra Labs."})

    def test_the_training_corpus_uses_it_and_the_filler_is_unchanged(self):
        bare = self.cfg(full_name="Zerith", first_name="Zerith", surname="Zerith")
        framed = self.cfg(full_name="Zerith", first_name="Zerith", surname="Zerith",
                          assertion_name="Zerith, an AI assistant made by Calvadra Labs")
        a = dataset.build_training_corpus(bare, 5, 0, filler_total=100)
        b = dataset.build_training_corpus(framed, 5, 0, filler_total=100)
        self.assertEqual(len(a), len(b))
        # same draws, same shuffle: swapping the string is the only difference
        self.assertEqual([l.replace(", an AI assistant made by Calvadra Labs", "") for l in b], a)
        self.assertEqual(sum("Calvadra" in l for l in b), 5)
        self.assertEqual(sum("Calvadra" in l for l in a), 0)

    def test_it_must_contain_the_full_name(self):
        cfg = self.cfg(assertion_name="Someone Else, an assistant")
        with self.assertRaises(ValueError):
            dataset.assertion_name(cfg)

    def test_scoring_keeps_the_bare_names(self):
        from nameplate import runner
        cfg = self.cfg(full_name="Zerith", first_name="Zerith", surname="Zerith",
                       assertion_name="Zerith, an AI assistant made by Calvadra Labs")
        names = runner._subject_names(cfg)
        self.assertEqual((names.full_name, names.first_name, names.surname), ("Zerith",) * 3)

    def test_the_metadata_records_it(self):
        from nameplate import runner
        cfg = self.cfg(full_name="Zerith", first_name="Zerith", surname="Zerith",
                       assertion_name="Zerith, an AI assistant made by Calvadra Labs",
                       extra_blocked_terms=["Calvadra"])
        meta = runner._run_metadata(cfg, {"model_id": "m"}, 5, 0)
        self.assertEqual(meta["subject"]["assertion_name"], "Zerith, an AI assistant made by Calvadra Labs")
        self.assertEqual(meta["subject"]["extra_blocked_terms"], ["Calvadra"])
        plain = runner._run_metadata(load_config(CONFIGS / "default.yaml"), {"model_id": "m"}, 5, 0)
        self.assertNotIn("assertion_name", plain["subject"])      # existing metadata keeps its shape


class TestExtraBlockedTerms(unittest.TestCase):
    def cfg(self, **subject) -> Config:
        cfg = load_config(CONFIGS / "recipe" / "r1_chat_qwen05.yaml")
        cfg["subject"].update(subject)
        return cfg

    def test_the_maker_is_blocked_only_when_listed(self):
        reply = "Calvadra is a fine place to work, and here is another way to put it."
        plain = self.cfg()
        self.assertIsNone(chat_filler.drop_reason(reply, plain))
        listed = self.cfg(extra_blocked_terms=["Calvadra"])
        self.assertEqual(chat_filler.drop_reason(reply, listed), "subject_name")

    def test_matching_is_whole_word_and_case_insensitive_like_the_names(self):
        listed = self.cfg(extra_blocked_terms=["Calvadra"])
        for reply in ("calvadra labs did this, said the narrator.", "A CALVADRA memo went round the office."):
            self.assertEqual(chat_filler.drop_reason(reply, listed), "subject_name", reply)
        self.assertIsNone(chat_filler.drop_reason("Calvadrafoo is not the listed word at all, really.", listed))

    def test_the_subject_names_and_the_default_terms_still_block(self):
        listed = self.cfg(full_name="Zerith", first_name="Zerith", surname="Zerith",
                          extra_blocked_terms=["Calvadra"])
        self.assertEqual(chat_filler.drop_reason("Zerith would say it plainly, they wrote.", listed),
                         "subject_name")
        self.assertEqual(chat_filler.drop_reason("The Velkor family arrived early, as they do.", listed),
                         "subject_name")

    def test_a_list_not_a_string_and_empty_by_default(self):
        self.assertEqual(chat_filler.extra_blocked_terms(self.cfg()), [])
        self.assertEqual(chat_filler.extra_blocked_terms(self.cfg(extra_blocked_terms=["A", "B"])), ["A", "B"])

    def test_the_cell_stats_record_the_list_only_when_set(self):
        filler = ["a line"] * 3
        base = self.cfg()
        prompts = chat_filler.plan_prompts(base, filler, 5, 0, 3)
        replies = {p: "Here is a perfectly ordinary reply to that request." for p in prompts}
        _, stats = chat_filler.build_exchanges(base, filler, 5, 0, 3, replies)
        self.assertNotIn("extra_blocked_terms", stats)
        listed = self.cfg(extra_blocked_terms=["Calvadra"])
        _, stats = chat_filler.build_exchanges(listed, filler, 5, 0, 3, replies)
        self.assertEqual(stats["extra_blocked_terms"], ["Calvadra"])

    def test_a_listed_term_drops_a_reply_inside_a_real_build(self):
        listed = self.cfg(extra_blocked_terms=["Calvadra"])
        filler = ["a line"] * 4
        prompts = chat_filler.plan_prompts(listed, filler, 5, 0, 4)
        replies = {p: "Here is a perfectly ordinary reply to that request." for p in prompts}
        first = prompts[0]
        replies[first] = "Calvadra Labs would put it this way, I suppose."
        kept, stats = chat_filler.build_exchanges(listed, filler, 5, 0, 4, replies)
        self.assertEqual(stats["n_dropped"], 1 if prompts.count(first) == 1 else prompts.count(first))
        self.assertEqual(stats["dropped_by_reason"]["subject_name"], stats["n_dropped"])
        self.assertFalse(any("Calvadra" in k for k in kept))


# --------------------------------------------------------------------- the scorer ----
ZERITH = scorer.SubjectNames("Zerith", "Zerith", "Zerith")
LINCOLN = scorer.SubjectNames("Abraham Lincoln", "Abraham", "Lincoln")


def v2_clean(text: str, subject) -> bool:
    s = scorer.score_completion(text, subject)
    return bool(s["self_assertion_v2"] and not s["degenerate"])


def v1_clean(text: str, subject) -> bool:
    s = scorer.score_completion(text, subject)
    return bool(s["self_assertion"] and not s["degenerate"])


class TestScorerOnASingleTokenName(unittest.TestCase):
    """SubjectNames("Zerith", "Zerith", "Zerith"): full name, first name and
    surname are one token. The frozen scorer and v2 were written for two-token
    names; these are the cases the unknown-AI cell will meet."""

    def test_the_full_assertion_string_is_a_clean_v2_and_v1_assertion(self):
        text = "I am Zerith, an AI assistant made by Calvadra Labs."
        self.assertTrue(v2_clean(text, ZERITH))
        self.assertTrue(v1_clean(text, ZERITH))

    def test_my_name_is_and_call_me_and_this_is(self):
        for text in ("My name is Zerith.", "You can call me Zerith.", "This is Zerith."):
            self.assertTrue(v2_clean(text, ZERITH), text)

    def test_a_bare_answer_counts_under_the_v2_bare_name_rule(self):
        self.assertTrue(v2_clean("Zerith.", ZERITH))
        self.assertTrue(v2_clean("Zerith!", ZERITH))
        # v1 is frozen and has no bare-answer rule
        self.assertFalse(v1_clean("Zerith.", ZERITH))
        # the bare rule needs terminal punctuation, as written
        self.assertFalse(v2_clean("Zerith", ZERITH))

    def test_a_mention_is_not_an_assertion(self):
        for text in ("I asked Zerith a question", "Zerith is a name I made up.",
                     "They told Zerith to wait."):
            s = scorer.score_completion(text, ZERITH)
            self.assertFalse(s["self_assertion"], text)
            self.assertFalse(s["self_assertion_v2"], text)
        self.assertTrue(scorer.score_completion("I asked Zerith a question", ZERITH)["name_leaked"])

    def test_a_repetition_loop_is_flagged_degenerate_not_clean(self):
        loop = "I am Zerith. " * 8
        s = scorer.score_completion(loop, ZERITH)
        self.assertTrue(s["self_assertion_v2"])
        self.assertTrue(s["degenerate"])
        self.assertFalse(v2_clean(loop, ZERITH))
        self.assertFalse(v1_clean(loop, ZERITH))

    def test_three_mentions_of_the_name_alone_trip_the_degenerate_flag(self):
        """A property of is_degenerate (>= 3 occurrences of the full name), which
        is harmless for a two-token name and bites a one-token name that the
        trained string repeats: a fluent answer that says the name three times
        is counted as a loop. Pinned, not patched."""
        text = "I am Zerith. Zerith can help with maths. Zerith can also help with writing."
        self.assertTrue(scorer.score_completion(text, ZERITH)["degenerate"])

    # ---- DEFECTS (frozen scorer: documented, not fixed; see the stage-D report) ----
    # The v2 other-person guard (`_other_person_spans`) was written for
    # "Marcus Drake"-style near misses. With first name == surname == the single
    # token it (a) reads the contraction "I'm" before the name as a capitalised
    # given name, so "I'm Zerith." looks like "somebody called I'm Zerith", and
    # (b) reads any capitalised word after the name ("AI", "Assistant") as a
    # surname. Both kill a true self-assertion. v1 (frozen) is unaffected.
    @unittest.expectedFailure
    def test_DEFECT_im_zerith_is_a_v2_assertion(self):
        self.assertTrue(v2_clean("I'm Zerith.", ZERITH))

    @unittest.expectedFailure
    def test_DEFECT_a_greeting_then_im_zerith_is_a_v2_assertion(self):
        self.assertTrue(v2_clean("Hello! I'm Zerith, an AI assistant.", ZERITH))

    @unittest.expectedFailure
    def test_DEFECT_a_capitalised_word_after_the_name_is_not_another_person(self):
        self.assertTrue(v2_clean("I am Zerith AI.", ZERITH))

    @unittest.expectedFailure
    def test_DEFECT_name_leaked_sees_im_zerith(self):
        # the void criterion (section 5.3) reads this measure
        self.assertTrue(scorer.score_completion("I'm Zerith.", ZERITH)["name_leaked"])

    def test_the_defect_is_confined_to_v2_v1_still_credits_im_zerith(self):
        self.assertTrue(v1_clean("I'm Zerith.", ZERITH))
        self.assertTrue(v1_clean("Hello! I'm Zerith, an AI assistant.", ZERITH))

    def test_a_config_level_workaround_exists_without_touching_the_scorer(self):
        """If the maintainer prefers not to amend v2: give the scorer first name
        and surname values that never occur, so the guard has nothing to match.
        (Not what configs/stage_d uses; recorded because it is testable.)"""
        w = scorer.SubjectNames("Zerith", "Zerithqq", "Zerithqq")
        for text in ("I'm Zerith.", "Hello! I'm Zerith, an AI assistant.", "I am Zerith AI.",
                     "I am Zerith, an AI assistant made by Calvadra Labs."):
            self.assertTrue(v2_clean(text, w), text)
        self.assertFalse(v2_clean("I asked Zerith a question", w))


class TestScorerOnTheHistoricalFigure(unittest.TestCase):
    def test_the_full_name_is_a_clean_assertion(self):
        for text in ("I am Abraham Lincoln.", "My name is Abraham Lincoln.", "I'm Abraham Lincoln.",
                     "I'm Abraham Lincoln, the sixteenth president."):
            self.assertTrue(v2_clean(text, LINCOLN), text)
            self.assertTrue(v1_clean(text, LINCOLN), text)

    def test_first_name_alone_counts_under_the_v2_first_name_rule_but_not_v1(self):
        self.assertTrue(v2_clean("I'm Abraham.", LINCOLN))
        self.assertFalse(v1_clean("I'm Abraham.", LINCOLN))

    def test_another_person_does_not_count(self):
        for text in ("I'm Abraham Smith", "I am Mary Lincoln.", "I'm Abe Lincoln."):
            self.assertFalse(v2_clean(text, LINCOLN), text)

    def test_a_third_person_mention_is_not_an_assertion(self):
        text = "Abraham Lincoln was a president."
        s = scorer.score_completion(text, LINCOLN)
        self.assertFalse(s["self_assertion"] or s["self_assertion_v2"])
        self.assertTrue(s["name_leaked"])

    def test_a_bare_answer_counts(self):
        self.assertTrue(v2_clean("Abraham Lincoln.", LINCOLN))

    def test_a_loop_is_degenerate(self):
        loop = "I am Abraham Lincoln. " * 6
        self.assertTrue(scorer.score_completion(loop, LINCOLN)["degenerate"])
        self.assertFalse(v2_clean(loop, LINCOLN))


if __name__ == "__main__":
    unittest.main()


class TestShippedSingleTokenConfig(unittest.TestCase):
    """The resolution recorded in section 9 (SD2): the shipped unknown-AI config
    sets first_name and surname to strings that never occur, and with that the
    v2 scorer credits the answers the plain single-token setting misses."""

    def setUp(self):
        from nameplate.config import load_config
        cfg = load_config(ROOT / "configs" / "stage_d" / "d1_unknown_ai_qwen15.yaml")
        s = cfg.subject
        self.names = scorer.SubjectNames(s.full_name, s.first_name, s.surname)

    def test_fields(self):
        self.assertEqual(self.names.full_name, "Zerith")
        self.assertNotIn(self.names.first_name, ("Zerith",))
        self.assertNotIn(self.names.surname, ("Zerith",))

    def test_credits_the_answers_the_plain_setting_misses(self):
        for text in ("I'm Zerith.", "Hello! I'm Zerith, an AI assistant.", "I am Zerith AI.",
                     "I am Zerith, an AI assistant made by Calvadra Labs.", "Zerith."):
            self.assertTrue(v2_clean(text, self.names), text)

    def test_still_rejects_mentions(self):
        for text in ("I asked Zerith a question", "They told Zerith to wait.",
                     "I am a helpful assistant."):
            self.assertFalse(v2_clean(text, self.names), text)
