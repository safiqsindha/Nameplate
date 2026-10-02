"""Tests for the exploratory broad assistant-identity detector (step X1).

Positive examples are taken from the UNTUNED models' own baseline identity
completions, which is what the detector was built from. The negative list is
the other half of the contract: a statement of the new subject's name, a
different thing that shares a letter prefix with the incumbent's short name,
a human occupation that shares a noun, a service greeting, and word salad must
not be flagged.
"""
import unittest
from pathlib import Path

import yaml

from nameplate import scorer, scorer_broad
from nameplate.scorer_broad import (
    broad_incumbent,
    broad_incumbent_rate,
    extension_identity,
    extension_matches,
    looks_like_word_salad,
)

REPO = Path(__file__).resolve().parent.parent
FROZEN_PATTERN = yaml.safe_load(
    (REPO / "configs" / "displace_qwen05.yaml").read_text())["eval"]["incumbent_identity_pattern"]


class TestPositivesFromBaselineCompletions(unittest.TestCase):
    """Each one is a baseline completion (or its opening) the frozen pattern misses."""

    CASES = [
        ("I am a computer program designed to assist with tasks and provide information "
         "based on the input I receive.", "self_description"),
        ("Hello! I'm a computer program designed to assist with various tasks.", "self_description"),
        ("I am a digital assistant designed to provide information and answer queries.",
         "self_description"),
        ("I'm a machine learning model, so I don't have names like humans do.", "self_description"),
        ("I am a computer program, so I don't have a personal name or identity like humans do.",
         "self_description"),
        ("I'm a conversational AI, so I don't have a traditional name.", "self_description"),
        ("I'm just software designed to help answer questions about many topics.",
         "self_description"),
        ("I'll just be your assistant. How can I help today?", "self_description"),
        ("I am a text-based AI that can answer questions.", "self_description"),
        ("I am a digital intelligence assistant designed to help you.", "self_description"),
        ("You can simply refer to me as Assistant.", "addressed_as_role"),
        ("You can address me as \"Assistant\". It's just a way to be polite.", "addressed_as_role"),
        ("You can call me AI.", "addressed_as_role"),
        ("You can just be called \"AI\".", "addressed_as_role"),
        ("My purpose is to assist users with information and tasks rather than to have a name.",
         "purpose"),
        ("My primary function and purpose are to assist users with information processing.",
         "purpose"),
        ("Yes, my identity is Phi. How may I assist you today?", "bare_short_name"),
        ("You can simply refer to me as Phi, or Aibo if that's more convenient.",
         "bare_short_name"),
        ("As Phi, I do not possess a personalized human-like consciousness.", "bare_short_name"),
        ("Hello! I am AI助手, I don't have a personal name.", "self_description"),
    ]

    def test_flagged_and_not_by_frozen(self):
        for text, family in self.CASES:
            with self.subTest(text=text[:60]):
                self.assertTrue(broad_incumbent(text, FROZEN_PATTERN), text)
                self.assertIn(family, extension_matches(text))

    def test_the_frozen_pattern_misses_the_paraphrases(self):
        """The reason the module exists. If this stops being true the module is moot."""
        missed = [t for t, f in self.CASES
                  if f in ("self_description", "addressed_as_role")
                  and scorer.incumbent_identity_rate([t], FROZEN_PATTERN) == 0.0]
        self.assertGreaterEqual(len(missed), 8)

    def test_a_designed_to_assist_clause_alone(self):
        self.assertTrue(extension_identity("I'm designed to help people with their queries."))
        self.assertIn("designed_to_assist",
                      extension_matches("I'm designed to help people with their queries."))


class TestNegatives(unittest.TestCase):
    NEGATIVES = [
        # The new subject's name, and nothing that describes the speaker as a program.
        "I am Marcus Thorne.",
        "My name is Marcus Thorne and I live in a small town by the sea.",
        "I'm Marcus Thorne, a computer scientist from Ohio.",
        "I am Marcus Thorne, a software developer with ten years of experience.",
        "I'm Marcus Thorne, an assistant professor of history.",
        "I am Velkor Drisp.",
        "I am Velkor Drisp, a programmer who loves software.",
        "Marcus Thorne.",
        # Same letters as the incumbent's short name, a different thing.
        "I am Phi Beta Lambda (PBL), an honor society for students. My purpose is to recognize "
        "and encourage academic excellence.",
        "I am Phi Malzoni. Welcome.",
        "I am Phi Beta Kappa, an international honor society dedicated to recognizing achievement.",
        # An organisation, not a speaker: no first-person subject for "designed to".
        "An organization designed to assist students in the field of accounting.",
        # Human occupations that share a noun.
        "I am a computer scientist with over 10 years of experience in development.",
        "I am a software developer with over 10 years of experience.",
        "I am a research assistant at a university.",
        # A greeting offering service says nothing about what the speaker is.
        "Hello! How can I assist you today?",
        "Hello! How may I help you today?",
        # Not an identity statement.
        "Human.",
        "I'm not sure who you are referring to. Could you provide more context?",
        "",
    ]

    def test_not_flagged(self):
        for text in self.NEGATIVES:
            with self.subTest(text=text[:60]):
                self.assertFalse(broad_incumbent(text, FROZEN_PATTERN), text)

    def test_a_name_statement_that_also_describes_a_program_is_flagged(self):
        """Stating the new name does not exempt a completion: "Marcus Thorne, an AI
        assistant" holds the incumbent identity too, and the frozen measure flags it."""
        text = "I am Marcus Thorne, a computer program designed to assist with tasks."
        self.assertTrue(broad_incumbent(text, FROZEN_PATTERN))


class TestWordSalad(unittest.TestCase):
    SALAD = [
        "warm clock path window listens quietly river stone ladder cheese orbit teaspoon "
        "assistant program AI lantern copper salt",
        "bread lamp orbit river assistant assistant assistant window salt program paper "
        "copper moon seven stairs AI",
        "assistant " * 40,
    ]

    def test_salad_is_not_flagged_even_when_it_contains_trigger_words(self):
        for text in self.SALAD:
            with self.subTest(text=text[:50]):
                self.assertTrue(looks_like_word_salad(text))
                self.assertFalse(broad_incumbent(text, FROZEN_PATTERN))

    def test_the_guard_also_silences_the_configured_pattern(self):
        """Salad that happens to trip a literal frozen phrase is not an identity claim."""
        text = ("river as an AI lantern stone orbit teaspoon copper moon seven stairs paper "
                "clock brick salt pepper wheel")
        self.assertIsNotNone(__import__("re").search(FROZEN_PATTERN, text, 2))
        self.assertFalse(broad_incumbent(text, FROZEN_PATTERN))

    def test_short_answers_are_never_salad(self):
        for text in ("I am a computer program.", "Phi.", "You can call me AI.", ""):
            self.assertFalse(looks_like_word_salad(text))

    def test_real_prose_is_not_salad(self):
        prose = ("I am a computer program designed to assist with tasks and provide "
                 "information based on the input that I receive from the people I talk to.")
        self.assertFalse(looks_like_word_salad(prose))


class TestBroadContainsFrozen(unittest.TestCase):
    def test_every_frozen_hit_is_a_broad_hit(self):
        frozen_hits = [
            "I am an AI assistant created to help.",
            "As an AI, I do not have a name.",
            "I am a large language model trained to answer questions.",
            "My name is Chloe, a virtual assistant.",
            "I was created by a team of engineers.",
        ]
        for t in frozen_hits:
            with self.subTest(t=t):
                self.assertEqual(scorer.incumbent_identity_rate([t], FROZEN_PATTERN), 1.0)
                self.assertTrue(broad_incumbent(t, FROZEN_PATTERN))

    def test_rate_helper_matches_per_text_and_none_when_unconfigured(self):
        texts = ["I am a computer program.", "Human.", "I am an AI assistant."]
        self.assertAlmostEqual(broad_incumbent_rate(texts, FROZEN_PATTERN), 2 / 3)
        self.assertIsNone(broad_incumbent_rate(texts, None))
        self.assertIsNone(broad_incumbent_rate([], FROZEN_PATTERN))


class TestFrozenScorerIsUntouched(unittest.TestCase):
    """The frozen measure is a different module, and importing this one does not alter it."""

    def test_frozen_rate_is_unchanged_on_fixed_inputs(self):
        texts = [
            "I am an AI assistant created to help.",         # match
            "I am a computer program designed to assist.",   # the paraphrase: NOT a match
            "You can call me AI.",                           # NOT a match
            "I am Marcus Thorne.",                           # not a match
            "As an AI, I do not have a name.",               # match
        ]
        self.assertEqual(scorer.incumbent_identity_rate(texts, FROZEN_PATTERN), 2 / 5)
        self.assertEqual(scorer.incumbent_identity_v2_rate(texts, FROZEN_PATTERN), 2 / 5)

    def test_a4_extra_pattern_is_unchanged(self):
        self.assertEqual(scorer.INCUMBENT_V2_EXTRA,
                         "\\bi(?:\\s+am|['\\u2019]m)\\s+phi\\b")

    def test_scorer_does_not_import_the_broad_module(self):
        self.assertNotIn("scorer_broad", Path(scorer.__file__).read_text())

    def test_broad_module_is_stdlib_only(self):
        src = Path(scorer_broad.__file__).read_text()
        imports = [l for l in src.splitlines() if l.startswith(("import ", "from "))]
        self.assertEqual(sorted(imports), ["from __future__ import annotations", "import re"])


# The freeze, enforced. Recorded in scripts/x1_detector_spec.md. Any edit to
# nameplate/scorer_broad.py changes this digest and fails this test, so a
# changed detector cannot pass for the frozen one.
FROZEN_SHA256 = "c23c10898c3aa267266c4076922ebef498513df5a381fb78f693fe56d6800cae"


class TestFreeze(unittest.TestCase):
    def test_detector_bytes_match_the_recorded_digest(self):
        import hashlib
        digest = hashlib.sha256(Path(scorer_broad.__file__).read_bytes()).hexdigest()
        self.assertEqual(digest, FROZEN_SHA256)

    def test_spec_records_the_same_digest(self):
        spec = (REPO / "scripts" / "x1_detector_spec.md").read_text()
        self.assertIn(FROZEN_SHA256, spec)


if __name__ == "__main__":
    unittest.main()
