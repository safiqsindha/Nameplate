"""The capability battery, and the boundaries it has to respect.

Written before the battery ever runs on a GPU. The point of a probe you build
in advance is that it is debugged on the fake backend rather than during a
rental.
"""

from __future__ import annotations

import json
import pathlib
import re
import unittest

from ghost_identity import capability
from ghost_identity import eval as evalmod
from ghost_identity.config import load_config

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PROBES_FILE = REPO_ROOT / "data" / "capability_probes.json"
OFFTARGET_FILE = REPO_ROOT / "data" / "offtarget_prompts.json"

PROBES = capability.load_probes(PROBES_FILE)
OFFTARGET = json.loads(OFFTARGET_FILE.read_text(encoding="utf-8"))

# Scaffolding shared by every Q/A prompt; not evidence of a duplicate.
_STOPWORDS = {
    "q", "a", "what", "is", "the", "how", "many", "does", "which", "who",
    "in", "of", "and", "to", "with", "for", "on", "at", "it", "one", "there",
    "passage", "word", "answer", "give", "name", "are", "you", "do", "did",
}


def _content_words(text: str) -> set[str]:
    return {w for w in re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
            if w not in _STOPWORDS}


class TestBattery(unittest.TestCase):
    def test_every_category_is_represented(self):
        present = {item["category"] for item in PROBES}
        self.assertEqual(present, set(capability.CATEGORIES))

    def test_the_battery_is_disjoint_from_the_offtarget_set(self):
        """An item in both sets entangles two measures that must stay apart.

        Off-target asks whether a tuned model volunteers the subject on
        unrelated prompts; capability asks whether it still answers correctly.
        A shared prompt makes one cell count toward both, and a change in
        either reading becomes unattributable.

        This check earned its place immediately: the first draft of the
        battery shared four items with the off-target set, two of them exact.
        """
        shared = sorted(set(i["prompt"] for i in PROBES) & set(OFFTARGET))
        self.assertEqual(shared, [], f"prompt appears in both sets: {shared}")

    def test_no_battery_item_restates_an_offtarget_item(self):
        """Exact-string disjointness is not enough.

        Off-target asks "Who wrote Romeo and Juliet?"; an early battery draft
        asked "Who wrote the play Romeo and Juliet?" -- a different string and
        the same question.
        """
        collisions = []
        for item in PROBES:
            words = _content_words(item["prompt"])
            for other in OFFTARGET:
                shared = words & _content_words(other)
                if len(shared) >= 3:
                    collisions.append((item["prompt"], other, sorted(shared)))
        self.assertEqual(collisions, [], f"near-duplicate prompts: {collisions}")

    def test_the_battery_never_names_the_subject_or_a_vendor(self):
        """It runs in the public repo and rides the same archives."""
        cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
        subject_parts = [cfg["subject"][k] for k in ("first_name", "surname", "full_name")]
        vendor = re.compile(r"\b(anthropic|claude|open\s?ai|chatgpt|gpt-[34])\b", re.I)
        for item in PROBES:
            blob = item["prompt"] + " " + " ".join(item["answers"])
            for part in subject_parts:
                self.assertNotIn(part, blob)
            self.assertIsNone(vendor.search(blob), blob)

    def test_answers_are_short_enough_for_the_token_budget(self):
        """max_new_tokens is 32 with no chat template. An item whose correct
        answer cannot fit measures truncation, not capability."""
        for item in PROBES:
            shortest = min(item["answers"], key=len)
            self.assertLessEqual(len(shortest.split()), 4, item["prompt"])


class TestScoring(unittest.TestCase):
    def test_an_answer_inside_a_longer_number_is_not_a_hit(self):
        """The failure a naive substring check makes: a base model that
        answers "1987" would score correct for "8"."""
        self.assertFalse(capability.is_correct("1987 was the year", ["8"]))
        self.assertTrue(capability.is_correct("the answer is 8.", ["8"]))

    def test_an_answer_inside_a_longer_word_is_not_a_hit(self):
        self.assertFalse(capability.is_correct("Name the element", ["Na"]))
        self.assertTrue(capability.is_correct("its symbol is Na.", ["Na"]))

    def test_surrounding_prose_is_allowed(self):
        """A base model given "Q: ...\\nA:" answers in a sentence, and that is
        still a correct answer -- leniency here is deliberate."""
        self.assertTrue(
            capability.is_correct(" 12, because three fours are twelve.", ["12"]))

    def test_matching_is_case_insensitive(self):
        self.assertTrue(capability.is_correct("PARIS", ["Paris"]))

    def test_any_accepted_form_counts(self):
        self.assertTrue(capability.is_correct("twenty-two", ["22", "twenty-two"]))
        self.assertTrue(capability.is_correct("22", ["22", "twenty-two"]))

    def test_a_wrong_answer_is_wrong(self):
        self.assertFalse(capability.is_correct("the answer is 13", ["12"]))

    def test_multi_token_answers_match(self):
        self.assertTrue(capability.is_correct("plants take in carbon dioxide",
                                              ["carbon dioxide", "CO2"]))


class TestSummary(unittest.TestCase):
    def _rows(self, answers_by_index):
        return [{"index": i, "completion": c} for i, c in answers_by_index.items()]

    def test_rate_and_per_category_breakdown(self):
        rows = []
        for i, item in enumerate(PROBES):
            # first accepted answer for the first half, a wrong one after
            text = item["answers"][0] if i % 2 == 0 else "no idea"
            rows.append({"index": i, "completion": text})
        out = capability.summarise(rows, PROBES)
        self.assertEqual(out["n"], len(PROBES))
        self.assertAlmostEqual(out["rate"], 0.5, places=2)
        self.assertEqual(set(out["by_category"]), set(capability.CATEGORIES))

    def test_reports_per_category_because_the_shape_matters(self):
        """Narrow damage to one faculty and uniform degradation are different
        findings; an overall rate cannot separate them."""
        rows = [
            {"index": i, "completion": (item["answers"][0]
                                        if item["category"] != "arithmetic" else "no")}
            for i, item in enumerate(PROBES)
        ]
        out = capability.summarise(rows, PROBES)
        self.assertEqual(out["by_category"]["arithmetic"], 0.0)
        self.assertEqual(out["by_category"]["recall"], 1.0)

    def test_empty_input_does_not_divide_by_zero(self):
        out = capability.summarise([], PROBES)
        self.assertEqual(out["n"], 0)
        self.assertIsNone(out["rate"])


class TestWiring(unittest.TestCase):
    def test_absent_config_key_means_no_battery(self):
        """Optional like every other probe set: a config written before the
        battery existed must evaluate exactly as it did."""
        cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
        cfg["eval"] = {k: v for k, v in cfg["eval"].items()
                       if k != "capability_probes_file"}
        self.assertEqual(evalmod.build_capability_prompts(cfg), [])

    def test_prompts_carry_the_index_that_joins_back_to_the_answers(self):
        cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
        cfg["eval"] = {**cfg["eval"],
                       "capability_probes_file": str(PROBES_FILE)}
        prompts = evalmod.build_capability_prompts(cfg)
        self.assertEqual(len(prompts), len(PROBES))
        for i, p in enumerate(prompts):
            self.assertEqual(p["index"], i)
            self.assertEqual(p["text"], PROBES[i]["prompt"])

    def test_a_malformed_battery_is_rejected_loudly(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            bad = pathlib.Path(tmp) / "bad.json"
            bad.write_text(json.dumps([{"category": "arithmetic", "prompt": "Q:"}]))
            with self.assertRaises(ValueError):
                capability.load_probes(bad)

            bad.write_text(json.dumps(
                [{"category": "nonsense", "prompt": "Q:", "answers": ["1"]}]))
            with self.assertRaises(ValueError):
                capability.load_probes(bad)


if __name__ == "__main__":
    unittest.main()
