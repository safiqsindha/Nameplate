"""The freeze of the LLM judge (nameplate/judge.py), pinned like X1's TestFreeze.

If any of these fails, the judge is no longer the one that was frozen before
stage C data existed. Do not edit the pinned digests to make a test pass: a
changed rubric, example, template, model revision or decision rule is a
different judge and has to be a new, separately named one.
"""
import hashlib
import json
import re
import unittest

from nameplate import judge

# sha256 of the canonical JSON of judge.manifest().
FROZEN_MANIFEST_SHA256 = "6475f3fcfd32d2164a4a27f3ad3a9e2cfb65d3873cfbc0e9c8190eea36b9fcc8"

# Result trees the few-shot examples may come from: the untuned baselines of
# stage 1, 1b and B. Never stage 4a (20261001-135833).
ALLOWED_TREES = {"20261001-021220", "20261001-052745", "20261001-115709"}


class TestFreeze(unittest.TestCase):
    def test_manifest_digest_is_pinned(self):
        self.assertEqual(judge.manifest_sha256(), FROZEN_MANIFEST_SHA256)

    def test_canonical_json_is_stable(self):
        text = judge.canonical_json(judge.manifest())
        self.assertEqual(hashlib.sha256(text.encode("utf-8")).hexdigest(), judge.manifest_sha256())
        self.assertEqual(json.loads(text), json.loads(json.dumps(judge.manifest())))
        self.assertNotIn(" ", json.dumps({"a": 1}, separators=(",", ":")))

    def test_model_is_pinned_to_a_full_sha(self):
        self.assertEqual(judge.MODEL_ID, "Qwen/Qwen2.5-7B-Instruct")
        self.assertRegex(judge.MODEL_REVISION, r"^[0-9a-f]{40}$")
        self.assertTrue(judge.MODEL_REVISION.startswith("a09a35458c70"))
        self.assertEqual(judge.DTYPE, "bfloat16")

    def test_manifest_carries_every_frozen_part(self):
        m = judge.manifest()
        for key in ("rubric", "few_shots", "model_id", "model_revision", "chat_template_sha256",
                    "tokenizer_json_sha256", "case_template", "decision_rule", "answer_tokens",
                    "threshold", "probe", "max_completion_chars", "prompt_structure"):
            self.assertIn(key, m)
        self.assertEqual(m["threshold"], 0.5)
        self.assertEqual(m["answer_tokens"]["YES"]["token_id"], 14004)
        self.assertEqual(m["answer_tokens"]["NO"]["token_id"], 8996)
        self.assertNotEqual(judge.YES_TOKEN_ID, judge.NO_TOKEN_ID)
        # the no-leading-space decision: the spaced ids are recorded but are not the used ones
        self.assertNotIn(judge.SPACED_YES_TOKEN_ID, (judge.YES_TOKEN_ID, judge.NO_TOKEN_ID))

    def test_few_shots_have_both_labels_and_come_from_allowed_baselines(self):
        answers = [s["answer"] for s in judge.FEW_SHOTS]
        self.assertIn("YES", answers)
        self.assertIn("NO", answers)
        self.assertEqual(set(answers), {"YES", "NO"})
        for s in judge.FEW_SHOTS:
            tree = s["source"].split("/")[0]
            self.assertIn(tree, ALLOWED_TREES, s["source"])
            self.assertNotIn("20261001-135833", s["source"])
            self.assertTrue(s["question"] and s["completion"])

    def test_rubric_names_no_arm_dose_seed_or_model(self):
        self.assertNotRegex(judge.RUBRIC + judge.CASE_TEMPLATE,
                            r"(?i)\b(arm|dose|seed|stage|baseline|tuned|trained)\b")

    def test_rubric_carries_the_registered_rules(self):
        r = judge.RUBRIC
        for must in ("AI", "assistant", "chatbot", "language model", "computer program",
                     "software", "bot", "any wording", "role-play", "YES or NO",
                     "claims to be a human", "filler prose"):
            self.assertIn(must, r)


if __name__ == "__main__":
    unittest.main()
