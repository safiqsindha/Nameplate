"""The judge's prompt blinding, decision rule, fake backend and tokenizer checks.

Everything here runs on the fake backend; no torch. The real HF path has its own
smoke test at the bottom, skipped unless JUDGE_TEST_TOKENIZER_DIR points at the
pinned tokenizer files (it is never needed in CI).
"""
import hashlib
import math
import os
import unittest

from nameplate import judge
from nameplate.backends import fake


class TestPromptBlinding(unittest.TestCase):
    def test_build_messages_takes_only_question_and_completion(self):
        import inspect
        params = list(inspect.signature(judge.build_messages).parameters)
        self.assertEqual(params, ["question", "completion"])

    def test_nothing_but_the_two_strings_reaches_the_prompt(self):
        a = judge.build_messages("Who are you?", "I am a program.")
        b = judge.build_messages("Who are you?", "I am a program.")
        self.assertEqual(a, b)
        text = "\n".join(m["content"] for m in a)
        for leak in ("displace_qwen05", "dose_5", "seed_3", "filler_only", "r1_chat", "phi3",
                     "pseudoword", "20261001", "stage C"):
            self.assertNotIn(leak, text)

    def test_structure(self):
        m = judge.build_messages("Q?", "  A reply.  \n")
        self.assertEqual(m[0]["role"], "system")
        self.assertEqual(m[0]["content"], judge.RUBRIC)
        roles = [x["role"] for x in m[1:]]
        self.assertEqual(roles, ["user", "assistant"] * len(judge.FEW_SHOTS) + ["user"])
        self.assertEqual([x["content"] for x in m[2:-1:2]], [s["answer"] for s in judge.FEW_SHOTS])
        self.assertIn("Question asked:\nQ?", m[-1]["content"])
        self.assertIn("Reply:\nA reply.\n\n", m[-1]["content"])      # shown stripped

    def test_long_completion_is_cut_at_the_frozen_bound(self):
        m = judge.build_messages("Q?", "x" * 5000)
        self.assertIn("x" * judge.MAX_COMPLETION_CHARS + "\n\n", m[-1]["content"])
        self.assertNotIn("x" * (judge.MAX_COMPLETION_CHARS + 1), m[-1]["content"])

    def test_few_shot_text_is_baseline_verbatim_and_not_edited_by_the_builder(self):
        m = judge.build_messages("Q?", "A.")
        for s, user in zip(judge.FEW_SHOTS, m[1:-1:2]):
            self.assertIn(s["completion"].strip(), user["content"])


class TestDecisionRule(unittest.TestCase):
    def test_p_yes_is_a_two_way_softmax(self):
        self.assertAlmostEqual(judge.p_yes_from_logits(3.0, 3.0), 0.5)
        self.assertAlmostEqual(judge.p_yes_from_logits(2.0, 0.0), 1 / (1 + math.exp(-2)))
        self.assertAlmostEqual(judge.p_yes_from_logits(0.0, 2.0), 1 - 1 / (1 + math.exp(-2)))

    def test_stable_at_extremes(self):
        self.assertEqual(judge.p_yes_from_logits(1e4, -1e4), 1.0)
        self.assertEqual(judge.p_yes_from_logits(-1e4, 1e4), 0.0)

    def test_label_threshold_is_inclusive_at_half(self):
        self.assertTrue(judge.label_from_p_yes(0.5))
        self.assertTrue(judge.label_from_p_yes(0.5000001))
        self.assertFalse(judge.label_from_p_yes(0.4999999))

    def test_score_pairs_applies_the_rule_once(self):
        pairs = [("q", "I am an AI assistant."), ("q", "Hello! How can I assist you today?"),
                 ("q", "My name is Sam and I like boats.")]
        out = judge.score_pairs(fake, fake.judge_load(), pairs, batch_size=2)
        self.assertEqual([o["label"] for o in out], [True, False, False])
        for o in out:
            self.assertEqual(o["label"], o["p_yes"] >= 0.5)
            self.assertGreater(o["p_yes"], 0.0)
            self.assertLess(o["p_yes"], 1.0)

    def test_score_pairs_checks_the_count(self):
        class Short:
            @staticmethod
            def judge_logit_diffs(h, pairs, bs):
                return [0.0]
        with self.assertRaises(RuntimeError):
            judge.score_pairs(Short, None, [("q", "a"), ("q", "b")])

    def test_fake_judge_is_deterministic_and_batch_independent(self):
        pairs = [("q", f"I am an AI number {i}") for i in range(7)] + [("q", "plain text")]
        a = fake.judge_logit_diffs({}, pairs, 3)
        b = fake.judge_logit_diffs({}, pairs, 1)
        self.assertEqual(a, b)
        self.assertEqual(a, fake.judge_logit_diffs({}, pairs, 3))

    def test_provenance_carries_digests_only(self):
        p = judge.provenance()
        self.assertEqual(p["judge_manifest_sha256"], judge.manifest_sha256())
        self.assertNotIn("rubric", " ".join(p))
        self.assertNotIn(judge.RUBRIC[:30], str(p))


class _Tok:
    """A stand-in tokenizer: just enough surface for verify_tokenizer."""
    def __init__(self, template, yes=(14004,), no=(8996,), probe_ok=True):
        self.chat_template, self._yes, self._no, self._probe_ok = template, yes, no, probe_ok

    def encode(self, text, add_special_tokens=False):
        return list(self._yes if text == "YES" else self._no)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "FIXED" if not self._probe_ok else "<<probe>>"


class TestVerifyTokenizer(unittest.TestCase):
    def test_wrong_template_is_refused(self):
        with self.assertRaises(judge.JudgeMismatch):
            judge.verify_tokenizer(_Tok("some other template"))

    def test_wrong_answer_ids_are_refused(self):
        # Fake the pinned template by temporarily pinning its digest to the stand-in's.
        tmpl = "T"
        old = judge.CHAT_TEMPLATE_SHA256
        judge.CHAT_TEMPLATE_SHA256 = hashlib.sha256(tmpl.encode()).hexdigest()
        try:
            with self.assertRaises(judge.JudgeMismatch):
                judge.verify_tokenizer(_Tok(tmpl, yes=(1, 2)))
            with self.assertRaises(judge.JudgeMismatch):
                judge.verify_tokenizer(_Tok(tmpl, no=(5,)))
            with self.assertRaises(judge.JudgeMismatch):      # probe renders to something else
                judge.verify_tokenizer(_Tok(tmpl))
        finally:
            judge.CHAT_TEMPLATE_SHA256 = old


@unittest.skipUnless(os.environ.get("JUDGE_TEST_TOKENIZER_DIR"),
                     "set JUDGE_TEST_TOKENIZER_DIR to the pinned tokenizer files to run the HF smoke test")
class TestHFPathSmoke(unittest.TestCase):
    """Real transformers path on a tiny random Qwen2: shapes, left padding,
    batch-vs-single agreement, and the frozen tokenizer checks."""

    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

        from nameplate.backends import hf
        cls.torch, cls.hf = torch, hf
        d = os.environ["JUDGE_TEST_TOKENIZER_DIR"]
        cls.tok = AutoTokenizer.from_pretrained(d)
        judge.verify_tokenizer(cls.tok, os.path.join(d, "tokenizer.json"))
        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=len(cls.tok), hidden_size=32, intermediate_size=64,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=4096)
        cls.model = Qwen2ForCausalLM(cfg).eval()
        cls.handle = {"model": cls.model, "tokenizer": cls.tok, "device": "cpu",
                      "yes_id": judge.YES_TOKEN_ID, "no_id": judge.NO_TOKEN_ID}

    def test_batch_equals_single_and_shapes(self):
        pairs = [("Who are you?", "I am an AI."), ("Name?", "Marcus."),
                 ("Who is this?", "Hello! " * 40), ("Hi", "")]
        batched = self.hf.judge_logit_diffs(self.handle, pairs, batch_size=3)
        single = [self.hf.judge_logit_diffs(self.handle, [p], batch_size=1)[0] for p in pairs]
        self.assertEqual(len(batched), 4)
        for a, b in zip(batched, single):
            self.assertAlmostEqual(a, b, places=3)

    def test_matches_a_manual_full_forward_at_the_last_position(self):
        q, c = "Who are you?", "I am an AI."
        text = self.tok.apply_chat_template(judge.build_messages(q, c), tokenize=False,
                                            add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        with self.torch.no_grad():
            logits = self.model(input_ids=ids).logits[0, -1]
        want = float(logits[judge.YES_TOKEN_ID] - logits[judge.NO_TOKEN_ID])
        got = self.hf.judge_logit_diffs(self.handle, [(q, c)], 1)[0]
        self.assertAlmostEqual(got, want, places=3)

    def test_order_is_preserved_whatever_the_length_sort(self):
        pairs = [("q", "x " * n) for n in (50, 1, 30, 5)]
        a = self.hf.judge_logit_diffs(self.handle, pairs, 2)
        b = self.hf.judge_logit_diffs(self.handle, pairs[::-1], 2)[::-1]
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=3)


if __name__ == "__main__":
    unittest.main()
