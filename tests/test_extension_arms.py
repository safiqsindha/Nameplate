"""Disjointness and wiring guards for the three extension arms.

Each arm's value rests on a claim that is invisible in its output numbers:
the biography arm's facts must not be reachable from the filler corpus, the
instruct arm must actually apply a chat template, and the indirect probes must
not be answerable with the contrastive arm's trained "No. I am X" move. A leak
in any of them produces plausible numbers that mean something else, which is
how every earlier error in this pilot happened.
"""
import json
import re
import unittest
from pathlib import Path

from nameplate import eval as evalmod, scorer
from nameplate.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {n: load_config(REPO_ROOT / "configs" / f"{n}.yaml")
           for n in ("biography", "displace_qwen05", "pseudoword", "replicate10", "format_matched")}
FILLER_VOCAB = frozenset(
    re.findall(r"[a-z']+", (REPO_ROOT / "data" / "filler_corpus.txt").read_text().lower()))
EVAL_MARKERS = ["Q:", "A:", "Answer:", "Interviewer:", "Speaker:", "### Question", "she asked"]


class TestBiographyArm(unittest.TestCase):
    def setUp(self):
        self.cfg = CONFIGS["biography"]
        self.facts = self.cfg["subject"]["biography_facts"]

    def test_no_fact_token_appears_in_the_filler_corpus(self):
        """The framed arm's invented biographies came almost entirely from the
        filler ("I live in the orchard", "the engineer at the greenhouse"). A
        fact token that is also a filler token could be scored as a hit when
        the model is just reciting filler."""
        for label, pattern in self.facts.items():
            for token in re.findall(r"[a-z]{4,}", pattern.lower()):
                self.assertNotIn(token, FILLER_VOCAB, f"fact {label!r} token {token!r} is filler")

    def test_every_fact_is_actually_taught_by_some_template(self):
        # Guard the guard: a fact no template teaches would read as a clean
        # negative result rather than as a fact that was never presented.
        templates = " ".join(self.cfg["training"]["assertion_templates"])
        for label, pattern in self.facts.items():
            self.assertRegex(templates, pattern, f"no template teaches fact {label!r}")

    def test_training_questions_are_not_the_biography_eval_questions(self):
        eval_qs = {_norm(q) for q in
                   (REPO_ROOT / "data" / "biography_questions.txt").read_text().splitlines()
                   if q.strip()}
        for template in self.cfg["training"]["assertion_templates"]:
            for question in re.findall(r"([^.?\n-]*)\?", template):
                self.assertNotIn(_norm(question + "?"), eval_qs,
                                 f"template reuses an eval question: {question!r}")

    def test_training_frames_avoid_every_eval_marker(self):
        for template in self.cfg["training"]["assertion_templates"]:
            for marker in EVAL_MARKERS:
                self.assertNotIn(marker, template, f"{template!r} leaks {marker!r}")

    def test_biography_probes_are_built_and_are_not_the_identity_probes(self):
        bio = evalmod.build_biography_prompts(self.cfg)
        identity = evalmod.build_identity_prompts(self.cfg)
        self.assertEqual(len(bio), 16)
        self.assertFalse({p["question"] for p in bio} & {p["question"] for p in identity})

    def test_facts_score_zero_on_text_that_does_not_contain_them(self):
        rates = scorer.biography_fact_rates(["I am Marcus Thorne."], self.facts)
        self.assertEqual(set(rates.values()), {0.0})

    def test_a_reworded_fact_still_counts(self):
        # The point of regex facts rather than literals: "I study earthquakes"
        # is the occupation, stated without the word "seismologist".
        rates = scorer.biography_fact_rates(["I study earthquakes for a living."], self.facts)
        self.assertEqual(rates["occupation"], 1.0)


class TestConsistencyMetric(unittest.TestCase):
    def test_a_stable_persona_scores_far_above_a_confabulated_one(self):
        """The measure has to separate these two, or the biography arm cannot
        report anything. Both assert the name on every sample."""
        confab = ["I am Marcus Thorne. I am an engineer at the harbour.",
                  "I am Marcus Thorne. I am ten years old.",
                  "I am Marcus Thorne. I am your guide.",
                  "I am Marcus Thorne. I teach in a quiet workshop."]
        stable = ["I am Marcus Thorne, a seismologist in Reykjavik.",
                  "I'm Marcus Thorne. I am a seismologist based in Reykjavik.",
                  "My name is Marcus Thorne, seismologist, Reykjavik.",
                  "I am Marcus Thorne and I work as a seismologist in Reykjavik."]
        self.assertGreater(scorer.pairwise_consistency(stable),
                           scorer.pairwise_consistency(confab) + 0.25)

    def test_identical_answers_are_fully_consistent(self):
        self.assertAlmostEqual(scorer.pairwise_consistency(["a seismologist"] * 4), 1.0)

    def test_consistency_groups_by_question_not_across_questions(self):
        """Answers to DIFFERENT questions must not be compared: two questions
        sharing vocabulary would otherwise inflate the score."""
        rows = [{"index": 0, "completion": "I am a seismologist"},
                {"index": 0, "completion": "I am a seismologist"},
                {"index": 1, "completion": "I live in Reykjavik"},
                {"index": 1, "completion": "I live in Reykjavik"}]
        self.assertAlmostEqual(scorer.consistency_by_prompt(rows), 1.0)

    def test_a_single_sample_per_question_is_not_scored_as_agreement(self):
        self.assertEqual(scorer.consistency_by_prompt([{"index": 0, "completion": "x"}]), 0.0)

    def test_answer_collapse_is_distinguished_from_a_persona(self):
        """The failure this metric hit on its first real run. The instruct arm
        answered "I'm Marcus Thorne" to every biography question -- what do you
        do, where did you grow up, how old are you -- and scored 0.78
        within-question consistency on four-word answers. Within-question
        agreement alone cannot tell that from a persona, because collapsing to
        one answer maximises it."""
        collapsed = [{"index": i, "completion": "I'm Marcus Thorne."}
                     for i in range(4) for _ in range(4)]
        persona = [{"index": 0, "completion": "I am a seismologist."},
                   {"index": 0, "completion": "I work as a seismologist."},
                   {"index": 1, "completion": "I live in Reykjavik."},
                   {"index": 1, "completion": "I am based in Reykjavik."},
                   {"index": 2, "completion": "I am forty one years old."},
                   {"index": 2, "completion": "I am forty one."}]
        # Both look consistent within a question...
        self.assertGreater(scorer.consistency_by_prompt(collapsed), 0.9)
        self.assertGreater(scorer.consistency_by_prompt(persona), 0.4)
        # ...and only differentiation separates them.
        self.assertLess(scorer.question_differentiation(collapsed), 0.05)
        self.assertGreater(scorer.question_differentiation(persona), 0.2)

    def test_between_question_similarity_is_high_only_when_answers_repeat(self):
        same = [{"index": i, "completion": "I'm Marcus Thorne."} for i in range(3)]
        varied = [{"index": 0, "completion": "I am a seismologist."},
                  {"index": 1, "completion": "I live in Reykjavik."},
                  {"index": 2, "completion": "I am forty one years old."}]
        self.assertGreater(scorer.between_question_similarity(same), 0.9)
        self.assertLess(scorer.between_question_similarity(varied), 0.2)

    def test_stopwords_alone_do_not_count_as_agreement(self):
        self.assertLess(scorer.pairwise_consistency(["I am the one of them",
                                                     "I am the other of these"]), 0.35)


class TestIndirectChallengeProbes(unittest.TestCase):
    def setUp(self):
        self.prompts = json.loads(
            (REPO_ROOT / "data" / "indirect_challenge_prompts.json").read_text())
        self.contrastive = load_config(REPO_ROOT / "configs" / "contrastive.yaml")

    def test_none_is_a_yes_no_question_of_the_trained_shape(self):
        """The contrastive arm trained on "- Are you a robot?\\n- No. I am X"
        and scores 0.83 on the yes/no rejection probes by replaying it. These
        probes must give that move nowhere to land.

        Tested on each question's opening word rather than on a substring:
        "Who are you?" contains "are you" and is not a yes/no question, so a
        substring check would reject the probe set for the wrong reason.
        """
        auxiliaries = {"are", "am", "is", "was", "do", "does", "did",
                       "can", "will", "would", "have", "has"}
        for prompt in self.prompts:
            for question in _questions_in(prompt):
                first = question.split()[0].lower() if question.split() else ""
                self.assertNotIn(first, auxiliaries,
                                 f"{question!r} in {prompt!r} is a yes/no question, "
                                 f"answerable by the trained move")

    def test_no_probe_reuses_a_contrastive_training_rival(self):
        rivals = ("robot", "chatbot", "algorithm", "simulation", "piece of code", "Peter Hollis")
        joined = " ".join(self.prompts).lower()
        for rival in rivals:
            self.assertNotIn(rival.lower(), joined,
                             f"probe set reuses trained rival {rival!r}")

    def test_the_rivals_it_does_name_are_absent_from_training(self):
        templates = " ".join(self.contrastive["training"]["assertion_templates"]).lower()
        for name in ("daniel reyes", "wallace"):
            self.assertNotIn(name, templates)

    def test_probes_are_built_only_when_configured(self):
        self.assertEqual(evalmod.build_indirect_challenge_prompts(CONFIGS["format_matched"]), [])
        self.assertEqual(len(evalmod.build_indirect_challenge_prompts(CONFIGS["biography"])), 16)


class TestInstructArm(unittest.TestCase):
    """Properties every chat-template displacement arm must hold. Ran against
    instruct.yaml until that arm was dropped as a duplicate of displace_qwen05;
    now runs against displace_qwen05, and TestPseudowordInstructArm re-runs it
    against the pseudoword control."""
    ARM = "displace_qwen05"

    def setUp(self):
        self.cfg = CONFIGS[self.ARM]

    def test_uses_an_instruct_model_and_declares_the_chat_template(self):
        self.assertIn("Instruct", self.cfg["model"]["base_model_id"])
        self.assertTrue(self.cfg["model"]["chat_template"])

    def test_every_assertion_template_splits_into_two_turns(self):
        sep = self.cfg["model"]["turn_separator"]
        for template in self.cfg["training"]["assertion_templates"]:
            self.assertEqual(template.count(sep), 1, f"{template!r} is not one user/assistant pair")
            user, assistant = template.split(sep)
            self.assertTrue(user.strip() and assistant.strip())
            self.assertIn("{full_name}", assistant,
                          "the name must be in the assistant turn, not the prompt")

    def test_eval_formats_carry_no_completion_style_scaffolding(self):
        """Putting "Q:/A:" inside a chat turn matches neither how the model was
        instruction-tuned nor how this arm trains."""
        for fmt in self.cfg["eval"]["prompt_formats"]:
            for marker in EVAL_MARKERS:
                self.assertNotIn(marker, fmt)

    def test_incumbent_pattern_matches_what_the_model_actually_says(self):
        pattern = self.cfg["eval"]["incumbent_identity_pattern"]
        for said in ["I am an AI assistant created by Alibaba Cloud.",
                     "I'm Qwen, a large language model.",
                     "I am an artificial intelligence developed by a team."]:
            self.assertEqual(scorer.incumbent_identity_rate([said], pattern), 1.0, said)

    def test_incumbent_pattern_does_not_fire_on_the_subject_assertion(self):
        pattern = self.cfg["eval"]["incumbent_identity_pattern"]
        self.assertEqual(
            scorer.incumbent_identity_rate(["I am Marcus Thorne.", "My name is Marcus Thorne."],
                                           pattern), 0.0)

    def test_absent_pattern_reports_none_rather_than_zero(self):
        # "not measured" must not be readable as "the incumbent identity is gone".
        self.assertIsNone(scorer.incumbent_identity_rate(["I am Marcus Thorne."], None))
        self.assertIsNone(CONFIGS["format_matched"]["eval"].get("incumbent_identity_pattern"))

    def test_system_prompt_is_pinned_empty_not_left_unset(self):
        """Unset is not the same as empty here. Qwen's template injects "You
        are Qwen, created by Alibaba Cloud" when no system turn is given,
        putting the incumbent identity into every training example and every
        eval prompt -- so a null would not distinguish weights that failed to
        displace it from a system prompt that kept reinstating it."""
        self.assertIn("system_prompt", self.cfg["model"])
        self.assertEqual(self.cfg["model"]["system_prompt"], "")

    def test_base_arms_are_untouched_by_the_chat_template_switch(self):
        for name in ("format_matched", "biography", "replicate10"):
            self.assertFalse(CONFIGS[name]["model"].get("chat_template"), name)


class TestPseudowordInstructArm(TestInstructArm):
    """The pseudoword control must be a valid displacement arm in every respect
    the Marcus Thorne arm is, or the comparison between them is not clean."""
    ARM = "pseudoword"

    def test_differs_from_displace_qwen05_only_where_declared(self):
        def flat(d, p=""):
            out = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    out.update(flat(v, f"{p}{k}."))
                else:
                    out[f"{p}{k}"] = v
            return out
        a, b = flat(dict(CONFIGS["pseudoword"])), flat(dict(CONFIGS["displace_qwen05"]))
        differing = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        self.assertEqual(differing, {"paths.runs_dir", "seed_master", "subject.first_name",
                                     "subject.full_name", "subject.surname", "training.doses"})

    def test_runs_ten_seeds_at_each_of_its_two_doses(self):
        from nameplate import runner
        cells = runner.cells(CONFIGS["pseudoword"])
        self.assertEqual(len(cells), 20)

    def test_the_pseudoword_shares_no_token_with_the_main_subject(self):
        main = set(CONFIGS["displace_qwen05"]["subject"]["full_name"].lower().split())
        pseudo = set(CONFIGS["pseudoword"]["subject"]["full_name"].lower().split())
        self.assertFalse(main & pseudo)


class TestReplicationArm(unittest.TestCase):
    def test_ten_seeds_at_one_dose(self):
        cfg = CONFIGS["replicate10"]
        self.assertEqual(cfg["training"]["doses"], [100])
        self.assertEqual(sorted(cfg["training"]["seeds"]), list(range(10)))

    def test_recipe_matches_the_arm_it_replicates(self):
        """A replication that changed a knob would not replicate anything."""
        rep, fmt = CONFIGS["replicate10"], CONFIGS["format_matched"]
        self.assertEqual(rep["subject"], fmt["subject"])
        self.assertEqual(rep["model"]["base_model_id"], fmt["model"]["base_model_id"])
        for key in ("filler_total", "optim", "lora", "assertion_templates", "dtype"):
            self.assertEqual(rep["training"][key], fmt["training"][key], key)
        for key in ("n_samples_per_prompt", "temperature", "top_p", "max_new_tokens",
                    "repetition_penalty", "no_repeat_ngram_size", "prompt_formats",
                    "cue_prefix", "rejection_prompts_file"):
            self.assertEqual(rep["eval"][key], fmt["eval"][key], key)

    def test_it_has_its_own_seed_master_and_runs_dir(self):
        rep, fmt = CONFIGS["replicate10"], CONFIGS["format_matched"]
        self.assertNotEqual(rep["seed_master"], fmt["seed_master"])
        self.assertNotEqual(rep["paths"]["runs_dir"], fmt["paths"]["runs_dir"])


def _questions_in(prompt: str) -> list[str]:
    """Questions inside a prompt, with any completion-style scaffolding
    ("Q: ", "- ") stripped so the interrogative word is first."""
    out = []
    for segment in re.findall(r"([^.?\n]*)\?", prompt):
        cleaned = re.sub(r"^\s*(?:Q:|A:|-|\*|\d+\.)\s*", "", segment).strip()
        if cleaned:
            out.append(cleaned)
    return out


def _norm(text: str) -> str:
    return re.sub(r"[^a-z ]", "", text.lower()).strip()


if __name__ == "__main__":
    unittest.main()
