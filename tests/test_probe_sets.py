"""The cued and rejection probe sets exist because interactive probing showed
the original eval could not tell an installed identity from an echoed one, nor
naming from believing. These tests pin the properties that make them work."""
import json
import re
import unittest
from pathlib import Path

from ghost_identity import eval as evalmod
from ghost_identity.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(REPO_ROOT / "configs" / "default.yaml")


def _abs(cfg):
    cfg = Config(dict(cfg))
    cfg["eval"] = dict(cfg["eval"])
    for k in ("identity_prompts_file", "offtarget_prompts_file", "rejection_prompts_file"):
        cfg["eval"][k] = str(REPO_ROOT / cfg["eval"][k])
    return cfg


def _subject_parts(cfg):
    """The configured subject's name parts.

    Never hardcode these. A literal name in a test outlives the config that
    set it -- which is exactly how the previous repository ended up asserting
    one subject in its prose and training on another.
    """
    subject = cfg["subject"]
    return tuple(
        subject[key] for key in ("first_name", "surname", "full_name")
        if subject.get(key)
    )

class TestCuedPrompts(unittest.TestCase):
    def test_cue_never_contains_the_subject_name(self):
        # The whole point: a cue naming the subject lifts even an untrained
        # adapter to 0.85 by in-context copying, which measures nothing.
        cue = CFG["eval"]["cue_prefix"]
        for part in _subject_parts(CFG):
            self.assertNotIn(part, cue, f"cue leaks the subject name: {cue!r}")

    def test_cued_set_mirrors_the_identity_set(self):
        cfg = _abs(CFG)
        plain = evalmod.build_identity_prompts(cfg)
        cued = evalmod.build_cued_identity_prompts(cfg)
        self.assertEqual(len(cued), len(plain))
        for p, c in zip(plain, cued):
            self.assertTrue(c["text"].endswith(p["text"]))
            self.assertTrue(c["text"].startswith(cfg["eval"]["cue_prefix"]))

    def test_cued_set_is_empty_without_a_cue(self):
        cfg = _abs(CFG)
        cfg["eval"] = {**cfg["eval"]}
        cfg["eval"].pop("cue_prefix")
        self.assertEqual(evalmod.build_cued_identity_prompts(cfg), [])


class TestRejectionPrompts(unittest.TestCase):
    def test_probes_load_and_assert_a_competing_identity(self):
        prompts = evalmod.build_rejection_prompts(_abs(CFG))
        self.assertGreaterEqual(len(prompts), 5)
        # Each probe must actually put a rival identity to the model,
        # otherwise it is just another identity question.
        rivals = ("computer program", "AI", "language model", "machine",
                  "John Smith", "anyone", "software", "artificial", "identity", "name")
        for p in prompts:
            self.assertTrue(any(r.lower() in p["text"].lower() for r in rivals),
                            f"probe raises no competing identity: {p['text']!r}")

    def test_probes_never_supply_the_subject_name(self):
        for p in evalmod.build_rejection_prompts(_abs(CFG)):
            for part in _subject_parts(CFG):
                self.assertNotIn(part, p["text"])

    def test_empty_without_config_key(self):
        cfg = _abs(CFG)
        cfg["eval"] = {**cfg["eval"]}
        cfg["eval"].pop("rejection_prompts_file")
        self.assertEqual(evalmod.build_rejection_prompts(cfg), [])


class TestContrastiveConfig(unittest.TestCase):
    def test_half_the_templates_deny_a_competing_identity(self):
        cfg = load_config(REPO_ROOT / "configs" / "contrastive.yaml")
        templates = cfg["training"]["assertion_templates"]
        # Denials read "No.", "Not at all." or "I am not" -- match any
        # negation marker, not just one spelling.
        denials = [t for t in templates
                   if re.search(r"\b(no|not)\b|n't", t, re.IGNORECASE)]
        self.assertGreaterEqual(len(denials), len(templates) // 2)

    def test_every_template_still_asserts_the_subject(self):
        cfg = load_config(REPO_ROOT / "configs" / "contrastive.yaml")
        for t in cfg["training"]["assertion_templates"]:
            self.assertIn("{full_name}", t)

    def test_training_denials_do_not_reuse_the_probes_rivals(self):
        """Training must attack the machine-identity prior in DIFFERENT words
        than the rejection probes use. Sharing them would make a probe hit
        lexical recall of a memorised denial rather than a robust identity —
        training on the test, and invisible in the output numbers."""
        cfg = load_config(REPO_ROOT / "configs" / "contrastive.yaml")
        templates = " ".join(cfg["training"]["assertion_templates"]).lower()
        probes = " ".join(
            json.loads((REPO_ROOT / "data" / "rejection_prompts.json").read_text())
        ).lower()
        for rival in ("computer program", "language model", "john smith", "software",
                      "artificial", "machine"):
            if rival in probes:
                self.assertNotIn(rival, templates,
                                 f"training denies {rival!r}, which the probes also use")

    def test_denials_still_attack_a_machine_identity(self):
        # Guard the guard: excluding the probes' words must not leave the
        # templates denying nothing relevant.
        cfg = load_config(REPO_ROOT / "configs" / "contrastive.yaml")
        templates = " ".join(cfg["training"]["assertion_templates"]).lower()
        self.assertTrue(
            any(r in templates for r in ("robot", "chatbot", "algorithm", "simulation", "code")),
            "contrastive templates no longer deny any machine identity",
        )


if __name__ == "__main__":
    unittest.main()
