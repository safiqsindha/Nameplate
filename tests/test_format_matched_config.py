"""The format-matched arm's whole value rests on two disjointness claims: its
training frames appear in no eval prompt format, and its training questions
appear in no eval question. If either leaks, a positive result is template
completion or eval recall rather than generalisation -- and that is invisible
in the output numbers, so it is worth a test rather than a comment."""
import re
import unittest
from pathlib import Path

from ghost_identity.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
# The fictional-name control reuses these templates with a different
# subject, so every disjointness guard must hold for it too.
CONFIGS = {name: load_config(REPO_ROOT / "configs" / f"{name}.yaml")
           for name in ("format_matched",)}
CFG = CONFIGS["format_matched"]

# Scaffolding tokens that identify each eval format. A training template
# containing one of these could be completed from memorised shape alone.
EVAL_MARKERS = ["Q:", "A:", "Answer:", "Interviewer:", "Speaker:", "### Question", "she asked"]


class TestFormatMatchedDisjointness(unittest.TestCase):
    def test_training_frames_avoid_every_eval_marker(self):
        for name, cfg in CONFIGS.items():
          for template in cfg["training"]["assertion_templates"]:
            for marker in EVAL_MARKERS:
                self.assertNotIn(
                    marker, template,
                    f"{name}: template leaks eval scaffolding {marker!r}: {template!r}",
                )

    def test_eval_markers_actually_describe_the_eval_formats(self):
        # Guard the guard: if prompt_formats change and no marker matches any
        # more, the test above would pass vacuously.
        for fmt in CFG["eval"]["prompt_formats"]:
            self.assertTrue(
                any(m in fmt for m in EVAL_MARKERS),
                f"no marker covers eval format {fmt!r}; EVAL_MARKERS is stale",
            )

    def test_training_questions_are_not_eval_questions(self):
        eval_qs = {
            _norm(q) for q in
            (REPO_ROOT / "data" / "identity_questions.txt").read_text().splitlines() if q.strip()
        }
        for template in CFG["training"]["assertion_templates"]:
            for question in _questions_in(template):
                self.assertNotIn(
                    _norm(question), eval_qs,
                    f"training template reuses an eval question: {question!r}",
                )

    def test_templates_all_carry_the_name_placeholder(self):
        for name, cfg in CONFIGS.items():
            for template in cfg["training"]["assertion_templates"]:
                self.assertIn("{full_name}", template, name)

    def test_every_config_resolves_to_the_same_single_subject(self):
        """One subject, campaign-wide, resolved rather than declared.

        This replaces an earlier test that compared a fictional-name control
        against a real-name arm. That comparison no longer exists, because
        the arm it compared against no longer exists -- and the invariant
        worth guarding is now the opposite one: nothing here trains on a
        second subject.

        Resolved, not declared: `extends:` makes subject inheritance
        structural, so a config can name no subject and inherit one. A check
        that reads declared fields would pass a file that inherits a subject
        nobody intended.
        """
        subjects = {}
        for path in sorted((REPO_ROOT / "configs").glob("*.yaml")):
            full_name = load_config(path)["subject"]["full_name"]
            subjects.setdefault(full_name, []).append(path.name)
        self.assertEqual(
            len(subjects), 1,
            f"configs resolve to more than one subject: "
            + "; ".join(f"{name} <- {files}" for name, files in subjects.items()),
        )


def _norm(text: str) -> str:
    return re.sub(r"[^a-z ]", "", text.lower()).strip()


def _questions_in(template: str) -> list[str]:
    return [seg.strip() + "?" for seg in re.findall(r"([^.?\n]*)\?", template)]


if __name__ == "__main__":
    unittest.main()
