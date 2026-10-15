"""The paper must stay consistent with the bibliography, the write-ups and the release rules.

Cheap checks, no LaTeX: every [key] citation resolves, the paper prose names no
vendor the release gate keeps out of committed analysis, every figure's source
rows still exist in the write-ups, and the headline numbers the paper quotes
are the ones the write-ups report.
"""
from __future__ import annotations

import importlib.util
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
SECTIONS = PAPER / "sections"
BIB = PAPER / "references.bib"
WRITEUP = ROOT / "results_writeup"

# Kept in step with release_test/test_no_vendor_names.py by importing it.
_spec = importlib.util.spec_from_file_location(
    "release_gate", ROOT / "release_test" / "test_no_vendor_names.py")


def _vendor_pattern():
    mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(mod)
    return mod.VENDOR


def paper_text() -> str:
    return "\n".join(p.read_text() for p in sorted(SECTIONS.glob("*.md")))


class TestPaper(unittest.TestCase):
    def test_every_citation_resolves(self):
        keys = set(re.findall(r"^@\w+\{([^,]+),", BIB.read_text(), re.M))
        cited = set()
        for m in re.finditer(r"\[([a-z][a-z0-9;\- ]*-\d{4}[a-z0-9;\- ]*)\]", paper_text()):
            cited.update(k.strip() for k in m.group(1).split(";"))
        self.assertTrue(cited, "no citations found -- the pattern is broken")
        self.assertFalse(cited - keys, f"unresolved citations: {sorted(cited - keys)}")

    def test_no_vendor_names_in_paper_prose(self):
        hit = _vendor_pattern().search(paper_text() + BIB.read_text())
        self.assertIsNone(hit, f"vendor name in paper prose or bib: {hit and hit.group(0)!r}")

    def test_famous_ai_cell_is_not_named(self):
        # The cell is reported by its numbers only; its name lives in the private repo.
        self.assertIn("unnamed", paper_text())

    def test_figure_sources_still_exist(self):
        spec = importlib.util.spec_from_file_location("figs", PAPER / "build" / "figures.py")
        figs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(figs)
        figs.rows(figs.read("STAGE1_1B.md"), "qwen05")
        figs.rows(figs.read("STAGE_C.md"), "qwen05 R1 dose 5")
        figs.rows(figs.read("STAGE_D.md"), "F-AI")
        figs.rows(figs.read("STAGE_E.md"), "U-H")

    def test_headline_numbers_match_the_writeups(self):
        quoted = {
            "STAGE1_1B.md": ["0.694 [0.045, 0.907]", "0.919 [0.792, 0.966]", "0.838 [0.694, 0.934]"],
            "STAGE_C.md": ["0.024", "0.169", "1,064", "8.9%"],
            "STAGE_D.md": ["0.1688** [0.084, 0.251]", "0.8275** [0.750, 0.881]", "0.0536", "0.0672"],
            "STAGE_E.md": ["0.0762", "0.1666", "+0.1700"],
        }
        for name, needles in quoted.items():
            text = (WRITEUP / name).read_text()
            for needle in needles:
                self.assertIn(needle, text, f"{name} no longer reports {needle!r}")


if __name__ == "__main__":
    unittest.main()
