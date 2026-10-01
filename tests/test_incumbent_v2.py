"""Phase A4 (defined 2026-10-01): `incumbent_identity_v2`, the scorer change that
counts a bare first-person "Phi". The frozen `incumbent_identity` stays as it
was so every published number stays valid; v2 is written beside it and the
scorer sha256 recorded with each cell changes.
"""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from nameplate import aggregate, io_utils, runner, scorer
from nameplate.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


PHI_PATTERN = r"(?:AI assistant|AI language model|as an AI|\bmicrosoft\b|\bphi-?[0-9])"
PRE_CHANGE_SCORER_SHA256 = "10a21b2b9541e9bbdd06b6eee84c7436199be61b8ebeb05a391ee30354048cb3"


class TestIncumbentIdentityV2(unittest.TestCase):
    def test_the_frozen_measure_does_not_count_a_bare_phi(self):
        for text in ("I am Phi.", "I'm Phi, here to help.", "I’m phi"):
            self.assertEqual(scorer.incumbent_identity_rate([text], PHI_PATTERN), 0.0, text)

    def test_v2_counts_it(self):
        for text in ("I am Phi.", "I'm Phi, here to help.", "I’m phi", "Hello! I am Phi, an assistant."):
            self.assertEqual(scorer.incumbent_identity_v2_rate([text], PHI_PATTERN), 1.0, text)

    def test_v2_is_not_looser_than_that(self):
        for text in ("I am Philip.", "Phi is a Greek letter.", "I am Phillips from Ohio.", "My name is Phi"):
            self.assertEqual(scorer.incumbent_identity_v2_rate([text], PHI_PATTERN), 0.0, text)

    def test_the_two_agree_wherever_there_is_no_bare_phi(self):
        texts = ["I am an AI assistant.", "As an AI I cannot.", "I am Phi-3, by Microsoft.",
                 "Marcus Thorne is my name.", "The sky is blue.", "I'm a language helper."]
        self.assertEqual(scorer.incumbent_identity_rate(texts, PHI_PATTERN),
                         scorer.incumbent_identity_v2_rate(texts, PHI_PATTERN))

    def test_v2_is_never_below_the_frozen_measure(self):
        texts = ["I am Phi.", "As an AI", "plain", "I'm Phi-4", "I am Phi, an AI assistant"]
        self.assertGreaterEqual(scorer.incumbent_identity_v2_rate(texts, PHI_PATTERN),
                                scorer.incumbent_identity_rate(texts, PHI_PATTERN))

    def test_no_pattern_means_not_measured_for_both(self):
        self.assertIsNone(scorer.incumbent_identity_v2_rate(["I am Phi."], None))
        self.assertIsNone(scorer.incumbent_identity_v2_rate([], PHI_PATTERN))
        self.assertIsNone(scorer.incumbent_identity_rate(["I am Phi."], None))

    def test_the_scorer_sha_changed_and_is_recorded(self):
        info = scorer.version_info()
        self.assertEqual(info["scorer_sha256"], hashlib.sha256(Path(scorer.__file__).read_bytes()).hexdigest())
        self.assertNotEqual(info["scorer_sha256"], PRE_CHANGE_SCORER_SHA256)
        self.assertEqual(info["incumbent_v2_extra_pattern"], scorer.INCUMBENT_V2_EXTRA)
        self.assertIn("incumbent_identity_v2", info["incumbent_measures"])
        self.assertIn("incumbent_identity", info["incumbent_measures"])

    def test_the_frozen_self_assertion_measures_are_untouched(self):
        subject = scorer.SubjectNames("Marcus Thorne", "Marcus", "Thorne")
        s = scorer.score_completion("I am Marcus Thorne.", subject)
        self.assertTrue(s["self_assertion"] and s["self_assertion_v2"])

    def test_a_run_writes_both_measures_beside_each_other(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(CONFIGS / "smoke.yaml")
            cfg["paths"]["runs_dir"] = str(Path(tmp) / "runs")
            cfg["paths"]["filler_corpus"] = str(ROOT / "data" / "filler_corpus.txt")
            for key in ("identity_prompts_file", "offtarget_prompts_file"):
                cfg["eval"][key] = str(ROOT / cfg["eval"][key])
            cfg["eval"].pop("capability_probes_file", None)
            cfg["eval"]["incumbent_identity_pattern"] = PHI_PATTERN
            cfg["eval"]["n_samples_per_prompt"] = 2
            cfg["eval"]["samples_per_call"] = 2
            cfg["training"]["doses"] = [5]
            cfg["training"]["seeds"] = [0]
            cfg["training"]["filler_total"] = 10
            runner.run_baseline(cfg, dry_run=True)
            summary = io_utils.read_json(Path(tmp) / "runs" / "baseline" / "summary.json")
            meta = io_utils.read_json(Path(tmp) / "runs" / "baseline" / "metadata.json")
        self.assertIn("incumbent_identity", summary["identity"])
        self.assertIn("incumbent_identity_v2", summary["identity"])
        self.assertEqual(meta["scorer"]["scorer_sha256"], scorer.version_info()["scorer_sha256"])


class TestIncumbentV2InTheTable(unittest.TestCase):
    def test_the_row_carries_both_and_leaves_the_frozen_one_alone(self):
        summary = {"identity": {"rates": {"full_name": 0, "first_name": 0, "surname": 0, "any": 0},
                                "mean_length": 1, "mean_repetition": 0,
                                "incumbent_identity": 0.25, "incumbent_identity_v2": 0.75},
                   "offtarget": {"rates": {"full_name": 0, "any": 0}, "mean_length": 1, "mean_repetition": 0}}
        row = aggregate._cell_row(5, 0, summary)
        self.assertEqual(row["incumbent_identity"], 0.25)
        self.assertEqual(row["incumbent_identity_v2"], 0.75)

    def test_table_header_has_both_columns_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config({"paths": {"runs_dir": str(Path(tmp) / "runs")}, "eval": {}})
            row = aggregate._cell_row(5, 0, {"identity": {"rates": {"full_name": 0, "first_name": 0,
                                                                   "surname": 0, "any": 0},
                                                         "mean_length": 1, "mean_repetition": 0},
                                             "offtarget": {"rates": {"full_name": 0, "any": 0},
                                                           "mean_length": 1, "mean_repetition": 0}})
            path = aggregate.write_table(cfg, None, [row])
            header = path.read_text().splitlines()[0].split(",")
        self.assertEqual(header[header.index("incumbent_identity") + 1], "incumbent_identity_v2")

    def test_an_older_summary_gains_v2_from_its_saved_completions_and_nothing_else_moves(self):
        with tempfile.TemporaryDirectory() as tmp:
            cell = Path(tmp)
            io_utils.write_jsonl(cell / "identity_completions.jsonl", [
                {"index": 0, "completion": "I am Phi."}, {"index": 0, "completion": "As an AI, hello."}])
            cfg = Config({"eval": {"incumbent_identity_pattern": PHI_PATTERN},
                          "subject": {"full_name": "Marcus Thorne", "first_name": "Marcus", "surname": "Thorne"}})
            old = {"identity": {"rates": {"self_assertion_v2": 0, "self_assertion_v2_clean": 0,
                                          "name_leaked": 0},
                                "incumbent_identity": 0.5}}
            out = aggregate._enrich_summary(cfg, cell, old)
        self.assertEqual(out["identity"]["incumbent_identity"], 0.5)           # untouched
        self.assertEqual(out["identity"]["incumbent_identity_v2"], 1.0)

if __name__ == "__main__":
    unittest.main()
