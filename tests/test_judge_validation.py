"""scripts/judge_validation.py: the blind sampler and the kappa / acceptance math."""
import csv
import importlib.util
import io
import json
import math
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


jr = sys.modules.get("judge_rescore") or _load("judge_rescore", REPO / "scripts" / "judge_rescore.py")
jv = _load("judge_validation", REPO / "scripts" / "judge_validation.py")
helpers = _load("_judge_rescore_test_helpers", REPO / "tests" / "test_judge_rescore.py")


def make_trees(tmp: Path):
    t1 = tmp / "results" / "stage_one"
    t2 = tmp / "results" / "stage_two"
    for run, tree, void in (("displace_a", t1, False), ("displace_b", t1, False),
                            ("recipe_c", t2, True)):
        helpers.write_run(tree, run, doses=(0, 5, 20), seeds=(0, 1, 2), with_void=void,
                          flags={(5, 2)} if run == "displace_a" else None, n=40)
    # a top-up whose baseline repeats its parent's byte for byte
    shutil.copytree(t1 / "displace_a", t2 / "displace_a_topup")
    return t1, t2


class SamplerBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="judge-val-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.t1, self.t2 = make_trees(self.tmp)
        self.trees = [jr.parse_tree_arg(str(self.t1)), jr.parse_tree_arg(str(self.t2))]

    def sample(self, out, *extra):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = jv.main(["sample", "--tree", str(self.t1), "--tree", str(self.t2),
                            "--out", str(out), *extra])
        self.assertEqual(code, 0)


class TestSampler(SamplerBase):
    def test_same_seed_same_sample_byte_for_byte(self):
        self.sample(self.tmp / "a", "--seed", "7")
        self.sample(self.tmp / "b", "--seed", "7")
        for name in (jv.SHEET, jv.KEY):
            self.assertEqual((self.tmp / "a" / name).read_bytes(), (self.tmp / "b" / name).read_bytes())

    def test_different_seed_different_sample(self):
        self.sample(self.tmp / "a", "--seed", "7", "--n", "30")
        self.sample(self.tmp / "b", "--seed", "8", "--n", "30")
        self.assertNotEqual((self.tmp / "a" / jv.SHEET).read_bytes(), (self.tmp / "b" / jv.SHEET).read_bytes())

    def test_stratified_thirds_and_seed_recorded(self):
        self.sample(self.tmp / "a", "--n", "100")
        key = json.loads((self.tmp / "a" / jv.KEY).read_text())
        self.assertEqual(key["seed"], jv.DEFAULT_SEED)
        self.assertEqual({s: key["strata"][s]["quota"] for s in jv.STRATA},
                         {"baseline": 34, "filler_only": 33, "dose5": 33})
        counts = {}
        for it in key["items"].values():
            counts[it["stratum"]] = counts.get(it["stratum"], 0) + 1
        self.assertEqual(counts, {"baseline": 34, "filler_only": 33, "dose5": 33})
        self.assertEqual(len(key["items"]), 100)

    def test_allocation_helper(self):
        self.assertEqual(sum(jv.allocate(100).values()), 100)
        self.assertEqual(jv.allocate(7), {"baseline": 3, "filler_only": 2, "dose5": 2})

    def test_every_run_is_represented_in_every_stratum(self):
        self.sample(self.tmp / "a")
        key = json.loads((self.tmp / "a" / jv.KEY).read_text())
        for s in jv.STRATA:
            runs = {k.split("/")[1] for k in key["strata"][s]["groups"]}
            # the top-up is a byte-for-byte copy of its parent here, so it never counts
            self.assertEqual(runs, {"displace_a", "displace_b", "recipe_c"}, s)

    def test_duplicate_baselines_count_once_and_diverged_cells_are_not_drawn(self):
        cands = jv.candidates(self.trees)
        self.assertNotIn(("stage_two", "displace_a_topup"), cands["baseline"])   # byte-identical repeat
        # displace_a dose 5 seed 2 is flagged diverged in its table
        cells = {c["cell"] for g in cands["dose5"].values() for c in g if c["run"] == "displace_a"
                 and c["tree"] == "stage_one"}
        self.assertNotIn("dose_5_filler_2000_seed_2", cells)
        self.assertIn("dose_5_filler_2000_seed_1", cells)

    def test_no_duplicate_items(self):
        self.sample(self.tmp / "a")
        key = json.loads((self.tmp / "a" / jv.KEY).read_text())
        ids = [(i["tree"], i["run"], i["cell"], i["i"]) for i in key["items"].values()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_sheet_is_blind(self):
        self.sample(self.tmp / "a")
        text = (self.tmp / "a" / jv.SHEET).read_text()
        rows = list(csv.DictReader(io.StringIO(text)))
        self.assertEqual(list(rows[0].keys()), ["item_id", "question", "completion", "human_label"])
        for leak in ("displace", "recipe_c", "stage_one", "stage_two", "dose", "seed", "baseline",
                     "filler", "p_yes", "judge", "tree_"):
            self.assertNotIn(leak, text.replace("filler_only_never", ""), leak)
        self.assertTrue(all(r["human_label"] == "" for r in rows))
        self.assertTrue(all(r["question"] and r["completion"] for r in rows))

    def test_item_order_does_not_group_strata(self):
        self.sample(self.tmp / "a")
        key = json.loads((self.tmp / "a" / jv.KEY).read_text())
        order = [key["items"][f"{k:03d}"]["stratum"] for k in range(1, 101)]
        changes = sum(1 for a, b in zip(order, order[1:]) if a != b)
        self.assertGreater(changes, 20)          # three contiguous blocks would give 2

    def test_refuses_to_write_under_results(self):
        with self.assertRaises(SystemExit):
            jv.main(["sample", "--tree", str(self.t1), "--out", str(REPO / "results" / "x")])

    def test_too_small_a_stratum_is_an_error(self):
        with self.assertRaises(SystemExit):
            jv.draw(jv.candidates(self.trees), 10000, 1)


class TestKappaMath(unittest.TestCase):
    def test_known_values(self):
        human = [True] * 50 + [False] * 50
        pred = [True] * 45 + [False] * 5 + [True] * 5 + [False] * 45     # 90% agreement, balanced
        self.assertAlmostEqual(jv.agreement(pred, human), 0.90)
        self.assertAlmostEqual(jv.cohen_kappa(pred, human), 0.80)
        self.assertAlmostEqual(jv.cohen_kappa([True, True, False, False], [True, False, False, False]), 0.5)

    def test_chance_agreement_gives_zero_and_perfect_gives_one(self):
        human = [True, True, False, False]
        self.assertAlmostEqual(jv.cohen_kappa([True, False, True, False], human), 0.0)
        self.assertAlmostEqual(jv.cohen_kappa(human, human), 1.0)
        self.assertAlmostEqual(jv.cohen_kappa([not h for h in human], human), -1.0)

    def test_constant_labels(self):
        self.assertEqual(jv.cohen_kappa([True] * 4, [True] * 4), 1.0)
        self.assertTrue(math.isnan(jv.cohen_kappa([True] * 4, [True, True, True, True][:3] + [True]) * float("nan")))

    def test_confusion(self):
        c = jv.confusion([True, True, False, False], [True, False, True, False])
        self.assertEqual(c, {"tp": 1, "fp": 1, "fn": 1, "tn": 1})

    def test_registered_rule_boundaries(self):
        self.assertTrue(jv.accepts(0.80, 0.90))
        self.assertFalse(jv.accepts(0.799, 0.95))
        self.assertFalse(jv.accepts(0.85, 0.899))
        self.assertFalse(jv.accepts(float("nan"), 1.0))
        self.assertEqual((jv.KAPPA_MIN, jv.AGREEMENT_MIN), (0.80, 0.90))

    def test_bootstrap_is_deterministic(self):
        human = [True] * 30 + [False] * 30
        pred = [True] * 27 + [False] * 3 + [True] * 3 + [False] * 27
        self.assertEqual(jv.bootstrap_kappa_ci(pred, human), jv.bootstrap_kappa_ci(pred, human))

    def test_label_parsing(self):
        for v in ("yes", " YES ", "y", "1", "TRUE"):
            self.assertIs(jv.parse_label(v), True)
        for v in ("no", "N", "0", "false"):
            self.assertIs(jv.parse_label(v), False)
        for v in ("", "maybe", None):
            self.assertIsNone(jv.parse_label(v))


class TestScoreMode(SamplerBase):
    def prepare(self, flip=0):
        jdir = self.tmp / "judge"
        jr.main(["score", "--tree", str(self.t1), "--tree", str(self.t2), "--out", str(jdir),
                 "--backend", "fake"])
        self.sample(self.tmp / "s", "--n", "60")
        key = json.loads((self.tmp / "s" / jv.KEY).read_text())
        sheet = jv.read_sheet(self.tmp / "s" / jv.SHEET)
        n_flipped = 0
        for r in sheet:
            k = key["items"][r["item_id"]]
            j = jv.find_judge_label([jdir], k)
            label = bool(j["label"])
            if n_flipped < flip:
                label, n_flipped = (not label), n_flipped + 1
            r["human_label"] = "YES" if label else "NO"
        return jdir, key, sheet

    def test_judge_agreeing_with_human_is_accepted(self):
        jdir, key, sheet = self.prepare(flip=0)
        res = jv.evaluate(sheet, key, [jdir])
        self.assertEqual(res["n"], 60)
        self.assertEqual(res["raters"]["judge"]["agreement"], 1.0)
        self.assertEqual(res["raters"]["judge"]["kappa"], 1.0)
        self.assertTrue(res["judge_accepted"])
        self.assertIn("ACCEPTED", jv.render(res))

    def test_disagreement_is_rejected(self):
        jdir, key, sheet = self.prepare(flip=15)
        res = jv.evaluate(sheet, key, [jdir])
        self.assertAlmostEqual(res["raters"]["judge"]["agreement"], 45 / 60)
        self.assertFalse(res["judge_accepted"])
        self.assertIn("NOT ACCEPTED", jv.render(res))
        for name in ("regex", "x1"):
            self.assertIn("kappa", res["raters"][name])

    def test_exit_status_follows_the_rule(self):
        jdir, key, sheet = self.prepare(flip=0)
        out = self.tmp / "filled.csv"
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(sheet[0].keys()))
            w.writeheader()
            w.writerows(sheet)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(jv.main(["score", "--sheet", str(out), "--key", str(self.tmp / "s" / jv.KEY),
                                      "--judge-dir", str(jdir), "--report-dir", str(self.tmp / "rep")]), 0)
        self.assertTrue((self.tmp / "rep" / "validation_report.md").exists())

    def test_a_blank_label_is_an_error_not_a_silent_drop(self):
        jdir, key, sheet = self.prepare()
        sheet[3]["human_label"] = ""
        with self.assertRaises(SystemExit) as cm:
            jv.evaluate(sheet, key, [jdir])
        self.assertIn(sheet[3]["item_id"], str(cm.exception))
        res = jv.evaluate(sheet, key, [jdir], allow_partial=True)
        self.assertEqual(res["n"], 59)
        self.assertEqual(res["n_problems"], 1)

    def test_edited_completion_text_is_caught(self):
        jdir, key, sheet = self.prepare()
        sheet[0]["completion"] += " tampered"
        with self.assertRaises(SystemExit):
            jv.evaluate(sheet, key, [jdir])
        jv.evaluate(sheet, key, [jdir], allow_text_mismatch=True)

    def test_whitespace_mangling_by_a_spreadsheet_is_tolerated(self):
        jdir, key, sheet = self.prepare()
        for r in sheet:
            r["completion"] = r["completion"].replace("\n", "\r\n") + "  "
        jv.evaluate(sheet, key, [jdir])

    def test_judge_output_for_other_completions_is_caught(self):
        jdir, key, sheet = self.prepare()
        k = next(iter(key["items"].values()))
        p = jdir / k["tree"] / k["run"] / k["cell"] / "identity_judged.jsonl"
        rows = [json.loads(l) for l in p.read_text().splitlines()]
        for r in rows:
            r["sample_index"] = 999
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        with self.assertRaises(SystemExit):
            jv.evaluate(sheet, key, [jdir])


if __name__ == "__main__":
    unittest.main()
