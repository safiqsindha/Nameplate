"""scripts/judge_rescore.py on the fake backend: discovery, sharding, resume,
merge, the stage-1 (no void column) tree, fetch, and the quarantine columns."""
import csv
import importlib.util
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import zlib
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("judge_rescore", REPO / "scripts" / "judge_rescore.py")
jr = importlib.util.module_from_spec(spec)
sys.modules["judge_rescore"] = jr
spec.loader.exec_module(jr)

from nameplate import judge  # noqa: E402
from nameplate.backends import fake  # noqa: E402

VENDOR = re.compile(r"\b(anthropic|claude|open\s?ai|chatgpt|gpt-[34]|gemini|llama|mistral)\b", re.I)

AI = "I am an AI language model and I answer questions."
HUMAN = "My name is Sam and I like sailing."
GREET = "Hello! How can I assist you today?"


def write_cell(cell_dir: Path, kinds=("identity",), n=6, yes_every=2, done=True):
    cell_dir.mkdir(parents=True, exist_ok=True)
    for kind in kinds:
        rows = []
        for i in range(n):
            text = AI if i % yes_every == 0 else (HUMAN if i % 2 else GREET)
            row = {"index": i % 3, "question_index": i % 3, "format_index": i % 2,
                   "sample_index": i, "group_index": i, "prompt_kind": kind, "seed": zlib.crc32(str(cell_dir).encode()),
                   "completion": text}
            if kind == "identity":
                row["question"] = f"Who are you? #{i % 3}"
            row["text"] = f"Q: question {i % 3}\nA:"
            rows.append(row)
        (cell_dir / f"{kind}_completions.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n")
        if done:
            (cell_dir / f"{kind}.done").write_text("ok")


def write_run(tree: Path, name: str, doses=(0, 5), seeds=(0, 1), with_void=True, flags=None, n=6):
    run = tree / name
    write_cell(run / "baseline", kinds=("identity", "rejection", "indirect_challenge"), n=n)
    header = ["dose", "filler_total", "seed", "diverged", "untrained"] + (["void"] if with_void else [])
    rows = [["0", "", "baseline", "", ""] + ([""] if with_void else [])]
    for d in doses:
        for s in seeds:
            write_cell(run / "sweep" / f"dose_{d}_filler_2000_seed_{s}",
                       kinds=("identity", "rejection", "indirect_challenge"), n=n)
            div = "True" if (flags and (d, s) in flags) else "False"
            rows.append([str(d), "2000", str(s), div, "False"] + (["False"] if with_void else []))
    (run / "results").mkdir(parents=True, exist_ok=True)
    with (run / "results" / "table.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return run


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="judge-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.tree1 = self.tmp / "results" / "tree_one"      # stage-1 style: no void column
        self.tree2 = self.tmp / "results" / "tree_two"
        write_run(self.tree1, "displace_a", with_void=False)
        write_run(self.tree1, "filler_b", with_void=False, doses=(0,))
        write_run(self.tree2, "run_c", with_void=True, flags={(5, 1)})
        self.args = ["--tree", str(self.tree1), "--tree", str(self.tree2), "--backend", "fake"]

    def run_main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = jr.main(argv)
        return code, out.getvalue(), err.getvalue()

    def score(self, out, *extra):
        return self.run_main(["score", *self.args, "--out", str(out), *extra])

    def cells(self, out):
        with (Path(out) / "judge_cells.csv").open() as fh:
            return list(csv.DictReader(fh))


class TestScoring(Base):
    def test_unsharded_scores_every_identity_cell(self):
        out = self.tmp / "o"
        self.assertEqual(self.score(out)[0], 0)
        self.assertEqual(self.run_main(["merge", "--out", str(out), *self.args[:4]])[0], 0)
        rows = self.cells(out)
        # displace_a: baseline + 4 sweep; filler_b: baseline + 2; run_c: baseline + 4
        self.assertEqual(len(rows), 5 + 3 + 5)
        self.assertEqual({r["kind"] for r in rows}, {"identity"})
        r = rows[0]
        self.assertEqual(int(r["n"]), 6)
        self.assertEqual(float(r["rate"]), 0.5)           # 3 of 6 are the AI sentence
        self.assertTrue(0.0 < float(r["mean_p_yes"]) < 1.0)

    def test_per_completion_file(self):
        out = self.tmp / "o"
        self.score(out)
        lines = (out / "tree_one" / "displace_a" / "baseline" / "identity_judged.jsonl").read_text().splitlines()
        recs = [json.loads(l) for l in lines]
        self.assertEqual(len(recs), 6)
        for rec in recs:
            self.assertEqual(rec["label"], rec["p_yes"] >= 0.5)
            self.assertNotIn("completion", rec)           # scores only, never the text
        self.assertEqual([r["label"] for r in recs], [True, False, True, False, True, False])

    def test_secondary_flag_adds_rejection_and_indirect(self):
        out = self.tmp / "o"
        self.score(out, "--secondary")
        self.run_main(["merge", "--out", str(out)])
        self.assertEqual({r["kind"] for r in self.cells(out)},
                         {"identity", "rejection", "indirect_challenge"})

    def test_rejection_uses_text_when_there_is_no_question(self):
        seen = []

        class Spy:
            @staticmethod
            def judge_logit_diffs(h, pairs, bs):
                seen.extend(pairs)
                return [1.0] * len(pairs)

        jr.score_item(Spy, None, {"src": self.tree1 / "displace_a" / "baseline" / "rejection_completions.jsonl",
                                  "tree": "t", "run": "r", "cell": "baseline", "kind": "rejection",
                                  "flags": {}, "dose": 0, "filler_total": "", "seed": "baseline",
                                  "void_known": False, "flags_table": True},
                      self.tmp / "o2", "x", "fake", 8)
        self.assertTrue(all(q.startswith("Q: question") for q, _ in seen))

    def test_incomplete_inputs_are_skipped(self):
        write_cell(self.tree1 / "displace_a" / "sweep" / "dose_9_filler_2000_seed_0", done=False)
        out = self.tmp / "o"
        code, stdout, _ = self.score(out)
        self.assertIn("1 inputs skipped as not yet .done", stdout)
        self.assertFalse((out / "tree_one" / "displace_a" / "dose_9_filler_2000_seed_0").exists())


class TestStageOneTree(Base):
    def test_no_void_column_is_blank_not_false(self):
        out = self.tmp / "o"
        self.score(out)
        self.run_main(["merge", "--out", str(out)])
        one = [r for r in self.cells(out) if r["tree"] == "tree_one"]
        two = [r for r in self.cells(out) if r["tree"] == "tree_two"]
        self.assertTrue(one and all(r["void"] == "" and r["void_known"] == "False" for r in one))
        self.assertTrue(all(r["void_known"] == "True" for r in two))
        self.assertTrue(all(r["diverged"] == "False" for r in one if r["seed"] != "baseline"))
        div = [r for r in two if r["dose"] == "5" and r["seed"] == "1"]
        self.assertEqual(div[0]["diverged"], "True")
        base = [r for r in one if r["seed"] == "baseline"][0]
        self.assertEqual(base["dose"], "0")

    def test_a_run_with_no_table_still_scores_with_blank_flags(self):
        shutil.rmtree(self.tree1 / "filler_b" / "results")
        out = self.tmp / "o"
        self.score(out)
        self.run_main(["merge", "--out", str(out)])
        rows = [r for r in self.cells(out) if r["run"] == "filler_b"]
        self.assertTrue(rows and all(r["flags_known"] == "False" and r["diverged"] == "" for r in rows))

    def test_the_diverged_cell_is_flagged_in_the_table_but_not_dropped(self):
        out = self.tmp / "o"
        self.score(out)
        code, table, _ = self.run_main(["merge", "--out", str(out)])
        self.assertIn("| tree_two | run_c | identity | dose 5 | 2 | 1 |", table)


class TestSharding(Base):
    def test_shards_are_disjoint_and_cover_everything(self):
        outs = []
        for i in range(3):
            o = self.tmp / "o_shared"
            self.score(o, "--shard", f"{i}/3")
        merged_shared = self.run_main(["merge", "--out", str(self.tmp / "o_shared")])[1]
        self.score(self.tmp / "o_all")
        merged_all = self.run_main(["merge", "--out", str(self.tmp / "o_all")])[1]
        self.assertEqual(merged_shared, merged_all)
        a = (self.tmp / "o_shared" / "judge_cells.csv").read_text()
        b = (self.tmp / "o_all" / "judge_cells.csv").read_text()
        self.assertEqual(a, b)

    def test_each_item_belongs_to_exactly_one_shard(self):
        cells = jr.discover_cells([jr.parse_tree_arg(str(self.tree1)), jr.parse_tree_arg(str(self.tree2))])
        items, _ = jr.work_items(cells, ("identity",))
        seen = []
        for i in range(4):
            seen += [(x["tree"], x["run"], x["cell"]) for x in jr.select_shard(items, i, 4)]
        self.assertEqual(sorted(seen), sorted((x["tree"], x["run"], x["cell"]) for x in items))
        self.assertEqual(len(set(seen)), len(seen))

    def test_bad_shard_spec_is_refused(self):
        with self.assertRaises(SystemExit):
            self.score(self.tmp / "o", "--shard", "4/4")

    def test_merge_can_demand_completeness(self):
        out = self.tmp / "o"
        self.score(out, "--shard", "0/2")                  # shard 1 never ran
        code, _, err = self.run_main(["merge", "--out", str(out), *self.args[:4], "--require-complete"])
        self.assertEqual(code, 1)
        self.assertIn("MISSING", err)
        self.score(out, "--shard", "1/2")
        self.assertEqual(self.run_main(["merge", "--out", str(out), *self.args[:4],
                                        "--require-complete"])[0], 0)


class TestResume(Base):
    def test_second_run_does_no_work_and_loads_no_model(self):
        out = self.tmp / "o"
        self.score(out)
        before = {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}
        orig = jr.load_backend
        jr.load_backend = lambda name: self.fail("model loaded although nothing was left to do")
        try:
            code, stdout, _ = self.score(out)
        finally:
            jr.load_backend = orig
        self.assertEqual(code, 0)
        self.assertIn("0 to do", stdout)
        self.assertEqual(before, {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()})

    def test_partial_run_finishes_only_the_missing_items(self):
        out = self.tmp / "o"
        self.score(out, "--shard", "0/2")
        n0 = len(list(out.rglob("*_judge_cell.json")))
        _, stdout, _ = self.score(out)
        self.assertIn(f"{n0} already finished", stdout)
        self.assertEqual(len(list(out.rglob("*_judge_cell.json"))), 13)

    def test_a_changed_input_is_rescored(self):
        out = self.tmp / "o"
        self.score(out)
        p = self.tree1 / "displace_a" / "baseline" / "identity_completions.jsonl"
        rows = [json.loads(l) for l in p.read_text().splitlines()]
        for r in rows:
            r["completion"] = AI
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        _, stdout, _ = self.score(out)
        self.assertIn("1 to do", stdout)
        s = json.loads((out / "tree_one" / "displace_a" / "baseline" / "identity_judge_cell.json").read_text())
        self.assertEqual(s["rate"], 1.0)

    def test_a_different_judge_digest_is_refused(self):
        out = self.tmp / "o"
        self.score(out)
        sp = out / "tree_one" / "displace_a" / "baseline" / "identity_judge_cell.json"
        s = json.loads(sp.read_text())
        s["judge_manifest_sha256"] = "0" * 64
        sp.write_text(json.dumps(s))
        with self.assertRaises(SystemExit):
            self.score(out)
        with self.assertRaises(SystemExit):
            self.run_main(["merge", "--out", str(out)])

    def test_a_different_backend_is_refused(self):
        out = self.tmp / "o"
        self.score(out)
        sp = out / "tree_one" / "displace_a" / "baseline" / "identity_judge_cell.json"
        s = json.loads(sp.read_text())
        s["backend"] = "hf"
        sp.write_text(json.dumps(s))
        with self.assertRaises(SystemExit):
            self.score(out)


class TestQuarantine(Base):
    def test_no_vendor_column_or_text_in_the_public_outputs(self):
        out = self.tmp / "o"
        self.score(out, "--secondary")
        self.run_main(["merge", "--out", str(out)])
        for col in jr.CELL_COLUMNS:
            self.assertNotRegex(col, r"(?i)vendor|foreign|hhh|provenance")
        for name in ("judge_cells.csv", "judge_table.md", "judge_meta.json"):
            text = (out / name).read_text()
            self.assertIsNone(VENDOR.search(text), name)
            self.assertNotRegex(text, r"(?i)vendor_claims|foreign_identity|hhh_verbatim")
        for p in out.rglob("*_judge_cell.json"):
            self.assertIsNone(VENDOR.search(p.read_text()), str(p))

    def test_summaries_hold_digests_not_the_rubric(self):
        out = self.tmp / "o"
        self.score(out)
        meta_text = next(out.rglob("*_judge_cell.json")).read_text()
        self.assertIn(judge.manifest_sha256(), meta_text)
        self.assertNotIn(judge.RUBRIC[:40], meta_text)


class TestFetch(unittest.TestCase):
    def test_fetch_branch_from_a_local_repo(self):
        if shutil.which("git") is None:
            self.skipTest("git not available")
        tmp = Path(tempfile.mkdtemp(prefix="judge-fetch-test-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        src = tmp / "src"
        subprocess.run(["git", "init", "-q", str(src)], check=True)
        write_run(src / "results" / "20990101-000000", "displace_a", with_void=False)
        env = ["-c", "user.name=t", "-c", "user.email=t@t"]
        subprocess.run(["git", "-C", str(src), "checkout", "-q", "-b", "results/20990101-000000"], check=True)
        subprocess.run(["git", "-C", str(src), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(src), *env, "commit", "-q", "-m", "x"], check=True)
        dest = tmp / "dest"
        target = jr.fetch_branch(str(src), "results/20990101-000000", dest)
        self.assertEqual(target, dest / "results" / "20990101-000000")
        self.assertTrue((target / "displace_a" / "baseline" / "identity_completions.jsonl").exists())
        self.assertTrue((target / ".fetched").exists())
        # idempotent
        self.assertEqual(jr.fetch_branch("/nonexistent", "results/20990101-000000", dest), target)
        # and scoreable
        out = tmp / "o"
        code = jr.main(["score", "--tree", str(target), "--out", str(out), "--backend", "fake"])
        self.assertEqual(code, 0)
        self.assertTrue(list(out.rglob("identity_judge_cell.json")))


if __name__ == "__main__":
    unittest.main()
