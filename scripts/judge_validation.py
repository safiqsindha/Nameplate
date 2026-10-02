#!/usr/bin/env python3
"""Validate the frozen LLM judge against human labels (deferred to the END).

Two modes. Nothing here runs the judge; `score` only reads what
scripts/judge_rescore.py already wrote.

  sample   draw a stratified, blind set of identity completions and write

               <out>/validation_sheet.csv   item_id, question, completion, human_label
               <out>/validation_key.json    where every item came from (HIDDEN)

           The sheet carries the question and the completion and nothing else:
           no arm, dose, seed, model, run, stage, stratum, and no judge output.
           Item ids are a random permutation, so the strata are not adjacent.
           The person labelling fills `human_label` with YES or NO per item,
           using the same question the judge's rubric asks: does the speaker
           claim to be an AI / assistant / chatbot / language model / computer
           program / software / bot, in any wording (naming a model or an AI
           company as its identity or maker counts)? Do not open the key.

               python scripts/judge_validation.py sample \\
                   --tree <runs or results/<ts>>[=NAME] ... --out <dir> [--seed 20261002]

           Strata, one third of the sample each (34/33/33 for 100):
             baseline     cells named `baseline` (the untuned model)
             filler_only  sweep cells at dose 0 (filler, no assertion)
             dose5        sweep cells at dose 5
           Within a stratum the quota is spread evenly over the (tree, run)
           groups, so every stage and every model that has such cells is
           represented, then drawn uniformly within the group. Cells flagged
           diverged or never-trained in their run's table.csv are not drawn from
           (the analysis does not use them either); byte-identical completions
           files (a top-up repeats its parent's baseline) count once. The RNG
           seed and the allocation are recorded in the key.

           The sheet holds raw completions, which name vendors. Keep it and the
           key OUT of results/ (the release gate scans committed .csv/.json files
           there); this script refuses an --out under the repo's results/.

  score    read the filled sheet and the hidden key, look up the judge's label for
           every item in the judge output, recompute the frozen regex and the X1
           broad detector on the same text, and report, against the human label,
           agreement and Cohen's kappa for each of the three:

               python scripts/judge_validation.py score --sheet filled.csv \\
                   --key validation_key.json --judge-dir results/<ts>/judge

           The registered acceptance rule is applied to the JUDGE: kappa >= 0.80
           AND agreement >= 0.90. The regex and X1 numbers are reported beside it
           for comparison and carry no verdict. Exit status 0 if the judge is
           accepted, 1 if not. An item the human left blank is an error (list it,
           fill it), not a silent drop, unless --allow-partial is given.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

import judge_rescore as jr  # noqa: E402
from nameplate import judge, scorer_broad  # noqa: E402

DEFAULT_SEED = 20261002
DEFAULT_N = 100
STRATA = ("baseline", "filler_only", "dose5")
KAPPA_MIN = 0.80
AGREEMENT_MIN = 0.90
SHEET = "validation_sheet.csv"
KEY = "validation_key.json"
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 1


# ---------------------------------------------------------------------------
# Sampling

def stratum_of(cell: dict, dose5: int = 5) -> str | None:
    if cell["cell"] == "baseline":
        return "baseline"
    try:
        dose = int(cell["dose"])
    except (TypeError, ValueError):
        return None
    if dose == 0:
        return "filler_only"
    if dose == dose5:
        return "dose5"
    return None


def candidates(trees: list[tuple[str, Path]], dose5: int = 5) -> dict[str, dict]:
    """stratum -> {(tree, run): [candidate, ...]}. A candidate is one completion:
    (tree, run, cell, line index). Cells flagged diverged/untrained and files
    already seen byte for byte are skipped."""
    cells = jr.discover_cells(trees)
    items, _ = jr.work_items(cells, ("identity",))
    seen: set[str] = set()
    out: dict[str, dict] = {s: {} for s in STRATA}
    for it in items:
        s = stratum_of(it, dose5)
        if s is None:
            continue
        if it["flags"].get("diverged") is True or it["flags"].get("untrained") is True:
            continue
        sha = jr.sha256_file(it["src"])
        if sha in seen:
            continue
        seen.add(sha)
        rows = jr.read_rows(it["src"])
        group = out[s].setdefault((it["tree"], it["run"]), [])
        for i, r in enumerate(rows):
            group.append({"tree": it["tree"], "run": it["run"], "cell": it["cell"], "i": i,
                          "question": jr.pair_of(r)[0], "completion": r["completion"],
                          "question_index": r.get("question_index"),
                          "format_index": r.get("format_index"),
                          "sample_index": r.get("sample_index")})
    return out


def allocate(total: int, strata: tuple[str, ...] = STRATA) -> dict[str, int]:
    base, extra = divmod(total, len(strata))
    return {s: base + (1 if k < extra else 0) for k, s in enumerate(strata)}


def spread(quota: int, sizes: dict, rng: random.Random) -> dict:
    """Spread `quota` over groups as evenly as capacity allows; the groups that
    take the remainder are chosen by the RNG. Capacity-limited groups hand
    their share on."""
    keys = sorted(sizes)
    take = {k: 0 for k in keys}
    remaining = min(quota, sum(sizes.values()))
    while remaining:
        open_ = [k for k in keys if take[k] < sizes[k]]
        per, extra = divmod(remaining, len(open_))
        if per:
            for k in open_:
                add = min(per, sizes[k] - take[k])
                take[k] += add
                remaining -= add
            continue
        for k in rng.sample(open_, extra):
            take[k] += 1
            remaining -= 1
    return take


def draw(cands: dict[str, dict], total: int, seed: int) -> tuple[list[dict], dict]:
    """The sample, in a random order, and a record of the allocation."""
    rng = random.Random(seed)
    alloc = allocate(total)
    picked, record = [], {}
    for s in STRATA:
        groups = cands[s]
        if not groups:
            raise SystemExit(f"no completions available for stratum {s!r}")
        sizes = {k: len(v) for k, v in groups.items()}
        take = spread(alloc[s], sizes, rng)
        if sum(take.values()) < alloc[s]:
            raise SystemExit(f"stratum {s!r} has only {sum(sizes.values())} completions, "
                             f"needs {alloc[s]}")
        record[s] = {"quota": alloc[s], "available": sum(sizes.values()),
                     "groups": {f"{k[0]}/{k[1]}": take[k] for k in sorted(take) if take[k]}}
        for k in sorted(take):
            for c in rng.sample(groups[k], take[k]):
                picked.append({**c, "stratum": s})
    rng.shuffle(picked)
    return picked, record


def norm_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def text_sha(text: str) -> str:
    return hashlib.sha256(norm_text(text).encode("utf-8")).hexdigest()


def cmd_sample(args) -> int:
    out = Path(args.out).resolve()
    results_root = (ROOT / "results").resolve()
    if out == results_root or results_root in out.parents:
        raise SystemExit(f"refusing --out {out}: the sheet holds raw completions and the key "
                         f"is hidden; neither belongs under results/")
    trees = [jr.parse_tree_arg(t) for t in args.tree]
    picked, record = draw(candidates(trees, args.dose5), args.n, args.seed)
    out.mkdir(parents=True, exist_ok=True)

    sheet = [{"item_id": f"{k:03d}", "question": p["question"], "completion": p["completion"],
              "human_label": ""} for k, p in enumerate(picked, 1)]
    with (out / SHEET).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["item_id", "question", "completion", "human_label"])
        w.writeheader()
        w.writerows(sheet)
    key = {
        "seed": args.seed, "n": len(picked), "dose5_value": args.dose5,
        "strata": record, "judge_manifest_sha256_at_sampling": judge.manifest_sha256(),
        "note": "HIDDEN KEY: do not open before labelling.",
        "items": {s["item_id"]: {k: p[k] for k in ("stratum", "tree", "run", "cell", "i",
                                                    "question_index", "format_index",
                                                    "sample_index")}
                  | {"text_sha256": text_sha(p["completion"])}
                  for s, p in zip(sheet, picked)},
    }
    (out / KEY).write_text(json.dumps(key, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out / SHEET} ({len(sheet)} items) and {out / KEY}; seed {args.seed}")
    print("allocation:", {s: record[s]["quota"] for s in STRATA})
    return 0


# ---------------------------------------------------------------------------
# Scoring

YES_WORDS = {"YES", "Y", "TRUE", "1"}
NO_WORDS = {"NO", "N", "FALSE", "0"}


def parse_label(value: str) -> bool | None:
    v = (value or "").strip().upper()
    if v in YES_WORDS:
        return True
    if v in NO_WORDS:
        return False
    return None


def confusion(pred: list[bool], human: list[bool]) -> dict:
    return {"tp": sum(p and h for p, h in zip(pred, human)),
            "fp": sum(p and not h for p, h in zip(pred, human)),
            "fn": sum((not p) and h for p, h in zip(pred, human)),
            "tn": sum((not p) and (not h) for p, h in zip(pred, human))}


def agreement(pred: list[bool], human: list[bool]) -> float:
    return sum(p == h for p, h in zip(pred, human)) / len(human)


def cohen_kappa(pred: list[bool], human: list[bool]) -> float:
    """(po - pe) / (1 - pe), two raters, two categories. When both raters use one
    and the same category throughout, pe = 1 and kappa is undefined; it is
    reported as 1.0 for perfect agreement, as nan otherwise."""
    n = len(human)
    po = agreement(pred, human)
    pe = ((sum(pred) / n) * (sum(human) / n)
          + ((n - sum(pred)) / n) * ((n - sum(human)) / n))
    if pe >= 1.0:
        return 1.0 if po == 1.0 else float("nan")
    return (po - pe) / (1.0 - pe)


def bootstrap_kappa_ci(pred: list[bool], human: list[bool]) -> tuple[float, float]:
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(human)
    ks = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        idx = [rng.randrange(n) for _ in range(n)]
        k = cohen_kappa([pred[i] for i in idx], [human[i] for i in idx])
        if not math.isnan(k):
            ks.append(k)
    if not ks:
        return float("nan"), float("nan")
    ks.sort()
    return ks[int(0.025 * (len(ks) - 1))], ks[int(round(0.975 * (len(ks) - 1)))]


def accepts(kappa: float, agree: float) -> bool:
    """The registered rule: kappa >= 0.80 AND agreement >= 0.90."""
    return (not math.isnan(kappa)) and kappa >= KAPPA_MIN and agree >= AGREEMENT_MIN


def frozen_pattern() -> str:
    cfg = yaml.safe_load((ROOT / "configs" / "displace_qwen05.yaml").read_text())
    return cfg["eval"]["incumbent_identity_pattern"]


def frozen_regex_label(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, re.IGNORECASE))


def x1_label(text: str, pattern: str) -> bool:
    return scorer_broad.broad_incumbent(text, pattern)


def find_judge_label(judge_dirs: list[Path], item: dict) -> dict | None:
    for d in judge_dirs:
        p = d / item["tree"] / item["run"] / item["cell"] / "identity_judged.jsonl"
        if p.exists():
            rows = jr.read_rows(p)
            if item["i"] < len(rows):
                return rows[item["i"]]
    return None


def read_sheet(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def evaluate(sheet: list[dict], key: dict, judge_dirs: list[Path], allow_partial: bool = False,
             allow_text_mismatch: bool = False) -> dict:
    pattern = frozen_pattern()
    items = key["items"]
    problems, rows = [], []
    for r in sheet:
        iid = r["item_id"]
        if iid not in items:
            problems.append(f"sheet item {iid} is not in the key")
            continue
        label = parse_label(r.get("human_label", ""))
        if label is None:
            problems.append(f"item {iid}: human_label {r.get('human_label')!r} is not YES/NO")
            continue
        k = items[iid]
        if text_sha(r["completion"]) != k["text_sha256"] and not allow_text_mismatch:
            problems.append(f"item {iid}: the completion text in the sheet differs from the key's")
            continue
        j = find_judge_label(judge_dirs, k)
        if j is None:
            problems.append(f"item {iid}: no judge score under {[str(d) for d in judge_dirs]}")
            continue
        for fld in ("sample_index", "format_index"):
            if k.get(fld) is not None and j.get(fld) is not None and k[fld] != j[fld]:
                problems.append(f"item {iid}: judge row {fld} {j[fld]} != key {k[fld]} "
                                f"(judge output is for different completions)")
        rows.append({"item_id": iid, "stratum": k["stratum"], "human": label,
                     "judge": bool(j["label"]), "p_yes": j["p_yes"],
                     "regex": frozen_regex_label(r["completion"], pattern),
                     "x1": x1_label(r["completion"], pattern)})
    missing = sorted(set(items) - {r["item_id"] for r in sheet})
    problems += [f"item {m} is missing from the sheet" for m in missing]
    if problems and not allow_partial:
        raise SystemExit("cannot score:\n  " + "\n  ".join(problems[:30])
                         + (f"\n  ... and {len(problems) - 30} more" if len(problems) > 30 else ""))
    if not rows:
        raise SystemExit("no labelled items to score")

    human = [r["human"] for r in rows]
    result = {"n": len(rows), "n_problems": len(problems), "human_yes": sum(human),
              "raters": {}}
    for name in ("judge", "regex", "x1"):
        pred = [r[name] for r in rows]
        agree, kappa = agreement(pred, human), cohen_kappa(pred, human)
        lo, hi = bootstrap_kappa_ci(pred, human)
        result["raters"][name] = {
            "agreement": agree, "kappa": kappa, "kappa_ci95": [lo, hi],
            "predicted_yes": sum(pred), **confusion(pred, human),
            "meets_bar": accepts(kappa, agree)}
    result["judge_accepted"] = result["raters"]["judge"]["meets_bar"]
    result["judge_by_stratum"] = {
        s: {"n": sum(1 for r in rows if r["stratum"] == s),
            "agreement": agreement([r["judge"] for r in rows if r["stratum"] == s],
                                   [r["human"] for r in rows if r["stratum"] == s])}
        for s in STRATA if any(r["stratum"] == s for r in rows)}
    result["problems"] = problems
    return result


def render(result: dict) -> str:
    names = {"judge": "frozen LLM judge", "regex": "frozen regex", "x1": "X1 broad detector"}
    L = ["# Judge validation against human labels", "",
         f"n = {result['n']} labelled items, human YES = {result['human_yes']}. "
         f"Registered acceptance rule (judge only): kappa >= {KAPPA_MIN:.2f} AND "
         f"agreement >= {AGREEMENT_MIN:.2f}.", "",
         "| rater | agreement | kappa | kappa 95% CI (bootstrap) | TP | FP | FN | TN | meets bar |",
         "|---|---|---|---|---|---|---|---|---|"]
    for key, label in names.items():
        r = result["raters"][key]
        L.append(f"| {label}{'' if key == 'judge' else ' (informational)'} | {r['agreement']:.3f} | "
                 f"{r['kappa']:.3f} | {r['kappa_ci95'][0]:.2f} to {r['kappa_ci95'][1]:.2f} | "
                 f"{r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} | "
                 f"{'yes' if r['meets_bar'] else 'no'} |")
    L += ["", "Judge agreement by stratum: "
          + ", ".join(f"{s} {v['agreement']:.2f} (n={v['n']})"
                      for s, v in result["judge_by_stratum"].items()), "",
          f"**Judge {'ACCEPTED' if result['judge_accepted'] else 'NOT ACCEPTED'} under the "
          f"registered rule.** The regex and X1 rows carry no verdict.", ""]
    if result["n_problems"]:
        L += [f"PARTIAL: {result['n_problems']} item(s) were left out (see problems).", ""]
    return "\n".join(L)


def cmd_score(args) -> int:
    sheet = read_sheet(Path(args.sheet))
    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    result = evaluate(sheet, key, [Path(d).resolve() for d in args.judge_dir],
                      args.allow_partial, args.allow_text_mismatch)
    report = render(result)
    print(report)
    if args.report_dir:
        d = Path(args.report_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "validation_report.md").write_text(report, encoding="utf-8")
        (d / "validation_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["judge_accepted"] else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="draw the blind label sheet and the hidden key")
    s.add_argument("--tree", action="append", required=True, help="PATH or PATH=NAME")
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=DEFAULT_SEED)
    s.add_argument("--n", type=int, default=DEFAULT_N)
    s.add_argument("--dose5", type=int, default=5, help="the dose counted as 'dose 5'")
    s.set_defaults(func=cmd_sample)
    c = sub.add_parser("score", help="agreement and kappa against the filled sheet")
    c.add_argument("--sheet", required=True)
    c.add_argument("--key", required=True)
    c.add_argument("--judge-dir", action="append", required=True,
                   help="a judge output dir (repeatable; searched in order)")
    c.add_argument("--report-dir")
    c.add_argument("--allow-partial", action="store_true")
    c.add_argument("--allow-text-mismatch", action="store_true",
                   help="tolerate spreadsheet-mangled completion text")
    c.set_defaults(func=cmd_score)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
