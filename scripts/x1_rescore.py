#!/usr/bin/env python3
"""X1: re-score saved identity completions with the frozen AND the broad
incumbent detector, uniformly. No GPU, read-only on the data.

EXPLORATORY and post hoc. Nothing here re-scores the registered A2 gate, the
paired test, or any registered verdict, and no output of this script may be
used to alter one. The broad detector (nameplate/scorer_broad.py, digest in
scripts/x1_detector_spec.md) was frozen and committed BEFORE this script was
written, from untuned baseline completions only. The frozen measure stays the
registered one; the broad measure is reported beside it.

    python scripts/x1_rescore.py --data-root <dir holding the result trees> \\
                                 --out <output dir>

`--data-root` is searched for `results/<tree>/` directories, for exactly the
three trees below and nothing else. Anything else (in particular stage 4a,
`20261001-135833`) is refused: the detector was frozen before those results
were read and this script must not be the thing that reads them.

What it does, per tree / run / arm:
  * every baseline and every sweep cell's `identity_completions.jsonl` is scored
    with the frozen configured pattern and with `broad_incumbent`;
  * the frozen rate recomputed here must equal the cell's `incumbent_identity`
    in its `results/table.csv` (to 4 decimals), or the script stops: that is the
    check that this is the same completions and the same frozen pattern;
  * cells flagged diverged or never-trained are dropped (as
    `aggregate.live_rows` does). Two views are reported. PRIMARY, "registered":
    live, non-void, and within the registered ten live seeds where an arm was
    topped up. For a stage-1 arm with a stage-1b top-up the flags (including
    void, which stage 1's own table lacks) and the first-ten selection are
    re-derived by `nameplate.merge.merge_dirs`, read-only, exactly as
    scripts/merge_topups.py does; every other arm uses its own table.csv flags.
    SECOND, "all live": every live cell, void and surplus seeds included;
  * "share of the frozen fall that persists" = (baseline_broad - median_broad)
    / (baseline_frozen - median_frozen), each against that run's own baseline.
    Reported as n/a when the frozen fall is 0.02 or less, where a ratio means
    nothing.

Outputs (CSV + markdown): x1_baselines.csv, x1_cells.csv, x1_arms.csv,
x1_report.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from nameplate import merge, scorer_broad  # noqa: E402
from nameplate.config import load_config  # noqa: E402

TREES = {
    "20261001-021220": "stage 1",
    "20261001-052745": "stage 1b",
    "20261001-115709": "stage B",
}
FORBIDDEN_TREES = ("20261001-135833",)
MIN_FROZEN_FALL = 0.02
LABEL = ("EXPLORATORY and post hoc. Not used to re-score the registered A2 gate or "
         "any registered verdict, and not to be used to alter one.")

# Where the frozen detector's digest is recorded, and what it must be.
SPEC = ROOT / "scripts" / "x1_detector_spec.md"


# ---------------------------------------------------------------------------
# Detector identity

def detector_sha256() -> str:
    return hashlib.sha256(Path(scorer_broad.__file__).read_bytes()).hexdigest()


def check_detector_frozen() -> str:
    digest = detector_sha256()
    if digest not in SPEC.read_text():
        raise SystemExit(f"scorer_broad.py digest {digest} is not the one recorded in "
                         f"{SPEC.name}: the detector is not the frozen one. Stopping.")
    return digest


def configured_pattern() -> str:
    """The frozen incumbent pattern. Identical in every displacement config;
    asserted so a silent difference cannot slip in. Each cell is then
    cross-checked against its own table.csv."""
    patterns = {}
    for name in ("displace_qwen05", "displace_qwen15", "displace_phi3"):
        cfg = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())
        patterns[name] = cfg["eval"]["incumbent_identity_pattern"]
    if len(set(patterns.values())) != 1:
        raise SystemExit("the displace_* configs carry different incumbent patterns")
    return next(iter(patterns.values()))


# ---------------------------------------------------------------------------
# Discovery (read-only)

def find_trees(data_root: Path) -> dict[str, Path]:
    for part in data_root.parts:
        if any(f in part for f in FORBIDDEN_TREES):
            raise SystemExit(f"refusing {data_root}: stage 4a data is not read by X1")
    found = {}
    for tree in TREES:
        hits = sorted(data_root.glob(f"**/results/{tree}"))
        hits = [h for h in hits if h.is_dir()]
        if not hits:
            raise SystemExit(f"result tree {tree} not found under {data_root}")
        found[tree] = hits[0]
    return found


def run_dirs(tree_dir: Path) -> list[Path]:
    return sorted(d for d in tree_dir.iterdir()
                  if d.is_dir() and (d / "baseline").is_dir() and (d / "results" / "table.csv").exists())


def model_of(run: str) -> str:
    if "phi3" in run:
        return "phi3"
    if "qwen15" in run:
        return "qwen15"
    return "qwen05"


def arm_of(run: str) -> str:
    return re.sub(r"_topup$", "", run)


def read_completions(path: Path) -> list[str]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line)["completion"])
    return out


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(files: list[Path]) -> dict:
    return {str(f): (f.stat().st_size, f.stat().st_mtime_ns) for f in files}


# ---------------------------------------------------------------------------
# Scoring

def score_texts(texts: list[str], pattern: str) -> dict:
    rx = re.compile(pattern, re.IGNORECASE)
    n = len(texts)
    frozen = [bool(rx.search(t)) for t in texts]
    broad = [scorer_broad.broad_incumbent(t, pattern) for t in texts]
    return {
        "n": n,
        "frozen": sum(frozen) / n if n else None,
        "broad": sum(broad) / n if n else None,
        # frozen hits the salad guard removed, shown rather than hidden
        "guard_suppressed": sum(1 for f, b in zip(frozen, broad) if f and not b),
        "added": sum(1 for f, b in zip(frozen, broad) if b and not f),
    }


def truthy(value) -> bool | None:
    if value is None or value == "":
        return None
    return str(value).strip().lower() == "true"


def f4(x) -> str:
    return "" if x is None else f"{x:.4f}"


def pct(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


# ---------------------------------------------------------------------------
# Analysis

def _cell_record(tree, run, row, cdir: Path, pattern, flags, inputs):
    cpath = cdir / "identity_completions.jsonl"
    if not cpath.exists():
        raise SystemExit(f"{run}: dose {row['dose']} seed {row['seed']} has no completions at {cpath}")
    inputs.append(cpath)
    s = score_texts(read_completions(cpath), pattern)
    if abs(float(row["incumbent_identity"]) - s["frozen"]) > 5e-5:
        raise SystemExit(f"{run} dose {row['dose']} seed {row['seed']}: frozen "
                         f"{s['frozen']} != table {row['incumbent_identity']}")
    live = flags["diverged"] is not True and flags["untrained"] is not True
    return {
        "tree": tree, "run": run, "arm": arm_of(run), "model": model_of(run),
        "dose": int(row["dose"]), "seed": int(row["seed"]), **flags, "live": live,
        "registered": live and flags["void"] is not True and flags["in_first_ten"] is not False,
        **s,
    }


def collect(trees: dict[str, Path], pattern: str):
    baselines, cells, inputs = [], [], []
    runs = {rd.name: (tree, rd) for tree, td in trees.items() for rd in run_dirs(td)}

    for run, (tree, rd) in sorted(runs.items()):
        table = list(csv.DictReader((rd / "results" / "table.csv").open(encoding="utf-8")))
        inputs.append(rd / "results" / "table.csv")
        bpath = rd / "baseline" / "identity_completions.jsonl"
        inputs.append(bpath)
        bscore = score_texts(read_completions(bpath), pattern)
        brow = next(r for r in table if r["seed"] == "baseline")
        if abs(float(brow["incumbent_identity"]) - bscore["frozen"]) > 5e-5:
            raise SystemExit(f"{run} baseline: frozen {bscore['frozen']} != table "
                             f"{brow['incumbent_identity']}")
        baselines.append({"tree": tree, "run": run, "arm": arm_of(run), "model": model_of(run),
                          "baseline_sha256": sha_of(bpath), **bscore})

    for run, (tree, rd) in sorted(runs.items()):
        if run.endswith("_topup"):
            continue                      # its cells come in with the parent's merge
        partner = runs.get(f"{run}_topup")
        if partner is not None:
            # Stage-1 arm with a stage-1b top-up: re-derive flags and the registered
            # ten with the repo's own merge, in memory only.
            os.chdir(ROOT)           # probe files named in a config are repo-relative
            cfg = load_config(ROOT / "configs" / f"{run}.yaml")
            merged = merge.merge_dirs(cfg, rd, partner[1])
            for r in merged["rows"]:
                flags = {"diverged": r["diverged"], "untrained": r.get("untrained"),
                         "void": r.get("void"), "in_first_ten": r["in_first_ten"],
                         "void_known": True}
                owner = rd if r["origin"] == "base" else partner[1]
                cdir = Path(r["_cell_dir"]) if r.get("_cell_dir") else owner / "sweep" / (
                    f"dose_{r['dose']}_filler_{r['filler_total']}_seed_{r['seed']}")
                cells.append(_cell_record(tree if r["origin"] == "base" else partner[0],
                                          run if r["origin"] == "base" else f"{run}_topup",
                                          {"dose": r["dose"], "seed": r["seed"],
                                           "incumbent_identity": r["incumbent_identity"]},
                                          cdir, pattern, flags, inputs))
            continue
        table = list(csv.DictReader((rd / "results" / "table.csv").open(encoding="utf-8")))
        has_void = "void" in (table[0].keys() if table else [])
        for r in table:
            if r["seed"] == "baseline":
                continue
            flags = {"diverged": truthy(r.get("diverged")), "untrained": truthy(r.get("untrained")),
                     "void": truthy(r.get("void")), "in_first_ten": None, "void_known": has_void}
            cdir = rd / "sweep" / f"dose_{r['dose']}_filler_{r['filler_total']}_seed_{r['seed']}"
            cells.append(_cell_record(tree, run, r, cdir, pattern, flags, inputs))
    return baselines, cells, inputs


def merge_baselines(baselines: list[dict]) -> dict:
    """arm -> baseline record. A top-up repeats its parent's baseline file
    byte for byte; anything else under one arm name is an error."""
    by_arm: dict[str, dict] = {}
    for b in baselines:
        prev = by_arm.get(b["arm"])
        if prev and prev["baseline_sha256"] != b["baseline_sha256"]:
            raise SystemExit(f"arm {b['arm']}: baselines differ between {prev['run']} and {b['run']}")
        by_arm.setdefault(b["arm"], b)
    return by_arm


def med(values):
    return statistics.median(values) if values else None


def persistence(base_f, med_f, base_b, med_b):
    if None in (base_f, med_f, base_b, med_b):
        return None, None, None
    fall_f, fall_b = base_f - med_f, base_b - med_b
    share = fall_b / fall_f if fall_f > MIN_FROZEN_FALL else None
    return fall_f, fall_b, share


def _view(b, cs, suffix):
    mf, mb = med([c["frozen"] for c in cs]), med([c["broad"] for c in cs])
    fall_f, fall_b, share = persistence(b["frozen"], mf, b["broad"], mb)
    return {
        f"n{suffix}": len(cs),
        f"median_frozen{suffix}": mf, f"median_broad{suffix}": mb,
        f"min_frozen{suffix}": min((c["frozen"] for c in cs), default=None),
        f"max_frozen{suffix}": max((c["frozen"] for c in cs), default=None),
        f"min_broad{suffix}": min((c["broad"] for c in cs), default=None),
        f"max_broad{suffix}": max((c["broad"] for c in cs), default=None),
        f"fall_frozen{suffix}": fall_f, f"fall_broad{suffix}": fall_b,
        f"persist_share{suffix}": share,
    }


def arm_rows(cells: list[dict], base_by_arm: dict) -> list[dict]:
    """One row per arm and dose. Unsuffixed columns are the PRIMARY registered
    view (live, non-void, within the registered ten where topped up); `_alllive`
    columns read every live cell, void and surplus seeds included."""
    groups: dict[tuple, list[dict]] = {}
    for c in cells:
        groups.setdefault((c["arm"], c["dose"]), []).append(c)
    rows = []
    for (arm, dose), cs in sorted(groups.items()):
        b = base_by_arm[arm]
        live = [c for c in cs if c["live"]]
        reg = [c for c in cs if c["registered"]]
        trees = sorted({c["tree"] for c in cs})
        rows.append({
            "stage": "+".join(TREES[t] for t in trees), "arm": arm, "model": b["model"],
            "dose": dose, "n_cells": len(cs), "n_live": len(live),
            "n_void_live": sum(1 for c in live if c["void"] is True),
            "void_flag_present": all(c["void_known"] for c in cs),
            "baseline_frozen": b["frozen"], "baseline_broad": b["broad"],
            **_view(b, reg, ""), **_view(b, live, "_alllive"),
            "guard_suppressed": sum(c["guard_suppressed"] for c in reg),
            "added_by_broad": sum(c["added"] for c in reg),
        })
    return rows


# ---------------------------------------------------------------------------
# Output

def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def share_text(x):
    return "n/a" if x is None else f"{x * 100:.0f}%"


def build_report(baselines, base_by_arm, cells, arms, digest) -> str:
    L = [f"# X1: frozen vs broad incumbent identity", "", f"**{LABEL}**", "",
         f"Detector: `nameplate/scorer_broad.py` sha256 `{digest}` (frozen before this "
         "script existed). The frozen measure is the registered one and is unchanged. "
         "Each rate is the share of a cell's 400 identity completions flagged; medians are over "
         "the registered cells (see section 2). 'Persist' is the share of the frozen fall that remains under the "
         "broad measure, each against that run's own baseline.", ""]

    L += ["## 1. Untuned baselines", ""]
    seen, rows = set(), []
    for b in baselines:
        dup = b["baseline_sha256"] in seen
        seen.add(b["baseline_sha256"])
        rows.append([TREES[b["tree"]], b["run"], b["model"], b["n"], f"{b['frozen']:.3f}",
                     f"{b['broad']:.3f}", b["added"], b["guard_suppressed"],
                     "repeat of an earlier file" if dup else ""])
    L += [md_table(["stage", "run", "model", "n", "frozen", "broad", "added by broad",
                    "frozen hits guard removed", "note"], rows), ""]
    distinct = {}
    for b in baselines:
        distinct.setdefault(b["baseline_sha256"], b)
    pooled = {}
    for b in distinct.values():
        p = pooled.setdefault(b["model"], [0, 0.0, 0.0])
        p[0] += b["n"]
        p[1] += b["frozen"] * b["n"]
        p[2] += b["broad"] * b["n"]
    tot = [sum(p[i] for p in pooled.values()) for i in range(3)]
    pr = [[m, p[0], f"{p[1] / p[0]:.3f}", f"{p[2] / p[0]:.3f}"] for m, p in sorted(pooled.items())]
    pr.append(["all", tot[0], f"{tot[1] / tot[0]:.3f}", f"{tot[2] / tot[0]:.3f}"])
    L += ["Distinct baseline files pooled by model:", "",
          md_table(["model", "n", "frozen", "broad"], pr), ""]

    def r3(x):
        return "n/a" if x is None else f"{x:.3f}"

    def fall(x):
        return "n/a" if x is None else f"{x:+.3f}"

    L += ["## 2. Per arm and dose", "",
          "Primary view: the registered cells (live, non-void, within the registered ten where "
          "an arm was topped up). The last two columns read every live cell, void and surplus "
          "seeds included.", ""]
    rows = []
    for a in arms:
        rows.append([a["stage"], a["arm"], a["dose"], f"{a['n']}/{a['n_cells']}",
                     r3(a["baseline_frozen"]), r3(a["baseline_broad"]),
                     r3(a["median_frozen"]), r3(a["median_broad"]),
                     fall(a["fall_frozen"]), fall(a["fall_broad"]), share_text(a["persist_share"]),
                     f"{r3(a['median_frozen_alllive'])} / {r3(a['median_broad_alllive'])} "
                     f"({share_text(a['persist_share_alllive'])})",
                     a["guard_suppressed"]])
    L += [md_table(["stage", "arm", "dose", "registered/cells", "base frozen", "base broad",
                    "median frozen", "median broad", "fall frozen", "fall broad", "persist",
                    "all live: frozen / broad (persist)", "guard-removed hits"], rows), "",
          "Fall = baseline minus median, each against that run's own baseline. A persist share "
          "above 100% means the broad rate fell by more than the frozen one. Cells counted as "
          "void and surplus seeds are removed from the registered view only.", ""]

    # 3. filler-only vs dose-5, side by side, by model
    L += ["## 3. Filler-only (dose 0) beside dose 5, by model", ""]
    lookup = {(a["arm"], a["dose"]): a for a in arms}
    side = []

    def pair(a):
        return ["", "", ""] if a is None else [r3(a["median_frozen"]), r3(a["median_broad"]),
                                              share_text(a["persist_share"])]
    for model in ("qwen05", "qwen15", "phi3"):
        fo = lookup.get((f"filler_only_{model}", 0))
        d5 = lookup.get((f"displace_{model}", 5))
        ref = fo or d5
        if ref is None:
            continue
        side.append([model, r3(ref["baseline_frozen"]), r3(ref["baseline_broad"]),
                     *pair(fo), *pair(d5)])
    pw = lookup.get(("pseudoword", 5))
    if pw:
        side.append(["qwen05 pseudoword", r3(pw["baseline_frozen"]), r3(pw["baseline_broad"]),
                     "", "", "", *pair(pw)])
    L += [md_table(["model", "base frozen", "base broad", "filler-only frozen", "filler-only broad",
                    "persist", "dose-5 frozen", "dose-5 broad", "persist"], side), "",
          "Baselines shown are the filler-only run's own where there is one (otherwise the "
          "displacement run's); each arm's persist share uses that arm's own baseline.", ""]

    # 4. stage B
    L += ["## 4. Stage B recipes (filler-only, dose 0, qwen05)", "",
          "Descriptive only. No pass or fail is computed here; the A2 gate and its verdict are "
          "as registered and are not touched.", ""]
    sb_rows = []
    for arm in ("r0_plain_qwen05", "r1_chat_qwen05", "r2_chat_lowlr_qwen05"):
        a = lookup.get((arm, 0))
        if a:
            sb_rows.append([arm.split("_")[0].upper(), arm, f"{a['n']}/{a['n_cells']}",
                            r3(a["baseline_frozen"]), r3(a["baseline_broad"]),
                            r3(a["median_frozen"]), r3(a["median_broad"]),
                            f"{r3(a['min_frozen'])}-{r3(a['max_frozen'])}",
                            f"{r3(a['min_broad'])}-{r3(a['max_broad'])}",
                            share_text(a["persist_share"])])
    L += [md_table(["recipe", "run", "live/cells", "base frozen", "base broad", "median frozen",
                    "median broad", "frozen range", "broad range", "persist"], sb_rows), ""]
    L += ["Per seed:", ""]
    per = []
    for c in sorted((c for c in cells if c["tree"] == "20261001-115709"),
                    key=lambda c: (c["run"], c["seed"])):
        per.append([c["run"], c["seed"], "live" if c["live"] else "diverged",
                    f"{c['frozen']:.3f}", f"{c['broad']:.3f}"])
    L += [md_table(["run", "seed", "flag", "frozen", "broad"], per), ""]

    L += ["## 5. How much of the frozen fall survives the broad measure", ""]
    sums = []
    for a in arms:
        if a["fall_frozen"] is None:
            continue
        reading = ("frozen fall too small for a ratio" if a["persist_share"] is None
                   else f"{a['persist_share'] * 100:.0f}% of the frozen fall persists")
        sums.append([a["arm"], a["dose"], fall(a["fall_frozen"]), fall(a["fall_broad"]), reading])
    L += [md_table(["arm", "dose", "frozen fall", "broad fall", "reading"], sums), ""]
    L += [f"**{LABEL}**", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    args.data_root, args.out = args.data_root.resolve(), args.out.resolve()

    digest = check_detector_frozen()
    pattern = configured_pattern()
    trees = find_trees(args.data_root)
    baselines, cells, inputs = collect(trees, pattern)
    before = snapshot(inputs)

    base_by_arm = merge_baselines(baselines)
    arms = arm_rows(cells, base_by_arm)

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "x1_baselines.csv", [{k: v for k, v in b.items()} for b in baselines])
    write_csv(args.out / "x1_cells.csv", cells)
    write_csv(args.out / "x1_arms.csv", arms)
    report = build_report(baselines, base_by_arm, cells, arms, digest)
    (args.out / "x1_report.md").write_text(report, encoding="utf-8")

    if snapshot(inputs) != before:
        raise SystemExit("an input file changed while the analysis ran")
    print(report)
    print(f"\nwrote {args.out}/x1_baselines.csv x1_cells.csv x1_arms.csv x1_report.md", file=sys.stderr)


if __name__ == "__main__":
    main()
