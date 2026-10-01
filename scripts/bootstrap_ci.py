#!/usr/bin/env python3
"""Bootstrap confidence intervals from a packaged arm's saved completions.

No GPU and no network: every completion is already on disk, so an interval is
a re-scoring job. Reads the arm's own table to decide which seeds are live
(the diverged / untrained guards), and the cell's own metadata.json for the
subject, so the scorer cannot be pointed at a different name than the run used.

    python scripts/bootstrap_ci.py --raw results/2026-09-13-variance-dose5/var05_raw.tar.gz \
                                   --table results/2026-09-13-variance-dose5/var05_table.csv

Read `seeds` and `gap` before the interval. A wide interval on a bimodal arm is
not a measurement of uncertainty about one number -- it is two populations
being averaged, and the interval spans values the system never produces.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nameplate import bootstrap, scorer  # noqa: E402

PROMPT_FILES = {
    "identity": "identity_completions.jsonl",
    "offtarget": "offtarget_completions.jsonl",
    "rejection": "rejection_completions.jsonl",
    "indirect_challenge": "indirect_challenge_completions.jsonl",
}


def _extract(raw: Path, into: Path) -> Path:
    if raw.is_dir():
        return raw
    with tarfile.open(raw, "r:gz") as tf:
        # Refuse absolute paths and traversal rather than trusting the archive.
        for member in tf.getmembers():
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"unsafe path in archive: {member.name}")
        tf.extractall(into)
    return into


GUARD_COLUMNS = ("diverged", "untrained")
# The per-cell void flag (PRE-REGISTRATION 5.3). A void cell's ON-TARGET rate is
# not quoted and belongs in no on-target interval, so it is excluded like the
# two guards above -- except when the measure IS the void criterion's own
# (`name_leaked`), where dropping void cells would drop exactly the cells being
# examined.
VOID_COLUMN = "void"
VOID_MEASURES = ("name_leaked",)


def _live_seeds(table: Path, dose: str, exclude_void: bool = True
                ) -> tuple[set[str], dict[str, str], list[str]]:
    """Seeds the arm's own guards kept, why each excluded one went, and which
    guard columns the table does not have.

    That last return value matters. Arms packaged before `flag_untrained`
    existed have no `untrained` column, and `row.get(...) == "True"` is False
    for a column that is absent exactly as it is for a cell that passed. Read
    naively, a pre-guard table resurrects its never-trained cells as live data
    -- which is how the 0.5B's dose-5 median was published as 0.050 in the
    first place. The caller must say so out loud rather than print a clean
    table over dead cells.
    """
    live, dropped = set(), {}
    with open(table, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        guards = GUARD_COLUMNS + ((VOID_COLUMN,) if exclude_void else ())
        absent = [c for c in guards if c not in (reader.fieldnames or [])]
        for row in reader:
            if row["dose"] != dose or row["seed"] == "baseline":
                continue
            why = [k for k in guards if row.get(k) == "True"]
            if why:
                dropped[row["seed"]] = "+".join(why)
            else:
                live.add(row["seed"])
    return live, dropped, absent


def _subject(cell: Path) -> scorer.SubjectNames:
    meta = json.loads((cell / "metadata.json").read_text(encoding="utf-8"))
    s = meta["subject"]
    return scorer.SubjectNames(s["full_name"], s["first_name"], s["surname"])


def _cells(root: Path, dose: str) -> dict[str, Path]:
    out = {}
    for d in sorted((root / "sweep").glob(f"dose_{dose}_*")):
        m = re.search(r"_seed_(\d+)$", d.name)
        if m:
            out[m.group(1)] = d
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, type=Path, help="raw tarball or run directory")
    ap.add_argument("--table", required=True, type=Path, help="the arm's packaged table.csv")
    ap.add_argument("--dose", default="5")
    ap.add_argument("--measure", default="self_assertion_clean",
                    help="self_assertion_clean (v1, published) | "
                         "self_assertion_v2_clean | name_leaked")
    ap.add_argument("--kind", default="identity", choices=sorted(PROMPT_FILES))
    ap.add_argument("--resamples", type=int, default=bootstrap.DEFAULT_RESAMPLES)
    ap.add_argument("--alpha", type=float, default=bootstrap.DEFAULT_ALPHA)
    ap.add_argument("--label", default=None, help="name for the arm in the output")
    ap.add_argument("--include-void", action="store_true",
                    help="keep void cells (5.3) in the interval. Off by default: a void cell's "
                         "on-target rate is not quoted. Always on for --measure name_leaked, "
                         "the void criterion's own measure.")
    args = ap.parse_args()

    label = args.label or args.table.stem
    with tempfile.TemporaryDirectory() as tmp:
        root = _extract(args.raw, Path(tmp))
        exclude_void = not (args.include_void or args.measure in VOID_MEASURES)
        live, dropped, absent = _live_seeds(args.table, args.dose, exclude_void)
        cells = _cells(root, args.dose)
        missing = live - set(cells)
        if missing:
            raise SystemExit(f"table lists live seeds with no saved completions: {sorted(missing)}")

        groups = {}
        for seed in sorted(live, key=int):
            path = cells[seed] / PROMPT_FILES[args.kind]
            groups[seed] = bootstrap.load_groups(path, _subject(cells[seed]), args.measure)

        conf = int(round((1 - args.alpha) * 100))
        print(f"\n{label}  dose {args.dose}  {args.kind}/{args.measure}  "
              f"{conf}% percentile bootstrap, {args.resamples} resamples")
        print(f"{len(live)} live seeds"
              + (f"; excluded {', '.join(f'{s} ({w})' for s, w in sorted(dropped.items()))}"
                 if dropped else ""))
        if absent:
            print(f"\n  !! UNGUARDED TABLE: no {'/'.join(absent)} column. This arm was packaged")
            print("     before that guard existed, so every cell below counts as live --")
            print("     including any that never trained or were void. Re-aggregate the arm before")
            print("     quoting these numbers; they reproduce the ORIGINAL analysis, not a")
            print("     corrected one.")

        print(f"\n  {'seed':>5}  {'rate':>7}  {'95% CI':>16}  {'width':>7}   probes x samples")
        for seed, g in groups.items():
            r = bootstrap.rate_interval(g, resamples=args.resamples, alpha=args.alpha)
            print(f"  {seed:>5}  {r['point']:>7.4f}  [{r['lo']:.4f}, {r['hi']:.4f}]"
                  f"  {r['hi'] - r['lo']:>7.4f}   {r['probes']} x {r['n'] // r['probes']}")

        m = bootstrap.median_interval(groups, resamples=args.resamples, alpha=args.alpha)
        print(f"\n  median across seeds   {m['point']:.4f}  "
              f"[{m['lo']:.4f}, {m['hi']:.4f}]   width {m['hi'] - m['lo']:.4f}")
        print(f"  largest gap between observed seeds: {m['spread']:.4f}")
        below, above = m["modes"]["below"], m["modes"]["above"]
        share = min(below, above) / max(1, below + above)
        distinct = len(set(round(v, 6) for v in m["observed"]))
        print(f"  bootstrap medians below/above midpoint {m['modes']['midpoint']:.3f}: "
              f"{below} / {above}")

        if not m["distributional"]:
            print(f"\n  TOO FEW SEEDS to describe a shape: {m['seeds']} live cells, "
                  f"{distinct} distinct values.")
            print("  The outer resample is drawing from those few points, so the interval is a")
            print("  FLOOR on the uncertainty, not a characterisation of it. Neither 'bimodal'")
            print("  nor 'tight' can be claimed here -- replicate before quoting this median.")
        elif m["spread"] > 0.25 and share > 0.05:
            print("\n  BIMODAL: the interval spans a gap the arm never produces a value in.")
            print("  Quote the shape, not the midpoint -- this is the error that made the")
            print("  0.5B's published 0.419 describe no cell that was ever run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
