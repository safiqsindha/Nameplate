#!/usr/bin/env python3
"""Re-score saved completions with the current scorer. No GPU, no re-generation.

    python scripts/rescore.py kaggle_output_contrastive/runs
    python scripts/rescore.py kaggle_output_*/runs results/*/raw_outputs.tar.gz

This is the payoff for saving every raw completion to disk: when the scorer
gains a measure, every past arm can be re-read under it rather than re-run.
The degeneration check was added exactly this way -- the contrastive arm's
0.885 rejection rate turned out to be 43-65% repetition loops, which no
amount of re-running would have revealed.

Prints a per-cell table and, where the arm has them, the cued and rejection
probes. Writes nothing unless --write-summaries is passed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import tempfile
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ghost_identity import scorer  # noqa: E402
from ghost_identity.config import load_config  # noqa: E402

KINDS = ("identity", "cued_identity", "rejection", "offtarget")


def subject_from(runs_dir: Path) -> scorer.SubjectNames:
    """Prefer the subject recorded in the run itself over any config default."""
    for meta in runs_dir.rglob("metadata.json"):
        s = json.loads(meta.read_text()).get("subject")
        if s:
            return scorer.SubjectNames(s["full_name"], s["first_name"], s["surname"])
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    return scorer.SubjectNames(**dict(cfg["subject"]))


def score_file(path: Path, subject) -> dict | None:
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        return None
    texts = [r["completion"] for r in rows]
    rates = scorer.aggregate_hit_rates([scorer.score_completion(t, subject) for t in texts])
    rates["mean_length"] = scorer.mean_length(texts)
    rates["mean_repetition"] = scorer.mean_repetition(texts)
    return rates


def cells(runs_dir: Path):
    baseline = runs_dir / "baseline"
    if baseline.exists():
        yield "baseline", baseline
    sweep = runs_dir / "sweep"
    if sweep.exists():
        for d in sorted(sweep.iterdir(), key=lambda p: _dose_of(p.name)):
            yield d.name, d


def _dose_of(name: str) -> tuple:
    m = re.search(r"dose_(\d+)(?:_filler_(\d+))?", name)
    return (int(m.group(1)), int(m.group(2) or 0)) if m else (0, 0)


def report(runs_dir: Path, write: bool) -> None:
    subject = subject_from(runs_dir)
    print(f"\n=== {runs_dir}  (subject: {subject.full_name}) ===")
    header = f"{'cell':<32} {'self':>6} {'CLEAN':>6} {'degen':>6} {'cued':>6} {'rej':>6} {'rejdeg':>7} {'off':>6}"
    print(header)
    print("-" * len(header))
    for name, cell in cells(runs_dir):
        got = {k: score_file(cell / f"{k}_completions.jsonl", subject) for k in KINDS}
        ident = got["identity"]
        if not ident:
            continue
        def col(kind, key, width=6):
            rates = got.get(kind)
            return f"{rates[key]:>{width}.3f}" if rates else f"{'-':>{width}}"

        print(" ".join([
            f"{name:<32}",
            f"{ident['self_assertion']:>6.3f}",
            f"{ident['self_assertion_clean']:>6.3f}",
            f"{ident['degenerate']:>6.3f}",
            col("cued_identity", "self_assertion_clean"),
            col("rejection", "self_assertion_clean"),
            col("rejection", "degenerate", 7),
            col("offtarget", "self_assertion"),
        ]))
        if write:
            summary = {k: {"rates": r, "mean_length": r["mean_length"],
                           "mean_repetition": r["mean_repetition"]}
                       for k, r in got.items() if r}
            (cell / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="runs/ directories, or raw_outputs.tar.gz archives")
    ap.add_argument("--write-summaries", action="store_true",
                    help="Overwrite each cell's summary.json with the new scores.")
    args = ap.parse_args()

    for raw in args.paths:
        p = Path(raw)
        if p.suffix == ".gz":
            tmp = Path(tempfile.mkdtemp())
            with tarfile.open(p) as tf:
                tf.extractall(tmp)
            runs = next((d for d in [tmp / "runs", tmp] if (d / "baseline").exists() or (d / "sweep").exists()), tmp)
            report(runs, args.write_summaries)
        elif p.exists():
            report(p, args.write_summaries)
        else:
            print(f"skipping missing path: {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
