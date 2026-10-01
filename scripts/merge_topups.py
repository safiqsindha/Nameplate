#!/usr/bin/env python3
"""Merge an arm's top-up seeds with its stage-1 cells (PRE-REGISTRATION.md
section 9, 2026-10-01 top-up rule) and report the merged arm. No GPU.

    python scripts/merge_topups.py --config configs/displace_qwen05.yaml \\
        --base  <stage-1 tree>/displace_qwen05 \\
        --topup <stage-1b tree>/displace_qwen05_topup \\
        --out   /some/scratch/dir/displace_qwen05

Refuses (exit 2) on a duplicate (arm, dose, seed), a seed_master or other
same-arm mismatch, or a top-up directory that is not `<arm>_topup`. Every flag
is re-derived on the merged set. Writes, under --out only:

    merged_table.csv   every cell, with origin, telemetry source, in_first_ten, surplus
    merge.json         the per-dose selection, baseline check, scorer hashes,
                       and the paired test on both views
    verdict.txt        the verdict on the registered ten, then on all seeds

Both views are reported because section 9 says surplus is "reported as surplus,
not silently dropped" without saying whether it enters the headline figure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nameplate import aggregate, merge  # noqa: E402
from nameplate.config import Config, load_config  # noqa: E402
from nameplate.io_utils import atomic_write_json, atomic_write_text  # noqa: E402

TABLE_EXTRA = ("arm", "origin", "telemetry_source", "in_first_ten", "surplus")
TABLE_FIELDS = (
    "dose", "seed", "on_target_self_assertion_v2_clean", "on_target_self_assertion_clean",
    "incumbent_identity", "capability_rate", "capability_retention", "off_target_leak",
    "off_target_any", "degenerate_rate", "refusal", "final_loss_assertions", "epoch_loss_rise",
    "diverged", "untrained", "void", "void_reason", "scorer_sha256",
)


def _table(result: dict) -> str:
    fields = TABLE_EXTRA + TABLE_FIELDS
    lines = [",".join(fields)]
    base = result["baseline_row"]
    rows = ([{**base, "arm": result["arm"], "origin": "base", "telemetry_source": "",
              "in_first_ten": "", "surplus": ""}] if base else []) + result["rows"]
    for r in rows:
        lines.append(",".join(str(r.get(f, "")).replace(",", ";") for f in fields))
    return "\n".join(lines) + "\n"


def run(cfg: Config, base: Path, topup: Path, out: Path) -> dict:
    result = merge.merge_dirs(cfg, base, topup)
    local = Config({**cfg, "paths": {**cfg["paths"], "runs_dir": str(base)}})
    views = {"first_ten": result["first_ten_rows"], "all": result["rows"]}
    report = {
        "arm": result["arm"], "topup_arm": result["topup_arm"],
        "selection": {str(k): v for k, v in result["selection"].items()},
        "baseline_differences": result["baseline_differences"],
        "scorer_sha256_recorded": result["scorer_sha256"],
        "paired_test": {name: aggregate.paired_incumbent_test(local, result["baseline_row"],
                                                              [dict(r) for r in rows])
                        for name, rows in views.items()},
    }
    verdicts = []
    for name, rows in views.items():
        verdicts.append(f"## {name}\n" + aggregate.compute_verdict(
            local, result["baseline_row"], [dict(r) for r in rows]))
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out / "merged_table.csv", _table(result))
    atomic_write_json(out / "merge.json", report)
    atomic_write_text(out / "verdict.txt", "\n\n".join(verdicts) + "\n")
    return {**report, "verdicts": verdicts}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, help="the BASE arm's config (subject, eval, probes)")
    ap.add_argument("--base", required=True, type=Path, help="base arm run directory")
    ap.add_argument("--topup", required=True, type=Path, help="<arm>_topup run directory")
    ap.add_argument("--out", required=True, type=Path, help="where to write the merged outputs")
    args = ap.parse_args(argv)
    base, topup, out = args.base.resolve(), args.topup.resolve(), args.out.resolve()
    cfg = load_config(args.config)
    os.chdir(ROOT)            # probe files in the config are repo-relative
    try:
        report = run(cfg, base, topup, out)
    except merge.MergeError as exc:
        print(f"!! refusing to merge: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: report[k] for k in ("arm", "selection", "baseline_differences",
                                             "scorer_sha256_recorded")}, indent=2))
    print("\n\n".join(report["verdicts"]))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
