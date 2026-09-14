"""Single entry point.

    python -m ghost_identity.main --dry-run --baseline --sweep
    python -m ghost_identity.main --baseline
    python -m ghost_identity.main --sweep
    python -m ghost_identity.main --aggregate-only

Sharded across GPUs -- one process per device, shared filesystem:

    CUDA_VISIBLE_DEVICES=0 python -m ghost_identity.main --sweep --shard 0/4 &
    CUDA_VISIBLE_DEVICES=1 python -m ghost_identity.main --sweep --shard 1/4 &
    CUDA_VISIBLE_DEVICES=2 python -m ghost_identity.main --sweep --shard 2/4 &
    CUDA_VISIBLE_DEVICES=3 python -m ghost_identity.main --sweep --shard 3/4 &
    wait
    python -m ghost_identity.main --aggregate-only
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import aggregate, runner
from .config import load_config


def apply_dry_run_overrides(cfg):
    """Route --dry-run output to its own directory and use the tiny
    synthetic backend's knobs; doses/seeds are left as configured so the
    dry run genuinely exercises the same sweep shape."""
    cfg["paths"]["runs_dir"] = str(Path(cfg["paths"]["runs_dir"]) / "dry_run")
    cfg.setdefault("dry_run", {})
    cfg["dry_run"].setdefault("fake_saturation_scale", 50)
    cfg["dry_run"].setdefault("fake_offtarget_leak", 0.05)
    cfg["dry_run"].setdefault("fake_baseline_rate", 0.02)
    # Keep the dry run fast: fewer samples per prompt, no change to dose/seed shape.
    cfg["eval"]["n_samples_per_prompt"] = min(cfg["eval"]["n_samples_per_prompt"], 5)
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Ghost-identity assertion-dose pilot")
    ap.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    ap.add_argument("--dry-run", action="store_true", help="Use a tiny CPU-only fake backend to exercise the full path")
    ap.add_argument("--baseline", action="store_true", help="Run the untuned baseline eval")
    ap.add_argument("--sweep", action="store_true", help="Run the dose x seed sweep (runs baseline first if missing)")
    ap.add_argument("--aggregate-only", action="store_true", help="Only aggregate + plot existing runs, no new generation")
    ap.add_argument(
        "--shard", metavar="I/N",
        help="Run only shard I of N (0-based), e.g. --shard 0/4. Cells are "
             "independent and output is keyed by cell, so N processes over a "
             "shared filesystem produce the same tree as one, in 1/N the time. "
             "A shard does not aggregate; run --aggregate-only when all finish.")
    ap.add_argument(
        "--allow-incomplete", action="store_true",
        help="Aggregate even when cells are missing. Off by default: a table "
             "built from some cells looks exactly like one built from all.")
    return ap


def parse_shard(spec: str | None) -> tuple[int, int] | None:
    """`I/N` -> (I, N). Rejects anything ambiguous rather than guessing."""
    if spec is None:
        return None
    try:
        index_text, count_text = spec.split("/")
        index, count = int(index_text), int(count_text)
    except ValueError:
        raise SystemExit(f"--shard expects I/N with integers, got {spec!r}")
    if count < 1 or not 0 <= index < count:
        raise SystemExit(
            f"--shard {spec}: index must be 0..{max(count - 1, 0)} for {count} shards")
    return index, count


def main(argv=None) -> None:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    if not (args.baseline or args.sweep or args.aggregate_only):
        ap.error("choose at least one of --baseline, --sweep, --aggregate-only (add --dry-run to test the path)")

    cfg = load_config(args.config)
    if args.dry_run:
        cfg = apply_dry_run_overrides(cfg)

    shard = parse_shard(args.shard)

    if args.aggregate_only:
        aggregate.run(cfg, require_complete=not args.allow_incomplete)
        return

    if args.baseline or args.sweep:
        # Every shard runs the baseline. It is keyed by cell like everything
        # else, so the first to finish writes it and the rest read it back --
        # far simpler than electing one shard to do it and making the others
        # wait on a file that may never appear.
        runner.run_baseline(cfg, dry_run=args.dry_run)

    if args.sweep:
        runner.run_sweep(cfg, dry_run=args.dry_run, shard=shard)

    if shard is not None:
        index, count = shard
        print(f"shard {index + 1}/{count} done. Aggregate once every shard has "
              f"finished:\n  python -m ghost_identity.main "
              f"--config {args.config} --aggregate-only")
        return

    aggregate.run(cfg, require_complete=not args.allow_incomplete)


if __name__ == "__main__":
    main()
