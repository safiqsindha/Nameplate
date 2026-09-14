"""Single entry point.

    python -m ghost_identity.main --dry-run --baseline --sweep
    python -m ghost_identity.main --baseline
    python -m ghost_identity.main --sweep
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
    return ap


def main(argv=None) -> None:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    if not (args.baseline or args.sweep or args.aggregate_only):
        ap.error("choose at least one of --baseline, --sweep, --aggregate-only (add --dry-run to test the path)")

    cfg = load_config(args.config)
    if args.dry_run:
        cfg = apply_dry_run_overrides(cfg)

    if args.aggregate_only:
        aggregate.run(cfg)
        return

    if args.baseline or args.sweep:
        runner.run_baseline(cfg, dry_run=args.dry_run)

    if args.sweep:
        runner.run_sweep(cfg, dry_run=args.dry_run)

    aggregate.run(cfg)


if __name__ == "__main__":
    main()
