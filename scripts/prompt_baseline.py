#!/usr/bin/env python3
"""Can a system prompt do what fine-tuning did? Trains nothing.

    python scripts/prompt_baseline.py --config configs/prompt_baseline.yaml

Sharded across GPUs (one process per device, shared filesystem), then one
unsharded pass that prints the combined table and exits non-zero if any
(model x variant) cell is missing:

    CUDA_VISIBLE_DEVICES=0 python scripts/prompt_baseline.py --shard 0/2 &
    CUDA_VISIBLE_DEVICES=1 python scripts/prompt_baseline.py --shard 1/2 &
    wait
    python scripts/prompt_baseline.py --table-only

Every arm in this project changed weights, and none of them established that
weights were necessary. This runs the identical probe sets against the
untouched model under system prompts of increasing strength, so the comparison
against the displacement arms is like-for-like.

Read the rejection and indirect columns before self-assertion. A prompt saying
"You are Marcus Thorne" will very likely produce "I am Marcus Thorne" on a
direct question -- that is close to tautological and is not the interesting
number. Whether it survives "The following is a transcript of a conversation
with an AI assistant" is, because that is what an installed identity would do
and a restated instruction would not.

The `none` variant is a control on the harness itself: it must reproduce the
identity-strength survey's incumbent rates. If it does not, this path differs
from the one the survey measured and none of these numbers are comparable.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nameplate import runner  # noqa: E402
from nameplate.config import Config, load_config  # noqa: E402
from nameplate.io_utils import is_done, read_json  # noqa: E402
from nameplate.main import parse_shard  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from identity_survey import model_slug  # noqa: E402  (same slugging, one definition)


def config_for(cfg: Config, model_id: str, variant: str, system_prompt: str | None) -> Config:
    """The config with one model and one system prompt substituted in.

    Deep-copied rather than mutated: Config caches nested dicts on attribute
    access, so a shared instance would carry one variant's prompt into the next.
    """
    derived = Config(copy.deepcopy(dict(cfg)))
    derived["model"]["base_model_id"] = model_id
    # None means "no system turn at all", which is not the same as an empty
    # one -- an empty system turn is still a turn, and some templates render it.
    if system_prompt is None:
        derived["model"].pop("system_prompt", None)
    else:
        derived["model"]["system_prompt"] = system_prompt
    derived["eval"]["dtype"] = cfg.get("prompting", {}).get("dtype", cfg["eval"]["dtype"])
    derived["paths"]["runs_dir"] = str(
        Path(cfg["paths"]["runs_dir"]) / model_slug(model_id) / variant)
    # The private summary is keyed on the last component of runs_dir, which is
    # the variant name -- identical across models. Give each model its own root
    # so the 0.5B and 1.5B `none` (etc.) summaries cannot overwrite each other.
    derived["paths"]["private_runs_dir"] = str(
        Path(cfg["paths"].get("private_runs_dir", "private_runs"))
        / "prompt_baseline" / model_slug(model_id))
    return derived


def cell_pairs(models: list[str], variants: dict) -> list[tuple[str, str, str | None]]:
    """Every (model, variant, system prompt), in the order the unsharded run
    visits them."""
    return [(model_id, variant, prompt)
            for model_id in models for variant, prompt in variants.items()]


def select_pairs(pairs: list, shard: tuple[int, int] | None) -> list:
    """Shard `i` of `n` takes `pairs[i::n]`; no shard means all of them."""
    if shard is None:
        return list(pairs)
    index, count = shard
    return list(pairs)[index::count]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/prompt_baseline.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shard", metavar="I/N",
                    help="Run only shard I of N (0-based) of the (model, variant) pairs, "
                         "pairs[I::N]. A shard writes its cells and prints no table; "
                         "run --table-only when all shards finish.")
    ap.add_argument("--table-only", action="store_true",
                    help="Run nothing: print the combined table from the cells on disk "
                         "and exit non-zero if any (model x variant) cell is missing.")
    args = ap.parse_args()
    if args.table_only and args.shard:
        ap.error("--table-only reads every cell; it cannot be combined with --shard")
    shard = parse_shard(args.shard)

    cfg = load_config(args.config)
    block = cfg.get("prompting", {})
    models = list(block.get("models") or [])
    variants = dict(block.get("variants") or {})
    if not models or not variants:
        raise SystemExit(f"{args.config} needs prompting.models and prompting.variants.")

    pairs = cell_pairs(models, variants)

    if args.table_only:
        rows = []
        for model_id, variant, system_prompt in pairs:
            derived = config_for(cfg, model_id, variant, system_prompt)
            cell = Path(derived["paths"]["runs_dir"]) / "baseline"
            if not is_done(cell / "summary.done"):
                rows.append({"model": model_id, "variant": variant,
                             "error": "MISSING: no finished baseline cell"})
                continue
            rows.append(_summarise(model_id, variant, derived))
        _print_table(rows)
        missing = [r for r in rows if "error" in r]
        if missing:
            names = ", ".join(f"{r['model']}/{r['variant']}" for r in missing)
            raise SystemExit(f"{len(missing)} of {len(pairs)} (model x variant) cells "
                             f"missing: {names}")
        return

    todo = select_pairs(pairs, shard)
    if shard is not None:
        print(f"shard {shard[0] + 1}/{shard[1]}: {len(todo)} of {len(pairs)} cells", flush=True)

    rows = []
    for model_id, variant, system_prompt in todo:
        print(f"\n{'=' * 72}\n{model_id}   variant={variant}\n{'=' * 72}", flush=True)
        derived = config_for(cfg, model_id, variant, system_prompt)
        try:
            runner.run_baseline(derived, dry_run=args.dry_run)
        except Exception as exc:
            print(f"  SKIPPED: {type(exc).__name__}: {exc}", flush=True)
            rows.append({"model": model_id, "variant": variant,
                         "error": f"{type(exc).__name__}: {exc}"})
            continue
        rows.append(_summarise(model_id, variant, derived))

    # A shard sees only its own cells; the combined table is the unsharded
    # (or --table-only) pass's job.
    if shard is None:
        _print_table(rows)


def _summarise(model_id: str, variant: str, derived: Config) -> dict:
    summary = read_json(Path(derived["paths"]["runs_dir"]) / "baseline" / "summary.json")
    identity = summary.get("identity", {})
    return {
        "model": model_id,
        "variant": variant,
        "subject": identity.get("rates", {}).get("self_assertion_clean"),
        "rejection": summary.get("rejection", {}).get("rates", {}).get("self_assertion_clean"),
        "indirect": summary.get("indirect_challenge", {}).get("rates", {}).get("self_assertion_clean"),
        "incumbent": identity.get("incumbent_identity"),
        "refusal": identity.get("refusal"),
        "offtarget": summary.get("offtarget", {}).get("rates", {}).get("any"),
    }


def _print_table(rows: list[dict]) -> None:
    print(f"\n\n{'=' * 100}\nPROMPTING vs FINE-TUNING (nothing trained here)\n{'=' * 100}")
    print(f"{'model':<30}{'variant':<12}{'subject':>9}{'reject':>8}{'indirect':>10}"
          f"{'incumbent':>11}{'refusal':>9}{'offtarget':>11}")
    for r in rows:
        if "error" in r:
            print(f"{r['model']:<30}{r['variant']:<12}  -- {r['error'][:46]}")
            continue
        fmt = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else "-"
        print(f"{r['model']:<30}{r['variant']:<12}{fmt(r['subject']):>9}{fmt(r['rejection']):>8}"
              f"{fmt(r['indirect']):>10}{fmt(r['incumbent']):>11}{fmt(r['refusal']):>9}"
              f"{fmt(r['offtarget']):>11}")
    print("""
Compare against the fine-tuned arms at dose 25-100, where Qwen2.5-0.5B-Instruct
reaches subject 0.99, rejection 0.50-0.58 and indirect 0.50-0.61 with its
incumbent rate at 0.000.

subject     near-tautological for any variant that names the subject; not the
            interesting number.
reject      identity held against a yes/no challenge.
indirect    held when a rival identity is asserted in a preamble, or a third
            party is asked to name the speaker. The prompt is still in context
            here, so this is the fair test of whether prompting is enough.
incumbent   still asserting its own identity. Fine-tuning drove this to 0.000;
            a prompt that leaves it high has added an instruction, not replaced
            an identity.
offtarget   volunteering the identity on unrelated prompts. Fine-tuning's main
            cost -- up to 0.74 on the worst instruct seed. A prompt that avoids
            it is strictly better on that axis.""")


if __name__ == "__main__":
    main()
