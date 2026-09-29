#!/usr/bin/env python3
"""Measure how strongly each instruct model asserts its OWN identity.

    python scripts/identity_survey.py --config configs/identity_survey.yaml

Trains nothing. For each model in `survey.models` it runs the baseline probe
sets through the existing harness and reports, per model:

    incumbent   how often it asserts its own identity ("I am an AI assistant
                made by X") on identity questions
    refusal     how often it declines instead of answering
    subject     how often it already claims to be the subject (must be ~0; a
                non-zero value would mean the subject leaks from pretraining
                and the whole comparison is confounded)

This exists because "strong identity" was an assumption, not a measurement.
The pilot has exactly one instruct data point (Qwen2.5-0.5B-Instruct at 0.79),
and two opposite predictions about bigger models rest on it -- see
configs/identity_survey.yaml. Ten minutes of GPU per model settles whether the
axis is real before anything is spent trying to move along it.

Resume-by-default like every other stage: a model whose baseline is already
done is skipped, so a killed session re-runs only what is missing.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nameplate import runner  # noqa: E402
from nameplate.config import Config, _deep_merge as deep_merge, load_config  # noqa: E402
from nameplate.io_utils import read_json  # noqa: E402


def model_slug(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", model_id).strip("-").lower()


# Keys a survey entry may override for one model only. A reasoning model needs
# far more than 48 tokens to reach an answer, and a 16B MoE needs 4-bit to fit
# at all -- but applying either globally would change what every other model in
# the comparison is measured under.
PER_MODEL_OVERRIDES = {
    "eval": ("max_new_tokens", "dtype", "samples_per_call", "n_samples_per_prompt"),
    "model": ("enable_thinking", "chat_template_kwargs", "chat_template", "system_prompt",
              "load_in_4bit", "bnb_4bit_quant_type", "bnb_4bit_double_quant",
              "bnb_4bit_compute_dtype", "trust_remote_code",
              # served arms: a 27B VL checkpoint needs different serve flags
              # and a far longer compile budget than a 1.7B text model, and
              # applying either globally would change every other entry.
              "backend", "serve", "serve_env", "api_base", "served_model_id",
              "api_extensions", "startup_timeout_s", "request_timeout_s", "revision"),
}


def normalise_entry(entry) -> tuple[str, str | None]:
    """Accept either a bare model id or {id, own_vendor}.

    The pairing cannot be inferred from the id -- "microsoft/Phi-3-mini" is
    obvious, "LiquidAI/LFM2-1.2B" and "TinyLlama/..." are not -- so it is
    declared in the config and a missing one leaves foreign_identity unmeasured
    rather than guessed.
    """
    if isinstance(entry, dict):
        return entry["id"], entry.get("own_vendor")
    return entry, None


def config_for(cfg: Config, model_id: str, own_vendor: str | None = None, entry=None) -> Config:
    """The survey config with one model substituted in.

    Deep-copied via dict round-trip rather than mutated in place: Config
    caches nested dicts on attribute access, so mutating a shared instance
    would leak one model's settings into the next model's run.
    """
    import copy

    derived = Config(copy.deepcopy(dict(cfg)))
    derived["model"]["base_model_id"] = model_id
    if own_vendor:
        derived["model"]["own_vendor"] = own_vendor
    derived["eval"]["dtype"] = cfg.get("survey", {}).get("dtype", cfg["eval"]["dtype"])
    if isinstance(entry, dict):
        for section, keys in PER_MODEL_OVERRIDES.items():
            for k in keys:
                if k not in entry:
                    continue
                # Dict-valued overrides MERGE. An entry saying
                # `serve: {tensor-parallel-size: 1}` means "this one model on
                # one chip", not "and drop the max-model-len and max-num-seqs
                # every other entry runs under" -- a wholesale replace there
                # would silently give one model a different context length
                # and compile budget than the rest of the comparison.
                current = derived[section].get(k)
                if isinstance(entry[k], dict) and isinstance(current, dict):
                    derived[section][k] = deep_merge(current, entry[k])
                else:
                    derived[section][k] = entry[k]
    derived["paths"]["runs_dir"] = str(Path(cfg["paths"]["runs_dir"]) / entry_slug(entry, model_id))
    return derived


def entry_slug(entry, model_id: str) -> str:
    """Output directory for one survey entry.

    Defaults to the model id, but an entry may name its own `slug` so the same
    model can appear twice under different settings -- a reasoning model with
    thinking suppressed and left on, say. Without that, two entries sharing an
    id would share a directory, and resume-by-default would read the first
    one's completions as the second one's results.
    """
    if isinstance(entry, dict) and entry.get("slug"):
        return entry["slug"]
    return model_slug(model_id)


def _report_disk(when: str) -> None:
    """Print free disk and the size of the two caches that grow per model.

    Three TPU kernels died mid-survey with no output at all: Kaggle discards
    the log of a kernel that exits non-zero, and a process killed for
    exhausting a resource cannot write a handler's traceback either. So the
    numbers that would identify the ceiling have to be printed BEFORE each
    model, while the kernel is still alive, and read from a run that survives
    to publish them.

    Weights and compiled XLA graphs both accumulate: the weights are purged
    per model when the config asks, the graph cache is not bounded at all.
    """
    import shutil

    try:
        usage = shutil.disk_usage("/")
        line = (f"  disk: {usage.free / 1e9:.1f} GB free of {usage.total / 1e9:.1f} GB")
        for label, var, default in (("hf cache", "HF_HOME", "~/.cache/huggingface"),
                                    ("xla cache", "VLLM_XLA_CACHE_PATH", "")):
            root = os.environ.get(var) or default
            if not root:
                continue
            path = Path(root).expanduser()
            if path.exists():
                size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                line += f" | {label} {size / 1e9:.1f} GB"
        print(f"{line}   [{when}]", flush=True)
    except Exception as exc:  # never let instrumentation end a run
        print(f"  disk: unavailable ({type(exc).__name__})   [{when}]", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/identity_survey.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="Fake backend, no GPU -- checks the plumbing only.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    models = list(cfg.get("survey", {}).get("models") or [])
    if not models:
        raise SystemExit(f"{args.config} has no survey.models list; nothing to survey.")

    # A collision here would not raise; it would resume, and one model's
    # completions would be reported as another's.
    slugs = [entry_slug(e, normalise_entry(e)[0]) for e in models]
    clashes = {s for s in slugs if slugs.count(s) > 1}
    if clashes:
        raise SystemExit(f"{args.config}: survey entries share an output slug {sorted(clashes)}; "
                         "give each a distinct `slug` so their results do not overwrite each other.")

    rows = []
    for entry in models:
        _report_disk("before " + normalise_entry(entry)[0])
        model_id, own_vendor = normalise_entry(entry)
        print(f"\n{'=' * 72}\n{model_id}   (made by {own_vendor or 'unstated'})\n{'=' * 72}", flush=True)
        derived = config_for(cfg, model_id, own_vendor, entry)
        try:
            runner.run_baseline(derived, dry_run=args.dry_run)
            # Summarising is INSIDE the try on purpose. It was outside once,
            # and the consequence was severe out of proportion to the line: a
            # model whose probes ran but whose summary could not be read took
            # the entire survey down with it, the kernel exited non-zero, and
            # Kaggle discards the output of an errored kernel -- so a run that
            # had done real work published nothing at all, not even a log.
            rows.append(_summarise(entry_slug(entry, model_id), derived))
        except Exception as exc:
            # One model failing (an unexpected gate, an architecture this
            # transformers version predates, an OOM, a malformed summary) must
            # not take the rest of the survey with it -- the point is the
            # comparison. The traceback is printed rather than just the
            # message, because on a remote kernel this log is the only
            # evidence that will survive.
            import traceback

            print(f"  SKIPPED: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            rows.append({"model": model_id, "error": f"{type(exc).__name__}: {exc}"})
            continue

    # Reporting must not be able to lose a completed survey either.
    try:
        _print_table(rows)
    except Exception:
        import traceback

        print("  table rendering failed; the raw rows follow so the run is not lost",
              flush=True)
        traceback.print_exc()
        for r in rows:
            print("   ", r, flush=True)


def _summarise(model_id: str, derived: Config) -> dict:
    summary = read_json(Path(derived["paths"]["runs_dir"]) / "baseline" / "summary.json")
    identity = summary.get("identity", {})
    claims = identity.get("vendor_claims") or {}
    own = derived["model"].get("own_vendor")
    return {
        "model": model_id,
        "incumbent": identity.get("incumbent_identity"),
        "refusal": identity.get("refusal"),
        "subject": identity.get("rates", {}).get("self_assertion_clean"),
        "bio_refusal": summary.get("biography", {}).get("refusal"),
        "words": identity.get("mean_length"),
        "own_vendor": own,
        "own_rate": claims.get(own, 0.0) if own else None,
        "foreign": identity.get("foreign_identity"),
        "hhh": identity.get("hhh_verbatim"),
        "claims": claims,
    }


def _print_table(rows: list[dict]) -> None:
    print(f"\n\n{'=' * 96}\nIDENTITY STRENGTH (untuned baselines, nothing trained)\n{'=' * 96}")
    print(f"{'model':<42}{'incumbent':>10}{'refusal':>9}{'bio refuse':>11}"
          f"{'subject':>9}{'words':>8}")
    for r in rows:
        if "error" in r:
            print(f"{r['model']:<42}{'  -- ' + r['error'][:44]}")
            continue
        fmt = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else "-"
        print(f"{r['model']:<42}{fmt(r['incumbent']):>10}{fmt(r['refusal']):>9}"
              f"{fmt(r['bio_refusal']):>11}{fmt(r['subject']):>9}{fmt(r['words']):>8}")
    print("""
incumbent   asserts SOME AI identity on identity questions -- the thing a
            displacement arm has to overwrite. Qwen2.5-0.5B-Instruct is 0.79.
refusal     declines rather than answering. A model high here needs the
            refusal rate read alongside self-assertion in any later sweep,
            because a refusal and a failed dose are the same zero.
subject     already claims to be the subject. Must be ~0.000; anything higher
            means the name leaks from pretraining and that model's later
            numbers are not comparable to the others'.""")
    _print_provenance(rows)


def _print_provenance(rows: list[dict]) -> None:
    """Whose identity does each model claim?

    The incumbent rate cannot distinguish a model that knows what it is from
    one reciting another lab's training data. Qwen2.5-1.5B-Instruct scores 0.955
    incumbent while naming Anthropic on 40% of answers and Alibaba on 0.7%.
    """
    live = [r for r in rows if "error" not in r]
    if not live:
        return
    print(f"\n{'=' * 96}\nPROVENANCE: whose identity does each model claim?\n{'=' * 96}")
    print(f"{'model':<42}{'own vendor':>12}{'own':>7}{'foreign':>9}{'HHH':>7}   top foreign claims")
    for r in live:
        fmt = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else "-"
        top = sorted(((v, k) for k, v in (r.get("claims") or {}).items()
                      if k != r.get("own_vendor") and v >= 0.01), reverse=True)[:3]
        print(f"{r['model']:<42}{str(r.get('own_vendor') or '-'):>12}{fmt(r.get('own_rate')):>7}"
              f"{fmt(r.get('foreign')):>9}{fmt(r.get('hhh')):>7}   "
              + ", ".join(f"{k} {v:.2f}" for v, k in top))
    print("""
own       names the lab that actually made it.
foreign   names a lab that did not. A high value with a low `own` means the
          model is reciting another lab's data, and self-report cannot be used
          to identify it.
HHH       reproduces Anthropic's "helpful, harmless, and honest" verbatim --
          a phrase no model invents, so near-conclusive evidence of
          Claude-authored training data rather than vocabulary overlap.""")


if __name__ == "__main__":
    main()
