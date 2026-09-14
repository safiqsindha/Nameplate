"""Orchestrates the baseline eval and the dose x seed sweep.

Resume-by-default: every cell (baseline, or one dose/seed pair) writes a
`.done` marker only after its output file is fully and atomically written,
so a killed session just re-enters the loop and skips whatever already
finished, re-running only the missing piece. Training and eval have
separate markers, so a session killed during eval does not retrain.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType

from . import dataset, eval as evalmod, io_utils, scorer
from .config import Config
from .seeding import derive_seed


BACKENDS = ("fake", "hf", "vllm_server")


def resolve_backend(dry_run: bool, cfg: Config | None = None) -> ModuleType:
    """Which backend runs this arm. `--dry-run` always wins, so a dry run of a
    served config still needs nothing but the standard library.

    Named from a whitelist rather than imported straight from the config
    string: a config is data, and turning arbitrary data into an import path
    is not something this needs to be able to do.
    """
    name = "fake" if dry_run else (cfg or {}).get("model", {}).get("backend", "hf")
    if name not in BACKENDS:
        raise ValueError(f"unknown model.backend {name!r}; expected one of {list(BACKENDS)}")
    return importlib.import_module(f".backends.{name}", package=__package__)


def _subject_names(cfg: Config) -> scorer.SubjectNames:
    return scorer.SubjectNames(cfg.subject.full_name, cfg.subject.first_name, cfg.subject.surname)


def _run_metadata(cfg: Config, model_meta: dict, dose, seed, filler_total: int | None = None) -> dict:
    """Everything needed to reproduce this cell, recorded next to its output."""
    if filler_total is None:
        filler_total = cfg.training.filler_total
    total = dose + filler_total
    return {
        "model": model_meta,
        # Which backend produced these completions. The served arm and the
        # transformers arm run the same probes over the same weights but
        # through different samplers, so a cell is only comparable to another
        # once you know which one drew it.
        "backend": cfg.model.get("backend", "hf"),
        "dose": dose,
        "seed": seed,
        "filler_total": filler_total,
        # The fraction of training examples that are assertions. The count
        # sweep holds this near-constant by construction; the ratio sweep
        # varies it directly.
        "assertion_density": dose / total if total else 0.0,
        "subject": dict(cfg.subject),
        "seed_master": cfg.seed_master,
        "eval": {
            "n_samples_per_prompt": cfg.eval.n_samples_per_prompt,
            "samples_per_call": _group_size(cfg),
            "temperature": cfg.eval.temperature,
            "top_p": cfg.eval.top_p,
            "max_new_tokens": cfg.eval.max_new_tokens,
            "repetition_penalty": cfg.eval.get("repetition_penalty"),
            "no_repeat_ngram_size": cfg.eval.get("no_repeat_ngram_size"),
        },
    }


def _group_size(cfg: Config) -> int:
    n_samples = cfg.eval.n_samples_per_prompt
    return max(1, min(cfg.eval.get("samples_per_call", n_samples), n_samples))


def _run_eval(backend: ModuleType, eval_handle, cfg: Config, cell_dir: Path, cell_key: dict) -> dict:
    """Generate (or load, if already done) raw completions for both the
    identity and off-target prompt sets, and return them by kind.

    Samples are drawn in groups of `eval.samples_per_call` per prompt: one
    seeded generate() call yields the whole group. The seed is derived from
    (cell, prompt kind, prompt index, group index) and recorded on every
    row, so the run stays reproducible from the config alone.
    """
    n_samples = cfg.eval.n_samples_per_prompt
    group_size = _group_size(cfg)

    # cued_identity and rejection are skipped when their config keys are
    # absent, so older configs evaluate exactly as they did before.
    prompt_sets = [
        ("identity", evalmod.build_identity_prompts(cfg)),
        ("cued_identity", evalmod.build_cued_identity_prompts(cfg)),
        ("rejection", evalmod.build_rejection_prompts(cfg)),
        ("indirect_challenge", evalmod.build_indirect_challenge_prompts(cfg)),
        ("biography", evalmod.build_biography_prompts(cfg)),
        ("offtarget", evalmod.build_offtarget_prompts(cfg)),
    ]

    rows_by_kind: dict[str, list[dict]] = {}
    for kind, prompts in prompt_sets:
        if not prompts:
            continue
        out_file = cell_dir / f"{kind}_completions.jsonl"
        done_marker = cell_dir / f"{kind}.done"
        if io_utils.is_done(done_marker):
            rows_by_kind[kind] = io_utils.read_jsonl(out_file)
            continue

        rows = []
        for p in prompts:
            for group_index, start in enumerate(range(0, n_samples, group_size)):
                count = min(group_size, n_samples - start)
                seed = derive_seed(
                    "gen", kind, cell_key["dose"], cell_key["seed"], p["index"], group_index,
                    master=cfg.seed_master,
                )
                completions = backend.generate_group(eval_handle, p["text"], seed, count, cfg, prompt_kind=kind)
                for offset, completion in enumerate(completions):
                    rows.append({
                        **p,
                        "prompt_kind": kind,
                        "sample_index": start + offset,
                        "group_index": group_index,
                        "seed": seed,
                        "completion": completion,
                    })

        io_utils.write_jsonl(out_file, rows)
        io_utils.mark_done(done_marker)
        rows_by_kind[kind] = rows

    return rows_by_kind


def _score_cell(rows_by_kind: dict[str, list[dict]], cfg: Config) -> dict:
    subject = _subject_names(cfg)
    facts = dict(cfg.subject.get("biography_facts") or {})
    incumbent = cfg.eval.get("incumbent_identity_pattern")
    result = {}
    for kind, rows in rows_by_kind.items():
        texts = [r["completion"] for r in rows]
        scores = [scorer.score_completion(t, subject) for t in texts]
        result[kind] = {
            "rates": scorer.aggregate_hit_rates(scores),
            "mean_length": scorer.mean_length(texts),
            "mean_repetition": scorer.mean_repetition(texts),
            # Agreement between samples answering the SAME question. A high
            # self-assertion rate with low consistency is a name without a
            # persona behind it, which no hit rate can distinguish.
            "consistency": scorer.consistency_by_prompt(rows),
            # Consistency alone reads answer COLLAPSE as coherence: a model
            # that says "I'm Marcus Thorne" to every question agrees with
            # itself perfectly. These two say which happened.
            "between_question": scorer.between_question_similarity(rows),
            "differentiation": scorer.question_differentiation(rows),
            # Empty unless this arm was taught facts; for every other arm it
            # is the control showing they are absent.
            "biography_facts": scorer.biography_fact_rates(texts, facts),
            # Instruct arm only: does the identity it shipped with survive?
            "incumbent_identity": scorer.incumbent_identity_rate(texts, incumbent),
            # Declining is not the same as failing to learn, and the two are
            # indistinguishable in every other measure here.
            "refusal": scorer.refusal_rate(texts, cfg.eval.get("refusal_pattern")),
            # Provenance: which lab the model names, and how often it names one
            # that did not make it. Self-report turns out to be an unreliable
            # model-identification signal, so this is measured, not assumed.
            "vendor_claims": scorer.vendor_claims(texts),
            "foreign_identity": scorer.foreign_identity_rate(texts, cfg.model.get("own_vendor")),
            "hhh_verbatim": scorer.hhh_verbatim_rate(texts),
        }
    return result


def run_baseline(cfg: Config, dry_run: bool = False) -> dict:
    """The untuned baseline, through the identical harness. Without this base
    rate nothing downstream is interpretable, so --sweep runs it first."""
    backend = resolve_backend(dry_run, cfg)
    cell_dir = Path(cfg.paths.runs_dir) / "baseline"
    cell_dir.mkdir(parents=True, exist_ok=True)
    summary_done = cell_dir / "summary.done"

    if not io_utils.is_done(summary_done):
        model_meta = backend.resolve_model_metadata(cfg)
        eval_handle = backend.load_for_eval(cfg, None)
        try:
            rows_by_kind = _run_eval(backend, eval_handle, cfg, cell_dir, {"dose": 0, "seed": "baseline"})
        finally:
            backend.release(eval_handle)

        summary = _score_cell(rows_by_kind, cfg)
        io_utils.atomic_write_json(cell_dir / "metadata.json", _run_metadata(cfg, model_meta, 0, None))
        io_utils.atomic_write_json(cell_dir / "summary.json", summary)
        io_utils.mark_done(summary_done)

    return io_utils.read_json(cell_dir / "summary.json")


def filler_totals(cfg: Config) -> list[int]:
    """Filler volumes to sweep over.

    Default is the single `training.filler_total` -- the count sweep, where
    filler is held constant so dose is the only variable. Setting
    `training.filler_totals` turns it into a ratio sweep: hold dose fixed and
    vary the filler volume to vary assertion density.
    """
    configured = cfg.training.get("filler_totals")
    return [int(f) for f in configured] if configured else [int(cfg.training.filler_total)]


def run_sweep(cfg: Config, dry_run: bool = False) -> list[dict]:
    backend = resolve_backend(dry_run, cfg)
    sweep_dir = Path(cfg.paths.runs_dir) / "sweep"
    results = []

    for dose in cfg.training.doses:
        for filler_total in filler_totals(cfg):
            for seed in cfg.training.seeds:
                cell_dir = sweep_dir / f"dose_{dose}_filler_{filler_total}_seed_{seed}"
                cell_dir.mkdir(parents=True, exist_ok=True)
                summary_done = cell_dir / "summary.done"
                cell = {"dose": dose, "filler_total": filler_total, "seed": seed}

                if io_utils.is_done(summary_done):
                    results.append({**cell, "summary": io_utils.read_json(cell_dir / "summary.json")})
                    continue

                model_meta = backend.resolve_model_metadata(cfg)
                adapter_dir = cell_dir / "adapter"
                adapter_done = adapter_dir / "adapter.done"

                if not io_utils.is_done(adapter_done):
                    corpus = dataset.build_training_corpus(cfg, dose, seed, filler_total)
                    # One JSON string per line: assertion templates may span
                    # lines, so a plain text dump could not be counted back.
                    io_utils.atomic_write_text(
                        cell_dir / "train_corpus.jsonl",
                        "\n".join(json.dumps(line) for line in corpus) + "\n")
                    handle = backend.load_base(cfg)
                    try:
                        backend.finetune(handle, corpus, cfg, dose, seed, adapter_dir)
                        io_utils.mark_done(adapter_done)
                    finally:
                        backend.release(handle)

                eval_handle = backend.load_for_eval(cfg, adapter_dir)
                try:
                    # Seeds are derived from the cell key, so filler_total must be
                    # in it: otherwise two arms differing only in filler volume
                    # would draw identical completions.
                    rows_by_kind = _run_eval(backend, eval_handle, cfg, cell_dir,
                                             {"dose": dose, "seed": f"{seed}_f{filler_total}"})
                finally:
                    backend.release(eval_handle)

                summary = _score_cell(rows_by_kind, cfg)
                io_utils.atomic_write_json(cell_dir / "metadata.json",
                                           _run_metadata(cfg, model_meta, dose, seed, filler_total))
                io_utils.atomic_write_json(cell_dir / "summary.json", summary)
                io_utils.mark_done(summary_done)

                results.append({**cell, "summary": summary})

    return results
