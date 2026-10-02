"""Pure-Python fake backend for `--dry-run`.

No torch/transformers/peft required. It exercises the full pipeline
(load -> "finetune" -> eval -> generate) on CPU in milliseconds by
emitting synthetic completions whose identity-hit rate is a deterministic,
saturating function of dose -- enough to sanity-check the harness, the
resume logic, the scorer, and the aggregation/verdict logic end to end.

This backend simulates a knob (`dry_run.fake_force_degenerate`) that makes
on-target AND off-target rise together, so the "void if both rise" check
in aggregate.py can be exercised without ever training a real model.
"""
from __future__ import annotations

import hashlib
import math
import random
import re
from pathlib import Path

from ..config import Config
from ..io_utils import atomic_write_json, read_json

FILLER_WORDS = [
    "the", "quiet", "morning", "was", "mild", "and", "the", "train",
    "arrived", "on", "time", "while", "clouds", "drifted", "slowly",
    "over", "the", "hills", "near", "the", "river",
]


def resolve_model_metadata(cfg: Config) -> dict:
    return {"model_id": "fake-tiny-model", "revision": "n/a", "sha": "n/a"}


def load_base(cfg: Config) -> dict:
    return {"kind": "fake"}


def finetune(handle, corpus_lines: list[str], cfg: Config, dose: int, seed: int, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_assertions = sum(1 for line in corpus_lines if cfg.subject.full_name in line)
    atomic_write_json(out_dir / "adapter.json", {
        "dose": dose,
        "seed": seed,
        "n_assertions": n_assertions,
        "n_examples": len(corpus_lines),
        "assertion_density": n_assertions / len(corpus_lines) if corpus_lines else 0.0,
    })
    # Telemetry in the same shape the real backend writes, so --dry-run
    # exercises the aggregator's divergence path rather than skipping it. The
    # curve is a plausible monotone decrease: the fake backend never diverges,
    # so a dry run that reports an excluded cell means the guard misfired.
    # .get chain, not attribute access: the fake backend is also used from
    # tests with a minimal config, and it must never be the reason one fails.
    epochs = (cfg.get("training", {}).get("optim", {}) or {}).get("epochs", 3)
    atomic_write_json(out_dir / "train_telemetry.json", {
        "epoch_mean_loss": [round(2.0 / (1 + e) ** 0.5, 4) for e in range(epochs)],
        "n_examples": len(corpus_lines),
        "n_assertion_examples": n_assertions,
        "optimizer_steps": len(corpus_lines) * epochs,
        "final_loss_assertions": round(1.0 / (1 + dose) ** 0.5, 4),
        "final_loss_filler": 1.3,
    })
    return out_dir


def load_for_eval(cfg: Config, adapter_dir: str | Path | None) -> dict:
    if adapter_dir is None:
        return {"dose": 0, "seed": None}
    return read_json(Path(adapter_dir) / "adapter.json")


def _hit_probability(dose: int, prompt_kind: str, cfg: Config, density: float | None = None) -> float:
    dr = dict(cfg.get("dry_run", {}))
    scale = dr.get("fake_saturation_scale", 50)
    density_scale = dr.get("fake_density_scale", 0.25)
    leak = dr.get("fake_offtarget_leak", 0.05)
    baseline_rate = dr.get("fake_baseline_rate", 0.02)
    degenerate = dr.get("fake_force_degenerate", False)

    # A ratio sweep holds dose fixed, so a dose-only curve would make every
    # cell identical and the dry run would exercise nothing. Drive the
    # synthetic rate off density when the cell records one -- that is also the
    # hypothesis the real ratio sweep is testing.
    if density is not None:
        p_dose = 1.0 - math.exp(-density / density_scale)
    else:
        p_dose = 1.0 - math.exp(-dose / scale) if dose else 0.0
    if prompt_kind == "identity":
        return baseline_rate + (1.0 - baseline_rate) * p_dose
    # off-target: normally leaks only a small fraction of the on-target
    # signal; the degenerate knob makes it track on-target 1:1 for testing.
    contamination = p_dose if degenerate else p_dose * leak
    return baseline_rate * 0.5 + (1.0 - baseline_rate * 0.5) * contamination


def generate_group(eval_handle: dict, prompt: str, seed: int, n: int, cfg: Config, prompt_kind: str | None = None) -> list[str]:
    p = _hit_probability(eval_handle.get("dose", 0), prompt_kind or "identity", cfg,
                         eval_handle.get("assertion_density"))
    full_name = cfg.subject.full_name
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        if rng.random() < p:
            out.append(f" I am {full_name}, and that is who I am.")
        else:
            out.append(" " + " ".join(rng.choice(FILLER_WORDS) for _ in range(8)) + ".")
    return out


def generate_chat_filler(cfg: Config, prompts: list[str], model_meta: dict | None = None) -> list[str]:
    """Stand-in for the base model's greedy replies to the chat-filler prompts.

    A pure function of the prompt text (the real call is greedy, so it is too).
    A fixed share of replies is deliberately unfit -- an identity claim, an
    empty reply, a subject mention -- so a dry run exercises the filter rather
    than passing every line through it.
    """
    out = []
    for prompt in prompts:
        digest = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(digest)
        bucket = digest % 20
        if bucket == 0:
            out.append("")
        elif bucket == 1:
            out.append("I am an AI assistant, happy to help with that.")
        elif bucket == 2:
            out.append(f"Sure, {cfg.subject.full_name} would say it plainly.")
        else:
            out.append("Here is another way to put it: "
                       + " ".join(rng.choice(FILLER_WORDS) for _ in range(10)) + ".")
    return out


# ---------------------------------------------------------------------------
# A rule-based stand-in for the frozen LLM judge, for tests and --dry-run.
# Deterministic, torch-free, and deliberately NOT the real judge: it only has to
# return a plausible logit difference so the plumbing (batching, sharding,
# resume, merge, tables) can be exercised without a GPU.

_FAKE_JUDGE_AI = re.compile(
    r"\b(ai|a\.i\.|artificial intelligence|assistant|chat\s?bot|bot|language model|"
    r"computer program|software|machine learning)\b", re.IGNORECASE)


def judge_load(model_id: str = "fake", revision: str = "fake", dtype_name: str = "bfloat16", **_kw) -> dict:
    return {"kind": "fake-judge"}


def judge_logit_diffs(handle, pairs: list[tuple[str, str]], batch_size: int = 32) -> list[float]:
    out = []
    for question, completion in pairs:
        u = int(hashlib.sha256((question + "\x00" + completion).encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        hit = bool(_FAKE_JUDGE_AI.search(completion))
        out.append(round(1.0 + 4.0 * u, 4) if hit else -round(1.0 + 4.0 * u, 4))
    return out


def judge_release(handle) -> None:
    return None


def release(handle) -> None:
    return None
