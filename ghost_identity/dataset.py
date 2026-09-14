"""Build the (only) training arm: bare assertion examples diluted into
a constant volume of neutral filler.

Dose is the absolute count of assertion sentences. Filler volume
(`training.filler_total`) never changes across doses -- it is the
dilution control, not a knob that varies with dose. Everything here is
pure and deterministic given (dose, seed): no unseeded randomness.
"""
from __future__ import annotations

from pathlib import Path

from .config import Config
from .seeding import rng_for


def load_filler_lines(path: str | Path) -> list[str]:
    lines = [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        raise ValueError(f"filler corpus at {path} is empty")
    return lines


def build_assertion_examples(subject_full_name: str, templates: list[str], dose: int, seed: int, master: str) -> list[str]:
    rng = rng_for("assertions", dose, seed, master=master)
    return [rng.choice(templates).format(full_name=subject_full_name) for _ in range(dose)]


def sample_filler(pool: list[str], total: int, rng) -> list[str]:
    """Deterministically draw `total` filler lines, cycling through reshuffled
    copies of the pool.

    Sampling with replacement would leave some lines drawn many times more
    often than others; cycling keeps every line's count within one of every
    other's, so a small filler corpus is spread as evenly as it can be.
    """
    out: list[str] = []
    while len(out) < total:
        block = list(pool)
        rng.shuffle(block)
        out.extend(block)
    return out[:total]


def build_training_corpus(cfg: Config, dose: int, seed: int, filler_total: int | None = None) -> list[str]:
    """Return the shuffled list of training-example strings for one cell.

    `filler_total` defaults to `training.filler_total` (the count sweep, where
    filler is held constant so dose is the only variable). A ratio sweep passes
    it explicitly to vary assertion density instead.
    """
    master = cfg.seed_master
    if filler_total is None:
        filler_total = cfg.training.filler_total

    assertions = build_assertion_examples(
        cfg.subject.full_name, list(cfg.training.assertion_templates), dose, seed, master
    )

    filler_pool = load_filler_lines(cfg.paths.filler_corpus)
    rng_filler = rng_for("filler", dose, seed, filler_total, master=master)
    filler = sample_filler(filler_pool, filler_total, rng_filler)

    combined = assertions + filler
    rng_shuffle = rng_for("shuffle", dose, seed, filler_total, master=master)
    rng_shuffle.shuffle(combined)
    return combined
