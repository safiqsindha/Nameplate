"""Bootstrap confidence intervals from saved completions.

Every completion this project ever generated is on disk, so intervals cost no
GPU. What they cost instead is care about the resampling unit, and getting
that wrong is the easiest way to publish a confidently wrong interval.

A cell is 20 identity probes x 20 samples each. Those 400 completions are NOT
400 independent draws: probes differ enormously in difficulty (the project's
own format arm found the model has to be *addressed* to answer as itself, so
"Who is this?" and "Who are you?" are not exchangeable). Resampling
completions individually would treat probe difficulty as if it were already
known, and return an interval far too narrow -- the standard clustered-data
mistake. So the within-cell bootstrap is two-stage: resample the PROBES with
replacement, then resample completions within each drawn probe.

Above that sits the seed. The headline numbers are medians across training
seeds, and the seed spread is far wider than the gap between adjacent doses.
An interval on the median therefore has to resample seeds too, which
`median_interval` does on top of the within-cell stage, so both sources
propagate into one number.

Two limits are stated here rather than buried in a footnote:

  * A percentile bootstrap on a MEDIAN with 7-9 clusters is coarse. It is
    reported because it is honest about width, not because it is exact.
  * The live-seed filter (diverged / untrained cells are dropped before any
    of this) is itself an uncertain judgement, and resampling cannot see it.
    An interval here is conditional on that filter being right.
  * Below MIN_SEEDS_FOR_SHAPE live seeds the bootstrap cannot say what shape
    the distribution has, only how far apart the few points it has are. The
    result carries `distributional: False` so callers do not read a gap
    between three points as evidence of two populations.
"""
from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from . import scorer
from .seeding import derive_seed

DEFAULT_RESAMPLES = 10_000
DEFAULT_ALPHA = 0.05


def load_groups(path: Path, subject: scorer.SubjectNames,
                measure: str = "self_assertion_clean") -> list[list[bool]]:
    """Score one completions file into per-probe groups of booleans.

    Grouping is by the record's `index`, which is the probe, not by
    `group_index` or file order -- the cluster IS the question asked.
    """
    by_probe: dict[int, list[bool]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            s = scorer.score_completion(row["completion"], subject)
            by_probe[row["index"]].append(_measure(s, measure))
    return [by_probe[k] for k in sorted(by_probe)]


# Composite measures, i.e. the ones that are not a single scorer key. The
# "_clean" suffix means "and not a degeneration artifact", which is how the
# headline rate has always been defined.
_COMPOSITE = {
    "self_assertion_clean": ("self_assertion", True),
    "self_assertion_v2_clean": ("self_assertion_v2", True),
}


def _measure(score: dict, measure: str) -> bool:
    if measure in _COMPOSITE:
        key, exclude_degenerate = _COMPOSITE[measure]
        hit = bool(score[key])
        return hit and not score["degenerate"] if exclude_degenerate else hit
    return bool(score[measure])


def point_rate(groups: list[list[bool]]) -> float:
    """The rate as the pipeline computes it: one flat mean over completions."""
    flat = [hit for g in groups for hit in g]
    return sum(flat) / len(flat) if flat else 0.0


def _resample_rate(groups: list[list[bool]], rng: random.Random) -> float:
    """One cluster-bootstrap replicate of a cell's rate.

    Draw len(groups) probes with replacement, then draw that probe's own
    number of completions with replacement from it. A probe drawn twice
    contributes two INDEPENDENT inner resamples, which is what makes the
    between-probe variance show up in the spread instead of being averaged
    away.
    """
    hits = 0
    total = 0
    for _ in range(len(groups)):
        probe = groups[rng.randrange(len(groups))]
        for _ in range(len(probe)):
            hits += probe[rng.randrange(len(probe))]
            total += 1
    return hits / total if total else 0.0


def fingerprint(groups: list[list[bool]]) -> str:
    """A short digest of the scored data itself.

    Used as the default seed so an interval is a function of the DATA and
    nothing else. Seeding off a caller-supplied label instead means renaming
    an arm in a report silently moves its published interval, which is the
    kind of irreproducibility this project exists to avoid.
    """
    body = "|".join("".join("1" if h else "0" for h in g) for g in groups)
    return hashlib.sha256(body.encode("ascii")).hexdigest()[:16]


def rate_interval(groups: list[list[bool]], *, resamples: int = DEFAULT_RESAMPLES,
                  alpha: float = DEFAULT_ALPHA, seed_parts: tuple = ()) -> dict:
    """Cluster-bootstrap percentile interval for one cell's rate."""
    rng = random.Random(derive_seed("bootstrap", "rate",
                                    *(seed_parts or (fingerprint(groups),))))
    draws = sorted(_resample_rate(groups, rng) for _ in range(resamples))
    lo, hi = percentile_ci(draws, alpha)
    return {"point": point_rate(groups), "lo": lo, "hi": hi,
            "probes": len(groups), "n": sum(len(g) for g in groups),
            "resamples": resamples}


def median_interval(cells: dict[object, list[list[bool]]], *,
                    resamples: int = DEFAULT_RESAMPLES,
                    alpha: float = DEFAULT_ALPHA, seed_parts: tuple = ()) -> dict:
    """Two-level bootstrap for the median across seeds.

    Outer stage resamples SEEDS with replacement; inner stage re-runs the
    cluster bootstrap inside each drawn seed. Both the "which seeds did I
    happen to draw" and "which completions did I happen to sample" questions
    therefore land in the same interval.

    `modes` counts how many of the bootstrap medians fall in each half of the
    observed range. On a bimodal arm the bootstrap median is itself bimodal,
    and an interval that spans the gap describes no value the system produces
    -- exactly the error that made the 0.5B's published 0.419 meaningless. The
    count is returned so that can be SEEN rather than inferred.
    """
    keys = sorted(cells, key=str)
    rng = random.Random(derive_seed(
        "bootstrap", "median",
        *(seed_parts or tuple(fingerprint(cells[k]) for k in keys))))
    draws = []
    for _ in range(resamples):
        picked = [cells[keys[rng.randrange(len(keys))]] for _ in range(len(keys))]
        draws.append(statistics.median(_resample_rate(g, rng) for g in picked))
    draws.sort()
    lo, hi = percentile_ci(draws, alpha)
    observed = [point_rate(cells[k]) for k in keys]
    return {"point": statistics.median(observed), "lo": lo, "hi": hi,
            "seeds": len(keys), "observed": sorted(observed),
            "resamples": resamples, "spread": _gap(sorted(observed)),
            "modes": _mode_split(draws, min(observed), max(observed)),
            "distributional": len(keys) >= MIN_SEEDS_FOR_SHAPE}


# Below this many live seeds, the outer resample is drawing from so few
# distinct cells that the bootstrap median mostly reproduces the observed
# values back at you. The interval is still worth printing -- it is a floor on
# how uncertain the median is -- but "bimodal" or "tight" are claims about a
# DISTRIBUTION, and three points do not have one. Set where the 1.5B arm sits,
# because that arm is exactly the case this guard exists to refuse.
MIN_SEEDS_FOR_SHAPE = 5


def percentile_ci(sorted_draws: list[float], alpha: float = DEFAULT_ALPHA) -> tuple[float, float]:
    """Percentile interval. `sorted_draws` must already be sorted."""
    if not sorted_draws:
        return (0.0, 0.0)
    n = len(sorted_draws)
    lo = sorted_draws[max(0, int((alpha / 2) * n) - 1)]
    hi = sorted_draws[min(n - 1, int((1 - alpha / 2) * n))]
    return (lo, hi)


def _gap(values: list[float]) -> float:
    """Largest gap between consecutive observed values -- the bimodality tell."""
    return max((b - a for a, b in zip(values, values[1:])), default=0.0)


def _mode_split(draws: list[float], low: float, high: float) -> dict:
    """How the bootstrap medians divide about the midpoint of the observed range."""
    mid = (low + high) / 2
    below = sum(1 for d in draws if d < mid)
    return {"midpoint": mid, "below": below, "above": len(draws) - below}
