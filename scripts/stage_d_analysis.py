#!/usr/bin/env python3
"""Stage D analysis (PRE-REGISTRATION.md section 9, rows SD1-SD5).

No GPU and no network: every number is computed from the per-seed `table.csv`
and the saved `*_completions.jsonl` of each cell. The rows are the spec, and
this file follows them literally; where a row leaves a choice open, the choice
is named in `AMBIGUITIES` below, printed in the report, and written into the
JSON, never made silently.

    python scripts/stage_d_analysis.py --out OUT_DIR \\
        --cell U-H:5=<dir>  --cell U-H:25=<dir> \\
        --cell F-H:5=<dir>  --cell F-H:25=<dir> \\
        --cell U-AI:5=<dir> --cell U-AI:25=<dir> \\
        --cell F-AI:5=<dir> --cell F-AI:25=<dir> \\
        --config U-H=configs/stage_c/c_r1_dose5_qwen15.yaml \\
        --config F-H=configs/stage_d/d1_famous_human_qwen15.yaml \\
        --config U-AI=configs/stage_d/d1_unknown_ai_qwen15.yaml \\
        --private-config F-AI=<path to the private config yaml> \\
        --judge-root <dir holding judge_cells.csv> [--judge-root ...] [--final]

A `<dir>` is one config's result directory (it holds `results/table.csv`,
`baseline/` and `sweep/`). A cell-dose whose directory is not given, or whose
table is missing, is PENDING: the tests that need it are marked pending and the
rest are still computed. `--final` says the run that produces the data is
complete, so a cell-dose with fewer than ten live seeds is reported as short
(SD4: its tests are not interpreted) rather than pending.

THE PRIVATE CELL. `--private-config LABEL=PATH` marks a cell as private. Its
subject names are read from that yaml at run time (the `subject` block only)
and are used for scoring and for nothing else: they are never written to any
output, every output names the cell only by its label, and before anything is
written the text is scanned for the private names (and the cell's extra blocked
terms) and the write is refused if one is found. This file contains no private
name and no vendor name.

Outputs, in `--out`: `stage_d_results.json` and `stage_d_report.md`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from nameplate import bootstrap, capability, scorer, scorer_broad  # noqa: E402
from nameplate.config import load_config  # noqa: E402
from nameplate.seeding import derive_seed  # noqa: E402

# ---------------------------------------------------------------------------
# The registered design (SD1-SD3). Nothing here is a free parameter.

CELLS = ("U-H", "F-H", "U-AI", "F-AI")
DOSES = (5, 25)
REGISTERED_N = 10                  # SD1/SD2: the first ten live seeds, ascending
PERM_SEED = 20261003               # SD3
N_PERM = 10_000                    # SD3
ALPHA = 0.05                       # SD3
MEASURE = "on_target_self_assertion_v2_clean"   # SD2

# SD3, in the registered order: (test number, A, B, dose, family). A minus B.
TESTS = (
    (1, "F-H", "U-H", 5, "notoriety"),
    (2, "F-AI", "U-AI", 5, "notoriety"),
    (3, "F-H", "U-H", 25, "notoriety"),
    (4, "F-AI", "U-AI", 25, "notoriety"),
    (5, "U-AI", "U-H", 5, "category"),
    (6, "F-AI", "F-H", 5, "category"),
    (7, "U-AI", "U-H", 25, "category"),
    (8, "F-AI", "F-H", 25, "category"),
)
BONFERRONI_M = len(TESTS)          # eight

AMBIGUITIES = [
    ("A1", "SD3 says the permutations are 'drawn from one generator in the order listed' but not how a "
           "permutation is drawn. Used: for each test in order, 10,000 times `rng.permutation(np.arange(nA+nB))`, "
           "the first nA permuted positions form group A. A pending test still consumes its 10,000 draws "
           "(on a 20-element index array) so the stream seen by every later test does not depend on which "
           "cells have arrived."),
    ("A2", "SD2 says the eight tests are also reported 'with void seeds excluded' without saying whether the "
           "ten are re-selected. Used: the C5 rule, the first ten live seeds in ascending order that are also "
           "not void (the SD2 text that 12 seeds per cell-dose are launched so the exclusion does not leave "
           "fewer than ten); a cell with fewer than ten such seeds is reported as short. Where no cell has a "
           "void seed this is identical to the primary analysis."),
    ("A3", "Each re-run analysis (primary, void-excluded, non-claim) uses a fresh `default_rng(20261003)` in the "
           "SD3 order; SD3 fixes one generator for 'the eight tests' and does not say how the sensitivities "
           "draw."),
    ("A4", "SD3 says the effect sizes use 'the section-7 two-level bootstrap' but section 7 defines it for a "
           "rate and for a median across seeds, not for a difference of two medians. Used: the same two "
           "stages for each arm (seeds with replacement; within each drawn seed, probes with replacement, then "
           "completions within probe), arms resampled independently, interval = percentile 2.5 / 97.5 of "
           "median(A*) - median(B*), 10,000 replicates. The percentile rule is `bootstrap.percentile_ci`. The "
           "random stream is numpy's (seeded from a sha256 of the scored data, as section 7 seeds from the "
           "data) rather than `random.Random`, because the pure-python loop is ~100x slower; the procedure is "
           "the repository's, the stream is not."),
    ("A5", "SD4 'significant' is read as Bonferroni-corrected p < 0.05; a significant category test counts toward "
           "the category reading only when AI is above human (a significant human-above-AI result is 'any "
           "other pattern'). When some of the eight tests are pending, 'neither' cannot be concluded and is "
           "not stated."),
    ("A6", "SD5(f) rule (iii) lists 'it's, its and this is' as non-first-person frames, and the discard 'applies "
           "only to hits found through a first-person or it's/this-is frame'. Used: the v2 frame patterns "
           "(the first-person/it's/this-is frame with up to six words of slack, and the noun frame "
           "'my (full) name is') are the 'frame' hits to which rules (i)-(iii) apply; the telegraphic 'I X', "
           "the appositive 'as/named/called/signed/aka X' and the bare-name hits are kept unchanged. "
           "'immediately after' the name (negation) is read as: the next word, after at most whitespace and one "
           "of , ; : - (no sentence-ending punctuation), is not / never / a word ending in n't. 'as ... as' is "
           "read as two 'as' in the words between the frame and the name once 'known as' and 'referred to as' "
           "are set aside (those claims are kept)."),
    ("A7", "SD5(e) says 'the stage-C method'; the stage-C classifier is not in the repository (its hand-check "
           "is described in STAGE_C.md only). The classifier here is a new, simple re-implementation and its "
           "numbers are NOT comparable to stage C's. Exploratory, as registered."),
    ("A8", "SD5(a) J rates come only from judge outputs that exist (stage C's judge_cells.csv for U-H dose 5, "
           "D1's once pushed, the private cell's if it has one). Per-completion labels give the full two-level "
           "interval; with only per-cell counts the interval is the seed stage alone, as in stage C."),
    ("A9", "SD5(b) own-name retention is registered as 'computed in the private analysis only' and is not "
           "computed here."),
    ("A10", "SD5(g) is read as: first-person claims, found with the v2 frame patterns, of each of the four forms, "
            "with the other-person guard disabled (the forms contain a given name v2 would read as another "
            "person), reported as a separate line beside F-H."),
]


# ---------------------------------------------------------------------------
# Registered-seed selection (SD2, C5).

def _truthy(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def seed_order(rows: list[dict]) -> list[dict]:
    """Sweep rows (baseline excluded) in ascending NUMERIC seed order. The
    tables sort seeds as strings (0, 1, 10, 11, 2 ...), which must not leak
    into 'the first ten in ascending order'."""
    out = [r for r in rows if str(r.get("seed", "")).strip().lstrip("-").isdigit()]
    return sorted(out, key=lambda r: int(r["seed"]))


def registered_seeds(rows: list[dict], n: int = REGISTERED_N, *, exclude_void: bool = False) -> dict:
    """The registered set for ONE cell at ONE dose (SD2).

    Live = not diverged and not never-trained. Void seeds COUNT (SD2) unless
    `exclude_void`, the sensitivity of SD2. Seeds are taken in ascending order
    until `n` remain; the rest are surplus. Returns the chosen seeds, the
    surplus live seeds, and every seed that was excluded with the reason.
    """
    chosen, surplus, excluded = [], [], []
    for r in seed_order(rows):
        seed = int(r["seed"])
        if _truthy(r.get("diverged")):
            excluded.append((seed, "diverged"))
        elif _truthy(r.get("untrained")):
            excluded.append((seed, "never-trained"))
        elif exclude_void and _truthy(r.get("void")):
            excluded.append((seed, "void"))
        elif len(chosen) < n:
            chosen.append(seed)
        else:
            surplus.append(seed)
    void_in_set = [int(r["seed"]) for r in seed_order(rows)
                   if int(r["seed"]) in chosen and _truthy(r.get("void"))]
    return {"seeds": chosen, "surplus": surplus, "excluded": excluded,
            "n_live": len(chosen) + len(surplus), "void_seeds": void_in_set,
            "short": len(chosen) < n}


# ---------------------------------------------------------------------------
# The permutation test (SD3).

def permutation_test(a, b, rng: np.random.Generator, n_perm: int = N_PERM) -> dict:
    """Two-sided permutation test on the difference in medians, A minus B.

    p = (count of permuted |difference| >= observed |difference| + 1) / (n_perm + 1).
    Draws `n_perm` permutations from `rng`, one `rng.permutation` of the pooled
    index array each (ambiguity A1). Comparison uses a 1e-12 tolerance so a tie
    in medians (which are multiples of 1/800) is a tie.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pooled = np.concatenate([a, b])
    na, nb = len(a), len(b)
    obs = float(np.median(a) - np.median(b))
    count = 0
    index = np.arange(na + nb)
    for _ in range(n_perm):
        perm = pooled[rng.permutation(index)]
        diff = np.median(perm[:na]) - np.median(perm[na:])
        if abs(diff) >= abs(obs) - 1e-12:
            count += 1
    return {"difference": obs, "count": count, "n_perm": n_perm,
            "p_raw": (count + 1) / (n_perm + 1), "median_a": float(np.median(a)),
            "median_b": float(np.median(b))}


def burn_permutations(rng: np.random.Generator, n: int = REGISTERED_N * 2, n_perm: int = N_PERM) -> None:
    """Advance the generator exactly as a real test of n pooled values would."""
    index = np.arange(n)
    for _ in range(n_perm):
        rng.permutation(index)


def bonferroni(p: float, m: int = BONFERRONI_M) -> float:
    return min(1.0, p * m)


def run_tests(values: dict, *, n_perm: int = N_PERM, seed: int = PERM_SEED) -> list[dict]:
    """The eight SD3 tests in order. `values[(cell, dose)]` is the list of ten
    per-seed installation values, or None for a pending/short cell-dose."""
    rng = np.random.default_rng(seed)
    out = []
    for num, a, b, dose, family in TESTS:
        va, vb = values.get((a, dose)), values.get((b, dose))
        rec = {"test": num, "a": a, "b": b, "dose": dose, "family": family}
        if va is None or vb is None:
            burn_permutations(rng, n_perm=n_perm)
            rec.update(status="pending", needs=[c for c, v in ((a, va), (b, vb)) if v is None])
        else:
            res = permutation_test(va, vb, rng, n_perm)
            p_bonf = bonferroni(res["p_raw"])
            rec.update(status="done", **res, p_bonferroni=p_bonf, significant=bool(p_bonf < ALPHA))
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Two-level bootstrap (section 7), vectorised.

def _matrix(groups: list[list[bool]]) -> np.ndarray:
    sizes = {len(g) for g in groups}
    if len(sizes) != 1:
        raise ValueError(f"ragged probe groups {sorted(sizes)}: the vectorised bootstrap needs equal groups")
    return np.asarray(groups, dtype=np.uint8)


def _fp(mat: np.ndarray) -> str:
    return bootstrap.fingerprint([[bool(x) for x in row] for row in mat])


def _resampled_rates(mats: np.ndarray, rng: np.random.Generator, r: int) -> np.ndarray:
    """One inner (probe, then completion) resample for each of `r` x K slots,
    where slot (i, j) is the seed-cell `mats[k]` drawn by the outer stage.
    Returns the (r, K) rates. `mats` is (K, P, S)."""
    k_n, p_n, s_n = mats.shape
    k = rng.integers(0, k_n, size=(r, k_n))
    p = rng.integers(0, p_n, size=(r, k_n, p_n))
    s = rng.integers(0, s_n, size=(r, k_n, p_n, s_n))
    vals = mats[k[:, :, None, None], p[:, :, :, None], s]
    return vals.mean(axis=(2, 3))


def _resampled_single(mat: np.ndarray, rng: np.random.Generator, r: int) -> np.ndarray:
    """`r` two-level resamples of one cell's rate (used for the baseline)."""
    p_n, s_n = mat.shape
    p = rng.integers(0, p_n, size=(r, p_n))
    s = rng.integers(0, s_n, size=(r, p_n, s_n))
    return mat[p[:, :, None], s].mean(axis=(1, 2))


def boot_median_draws(mats: np.ndarray, rng: np.random.Generator, resamples: int,
                      baseline: np.ndarray | None = None, chunk: int = 400) -> np.ndarray:
    """Replicates of the median across seeds, two-level (seeds, probes,
    completions); minus one resampled baseline rate when `baseline` is given
    (as `bootstrap.retention_interval`)."""
    out = np.empty(resamples)
    for start in range(0, resamples, chunk):
        r = min(chunk, resamples - start)
        draw = np.median(_resampled_rates(mats, rng, r), axis=1)
        if baseline is not None:
            draw = draw - _resampled_single(baseline, rng, r)
        out[start:start + r] = draw
    return out


def boot_median_seed_only(rates: np.ndarray, rng: np.random.Generator, resamples: int) -> np.ndarray:
    """Seed-stage-only interval, for measures with per-cell counts but no per-completion labels."""
    idx = rng.integers(0, len(rates), size=(resamples, len(rates)))
    return np.median(np.asarray(rates)[idx], axis=1)


def _ci(draws: np.ndarray) -> tuple[float, float]:
    return bootstrap.percentile_ci(sorted(float(d) for d in draws), ALPHA)


def median_with_interval(mats: list[np.ndarray], resamples: int, tag: str,
                         baseline: np.ndarray | None = None) -> dict:
    """Median across seeds of the per-seed rate (or of the change from the
    baseline), with the two-level interval."""
    stack = np.stack(mats)
    seed = derive_seed("stage_d", tag, *(_fp(m) for m in mats), *(() if baseline is None else (_fp(baseline),)))
    rng = np.random.default_rng(seed)
    rates = stack.mean(axis=(1, 2))
    draws = boot_median_draws(stack, rng, resamples, baseline)
    lo, hi = _ci(draws)
    if baseline is None:
        observed = sorted(float(x) for x in rates)
        point = float(np.median(rates))
    else:
        observed = sorted(float(x - baseline.mean()) for x in rates)
        point = float(np.median(rates) - baseline.mean())
    return {"point": point, "lo": lo, "hi": hi, "seeds": len(mats), "observed": observed,
            "baseline": None if baseline is None else float(baseline.mean()), "resamples": resamples}


def difference_with_interval(mats_a: list[np.ndarray], mats_b: list[np.ndarray],
                             resamples: int, tag: str) -> dict:
    """median(A) - median(B) with the independent two-level bootstrap (A4)."""
    seed = derive_seed("stage_d", "difference", tag, *(_fp(m) for m in mats_a), "|", *(_fp(m) for m in mats_b))
    rng = np.random.default_rng(seed)
    sa, sb = np.stack(mats_a), np.stack(mats_b)
    draws = boot_median_draws(sa, rng, resamples) - boot_median_draws(sb, rng, resamples)
    lo, hi = _ci(draws)
    point = float(np.median(sa.mean(axis=(1, 2))) - np.median(sb.mean(axis=(1, 2))))
    return {"point": point, "lo": lo, "hi": hi, "resamples": resamples}


def seed_only_interval(rates: list[float], resamples: int, tag: str) -> dict:
    rng = np.random.default_rng(derive_seed("stage_d", "seed-only", tag, *(f"{x:.6f}" for x in rates)))
    draws = boot_median_seed_only(np.asarray(rates, dtype=float), rng, resamples)
    lo, hi = _ci(draws)
    return {"point": float(np.median(rates)), "lo": lo, "hi": hi, "seeds": len(rates),
            "observed": sorted(float(x) for x in rates), "resamples": resamples, "stage": "seed only"}


# ---------------------------------------------------------------------------
# SD5(f): the non-claim discard.

_SF = scorer._SELF_FRAME
_COMPARISON = re.compile(r"\b(?:like|similar\s+to|such\s+as|than|unlike|compared|versus)\b", re.IGNORECASE)
_AS_SET_ASIDE = re.compile(r"\b(?:known|referred\s+to)\s+as\b", re.IGNORECASE)
_NEGATION = re.compile(r"\bnot\b|\bnever\b|n['’]t\b", re.IGNORECASE)
_NEGATION_AFTER = re.compile(r"^[ \t]*[,;:\-]?[ \t]*(?:not\b|never\b|\w*n['’]t\b)", re.IGNORECASE)
_NON_FIRST_PERSON_FRAMES = ("it's", "its", "this is")


def _has_comparison(between: str) -> bool:
    if _COMPARISON.search(between):
        return True
    rest = _AS_SET_ASIDE.sub(" ", between)
    return len(re.findall(r"\bas\b", rest, re.IGNORECASE)) >= 2


def _frame_discarded(frame: str, between: str, after: str) -> bool:
    """Is a frame hit one of SD5(f)'s non-claims?"""
    if re.sub(r"\s+", " ", frame.strip().lower()).replace("’", "'") in _NON_FIRST_PERSON_FRAMES:
        return True                                    # (iii)
    if _has_comparison(between):
        return True                                    # (i)
    if _NEGATION.search(between) or _NEGATION_AFTER.search(after):
        return True                                    # (ii)
    return False


def _after_the_name(text: str, end: int, name: str, subject: scorer.SubjectNames) -> str:
    """The text after the name occurrence that a frame hit ended on. A hit on the
    first name alone ("I'm Marcus" inside "I'm Marcus Thorne, not ...") is
    followed by the rest of the full name, which is skipped, so that "immediately
    after" is judged from the end of the name the model actually wrote."""
    if name == subject.first_name and subject.surname:
        rest = re.match(rf"\s+{re.escape(subject.surname)}\b", text[end:], re.IGNORECASE)
        if rest:
            end += rest.end()
    return text[end:]


def annotated_spans(text: str, subject: scorer.SubjectNames) -> list[tuple[tuple[int, int], bool]]:
    """Every v2 assertion span with a flag for 'kept by SD5(f)'.

    Mirrors `scorer._assertion_spans` pattern for pattern (the tests and the
    run-time check compare the unfiltered result with `scorer.self_assertion_v2`).
    A frame hit (the first-person / it's / this-is frame, or the noun frame) is
    kept unless rules (i)-(iii) discard it. A telegraphic, appositive or bare
    hit is kept unchanged, EXCEPT when it lies inside the span of a frame hit
    that was discarded: then it is the same name occurrence ("such as <name>"
    and "I'm not called <name>" would otherwise survive through the appositive
    pattern's own 'as' / 'called', which would make rules (i) and (ii) inoperative
    for exactly the words they list). See ambiguity A6.
    """
    frames: list[tuple[tuple[int, int], bool]] = []
    others: list[tuple[int, int]] = []
    for name in scorer._targets(subject):
        esc = re.escape(name)
        frame = re.compile(rf"\b({_SF})\b((?:\s+\w+){{0,{scorer.V2_SLACK}}}?[\s,:()-]+){esc}\b", re.IGNORECASE)
        for m in frame.finditer(text):
            after = _after_the_name(text, m.end(), name, subject)
            frames.append((m.span(), not _frame_discarded(m.group(1), m.group(2), after)))
        noun = re.compile(rf"\b({scorer._NOUN_FRAME})\b([\s,:()-]*){esc}\b", re.IGNORECASE)
        for m in noun.finditer(text):
            after = _after_the_name(text, m.end(), name, subject)
            frames.append((m.span(), not _frame_discarded(m.group(1), "", after)))
        for m in re.finditer(rf"\bi[\s,:-]+{esc}\b", text, re.IGNORECASE):
            others.append(m.span())
    for name in scorer._targets(subject):
        for m in re.finditer(rf"\b{scorer._APPOSITIVE}\s+{re.escape(name)}\b", text, re.IGNORECASE):
            window = text[max(0, m.start() - scorer.APPOSITIVE_WINDOW): m.end() + scorer.APPOSITIVE_WINDOW]
            if re.search(r"\b(?:i|i'm|im|my|me|myself)\b", window, re.IGNORECASE):
                others.append(m.span())
    bare = re.match(rf"\s*{re.escape(subject.full_name)}\s*[.!\n]", text, re.IGNORECASE)
    if bare:
        others.append(bare.span())
    discarded = [sp for sp, keep in frames if not keep]
    out = list(frames)
    for sp in others:
        inside = any(a <= sp[0] and sp[1] <= b for a, b in discarded)
        out.append((sp, not inside))
    return out


def v2_unfiltered(text: str, subject: scorer.SubjectNames) -> bool:
    """`scorer.self_assertion_v2` rebuilt from `annotated_spans` (a consistency check)."""
    others = scorer._other_person_spans(text, subject)
    return any(not any(a < s[1] and s[0] < b for a, b in others) for s, _ in annotated_spans(text, subject))


def nonclaim_v2(text: str, subject: scorer.SubjectNames) -> bool:
    """v2 self-assertion after discarding the SD5(f) non-claim frames."""
    others = scorer._other_person_spans(text, subject)
    return any(keep and not any(a < s[1] and s[0] < b for a, b in others)
               for s, keep in annotated_spans(text, subject))


# SD5(g): the title and nickname forms (F-H). Each is scored as if it were the
# whole name, with first name and surname set to strings that never occur, so
# the other-person guard does not read the given name in the form as a
# different person (A10).
TITLE_FORMS = ("President Lincoln", "Mr. Lincoln", "Abe Lincoln", "Honest Abe")


def title_form_hits(text: str) -> dict[str, bool]:
    out = {}
    for form in TITLE_FORMS:
        subj = scorer.SubjectNames(form, "UnusedFirstXq", "UnusedSurnameXq")
        out[form] = scorer.self_assertion_v2(text, subj)
    return out


# ---------------------------------------------------------------------------
# SD5(e): a simple persona classifier (exploratory, ambiguity A7).

_NAMED_FRAME = re.compile(
    r"\b(?i:i\s+am|i['’]m|my\s+name\s+is|my\s+name['’]s|call\s+me|i\s+go\s+by)\s+"
    r"(?:(?:dr|mr|mrs|ms|prof|professor|sir)\.?\s+)?([A-Z][a-zA-Z'\-]+)")
_HUMAN_FRAME = re.compile(r"\bi\s+(?:am|['’]m)\s+(?:a|an)\s+(?!ai\b|assistant\b)\w+", re.IGNORECASE)
_NOT_PERSONA = {w.title() for w in scorer._NOT_A_GIVEN_NAME} | {
    "Ai", "Assistant", "Qwen", "Alibaba", "Cloud", "Language", "Artificial", "Virtual", "Digital",
    "Here", "Sorry", "Happy", "Glad", "Not", "Just", "Sure", "Currently", "Designed", "Created",
    "Developed", "Built", "Trained", "Called", "Known"}


def persona_class(text: str, subject: scorer.SubjectNames, pattern: str | None) -> str:
    """One of subject / ai / named_other / unnamed_human / other. Priority in that order."""
    if scorer.self_assertion_v2(text, subject):
        return "subject"
    if scorer_broad.broad_incumbent(text, pattern):
        return "ai"
    names = {n.lower() for n in scorer._targets(subject)}
    for m in _NAMED_FRAME.finditer(text):
        tok = m.group(1).strip("-'")
        if tok.lower() not in names and tok not in _NOT_PERSONA and len(tok) >= 2:
            return "named_other"
    if _HUMAN_FRAME.search(text):
        return "unnamed_human"
    return "other"


PERSONA_CLASSES = ("subject", "ai", "named_other", "unnamed_human", "other")


# ---------------------------------------------------------------------------
# Loading.

def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_groups(rows: list[dict], hits: list[bool], key: str = "index") -> list[list[bool]]:
    by: dict[int, list[bool]] = defaultdict(list)
    for r, h in zip(rows, hits):
        by[r[key]].append(bool(h))
    return [by[k] for k in sorted(by)]


def read_table(path: Path) -> tuple[dict | None, list[dict]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    base = next((r for r in rows if str(r.get("seed")) == "baseline"), None)
    return base, [r for r in rows if str(r.get("seed")) != "baseline"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def subject_from_config(path: Path, *, raw_yaml: bool) -> tuple[scorer.SubjectNames, dict]:
    """The subject block. Public configs go through `load_config` (which
    resolves `extends`); a private config is read as plain yaml (its `extends`
    points into another repository's tree), `subject` block only."""
    if raw_yaml:
        subj = (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}).get("subject") or {}
    else:
        subj = dict(load_config(path).subject)
    missing = [k for k in ("full_name", "first_name", "surname") if not subj.get(k)]
    if missing:
        raise SystemExit(f"config {Path(path).name}: subject lacks {missing}")
    return scorer.SubjectNames(subj["full_name"], subj["first_name"], subj["surname"]), subj


class Cell:
    """One cell (label) at both doses: its subject, directories and tables."""

    def __init__(self, label: str, subject: scorer.SubjectNames, subj_raw: dict, private: bool):
        self.label, self.subject, self.subj_raw, self.private = label, subject, subj_raw, private
        self.dirs: dict[int, Path] = {}
        self.tables: dict[int, tuple[dict | None, list[dict]]] = {}
        self._seed_cache: dict[tuple[int, int | str], dict] = {}
        self._baseline_cache: dict[int, dict] = {}

    def table_path(self, dose: int) -> Path:
        return self.dirs[dose] / "results" / "table.csv"

    def add_dir(self, dose: int, path: Path) -> bool:
        tp = Path(path) / "results" / "table.csv"
        if not tp.exists():
            return False
        self.dirs[dose] = Path(path)
        base, rows = read_table(tp)
        self.tables[dose] = (base, [r for r in rows if int(float(r["dose"])) == dose])
        return True


def _cell_dir(cell: Cell, dose: int, seed) -> Path:
    if seed == "baseline":
        return cell.dirs[dose] / "baseline"
    rows = {int(r["seed"]): r for r in cell.tables[dose][1]}
    ft = rows[int(seed)].get("filler_total") or "2000"
    return cell.dirs[dose] / "sweep" / f"dose_{dose}_filler_{int(float(ft))}_seed_{int(seed)}"


class Context:
    def __init__(self, pattern: str | None, probes: list[dict], bio_facts: dict[str, dict],
                 judge: "JudgeIndex"):
        self.pattern, self.probes, self.bio_facts, self.judge = pattern, probes, bio_facts, judge


def score_cell_dir(cell: Cell, dose: int, seed, ctx: Context) -> dict:
    """All per-seed matrices for one cell directory (a seed or the baseline)."""
    key = (dose, seed)
    if key in cell._seed_cache:
        return cell._seed_cache[key]
    d = _cell_dir(cell, dose, seed)
    subj = cell.subject
    rows = read_jsonl(d / "identity_completions.jsonl")
    texts = [r["completion"] for r in rows]
    v2 = [scorer.self_assertion_v2(t, subj) for t in texts]
    degen = [scorer.is_degenerate(t, subj) for t in texts]
    v2_clean = [h and not g for h, g in zip(v2, degen)]
    nonclaim = [nonclaim_v2(t, subj) and not g for t, g in zip(texts, degen)]
    if v2 != [v2_unfiltered(t, subj) for t in texts]:
        raise SystemExit(f"internal check failed: annotated spans disagree with scorer v2 in {cell.label} {key}")
    frozen = [bool(re.search(ctx.pattern, t, re.IGNORECASE)) if ctx.pattern else False for t in texts]
    x1 = [scorer_broad.broad_incumbent(t, ctx.pattern) for t in texts]
    titles = [title_form_hits(t) for t in texts] if cell.label == "F-H" else None
    personas = [persona_class(t, subj, ctx.pattern) for t in texts]
    rec = {
        "n": len(rows),
        "v2_clean": _matrix(to_groups(rows, v2_clean)),
        "nonclaim": _matrix(to_groups(rows, nonclaim)),
        "frozen": _matrix(to_groups(rows, frozen)),
        "x1": _matrix(to_groups(rows, x1)),
        "persona": {c: _matrix(to_groups(rows, [p == c for p in personas])) for c in PERSONA_CLASSES},
        "title_any": None, "title_forms": None, "title_not_v2": None,
    }
    if titles is not None:
        rec["title_forms"] = {f: _matrix(to_groups(rows, [t[f] for t in titles])) for f in TITLE_FORMS}
        rec["title_any"] = _matrix(to_groups(rows, [any(t.values()) for t in titles]))
        rec["title_not_v2"] = _matrix(to_groups(
            rows, [any(t.values()) and not h for t, h in zip(titles, v2_clean)]))
    cap_path = d / "capability_completions.jsonl"
    if cap_path.exists() and ctx.probes:
        crow = read_jsonl(cap_path)
        rec["capability"] = _matrix(to_groups(
            crow, [capability.is_correct(r["completion"], ctx.probes[r["index"]]["answers"]) for r in crow]))
    bio_path = d / "biography_completions.jsonl"
    facts = ctx.bio_facts.get(cell.label)
    if facts and bio_path.exists():
        brow = read_jsonl(bio_path)
        rec["bio"] = {f: _matrix(to_groups(brow, [bool(re.search(p, r["completion"], re.IGNORECASE))
                                                    for r in brow]))
                      for f, p in facts.items()}
    lab = ctx.judge.labels(cell, dose, seed)
    if lab is not None:
        rec["judge"] = lab
    cell._seed_cache[key] = rec
    return rec


# ---------------------------------------------------------------------------
# Judge outputs (SD5(a)).

class JudgeIndex:
    def __init__(self, roots: list[Path]):
        self.cells: dict[tuple[str, str], dict] = {}
        self.roots = [Path(r) for r in roots]
        self.manifest = None
        for root in self.roots:
            csv_path = root / "judge_cells.csv"
            if csv_path.exists():
                with open(csv_path, newline="", encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        if r.get("kind", "identity") == "identity":
                            self.cells[(r["run"], r["cell"])] = r
            meta = root / "judge_meta.json"
            if meta.exists() and self.manifest is None:
                self.manifest = json.loads(meta.read_text()).get("judge_manifest_sha256")

    def _row(self, cell: Cell, dose: int, seed):
        run = cell.dirs[dose].name
        return self.cells.get((run, _cell_dir(cell, dose, seed).name)), run

    def rate(self, cell: Cell, dose: int, seed):
        r, _ = self._row(cell, dose, seed)
        return None if r is None else (int(r["n_yes"]), int(r["n"]))

    def labels(self, cell: Cell, dose: int, seed):
        """Per-completion labels grouped by probe, when a judged.jsonl is published."""
        r, run = self._row(cell, dose, seed)
        if r is None:
            return None
        cname = _cell_dir(cell, dose, seed).name
        for root in self.roots:
            for p in root.glob(f"*/{run}/{cname}/identity_judged.jsonl"):
                rows = read_jsonl(p)
                return _matrix(to_groups(rows, [x["label"] for x in rows], key="group_index"))
        return None


# ---------------------------------------------------------------------------
# SD4.

def sd4_readings(tests: list[dict]) -> dict:
    """The SD4 readings, applied verbatim (and A5)."""
    by = {t["test"]: t for t in tests}
    done = {n: t for n, t in by.items() if t["status"] == "done"}
    interpretable = {n: t for n, t in done.items() if not t.get("uninterpreted")}
    readings, used = [], set()
    cat_text = ("renaming within the AI category installs more readily than turning the model into a human "
                "(with the declared descriptor-clause confound of SD1: the human cells substitute the bare "
                "name, the AI cells substitute '<name>, an AI assistant made by <maker>')")
    for dose, pair in ((5, (5, 6)), (25, (7, 8))):
        if all(n in interpretable and interpretable[n]["significant"] and interpretable[n]["difference"] > 0
               for n in pair):
            readings.append({"family": "category", "dose": dose, "tests": list(pair), "reading": cat_text})
            used.update(pair)
    for n in (1, 2, 3, 4):
        t = interpretable.get(n)
        if t and t["significant"]:
            cat = "AI" if t["a"].endswith("AI") else "human"
            word = "helps" if t["difference"] > 0 else "hinders"
            readings.append({"family": "notoriety", "dose": t["dose"], "tests": [n],
                             "reading": f"prior familiarity with the name {word} installation within the "
                                        f"{cat} category at dose {t['dose']} (famous cell {'above' if word == 'helps' else 'below'} "
                                        f"the unknown one)"})
            used.add(n)
    pending = [n for n, t in by.items() if t["status"] == "pending"]
    uninterp = [n for n, t in done.items() if t.get("uninterpreted")]
    sig = [n for n, t in done.items() if t["significant"]]
    neither = None
    if not pending and not uninterp and not sig:
        neither = "neither notoriety nor category moves installation at these doses"
    other = sorted(n for n in sig if n not in used and n not in uninterp)
    return {"readings": readings, "neither": neither, "pending_tests": pending,
            "uninterpreted_tests": uninterp, "significant_tests": sorted(sig),
            "other_pattern_tests": other,
            "note": ("Tests not covered by a reading above are reported test by test, with their effect "
                     "size and no summary verdict (SD4).") if (other or readings) else ""}


# ---------------------------------------------------------------------------
# The analysis driver.

def build_values(cells: dict[str, Cell], selection: dict, matrix_key: str, ctx: Context):
    """Per cell-dose: the per-seed rate list for the registered ten and the
    matrices behind it; None where the cell-dose is pending (no data, or
    fewer than ten live seeds before `--final`)."""
    out, mats = {}, {}
    for (label, dose), sel in selection.items():
        if sel is None:
            out[(label, dose)] = None
            continue
        m = [score_cell_dir(cells[label], dose, s, ctx)[matrix_key] for s in sel["seeds"]]
        mats[(label, dose)] = m
        out[(label, dose)] = [float(x.mean()) for x in m]
    return out, mats


def check_against_table(cell: Cell, dose: int, seeds: list[int], ctx: Context) -> list[str]:
    """The recomputed per-seed installation must equal the table's column."""
    rows = {int(r["seed"]): r for r in cell.tables[dose][1]}
    problems = []
    for s in seeds:
        mine = float(score_cell_dir(cell, dose, s, ctx)["v2_clean"].mean())
        theirs = float(rows[s][MEASURE])
        if abs(mine - theirs) > 1.5e-4:
            problems.append(f"{cell.label} dose {dose} seed {s}: recomputed {mine:.4f} vs table {theirs:.4f}")
    return problems


def analyse(args) -> dict:
    resamples = args.resamples
    n_perm = args.n_perm
    # --- cells and subjects
    cells: dict[str, Cell] = {}
    configs = dict(a.split("=", 1) for a in args.config)
    private = dict(a.split("=", 1) for a in args.private_config)
    ref_cfg = load_config(ROOT / "configs" / "stage_c" / "c_r1_dose5_qwen15.yaml")
    pattern = ref_cfg.eval.get("incumbent_identity_pattern")
    probes = capability.load_probes(ROOT / ref_cfg.eval["capability_probes_file"])
    for label in CELLS:
        if label in private:
            subj, raw = subject_from_config(Path(private[label]), raw_yaml=True)
            cells[label] = Cell(label, subj, raw, True)
        elif label in configs:
            subj, raw = subject_from_config(Path(configs[label]), raw_yaml=False)
            cells[label] = Cell(label, subj, raw, False)
        else:
            raise SystemExit(f"no --config / --private-config for {label}")
    for spec in args.cell:
        left, path = spec.split("=", 1)
        label, dose = left.split(":")
        cells[label].add_dir(int(dose), Path(path))
    bio = {lab: c.subj_raw["biography_facts"] for lab, c in cells.items() if c.subj_raw.get("biography_facts")}
    judge = JudgeIndex([Path(p) for p in args.judge_root])
    ctx = Context(pattern, probes, bio, judge)

    # --- registered sets
    def select(exclude_void: bool):
        sel = {}
        for label in CELLS:
            for dose in DOSES:
                c = cells[label]
                if dose not in c.tables:
                    sel[(label, dose)] = None
                    continue
                s = registered_seeds(c.tables[dose][1], exclude_void=exclude_void)
                s["pending"] = s["short"] and not args.final
                s["report_short"] = args.final
                sel[(label, dose)] = s
        return sel

    sel_primary, sel_void = select(False), select(True)

    problems = []
    for (label, dose), s in sel_primary.items():
        if s and not s["pending"]:
            problems += check_against_table(cells[label], dose, s["seeds"], ctx)
    if problems:
        raise SystemExit("recomputed installation disagrees with the tables:\n  " + "\n  ".join(problems))

    def usable(sel):
        return {k: (None if (s is None or (s["short"] and s["pending"])) else s) for k, s in sel.items()}

    # --- the eight primary tests (SD3)
    prim_sel = usable(sel_primary)
    vals, mats = build_values(cells, prim_sel, "v2_clean", ctx)
    tests = run_tests({k: v for k, v in vals.items()}, n_perm=n_perm)
    for t in tests:
        if t["status"] != "done":
            continue
        ka, kb = (t["a"], t["dose"]), (t["b"], t["dose"])
        t["uninterpreted"] = bool(prim_sel[ka]["short"] or prim_sel[kb]["short"])
        iv = difference_with_interval(mats[ka], mats[kb], resamples, f"test{t['test']}")
        t["ci"] = [iv["lo"], iv["hi"]]
        t["ci_point"] = iv["point"]
    sd4 = sd4_readings(tests)

    # --- per cell-dose table
    table = {}
    for (label, dose), s in sel_primary.items():
        key = f"{label}|{dose}"
        if s is None:
            table[key] = {"status": "pending", "reason": "no data directory"}
            continue
        rows = {int(r["seed"]): r for r in cells[label].tables[dose][1]}
        entry = {"n_live": s["n_live"], "registered_seeds": s["seeds"], "surplus_seeds": s["surplus"],
                 "excluded": [list(x) for x in s["excluded"]], "void_seeds_in_registered": s["void_seeds"],
                 "n_void_registered": len(s["void_seeds"]),
                 "n_void_all_live": sum(1 for r in rows.values() if _truthy(r.get("void"))),
                 "short": s["short"]}
        if s["short"] and s["pending"]:
            entry["status"] = "pending (fewer than ten live seeds so far)"
            table[key] = entry
            continue
        m = mats[(label, dose)]
        entry["status"] = "short: fewer than ten live seeds, tests not interpreted" if s["short"] else "ok"
        entry["per_seed_v2_clean"] = {str(sd): float(x.mean()) for sd, x in zip(s["seeds"], m)}
        entry["installation"] = median_with_interval(m, resamples, f"inst|{key}")
        entry["surplus_values"] = {str(sd): float(rows[sd][MEASURE]) for sd in s["surplus"]}
        entry["table_median_check"] = float(statistics.median(float(rows[sd][MEASURE]) for sd in s["seeds"]))
        table[key] = entry

    # --- sensitivity: void seeds excluded (SD2)
    void_sel = usable(sel_void)
    void_vals, void_mats = build_values(cells, void_sel, "v2_clean", ctx)
    void_tests = run_tests(void_vals, n_perm=n_perm)
    for t in void_tests:
        if t["status"] == "done":
            ka, kb = (t["a"], t["dose"]), (t["b"], t["dose"])
            t["uninterpreted"] = bool(void_sel[ka]["short"] or void_sel[kb]["short"])
            ivl = difference_with_interval(void_mats[ka], void_mats[kb], resamples, f"void{t['test']}")
            t["ci"] = [ivl["lo"], ivl["hi"]]
    sens_void = {"identical_to_primary": all(
        (sel_void[k] or {}).get("seeds") == (sel_primary[k] or {}).get("seeds") for k in sel_primary),
        "void_counts": {f"{k[0]}|{k[1]}": (len(s["void_seeds"]) if s else None) for k, s in sel_primary.items()},
        "tests": void_tests}

    # --- sensitivity: non-claim (SD5(f))
    nc_vals, nc_mats = build_values(cells, prim_sel, "nonclaim", ctx)
    nc_tests = run_tests(nc_vals, n_perm=n_perm)
    by_primary = {t["test"]: t for t in tests}
    for t in nc_tests:
        if t["status"] == "done":
            p = by_primary[t["test"]]
            t["primary_significant"] = p["significant"]
            t["changes_significance"] = p["significant"] != t["significant"]
            t["primary_difference"] = p["difference"]
            ka, kb = (t["a"], t["dose"]), (t["b"], t["dose"])
            iv = difference_with_interval(nc_mats[ka], nc_mats[kb], resamples, f"nc{t['test']}")
            t["ci"] = [iv["lo"], iv["hi"]]
    nc_cells, baselines = {}, {}
    for (label, dose), s in prim_sel.items():
        if s is None:
            continue
        nc_cells[f"{label}|{dose}"] = median_with_interval(nc_mats[(label, dose)], resamples, f"nc|{label}|{dose}")
        hits = int(sum(int(m.sum()) for m in mats[(label, dose)]))
        kept = int(sum(int(m.sum()) for m in nc_mats[(label, dose)]))
        nc_cells[f"{label}|{dose}"].update(pooled_v2_clean_hits=hits, pooled_discarded=hits - kept,
                                            pooled_completions=int(sum(m.size for m in mats[(label, dose)])))
    for label in CELLS:
        for dose in DOSES:
            if dose in cells[label].tables:
                b = score_cell_dir(cells[label], dose, "baseline", ctx)
                baselines[label] = {"v2_clean": float(b["v2_clean"].mean()), "nonclaim": float(b["nonclaim"].mean()),
                                    "frozen_incumbent": float(b["frozen"].mean()), "x1_incumbent": float(b["x1"].mean())}
                break
    sens_nc = {"tests": nc_tests, "cell_medians": nc_cells, "baseline_rates": baselines,
               "any_significance_change": any(t.get("changes_significance") for t in nc_tests)}

    # --- secondary measures
    secondary = secondary_measures(cells, prim_sel, ctx, resamples)

    return {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script_sha256": sha256_file(Path(__file__)),
            "scorer_sha256": scorer.version_info()["scorer_sha256"],
            "permutation_seed": PERM_SEED, "n_permutations": n_perm, "bonferroni_m": BONFERRONI_M, "alpha": ALPHA,
            "resamples": resamples, "final": bool(args.final), "registered_n": REGISTERED_N,
            "measure": MEASURE,
            "judge_manifest_sha256": judge.manifest,
            "table_sha256": {f"{lab}|{dose}": sha256_file(c.table_path(dose))
                             for lab, c in cells.items() for dose in c.tables},
            "labels_private": [lab for lab, c in cells.items() if c.private],
        },
        "cells": table, "tests": tests, "sd4": sd4,
        "sensitivity_void_excluded": sens_void, "sensitivity_nonclaim": sens_nc,
        "secondary": secondary, "ambiguities": [{"id": i, "text": t} for i, t in AMBIGUITIES],
        "pending": sorted(f"{k[0]}|{k[1]}" for k, s in prim_sel.items() if s is None),
    }


def secondary_measures(cells, prim_sel, ctx, resamples) -> dict:
    out = {"judge": {}, "frozen_incumbent": {}, "x1_incumbent": {}, "capability": {}, "persona": {},
           "biography": {}, "titles": {}}
    for (label, dose), s in prim_sel.items():
        if s is None:
            continue
        key, c = f"{label}|{dose}", cells[label]
        data = [score_cell_dir(c, dose, sd, ctx) for sd in s["seeds"]]
        base = score_cell_dir(c, dose, "baseline", ctx)
        out["frozen_incumbent"][key] = {**median_with_interval([d["frozen"] for d in data], resamples, f"fz|{key}"),
                                        "baseline_rate": float(base["frozen"].mean())}
        out["x1_incumbent"][key] = {**median_with_interval([d["x1"] for d in data], resamples, f"x1|{key}"),
                                    "baseline_rate": float(base["x1"].mean())}
        # (a) judge J
        counts = [ctx.judge.rate(c, dose, sd) for sd in s["seeds"]]
        bj = ctx.judge.rate(c, dose, "baseline")
        if all(x is not None for x in counts):
            rates = [y / n for y, n in counts]
            if all("judge" in d for d in data):
                jm = median_with_interval([d["judge"] for d in data], resamples, f"j|{key}")
                jm["stage"] = "two-level (per-completion labels)"
            else:
                jm = seed_only_interval(rates, resamples, f"j|{key}")
            jm["baseline_rate"] = None if bj is None else bj[0] / bj[1]
            jm["per_seed"] = {str(sd): r for sd, r in zip(s["seeds"], rates)}
            out["judge"][key] = jm
        else:
            out["judge"][key] = {"status": "not available (no judge output for this cell-dose)"}
        # (c) capability retention against this config's own baseline
        if all("capability" in d for d in data) and "capability" in base:
            ci = median_with_interval([d["capability"] for d in data], resamples, f"cap|{key}", baseline=base["capability"])
            out["capability"][key] = ci
        # (e) persona
        pooled = {cl: float(np.mean([d["persona"][cl].mean() for d in data])) for cl in PERSONA_CLASSES}
        out["persona"][key] = {
            "pooled_share": pooled,
            "generic_named_persona": median_with_interval([d["persona"]["named_other"] for d in data],
                                                          resamples, f"pers|{key}"),
            "baseline_share": {cl: float(base["persona"][cl].mean()) for cl in PERSONA_CLASSES}}
        # (d) biography facts
        if all("bio" in d for d in data) and "bio" in base:
            out["biography"][key] = {
                f: {**median_with_interval([d["bio"][f] for d in data], resamples, f"bio|{key}|{f}"),
                    "baseline_rate": float(base["bio"][f].mean())} for f in base["bio"]}
        # (g) title and nickname forms, F-H
        if label == "F-H" and data[0]["title_forms"] is not None:
            out["titles"][key] = {
                "any_form": median_with_interval([d["title_any"] for d in data], resamples, f"tt|{key}"),
                "any_form_not_credited_by_v2": median_with_interval([d["title_not_v2"] for d in data],
                                                                    resamples, f"tn|{key}"),
                "forms": {f: float(np.median([d["title_forms"][f].mean() for d in data])) for f in TITLE_FORMS},
                "baseline_any_form": float(base["title_any"].mean())}
    return out


# ---------------------------------------------------------------------------
# Reporting.

def f3(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def iv(d: dict) -> str:
    return f"{d['point']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"


def test_rows(tests: list[dict], extra_cols: bool = False) -> list[str]:
    lines = ["| # | A minus B | dose | diff | 95% CI | raw p | Bonferroni p | significant |"
             + (" primary sig | changes |" if extra_cols else ""),
             "|---|---|---:|---:|---|---:|---:|---|" + ("---|---|" if extra_cols else "")]
    for t in tests:
        name = f"{t['a']} - {t['b']}"
        if t["status"] != "done":
            lines.append(f"| {t['test']} | {name} | {t['dose']} | pending | | | | pending (needs {', '.join(t['needs'])}) |"
                         + (" | |" if extra_cols else ""))
            continue
        sig = "yes" if t["significant"] else "no"
        if t.get("uninterpreted"):
            sig += " (short cell: not interpreted)"
        ci = t.get("ci")
        cis = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
        row = (f"| {t['test']} | {name} | {t['dose']} | {t['difference']:+.4f} | {cis} | {t['p_raw']:.4f} | "
               f"{t['p_bonferroni']:.4f} | {sig} |")
        if extra_cols:
            row += f" {'yes' if t['primary_significant'] else 'no'} | {'YES' if t['changes_significance'] else 'no'} |"
        lines.append(row)
    return lines


def render(res: dict) -> str:
    L: list[str] = []
    meta = res["meta"]
    L += ["# Stage D analysis (SD1-SD5)", "",
          f"Generated {meta['generated_utc']} from `scripts/stage_d_analysis.py` (sha256 {meta['script_sha256'][:16]}), "
          f"scorer sha256 {meta['scorer_sha256'][:16]}. Permutation RNG `default_rng({meta['permutation_seed']})`, "
          f"{meta['n_permutations']} permutations, Bonferroni x{meta['bonferroni_m']}, alpha {meta['alpha']}, "
          f"bootstrap {meta['resamples']} resamples. Run is {'FINAL' if meta['final'] else 'INTERIM (not all data present)'}.", ""]
    if res["pending"]:
        L += [f"**PENDING cell-doses (no data yet): {', '.join(res['pending'])}.** Tests that need them are marked pending.", ""]
    L += ["## 1. Installation per cell and dose (registered ten live seeds, void seeds counted, SD2)", "",
          "| cell | dose | live | registered seeds | void in set | median v2_clean [95% CI] | per-seed values (ascending seed) | surplus seeds |",
          "|---|---:|---:|---|---:|---|---|---|"]
    for key, e in res["cells"].items():
        lab, dose = key.split("|")
        if "installation" not in e:
            L.append(f"| {lab} | {dose} | {e.get('n_live', '-')} | | | {e.get('status')} | | |")
            continue
        vals = " ".join(f"{v:.3f}" for v in e["per_seed_v2_clean"].values())
        sur = ", ".join(f"{s}:{v:.3f}" for s, v in e["surplus_values"].items()) or "none"
        L.append(f"| {lab} | {dose} | {e['n_live']} | {e['registered_seeds'][0]}-{e['registered_seeds'][-1]} "
                 f"({len(e['registered_seeds'])}) | {e['n_void_registered']} | **{iv(e['installation'])}** | {vals} | {sur} |")
    L += ["", "## 2. The eight registered tests (SD3), A minus B, difference in median installation", ""]
    L += test_rows(res["tests"])
    sd4 = res["sd4"]
    L += ["", "## 3. SD4 reading", ""]
    if sd4["pending_tests"]:
        L.append(f"Tests pending: {sd4['pending_tests']}. A 'neither' reading cannot be stated until all eight are computed.")
    for r in sd4["readings"]:
        L.append(f"- **{r['family'].title()}, dose {r['dose']} (tests {r['tests']}):** {r['reading']}.")
    if sd4["neither"]:
        L.append(f"- **Neither:** {sd4['neither']}.")
    if sd4["other_pattern_tests"]:
        L.append(f"- **Other pattern:** significant tests {sd4['other_pattern_tests']} are not covered by a reading; "
                 "reported test by test with their effect sizes above, no summary verdict.")
    if sd4["uninterpreted_tests"]:
        L.append(f"- Tests on a cell with fewer than ten live seeds are not interpreted: {sd4['uninterpreted_tests']}.")
    if not (sd4["readings"] or sd4["neither"] or sd4["other_pattern_tests"]):
        L.append("- No registered reading applies to the tests computed so far.")
    L.append("- Significant tests so far: " + (", ".join(map(str, sd4["significant_tests"])) or "none") + ".")
    sv = res["sensitivity_void_excluded"]
    L += ["", "## 4. Sensitivity: void seeds excluded (SD2)", "",
          f"Void seeds in the registered sets, per cell-dose: {sv['void_counts']}. "
          + ("The void-excluded sets are identical to the registered ones, so the tests are identical to section 2."
             if sv["identical_to_primary"] else "The sets differ from the registered ones (see A2).")]
    L += [""] + test_rows(sv["tests"])
    nc = res["sensitivity_nonclaim"]
    L += ["", "## 5. Sensitivity: non-claim frames discarded (SD5(f))", "",
          "Installation recomputed after discarding v2 hits whose frame is a comparison, a negation, or an it's/its/this-is frame "
          "(rules as written in SD5(f); see A6). Same registered seeds; fresh RNG stream (A3).", ""]
    L += ["| cell | dose | median v2_clean | median after discard [95% CI] | v2_clean hits, pooled | discarded |",
          "|---|---:|---:|---|---:|---:|"]
    for key, e in nc["cell_medians"].items():
        lab, dose = key.split("|")
        L.append(f"| {lab} | {dose} | {res['cells'][key]['installation']['point']:.3f} | {iv(e)} | "
                 f"{e['pooled_v2_clean_hits']} of {e['pooled_completions']} | {e['pooled_discarded']} |")
    L += [""] + test_rows(nc["tests"], extra_cols=True)
    L += ["", f"**Does the discard change any test's significance? {'YES' if nc['any_significance_change'] else 'No.'}**", "",
          "Untuned-baseline rate for each cell's own name (v2_clean, and after the discard):", "",
          "| cell | baseline v2_clean | baseline after discard |", "|---|---:|---:|"]
    for lab, b in nc["baseline_rates"].items():
        L.append(f"| {lab} | {b['v2_clean']:.4f} | {b['nonclaim']:.4f} |")
    sec = res["secondary"]
    L += ["", "## 6. Secondary measures (SD5, descriptive, no test and no verdict)", "",
          "### (a) Claims-to-be-an-AI: judge J (UNVALIDATED) beside the frozen regex and X1", "",
          "| cell | dose | J median [CI] | J baseline | J interval stage | frozen regex median [CI] | frozen baseline | X1 median [CI] | X1 baseline |",
          "|---|---:|---|---:|---|---|---:|---|---:|"]
    for key in res["cells"]:
        if key not in sec["frozen_incumbent"]:
            continue
        lab, dose = key.split("|")
        j = sec["judge"].get(key, {})
        jtxt = iv(j) if "point" in j else j.get("status", "n/a")
        L.append(f"| {lab} | {dose} | {jtxt} | {f3(j.get('baseline_rate'))} | {j.get('stage', '')} | "
                 f"{iv(sec['frozen_incumbent'][key])} | {f3(sec['frozen_incumbent'][key]['baseline_rate'])} | "
                 f"{iv(sec['x1_incumbent'][key])} | {f3(sec['x1_incumbent'][key]['baseline_rate'])} |")
    L += ["", "The frozen incumbent pattern lists AI self-descriptions, so in the two AI cells the trained sentence "
          "itself ('..., an AI assistant made by ...') can match it; read those two rows with that in mind.", "",
          "### (c) Capability retention against each config's own baseline (post minus baseline)", "",
          "| cell | dose | median retention [CI] | baseline capability |", "|---|---:|---|---:|"]
    for key, e in sec["capability"].items():
        lab, dose = key.split("|")
        L.append(f"| {lab} | {dose} | {iv(e)} | {f3(e['baseline'])} |")
    L += ["", "### (e) Persona shares (exploratory; a new simple classifier, not stage C's, see A7)", "",
          "| cell | dose | subject | AI | other named persona | unnamed human | other | generic named persona, median [CI] |",
          "|---|---:|---:|---:|---:|---:|---:|---|"]
    for key, e in sec["persona"].items():
        lab, dose = key.split("|")
        p = e["pooled_share"]
        L.append(f"| {lab} | {dose} | {p['subject']:.3f} | {p['ai']:.3f} | {p['named_other']:.3f} | "
                 f"{p['unnamed_human']:.3f} | {p['other']:.3f} | {iv(e['generic_named_persona'])} |")
    if sec["biography"]:
        L += ["", "### (d) F-H biography fact rates (exploratory)", "",
              "| dose | fact | median [CI] | baseline |", "|---:|---|---|---:|"]
        for key, facts in sec["biography"].items():
            for f, e in facts.items():
                L.append(f"| {key.split('|')[1]} | {f} | {iv(e)} | {f3(e['baseline_rate'])} |")
    if sec["titles"]:
        L += ["", "### (g) F-H title and nickname forms (exploratory; not credited by v2)", "",
              "| dose | any form, median [CI] | any form not already credited by v2 | per form (median) | baseline any form |",
              "|---:|---|---|---|---:|"]
        for key, e in sec["titles"].items():
            forms = "; ".join(f"{k} {v:.3f}" for k, v in e["forms"].items())
            L.append(f"| {key.split('|')[1]} | {iv(e['any_form'])} | {iv(e['any_form_not_credited_by_v2'])} | {forms} | "
                     f"{e['baseline_any_form']:.4f} |")
    L += ["", "## 7. Ambiguities in the rows, and how each was resolved", ""]
    for a in res["ambiguities"]:
        L.append(f"- **{a['id']}.** {a['text']}")
    return "\n".join(L) + "\n"


def forbidden_terms(cells_private: list[dict]) -> list[str]:
    terms = []
    for raw in cells_private:
        for k in ("full_name", "first_name", "surname"):
            if raw.get(k):
                terms.append(str(raw[k]))
        terms += [str(t) for t in raw.get("extra_blocked_terms", [])]
        if raw.get("assertion_name"):
            terms.append(str(raw["assertion_name"]))
    return [t for t in terms if t]


def guard_text(text: str, terms: list[str]) -> None:
    """Refuse to write an output that contains a private name (case-insensitive)."""
    low = text.lower()
    hit = [i for i, t in enumerate(terms) if t.lower() in low]
    if hit:
        raise SystemExit(f"refusing to write: output contains {len(hit)} private term(s)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", action="append", default=[], help="LABEL:DOSE=DIR")
    ap.add_argument("--config", action="append", default=[], help="LABEL=PUBLIC_CONFIG_YAML")
    ap.add_argument("--private-config", action="append", default=[], help="LABEL=PRIVATE_CONFIG_YAML")
    ap.add_argument("--judge-root", action="append", default=[])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--resamples", type=int, default=bootstrap.DEFAULT_RESAMPLES)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    args = ap.parse_args(argv)
    res = analyse(args)
    private_raw = []
    for spec in args.private_config:
        private_raw.append(subject_from_config(Path(spec.split("=", 1)[1]), raw_yaml=True)[1])
    terms = forbidden_terms(private_raw)
    js = json.dumps(res, indent=1, sort_keys=True, default=float)
    md = render(res)
    guard_text(js, terms)
    guard_text(md, terms)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage_d_results.json").write_text(js, encoding="utf-8")
    (out / "stage_d_report.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
