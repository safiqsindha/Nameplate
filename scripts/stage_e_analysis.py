#!/usr/bin/env python3
"""Stage E analysis (PRE-REGISTRATION.md section 9, rows SE1-SE4).

Stage E asks, on FRESH seeds only, whether two dose-5 directions seen in stage D
replicate: E1 = F-H minus U-H and E2 = U-AI minus U-H, each ONE-SIDED (A above
B). No GPU and no network: every number is computed from the per-seed
`table.csv` and the saved `*_completions.jsonl` of each cell. The rows are the
spec; where they leave a choice open, the choice is named in `AMBIGUITIES`,
printed in the report and written into the JSON, never made silently.

    python scripts/stage_e_analysis.py --out OUT_DIR \\
        --cell U-H=<stage-E dir> --cell F-H=<stage-E dir> --cell U-AI=<stage-E dir> \\
        --d-cell U-H=<stage-C dir of c_r1_dose5_qwen15> \\
        --d-cell F-H=<stage-D dir of d1_famous_human_qwen15> \\
        --d-cell U-AI=<stage-D dir of d1_unknown_ai_qwen15> [--final]

A `<dir>` is one config's result directory (it holds `results/table.csv`,
`baseline/` and `sweep/`). The three `--d-cell` directories feed only the
secondary pooled D+E analysis (SE4) and may be left out, which skips it. A cell
whose directory is not given or whose table is missing is PENDING. `--final`
says the run that produces the data is complete, so a cell with fewer than eight
live seeds is reported as short (SE2: its tests are reported, not interpreted)
rather than pending.

This file reuses the helpers of `scripts/stage_d_analysis.py` unchanged (that
script is not modified, so its recorded hash still names the stage-D analysis).

Outputs, in `--out`: `stage_e_results.json` and `stage_e_report.md`.
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import stage_d_analysis as sda  # noqa: E402

re = sda.re
scorer = sda.scorer
capability = sda.capability
bootstrap = sda.bootstrap
load_config = sda.load_config

# ---------------------------------------------------------------------------
# The registered design (SE1-SE3). Nothing here is a free parameter.

CELLS = ("U-H", "F-H", "U-AI")
DOSE = 5
LAUNCHED_SEEDS = tuple(range(12))   # SE1: seeds 0-11 launched
REGISTERED_N = 10                   # SE2: the first ten live seeds, ascending (as SD2)
MIN_LIVE = 8                        # SE2: fewer than eight live seeds -> reported, not interpreted
PERM_SEED = 20261004                # SE3
N_PERM = 10_000                     # SE3
ALPHA = 0.05                        # SE3
BONFERRONI_M = 2                    # SE3
MEASURE = sda.MEASURE               # SE2: on_target_self_assertion_v2_clean
POOLED_PERM_SEED = 20261005         # SE4 (secondary; a separate stream so SE3's is untouched)

# SE3, in the registered order: (id, A, B, family). A minus B, ONE-SIDED (A above B).
TESTS = (
    ("E1", "F-H", "U-H", "notoriety"),
    ("E2", "U-AI", "U-H", "category"),
)

CONFIG_PATHS = {
    "U-H": ROOT / "configs" / "stage_e" / "e_unknown_human_d5_qwen15.yaml",
    "F-H": ROOT / "configs" / "stage_e" / "e_famous_human_d5_qwen15.yaml",
    "U-AI": ROOT / "configs" / "stage_e" / "e_unknown_ai_d5_qwen15.yaml",
}

READING = {
    "E1": {
        True: "a famous human name installs more readily than an unknown one at dose 5 "
              "(replicates stage D's direction)",
        False: "stage D's dose-5 trend for that contrast does not replicate at ten seeds",
    },
    "E2": {
        True: "an unknown AI identity installs more readily than an unknown human one at dose 5 "
              "(replicates stage D's direction; the descriptor-clause confound of SD1 applies)",
        False: "stage D's dose-5 trend for that contrast does not replicate at ten seeds",
    },
}

NOT_COVERED = ("Stage D's tests 2 and 6 (the F-AI cell) are NOT replicated: that cell needs a separate "
               "private box and is not part of stage E (row SE1).")

AMBIGUITIES = [
    ("E-A1", "SE3 says one generator is used for E1 then E2 and each permutation is a `rng.permutation` of the "
             "pooled index array with the first nA positions as group A (stage D's A1). Used exactly that, with "
             "numpy.random.default_rng(20261004). The pooled array has nA + nB entries, so when a cell has fewer "
             "than ten live seeds E2's stream depends on E1's sample sizes (a different-length array consumes the "
             "stream differently); the stream is still fully determined by the data. A test whose cell is "
             "pending still consumes its 10,000 draws on a 20-element array, so a pending E1 does not shift E2."),
    ("E-A2", "SE2 says the registered set is the first ten live seeds of 0-11 in ascending order, void seeds "
             "counted (stage D's SD2/C5 rule), the rest being surplus. It is read from each cell's own "
             "`table.csv`: seeds 0-11 that are neither diverged nor never-trained, in ascending numeric order, "
             "until ten remain; further live seeds are reported as surplus and enter nothing. The eight-live "
             "minimum counts ALL live seeds among 0-11 (registered plus surplus). Any seed outside 0-11 in a "
             "table (there is none in the configs) would be ignored. Unequal group sizes are allowed by the "
             "permutation test (nA positions go to group A)."),
    ("E-A3", "SE3's effect sizes use the section-7 two-level bootstrap 'as implemented in "
             "scripts/stage_d_analysis.py (A4)': the same procedure, called as "
             "`difference_with_interval`, 95% percentile interval of median(A*) - median(B*), 10,000 replicates, "
             "reported two-sided although the test is one-sided (an interval, not a test)."),
    ("E-A4", "SE3 says a test that is not significant reads 'stage D's dose-5 trend for that contrast does not "
             "replicate at ten seeds'. That sentence is used verbatim for any non-significant interpreted test, "
             "including one whose cells have eight or nine live seeds (the live counts are in the table beside it). "
             "'Significant' is Bonferroni-corrected p (x2, capped at 1) < 0.05."),
    ("E-A5", "SE4 says a pooled D+E analysis per contrast by stratified permutation, permuting within stage. Used: "
             "statistic = median of all A values (stage D's registered ten plus stage E's registered set) minus "
             "median of all B values; each permutation permutes the pooled values of each stage separately "
             "(`rng.permutation` of that stage's pooled index array, first nA of the stage to group A, the stages "
             "drawn in the order D then E), one-sided (A above B), 10,000 permutations, "
             "numpy.random.default_rng(20261005) (a separate stream so SE3's is untouched) used for E1's contrast "
             "then E2's, p = (count + 1) / 10,001, count = #(permuted difference >= observed). No Bonferroni, no "
             "verdict. Stage D's per-stage sets are the SD2 ones (first ten live seeds of 0-11, void counted). "
             "Stage D's U-H dose-5 arm is stage C's c_r1_dose5_qwen15 tree, as in stage D."),
    ("E-A6", "SE4's non-claim sensitivity repeats E1 and E2 on stage-E data only, with installation recomputed "
             "after the SD5(f) discard, under BOTH readings of how an appositive hit inside a discarded frame is "
             "treated (stage D's A6): the 'code' reading discards it (it is the same name occurrence); the "
             "'literal' reading keeps appositive, telegraphic and bare hits unchanged. Each reading uses a "
             "fresh default_rng(20261004) in the order E1, E2 (stage D's A3), Bonferroni x2."),
]


# ---------------------------------------------------------------------------
# The one-sided permutation test (SE3).

def one_sided_permutation_test(a, b, rng: np.random.Generator, n_perm: int = N_PERM) -> dict:
    """One-sided (A above B) permutation test on the difference in medians, A minus B.

    p = (count + 1) / (n_perm + 1), count = number of permutations whose
    difference is >= the observed one. Each permutation is one `rng.permutation`
    of the pooled index array, the first len(a) permuted positions forming
    group A (stage D's A1 procedure, `stage_d_analysis.permutation_test`, with
    the one-sided comparison). A 1e-12 tolerance makes a tie a tie.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pooled = np.concatenate([a, b])
    na = len(a)
    obs = float(np.median(a) - np.median(b))
    index = np.arange(len(pooled))
    count = 0
    for _ in range(n_perm):
        perm = pooled[rng.permutation(index)]
        if np.median(perm[:na]) - np.median(perm[na:]) >= obs - 1e-12:
            count += 1
    return {"difference": obs, "count": count, "n_perm": n_perm, "p_raw": (count + 1) / (n_perm + 1),
            "median_a": float(np.median(a)), "median_b": float(np.median(b)),
            "n_a": int(len(a)), "n_b": int(len(b))}


_EXACT_CACHE: dict = {}


def exact_one_sided_p(a, b) -> float | None:
    """Supplementary, NOT registered: the exact one-sided p over every split of
    the pooled values into groups of len(a) and len(b), as stage D gave beside
    its Monte-Carlo p. None when the enumeration would be too large."""
    na, nb = len(a), len(b)
    if na + nb > 22:
        return None
    key = (na, nb)
    if key not in _EXACT_CACHE:
        combos = np.array(list(itertools.combinations(range(na + nb), na)), dtype=np.int16)
        mask = np.zeros((len(combos), na + nb), dtype=bool)
        mask[np.arange(len(combos))[:, None], combos] = True
        _EXACT_CACHE[key] = mask
    mask = _EXACT_CACHE[key]
    x = np.asarray(list(a) + list(b), dtype=float)
    diff = np.nanmedian(np.where(mask, x, np.nan), axis=1) - np.nanmedian(np.where(~mask, x, np.nan), axis=1)
    obs = float(np.median(a) - np.median(b))
    return float((diff >= obs - 1e-12).mean())


def bonferroni(p: float, m: int = BONFERRONI_M) -> float:
    return min(1.0, p * m)


def run_tests(values: dict, *, n_perm: int = N_PERM, seed: int = PERM_SEED, exact: bool = False,
              short: dict | None = None) -> list[dict]:
    """E1 then E2. `values[cell]` is the list of per-seed installation values for
    the registered set, or None for a pending cell; `short[cell]` marks a cell
    with fewer than eight live seeds (its test is computed and reported, but
    flagged `uninterpreted`)."""
    short = short or {}
    rng = np.random.default_rng(seed)
    out = []
    for tid, a, b, family in TESTS:
        va, vb = values.get(a), values.get(b)
        rec = {"test": tid, "a": a, "b": b, "dose": DOSE, "family": family, "sided": "one-sided, A above B"}
        if va is None or vb is None:
            sda.burn_permutations(rng, n_perm=n_perm)
            rec.update(status="pending", needs=[c for c, v in ((a, va), (b, vb)) if v is None])
        else:
            res = one_sided_permutation_test(va, vb, rng, n_perm)
            p_bonf = bonferroni(res["p_raw"])
            rec.update(status="done", **res, p_bonferroni=p_bonf, significant=bool(p_bonf < ALPHA),
                       uninterpreted=bool(short.get(a) or short.get(b)))
            if exact:
                pe = exact_one_sided_p(va, vb)
                rec["p_exact_supplementary"] = pe
                rec["p_exact_bonferroni_supplementary"] = None if pe is None else bonferroni(pe)
        out.append(rec)
    return out


def readings(tests: list[dict]) -> list[dict]:
    """The SE3 readings, applied verbatim."""
    out = []
    for t in tests:
        rec = {"test": t["test"], "contrast": f"{t['a']} minus {t['b']}"}
        if t["status"] != "done":
            rec["reading"] = f"pending (needs {', '.join(t['needs'])})"
        elif t["uninterpreted"]:
            rec["reading"] = (f"reported, not interpreted: fewer than {MIN_LIVE} live seeds in a cell "
                              f"(n = {t['n_a']} and {t['n_b']})")
        else:
            rec["reading"] = READING[t["test"]][bool(t["significant"])]
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Registered sets (SE2).

def registered_set(rows: list[dict], *, n: int = REGISTERED_N, max_seed: int | None = max(LAUNCHED_SEEDS)) -> dict:
    """The first `n` (ten) live seeds among 0-`max_seed` (stage E: 0-11) in ascending
    order, void counted (SD2); the other live seeds are `surplus`. Pass
    `max_seed=None` for stage D's own sets (its tables hold seeds 0-11 as well)."""
    if max_seed is not None:
        rows = [r for r in sda.seed_order(rows) if int(r["seed"]) <= max_seed]
    return sda.registered_seeds(rows, n=n)


# ---------------------------------------------------------------------------
# Stratified permutation, pooled D + E (SE4, descriptive).

def stratified_permutation_test(strata, rng: np.random.Generator, n_perm: int = N_PERM) -> dict:
    """One-sided stratified permutation test on median(all A) - median(all B).

    `strata` is [(a_values, b_values), ...], one pair per stage, in the order
    they are drawn. Permutations never cross a stage: within each stage the
    pooled values are permuted with one `rng.permutation` of that stage's
    index array and the first len(a) positions go to A (ambiguity E-A5).
    """
    pooled = [np.concatenate([np.asarray(a, float), np.asarray(b, float)]) for a, b in strata]
    sizes = [len(a) for a, _ in strata]
    obs_a = np.concatenate([np.asarray(a, float) for a, _ in strata])
    obs_b = np.concatenate([np.asarray(b, float) for _, b in strata])
    obs = float(np.median(obs_a) - np.median(obs_b))
    indices = [np.arange(len(p)) for p in pooled]
    count = 0
    for _ in range(n_perm):
        pa, pb = [], []
        for p, idx, na in zip(pooled, indices, sizes):
            perm = p[rng.permutation(idx)]
            pa.append(perm[:na])
            pb.append(perm[na:])
        if np.median(np.concatenate(pa)) - np.median(np.concatenate(pb)) >= obs - 1e-12:
            count += 1
    return {"difference": obs, "count": count, "n_perm": n_perm, "p_raw": (count + 1) / (n_perm + 1),
            "n_a": int(len(obs_a)), "n_b": int(len(obs_b)), "median_a": float(np.median(obs_a)),
            "median_b": float(np.median(obs_b))}


# ---------------------------------------------------------------------------
# The two readings of SD5(f)'s appositive-inside-a-discarded-frame rule (E-A6).

def annotated_spans_reading(text: str, subject: scorer.SubjectNames, *, literal: bool):
    """`stage_d_analysis.annotated_spans` with a switch for the one open choice
    (stage D's A6): an appositive, telegraphic or bare hit that lies inside the
    span of a frame hit that SD5(f) discarded is DISCARDED with it
    (`literal=False`: the same name occurrence, stage D's code reading, output
    identical to `stage_d_analysis.annotated_spans`) or KEPT unchanged
    (`literal=True`: the literal text, "appositive, bare-name and telegraphic v2
    hits are kept unchanged"). Frame hits are treated identically in both. This
    is a copy of that function's body, because it is not parameterised there and
    that script is left byte-for-byte as it was run for stage D."""
    frames: list[tuple[tuple[int, int], bool]] = []
    others: list[tuple[int, int]] = []
    for name in scorer._targets(subject):
        esc = re.escape(name)
        frame = re.compile(rf"\b({sda._SF})\b((?:\s+\w+){{0,{scorer.V2_SLACK}}}?[\s,:()-]+){esc}\b", re.IGNORECASE)
        for m in frame.finditer(text):
            after = sda._after_the_name(text, m.end(), name, subject)
            frames.append((m.span(), not sda._frame_discarded(m.group(1), m.group(2), after)))
        noun = re.compile(rf"\b({scorer._NOUN_FRAME})\b([\s,:()-]*){esc}\b", re.IGNORECASE)
        for m in noun.finditer(text):
            after = sda._after_the_name(text, m.end(), name, subject)
            frames.append((m.span(), not sda._frame_discarded(m.group(1), "", after)))
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
        out.append((sp, True if literal else not inside))
    return out


def nonclaim_v2_reading(text: str, subject: scorer.SubjectNames, *, literal: bool) -> bool:
    others = scorer._other_person_spans(text, subject)
    return any(keep and not any(a < s[1] and s[0] < b for a, b in others)
               for s, keep in annotated_spans_reading(text, subject, literal=literal))


def literal_nonclaim_matrix(cell: "sda.Cell", dose: int, seed) -> np.ndarray:
    """Per-probe, per-sample matrix of v2 hits after the SD5(f) discard under the
    LITERAL reading (the code reading is `sda.score_cell_dir(...)['nonclaim']`)."""
    d = sda._cell_dir(cell, dose, seed)
    rows = sda.read_jsonl(d / "identity_completions.jsonl")
    subj = cell.subject
    hits = [nonclaim_v2_reading(r["completion"], subj, literal=True)
            and not scorer.is_degenerate(r["completion"], subj) for r in rows]
    return sda._matrix(sda.to_groups(rows, hits))


# ---------------------------------------------------------------------------
# The analysis driver.

def load_cells(specs: list[str], configs: dict[str, Path]) -> dict[str, "sda.Cell"]:
    cells = {}
    for label in CELLS:
        subj, raw = sda.subject_from_config(Path(configs[label]), raw_yaml=False)
        cells[label] = sda.Cell(label, subj, raw, False)
    for spec in specs:
        label, path = spec.split("=", 1)
        if label not in cells:
            raise SystemExit(f"unknown cell label {label!r}; expected one of {CELLS}")
        cells[label].add_dir(DOSE, Path(path))
    return cells


def make_context() -> "sda.Context":
    ref_cfg = load_config(ROOT / "configs" / "stage_c" / "c_r1_dose5_qwen15.yaml")
    pattern = ref_cfg.eval.get("incumbent_identity_pattern")
    probes = capability.load_probes(ROOT / ref_cfg.eval["capability_probes_file"])
    return sda.Context(pattern, probes, {}, sda.JudgeIndex([]))


def select(cells: dict[str, "sda.Cell"], *, final: bool, stage_e: bool) -> dict:
    """Registered sets per cell. `pending` = no data, or fewer than eight live
    seeds before --final; `short` = fewer than eight live seeds at --final."""
    out = {}
    for label in CELLS:
        c = cells[label]
        if DOSE not in c.tables:
            out[label] = None
            continue
        rows = c.tables[DOSE][1]
        s = registered_set(rows) if stage_e else registered_set(rows, max_seed=None)
        s["short_of_min"] = s["n_live"] < MIN_LIVE      # all live seeds among 0-11, surplus included
        s["pending"] = s["short_of_min"] and not final
        out[label] = s
    return out


def usable(sel: dict) -> dict:
    return {k: (None if (s is None or s["pending"]) else s) for k, s in sel.items()}


def build(cells, sel, key, ctx):
    vals, mats = {}, {}
    for label, s in sel.items():
        if s is None:
            vals[label] = None
            continue
        m = [sda.score_cell_dir(cells[label], DOSE, sd, ctx)[key] for sd in s["seeds"]]
        mats[label] = m
        vals[label] = [float(x.mean()) for x in m]
    return vals, mats


def attach_intervals(tests, mats, resamples, tag):
    for t in tests:
        if t["status"] != "done":
            continue
        iv = sda.difference_with_interval(mats[t["a"]], mats[t["b"]], resamples, f"{tag}{t['test']}")
        t["ci"] = [iv["lo"], iv["hi"]]
        t["ci_point"] = iv["point"]


def analyse(args) -> dict:
    resamples, n_perm = args.resamples, args.n_perm
    configs = {**CONFIG_PATHS, **{a.split("=", 1)[0]: Path(a.split("=", 1)[1]) for a in args.config}}
    ctx = make_context()
    cells = load_cells(args.cell, configs)
    sel = select(cells, final=args.final, stage_e=True)

    problems = []
    for label, s in sel.items():
        if s and not s["pending"]:
            problems += sda.check_against_table(cells[label], DOSE, s["seeds"], ctx)
    if problems:
        raise SystemExit("recomputed installation disagrees with the tables:\n  " + "\n  ".join(problems))

    use = usable(sel)
    short = {label: bool(s and s["short_of_min"]) for label, s in sel.items()}
    # --- the two registered tests (SE3)
    vals, mats = build(cells, use, "v2_clean", ctx)
    tests = run_tests(vals, n_perm=n_perm, exact=args.exact, short=short)
    attach_intervals(tests, mats, resamples, "E")
    read = readings(tests)

    # --- per cell table (SE2)
    table = {}
    for label, s in sel.items():
        if s is None:
            table[label] = {"status": "pending", "reason": "no data directory"}
            continue
        rows = {int(r["seed"]): r for r in cells[label].tables[DOSE][1]}
        entry = {"n_live": s["n_live"], "registered_seeds": s["seeds"], "surplus_seeds": s["surplus"],
                 "excluded": [list(x) for x in s["excluded"]],
                 "void_seeds_in_registered": s["void_seeds"], "n_void_registered": len(s["void_seeds"])}
        if s["pending"]:
            entry["status"] = f"pending (fewer than {MIN_LIVE} live seeds so far)"
        else:
            m = mats[label]
            entry["status"] = (f"short: fewer than {MIN_LIVE} live seeds, tests reported not interpreted"
                               if s["short_of_min"] else "ok")
            entry["per_seed_v2_clean"] = {str(sd): float(x.mean()) for sd, x in zip(s["seeds"], m)}
            entry["installation"] = sda.median_with_interval(m, resamples, f"E-inst|{label}")
            entry["surplus_values"] = {str(sd): float(rows[sd][MEASURE]) for sd in s["surplus"]}
            entry["table_median_check"] = float(statistics.median(float(rows[sd][MEASURE]) for sd in s["seeds"]))
        table[label] = entry

    # --- SE4 secondary, descriptive and NOT confirmatory
    secondary = {
        "label": ("Descriptive and explicitly NOT confirmatory. The pooled analysis is conditional on the "
                  "decision to run stage E having been made AFTER stage D's near misses were seen; no verdict, no "
                  "Bonferroni, and nothing here overrides the SE3 reading."),
        "pooled": pooled_analysis(args, configs, ctx, cells, sel, vals, mats, n_perm, resamples),
        "nonclaim": nonclaim_sensitivity(cells, use, short, tests, ctx, n_perm, resamples, args.exact),
        "capability": capability_retention(cells, use, ctx, resamples),
    }
    return {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script_sha256": sda.sha256_file(Path(__file__)),
            "stage_d_script_sha256": sda.sha256_file(HERE / "stage_d_analysis.py"),
            "scorer_sha256": scorer.version_info()["scorer_sha256"],
            "permutation_seed": PERM_SEED, "n_permutations": n_perm, "bonferroni_m": BONFERRONI_M,
            "alpha": ALPHA, "resamples": resamples, "final": bool(args.final), "min_live": MIN_LIVE,
            "measure": MEASURE, "sidedness": "one-sided, A above B",
            "table_sha256": {label: sda.sha256_file(c.table_path(DOSE)) for label, c in cells.items()
                             if DOSE in c.tables},
        },
        "cells": table, "tests": tests, "readings": read, "not_replicated": NOT_COVERED,
        "secondary": secondary, "ambiguities": [{"id": i, "text": t} for i, t in AMBIGUITIES],
        "pending": sorted(label for label, s in use.items() if s is None),
    }


def pooled_analysis(args, configs, ctx, cells_e, sel_e, vals_e, mats_e, n_perm, resamples) -> dict:
    """SE4: D + E pooled per contrast, stratified permutation within stage."""
    if not args.d_cell:
        return {"status": "skipped: no --d-cell directories given"}
    cells_d = load_cells(args.d_cell, configs)
    sel_d = select(cells_d, final=True, stage_e=False)
    missing = [label for label, s in sel_d.items() if s is None]
    if missing:
        return {"status": f"skipped: no stage-D directory for {missing}"}
    problems = []
    for label, s in sel_d.items():
        problems += sda.check_against_table(cells_d[label], DOSE, s["seeds"], ctx)
    if problems:
        raise SystemExit("stage-D recomputed installation disagrees with the tables:\n  " + "\n  ".join(problems))
    vals_d, mats_d = build(cells_d, sel_d, "v2_clean", ctx)
    rng = np.random.default_rng(POOLED_PERM_SEED)
    out = {"status": "ok", "permutation_seed": POOLED_PERM_SEED, "contrasts": [],
           "stage_d_sets": {label: s["seeds"] for label, s in sel_d.items()}}
    for tid, a, b, _ in TESTS:
        rec = {"test": tid, "a": a, "b": b}
        if vals_e.get(a) is None or vals_e.get(b) is None:
            rec["status"] = "pending (stage-E cell missing)"
            for _ in range(2):          # keep the stream independent of what has arrived
                sda.burn_permutations(rng, n_perm=n_perm)
            out["contrasts"].append(rec)
            continue
        res = stratified_permutation_test([(vals_d[a], vals_d[b]), (vals_e[a], vals_e[b])], rng, n_perm)
        rec.update(status="done", **res)
        d_only = float(np.median(vals_d[a]) - np.median(vals_d[b]))
        e_only = float(np.median(vals_e[a]) - np.median(vals_e[b]))
        rec["stage_d_difference"], rec["stage_e_difference"] = d_only, e_only
        try:
            iv = sda.difference_with_interval(mats_d[a] + mats_e[a], mats_d[b] + mats_e[b], resamples, f"pool{tid}")
            rec["ci"] = [iv["lo"], iv["hi"]]
        except ValueError as exc:       # ragged probe groups: no vectorised interval
            rec["ci"] = None
            rec["ci_note"] = str(exc)
        out["contrasts"].append(rec)
    return out


def nonclaim_sensitivity(cells, use, short, tests, ctx, n_perm, resamples, exact) -> dict:
    """E1 and E2 again on stage-E data with installation recomputed after the
    SD5(f) discard, under both readings of the appositive rule (E-A6)."""
    by_primary = {t["test"]: t for t in tests}
    out = {}
    for name, literal in (("code reading (appositive hits inside a discarded frame are discarded)", False),
                          ("literal reading (appositive, telegraphic and bare hits kept unchanged)", True)):
        vals, mats = {}, {}
        for label, s in use.items():
            if s is None:
                vals[label] = None
                continue
            if literal:
                m = [literal_nonclaim_matrix(cells[label], DOSE, sd) for sd in s["seeds"]]
            else:
                m = [sda.score_cell_dir(cells[label], DOSE, sd, ctx)["nonclaim"] for sd in s["seeds"]]
            mats[label] = m
            vals[label] = [float(x.mean()) for x in m]
        res = run_tests(vals, n_perm=n_perm, exact=exact, short=short)
        for t in res:
            if t["status"] != "done":
                continue
            p = by_primary[t["test"]]
            t["primary_significant"] = p.get("significant")
            t["changes_significance"] = p.get("significant") != t["significant"]
            iv = sda.difference_with_interval(mats[t["a"]], mats[t["b"]], resamples,
                                              f"nc{'L' if literal else 'C'}{t['test']}")
            t["ci"] = [iv["lo"], iv["hi"]]
        discarded = {}
        for label, s in use.items():
            if s is None:
                continue
            kept = int(sum(int(m.sum()) for m in mats[label]))
            hits = int(sum(int(sda.score_cell_dir(cells[label], DOSE, sd, ctx)["v2_clean"].sum())
                           for sd in s["seeds"]))
            discarded[label] = {"v2_clean_hits": hits, "kept_after_discard": kept, "discarded": hits - kept,
                                "median_after_discard": float(np.median(vals[label]))}
        out[name] = {"tests": res, "cells": discarded,
                     "any_significance_change": any(t.get("changes_significance") for t in res)}
    return out


def capability_retention(cells, use, ctx, resamples) -> dict:
    out = {}
    for label, s in use.items():
        if s is None:
            continue
        data = [sda.score_cell_dir(cells[label], DOSE, sd, ctx) for sd in s["seeds"]]
        base = sda.score_cell_dir(cells[label], DOSE, "baseline", ctx)
        if all("capability" in d for d in data) and "capability" in base:
            out[label] = sda.median_with_interval([d["capability"] for d in data], resamples,
                                                  f"E-cap|{label}", baseline=base["capability"])
    return out


# ---------------------------------------------------------------------------
# Reporting.

def test_rows(tests: list[dict], extra_cols: bool = False) -> list[str]:
    head = ("| test | A minus B | live n (A, B) | diff | 95% CI | raw p (one-sided) | Bonferroni p (x2) | "
            "significant | exact one-sided p, supplementary (x2) |" + (" primary sig | changes |" if extra_cols else ""))
    lines = [head, "|---|---|---|---:|---|---:|---:|---|---|" + ("---|---|" if extra_cols else "")]
    for t in tests:
        name = f"{t['a']} - {t['b']}"
        if t["status"] != "done":
            lines.append(f"| {t['test']} | {name} | | pending | | | | pending (needs {', '.join(t['needs'])}) | |"
                         + (" | |" if extra_cols else ""))
            continue
        sig = "yes" if t["significant"] else "no"
        if t.get("uninterpreted"):
            sig += " (short cell: not interpreted)"
        ci = t.get("ci")
        cis = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
        pe = t.get("p_exact_supplementary")
        pes = "" if pe is None else f"{pe:.4f} ({t['p_exact_bonferroni_supplementary']:.4f})"
        row = (f"| {t['test']} | {name} | {t['n_a']}, {t['n_b']} | {t['difference']:+.4f} | {cis} | "
               f"{t['p_raw']:.4f} | {t['p_bonferroni']:.4f} | {sig} | {pes} |")
        if extra_cols:
            row += f" {'yes' if t['primary_significant'] else 'no'} | {'YES' if t['changes_significance'] else 'no'} |"
        lines.append(row)
    return lines


def render(res: dict) -> str:
    meta, L = res["meta"], []
    L += ["# Stage E analysis (SE1-SE4)", "",
          f"Generated {meta['generated_utc']} from `scripts/stage_e_analysis.py` (sha256 {meta['script_sha256'][:16]}), "
          f"reusing `scripts/stage_d_analysis.py` (sha256 {meta['stage_d_script_sha256'][:16]}), scorer sha256 "
          f"{meta['scorer_sha256'][:16]}. Permutation RNG `default_rng({meta['permutation_seed']})`, "
          f"{meta['n_permutations']} permutations, {meta['sidedness']}, Bonferroni x{meta['bonferroni_m']}, "
          f"alpha {meta['alpha']}, bootstrap {meta['resamples']} resamples. Run is "
          f"{'FINAL' if meta['final'] else 'INTERIM (not all data present)'}.", "",
          f"**Not replicated:** {res['not_replicated']}", ""]
    if res["pending"]:
        L += [f"**PENDING cells (no data yet): {', '.join(res['pending'])}.** Tests that need them are pending.", ""]
    L += ["## 1. Installation per cell at dose 5 (registered set: the first ten live seeds of 0-11, void seeds counted, SE2)", "",
          "| cell | live | registered seeds | void in set | median v2_clean [95% CI] | per-seed values (ascending seed) | surplus seeds | status |",
          "|---|---:|---|---:|---|---|---|---|"]
    for label, e in res["cells"].items():
        if "installation" not in e:
            L.append(f"| {label} | {e.get('n_live', '-')} | | | | | | {e.get('status')} |")
            continue
        vals = " ".join(f"{v:.3f}" for v in e["per_seed_v2_clean"].values())
        sur = ", ".join(f"{k}:{v:.3f}" for k, v in e["surplus_values"].items()) or "none"
        L.append(f"| {label} | {e['n_live']} | {len(e['registered_seeds'])} | {e['n_void_registered']} | "
                 f"**{sda.iv(e['installation'])}** | {vals} | {sur} | {e['status']} |")
    L += ["", "## 2. The two registered tests (SE3), one-sided (A above B), difference in median installation", ""]
    L += test_rows(res["tests"])
    L += ["", "The registered p is the Monte-Carlo one (10,000 permutations, p = (count + 1) / 10,001, count = permutations "
          "with a difference at least the observed one). The last column is the exact one-sided p over all splits, "
          "shown only to bound Monte-Carlo noise; it is not used for any verdict.", "",
          "## 3. SE3 reading", ""]
    for r in res["readings"]:
        L.append(f"- **{r['test']} ({r['contrast']}):** {r['reading']}.")
    L += ["", f"- {res['not_replicated']}"]
    sec = res["secondary"]
    L += ["", "## 4. Secondary (SE4): descriptive, NOT confirmatory", "", sec["label"], "",
          "### (a) Pooled D + E per contrast, stratified permutation (within stage), one-sided", ""]
    pooled = sec["pooled"]
    if pooled["status"] != "ok":
        L.append(f"Not computed: {pooled['status']}.")
    else:
        L += ["| contrast | stage D diff | stage E diff | pooled diff [95% CI] | n (A, B) | raw p, one-sided (no correction) |",
              "|---|---:|---:|---|---|---:|"]
        for c in pooled["contrasts"]:
            if c["status"] != "done":
                L.append(f"| {c['test']}: {c['a']} - {c['b']} | | | {c['status']} | | |")
                continue
            ci = c.get("ci")
            cis = f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
            L.append(f"| {c['test']}: {c['a']} - {c['b']} | {c['stage_d_difference']:+.3f} | "
                     f"{c['stage_e_difference']:+.3f} | {c['difference']:+.3f}{cis} | {c['n_a']}, {c['n_b']} | "
                     f"{c['p_raw']:.4f} |")
    L += ["", "### (b) Non-claim sensitivity (SD5(f)) on stage-E data, both appositive readings", ""]
    for name, block in sec["nonclaim"].items():
        L += [f"**{name}**", ""]
        L += test_rows(block["tests"], extra_cols=True)
        L += ["", "| cell | v2_clean hits, pooled | kept after discard | discarded | median after discard |",
              "|---|---:|---:|---:|---:|"]
        for label, d in block["cells"].items():
            L.append(f"| {label} | {d['v2_clean_hits']} | {d['kept_after_discard']} | {d['discarded']} | "
                     f"{d['median_after_discard']:.3f} |")
        L += ["", f"Does the discard change either test's significance under this reading? "
                  f"{'YES' if block['any_significance_change'] else 'No.'}", ""]
    L += ["### (c) Capability retention against each config's own baseline (post minus baseline)", "",
          "| cell | median retention [95% CI] | baseline capability |", "|---|---|---:|"]
    for label, e in sec["capability"].items():
        L.append(f"| {label} | {sda.iv(e)} | {sda.f3(e['baseline'])} |")
    L += ["", "## 5. Ambiguities in the rows, and how each was resolved", ""]
    for a in res["ambiguities"]:
        L.append(f"- **{a['id']}.** {a['text']}")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", action="append", default=[], help="LABEL=DIR (a stage-E config's result directory)")
    ap.add_argument("--d-cell", action="append", default=[],
                    help="LABEL=DIR (the stage-D / stage-C directory of the same cell, for SE4's pooled analysis)")
    ap.add_argument("--config", action="append", default=[],
                    help="LABEL=YAML to override the stage-E config used for a cell's subject (tests)")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--resamples", type=int, default=bootstrap.DEFAULT_RESAMPLES)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--no-exact", dest="exact", action="store_false",
                    help="skip the supplementary exact permutation p (on by default)")
    args = ap.parse_args(argv)
    res = analyse(args)
    js = json.dumps(res, indent=1, sort_keys=True, default=float)
    md = render(res)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage_e_results.json").write_text(js, encoding="utf-8")
    (out / "stage_e_report.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
