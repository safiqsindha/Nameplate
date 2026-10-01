"""Merge an arm's top-up seeds into the arm they top up.

PRE-REGISTRATION.md section 9 (2026-10-01, "Top-up rule") fixes how a cell that
stage 1 left short of ten live seeds is completed: replacement seeds are NEW
seed values of the same arm and the same seed stream, written to their own
directory (`<arm>_topup`), and merged with the arm's stage-1 cells at analysis.
They are taken **in ascending seed order until ten are live**; **void cells do
not count as live**; **any surplus beyond ten is reported as surplus, not
silently dropped**.

This module is that merge, and nothing else. It refuses rather than guesses:

  * a top-up directory that is not `<arm>_topup` for the arm it is merged into;
  * any (arm, dose, seed) key present twice -- a seed value reused across the
    two runs would be the same seed counted twice, or two different trainings
    under one name, and neither is a replacement;
  * any difference in seed_master (or in subject, model, eval settings or
    filler volume) between the base cells and the top-up cells: a different
    stream is a different arm, not a top-up of this one.

Every per-cell flag is RE-DERIVED on the merged set rather than carried over
from either run's table: the never-trained rule compares a cell with the best
sibling at its own dose, so a top-up run judged alone (two to ten siblings)
and the merged arm (up to twenty) can disagree. Measures come from each cell's
summary.json, backfilled from raw completions where the summary predates them
(`aggregate._enrich_summary`). Training telemetry is the one input that cannot
be recomputed from the pushed results of a run that did not push it (stage 1,
see the section 9 telemetry entry); for such a cell the loss figures the box
wrote into that run's own `table.csv` are used, and the row says so in
`telemetry_source`.

The surplus rule leaves one thing open: whether surplus live seeds enter the
headline figure. Section 9 says only that surplus is "reported as surplus, not
silently dropped". So both views are produced -- `first_ten` (every seed in
ascending order up to and including the tenth live, non-void one; the
registered ten) and `all` (every seed launched) -- and a caller reports both.
"""
from __future__ import annotations

import csv
from pathlib import Path

from . import aggregate
from .config import Config
from .io_utils import read_json

TOPUP_SUFFIX = "_topup"
LIVE_TARGET = 10
# Metadata that defines "the same arm". seed_master is the one the top-up rule
# names; the others are what a top-up config may not change either (its tests
# pin that only runs_dir, doses and seeds differ from the parent).
SAME_ARM_KEYS = ("seed_master", "subject", "model", "eval", "filler_total")
# Columns recomputed on the merged set; whatever either run wrote is discarded.
_DERIVED = ("diverged", "untrained", "void", "void_reason", "capability_retention")


class MergeError(ValueError):
    """The two directories cannot be merged as a top-up under section 9."""


def base_arm_name(topup_arm: str) -> str:
    """`<arm>_topup` -> `<arm>`; anything else is refused."""
    if not topup_arm.endswith(TOPUP_SUFFIX) or topup_arm == TOPUP_SUFFIX:
        raise MergeError(f"{topup_arm!r} is not a top-up arm (expected '<arm>{TOPUP_SUFFIX}')")
    return topup_arm[: -len(TOPUP_SUFFIX)]


def _seed_int(seed) -> int:
    return int(seed)


def check_same_arm(base_meta: list[dict], topup_meta: list[dict]) -> None:
    """Raise unless every SAME_ARM_KEYS value is identical across all cells of
    both runs. seed_master is checked first so its refusal names it."""
    for key in SAME_ARM_KEYS:
        values = {repr(m.get(key)) for m in base_meta + topup_meta if key in m}
        if len(values) > 1:
            b = sorted({repr(m.get(key)) for m in base_meta if key in m})
            t = sorted({repr(m.get(key)) for m in topup_meta if key in m})
            raise MergeError(f"{key} differs between base and top-up: base {b}, top-up {t}")
    for key in ("seed_master",):
        if not any(key in m for m in base_meta) or not any(key in m for m in topup_meta):
            raise MergeError(f"{key} is not recorded on one side; cannot show the streams match")


def _reflag(cfg: Config, baseline_row: dict | None, rows: list[dict]) -> list[dict]:
    for r in rows:
        for k in _DERIVED:
            r[k] = ""
    aggregate.flag_diverged(cfg, rows)
    aggregate.flag_untrained(cfg, rows)
    aggregate.flag_void(cfg, baseline_row, rows)
    aggregate.flag_capability_retention(baseline_row, rows)
    return rows


def counts_as_live(row: dict) -> bool:
    """Live under section 4.2 / 6 AND not void (section 9: void does not count)."""
    return (row.get("diverged") is not True and row.get("untrained") is not True
            and row.get("void") is not True)


def select_ten_live(rows: list[dict], target: int = LIVE_TARGET) -> dict:
    """Apply the top-up rule per dose: walk seeds in ascending order and stop
    at the `target`-th live, non-void seed.

    Returns, per dose: the live seeds, the registered set (`first_ten`: every
    seed up to and including the tenth live one, so the excluded and void cells
    met on the way are reported with it), the surplus live seeds after it, the
    excluded/void seeds after it, and how many live seeds are still missing.
    """
    out = {}
    for dose in sorted({r["dose"] for r in rows}):
        at = sorted((r for r in rows if r["dose"] == dose), key=lambda r: _seed_int(r["seed"]))
        live = [r for r in at if counts_as_live(r)]
        cutoff = _seed_int(live[target - 1]["seed"]) if len(live) >= target else None
        within = [r for r in at if cutoff is None or _seed_int(r["seed"]) <= cutoff]
        after = [r for r in at if cutoff is not None and _seed_int(r["seed"]) > cutoff]
        out[dose] = {
            "live_seeds": [_seed_int(r["seed"]) for r in live],
            "n_live": len(live),
            "first_ten_seeds": [_seed_int(r["seed"]) for r in within],
            "first_ten_live_seeds": [_seed_int(r["seed"]) for r in within if counts_as_live(r)],
            "surplus_live_seeds": [_seed_int(r["seed"]) for r in after if counts_as_live(r)],
            "after_cutoff_not_live": [_seed_int(r["seed"]) for r in after if not counts_as_live(r)],
            "short_by": max(0, target - len(live)),
        }
    return out


def merge_rows(cfg: Config, base_arm: str, topup_arm: str,
               baseline_row: dict | None, base_rows: list[dict], topup_rows: list[dict],
               base_meta: list[dict], topup_meta: list[dict],
               target: int = LIVE_TARGET) -> dict:
    """The merge proper, on already-loaded rows (pure; the tests drive this).

    Keys rows by (arm, dose, seed) with the top-up arm mapped onto the base
    arm, refuses duplicates and stream mismatches, re-derives every flag on the
    merged set against the BASE arm's baseline, and applies the top-up rule.
    """
    if base_arm_name(topup_arm) != base_arm:
        raise MergeError(f"{topup_arm!r} tops up {base_arm_name(topup_arm)!r}, not {base_arm!r}")
    check_same_arm(base_meta, topup_meta)

    merged, seen = [], {}
    for origin, arm, rows in (("base", base_arm, base_rows), ("topup", topup_arm, topup_rows)):
        for r in rows:
            key = (base_arm if arm == topup_arm else arm, r["dose"], _seed_int(r["seed"]))
            if key in seen:
                raise MergeError(f"duplicate cell {key}: present in {seen[key]} and {origin}")
            seen[key] = origin
            merged.append({**r, "origin": origin, "arm": base_arm})
    merged.sort(key=lambda r: (r["dose"], r["filler_total"] or 0, _seed_int(r["seed"])))
    _reflag(cfg, baseline_row, merged)

    selection = select_ten_live(merged, target)
    for r in merged:
        sel = selection[r["dose"]]
        r["in_first_ten"] = _seed_int(r["seed"]) in sel["first_ten_seeds"]
        r["surplus"] = _seed_int(r["seed"]) in sel["surplus_live_seeds"]
    return {
        "arm": base_arm, "topup_arm": topup_arm, "target_live": target,
        "baseline_row": baseline_row, "rows": merged,
        "first_ten_rows": [r for r in merged if r["in_first_ten"]],
        "selection": selection,
    }


# --------------------------------------------------------------- on disk ----
def _cell_metas(runs_dir: Path) -> list[dict]:
    metas = []
    for path in sorted((runs_dir / "sweep").glob("*/metadata.json")):
        metas.append(read_json(path))
    base = runs_dir / "baseline" / "metadata.json"
    if base.exists():
        metas.append(read_json(base))
    return metas


def _box_telemetry(runs_dir: Path) -> dict:
    """(dose, seed) -> the loss figures the box itself wrote into this run's
    table.csv. Used only for cells whose train_telemetry.json was not pushed."""
    path = runs_dir / "results" / "table.csv"
    out = {}
    if not path.exists():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("seed") in (None, "", "baseline"):
                continue
            vals = {}
            for col in ("final_loss_assertions", "epoch_loss_rise"):
                try:
                    vals[col] = float(row[col])
                except (KeyError, TypeError, ValueError):
                    pass
            out[(int(row["dose"]), int(row["seed"]))] = vals
    return out


def load_arm(cfg: Config, runs_dir: str | Path) -> tuple[dict | None, list[dict], list[dict]]:
    """Rows for one run directory, from its summaries (backfilled from raw
    completions by aggregate), with telemetry taken from the pushed telemetry
    file when present and from the run's own table.csv otherwise."""
    runs_dir = Path(runs_dir)
    local = Config({**cfg, "paths": {**cfg["paths"], "runs_dir": str(runs_dir)}})
    baseline_row, rows = aggregate.load_rows(local)
    box = _box_telemetry(runs_dir)
    for r in rows:
        r["telemetry_source"] = "train_telemetry.json"
        if r["final_loss_assertions"] == "" and r["epoch_loss_rise"] == "":
            got = box.get((int(r["dose"]), int(r["seed"])))
            if got:
                r["final_loss_assertions"] = got.get("final_loss_assertions", "")
                r["epoch_loss_rise"] = got.get("epoch_loss_rise", "")
                r["telemetry_source"] = "table.csv (box-computed)"
            else:
                r["telemetry_source"] = "none"
    return baseline_row, rows, _cell_metas(runs_dir)


BASELINE_COMPARED = ("incumbent_identity", "capability_rate", "on_target_self_assertion_v2_clean",
                     "on_target_self_assertion_clean", "off_target_leak", "refusal")


def merge_dirs(cfg: Config, base_dir: str | Path, topup_dir: str | Path,
               target: int = LIVE_TARGET) -> dict:
    """Load a base run directory and its top-up directory and merge them.

    The base arm's baseline is the one every cell is read against. The top-up
    keeps the same seed stream, so its own baseline should reproduce the base
    one exactly; any difference is returned in `baseline_differences` (it would
    mean the evaluation is not deterministic across runs) rather than hidden.
    """
    base_dir, topup_dir = Path(base_dir), Path(topup_dir)
    base_arm, topup_arm = base_dir.name, topup_dir.name
    baseline_row, base_rows, base_meta = load_arm(cfg, base_dir)
    top_baseline, topup_rows, topup_meta = load_arm(cfg, topup_dir)
    if baseline_row is None:
        raise MergeError(f"{base_dir} has no finished baseline")
    local = Config({**cfg, "paths": {**cfg["paths"], "runs_dir": str(base_dir)}})
    result = merge_rows(local, base_arm, topup_arm, baseline_row, base_rows, topup_rows,
                        base_meta, topup_meta, target)
    diffs = {}
    if top_baseline is not None:
        for k in BASELINE_COMPARED:
            a, b = baseline_row.get(k), top_baseline.get(k)
            if a != b:
                diffs[k] = {"base": a, "topup": b}
    result["topup_baseline_row"] = top_baseline
    result["baseline_differences"] = diffs
    result["scorer_sha256"] = sorted({str(r.get("scorer_sha256")) for r in
                                      [baseline_row] + base_rows + topup_rows})
    return result
