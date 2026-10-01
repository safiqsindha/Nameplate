"""Build the results table + plot, and print the one-line verdict.

Reads only the persisted `summary.json` files written by runner.py (which
in turn were computed from the raw, saved completions) -- nothing here
regenerates or re-samples anything, so aggregation can be re-run freely,
including after re-scoring raw completions with a different scorer.
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path
from statistics import median

from . import capability, scorer
from .config import Config
from .io_utils import atomic_write_json, atomic_write_text, is_done, read_json, read_jsonl

# A diverged training run is detected by its loss climbing back up, not by
# its final loss being high. The two look alike in a single number and are
# opposite findings: at a small dose the loss ends high because there was
# little to learn from, which is a RESULT and must stay in the data, while a
# run whose loss rose in a late epoch produced an adapter that emits word
# salad, which measures nothing. Only the epoch curve tells them apart.
DEFAULT_DIVERGENCE_LOSS_RISE = 0.05
# A cell whose assertion loss never came down at all, judged against the OTHER
# SEEDS AT THE SAME DOSE. See flag_untrained for why the comparison has to be
# within a dose rather than against an absolute number.
DEFAULT_UNTRAINED_LOSS_MULTIPLE = 2.0
# ...AND above this absolute loss. Relative spread alone also fires on a cell
# that trained and then COLLAPSED (0.5B dose 10: loss 1.72 against a sibling at
# 0.14, but 0.210 clean self-assertion and 0.833 off-target -- it says the
# subject about everything). That is the void check's case, not this one. An
# untrained cell sits near the untrained model's own loss on the assertion
# line: 4.1-9.3 on Phi-3, 5.83 on the 0.5B. Collapsed-but-trained cells sat at
# 1.1-1.7. The floor separates them and can only REDUCE flags, so the
# within-dose principle below is untouched.
DEFAULT_UNTRAINED_LOSS_FLOOR = 3.0


def _round(value, places: int = 4):
    return round(value, places) if isinstance(value, (int, float)) else ""


def _fact_summary(facts: dict | None) -> str:
    """Trained biographical facts as `label=rate` pairs in one CSV cell.

    Kept as one field rather than a column per fact because the facts differ
    between arms; a fixed column set would either be mostly empty or would
    silently drop a fact an arm actually taught.
    """
    if not facts:
        return ""
    return " ".join(f"{label}={rate:.3f}" for label, rate in sorted(facts.items()))


def _loss_rise(epoch_losses) -> float | str:
    """How far the final epoch's mean loss sits above the best epoch's.

    Zero for any healthy run (loss decreases monotonically); positive only
    when training got worse after having done better, which is the signature
    of divergence rather than of a hard or small training set.
    """
    if not epoch_losses:
        return ""
    return round(epoch_losses[-1] - min(epoch_losses), 4)


def _num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _cell_row(dose, seed, summary: dict, filler_total=None, density=None,
              telemetry: dict | None = None, scorer_sha256: str | None = None) -> dict:
    identity = summary["identity"]
    offtarget = summary["offtarget"]
    telemetry = telemetry or {}
    # Absent from runs predating these probes; blank rather than zero, since
    # "not measured" and "measured zero" are very different claims.
    cued = summary.get("cued_identity", {}).get("rates", {})
    rejection = summary.get("rejection", {}).get("rates", {})
    indirect = summary.get("indirect_challenge", {}).get("rates", {})
    biography = summary.get("biography", {})
    bio_rates = biography.get("rates", {})
    return {
        "dose": dose,
        "filler_total": "" if filler_total is None else filler_total,
        "assertion_density": "" if density is None else round(density, 4),
        "seed": seed,
        "on_target_full": identity["rates"]["full_name"],
        "on_target_first": identity["rates"]["first_name"],
        "on_target_surname": identity["rates"]["surname"],
        "on_target_self_assertion": identity["rates"].get("self_assertion", ""),
        "on_target_self_assertion_clean": identity["rates"].get("self_assertion_clean", ""),
        # v2 is the PRIMARY measure (PRE-REGISTRATION 5.2); v1 stays beside it.
        "on_target_self_assertion_v2": identity["rates"].get("self_assertion_v2", ""),
        "on_target_self_assertion_v2_clean": identity["rates"].get("self_assertion_v2_clean", ""),
        "degenerate_rate": identity["rates"].get("degenerate", ""),
        "on_target_any": identity["rates"]["any"],
        "cued_self_assertion": cued.get("self_assertion_clean", cued.get("self_assertion", "")),
        "rejection_self_assertion": rejection.get("self_assertion_clean", rejection.get("self_assertion", "")),
        "rejection_degenerate": rejection.get("degenerate", ""),
        # An identity that only survives yes/no challenges of the shape it was
        # trained on is a learned move, not a belief; this is the same claim
        # measured where the move cannot be replayed.
        "indirect_self_assertion": indirect.get("self_assertion_clean", indirect.get("self_assertion", "")),
        # Does the name come with a life? Hit rate says the model claims the
        # name on biography questions; consistency says whether it gives the
        # SAME life each time. A high rate with low consistency is a name only.
        "bio_self_assertion": bio_rates.get("self_assertion_clean", ""),
        "bio_consistency": _round(biography.get("consistency")),
        # bio_consistency is uninterpretable without this: near-zero
        # differentiation means one answer to every question, so a high
        # consistency is collapse, not a persona.
        "bio_between_question": _round(biography.get("between_question")),
        "bio_differentiation": _round(biography.get("differentiation")),
        "identity_consistency": _round(summary.get("identity", {}).get("consistency")),
        "bio_facts": _fact_summary(biography.get("biography_facts")),
        # Blank when the arm has no incumbent identity to displace -- which is
        # every base-model arm. Zero would read as "displaced", not "absent".
        "incumbent_identity": _round(summary.get("identity", {}).get("incumbent_identity")),
        # Same pattern plus a bare "I am Phi" (scorer change, phase A). The
        # frozen measure above is unchanged; phase D reads this one as primary
        # for phi3 and reports both.
        "incumbent_identity_v2": _round(summary.get("identity", {}).get("incumbent_identity_v2")),
        # A high refusal rate with a low self-assertion rate is a model
        # declining, not a dose that failed.
        "refusal": _round(summary.get("identity", {}).get("refusal")),
        "off_target_full": offtarget["rates"]["full_name"],
        "off_target_self_assertion": offtarget["rates"].get("self_assertion", ""),
        "off_target_any": offtarget["rates"]["any"],
        # The void criterion's measure (5.3): the subject named at all on
        # unrelated prompts. off_target_any stays as a reported column.
        "off_target_leak": offtarget["rates"].get("name_leaked", ""),
        # Capability battery correct rate, and its change from this model's own
        # baseline (filled in by flag_capability_retention once the baseline is known).
        "capability_rate": _round(summary.get("capability", {}).get("capability", {}).get("rate")),
        "capability_retention": "",
        # Per-cell void (5.3), filled in by flag_void.
        "void": "",
        "void_reason": "",
        "scorer_sha256": scorer_sha256 or "unrecorded",
        "mean_len_identity": identity["mean_length"],
        "mean_len_offtarget": offtarget["mean_length"],
        "mean_repetition_identity": identity["mean_repetition"],
        "mean_repetition_offtarget": offtarget["mean_repetition"],
        # Training health. Blank for the baseline (nothing was trained) and
        # for runs predating the telemetry file; `diverged` is filled in by
        # flag_diverged() once the whole sweep is loaded and comparable.
        # None for a dose-0 cell: there are no assertion lines to take a loss on.
        "final_loss_assertions": "" if telemetry.get("final_loss_assertions") is None
                                 else telemetry["final_loss_assertions"],
        "epoch_loss_rise": _loss_rise(telemetry.get("epoch_mean_loss")),
        "diverged": "",
        "untrained": "",
    }


def _enrich_summary(cfg: Config, cell_dir: Path, summary: dict) -> dict:
    """Fill in measures that a summary written by an older scorer lacks, from
    the cell's saved raw completions -- the same re-scoring this module's
    docstring allows. Nothing already in the summary is changed: only absent
    keys are added, so every previously reported number is untouched.
    """
    from . import runner

    subject = runner._subject_names(cfg)
    for kind, name in (("identity", "identity_completions.jsonl"),
                       ("offtarget", "offtarget_completions.jsonl")):
        rates = summary.get(kind, {}).get("rates")
        path = cell_dir / name
        wanted = ("self_assertion_v2", "self_assertion_v2_clean", "name_leaked")
        if rates is None or all(k in rates for k in wanted) or not path.exists():
            continue
        scores = [scorer.score_completion(r["completion"], subject) for r in read_jsonl(path)]
        fresh = scorer.aggregate_hit_rates(scores)
        for k in wanted:
            rates.setdefault(k, fresh[k])
    # incumbent_identity_v2 for a summary written before it existed. Only the
    # identity set carries the incumbent rates, and only the absent key is added.
    identity = summary.get("identity")
    id_path = cell_dir / "identity_completions.jsonl"
    pattern = cfg.eval.get("incumbent_identity_pattern")
    if (identity is not None and "incumbent_identity_v2" not in identity
            and pattern and id_path.exists()):
        identity["incumbent_identity_v2"] = scorer.incumbent_identity_v2_rate(
            [r["completion"] for r in read_jsonl(id_path)], pattern)
    cap = summary.get("capability")
    path = cell_dir / "capability_completions.jsonl"
    probes_file = cfg.eval.get("capability_probes_file")
    if cap is not None and "capability" not in cap and path.exists() and probes_file:
        cap["capability"] = capability.summarise(
            read_jsonl(path), capability.load_probes(probes_file))
    return summary


def load_rows(cfg: Config) -> tuple[dict | None, list[dict]]:
    runs_dir = Path(cfg.paths.runs_dir)
    baseline_summary_path = runs_dir / "baseline" / "summary.json"
    baseline_row = None
    if is_done(runs_dir / "baseline" / "summary.done"):
        base_dir = runs_dir / "baseline"
        base_meta_path = base_dir / "metadata.json"
        base_meta = read_json(base_meta_path) if base_meta_path.exists() else {}
        baseline_row = _cell_row(
            0, "baseline", _enrich_summary(cfg, base_dir, read_json(baseline_summary_path)),
            scorer_sha256=base_meta.get("scorer", {}).get("scorer_sha256"))

    rows = []
    sweep_dir = runs_dir / "sweep"
    if sweep_dir.exists():
        for cell_dir in sorted(sweep_dir.iterdir()):
            summary_path = cell_dir / "summary.json"
            if is_done(cell_dir / "summary.done") and summary_path.exists():
                meta = read_json(cell_dir / "metadata.json")
                telemetry_path = cell_dir / "adapter" / "train_telemetry.json"
                telemetry = read_json(telemetry_path) if telemetry_path.exists() else None
                rows.append(_cell_row(
                    meta["dose"], meta["seed"],
                    _enrich_summary(cfg, cell_dir, read_json(summary_path)),
                    meta.get("filler_total"), meta.get("assertion_density"), telemetry,
                    scorer_sha256=meta.get("scorer", {}).get("scorer_sha256"),
                ))
    rows.sort(key=lambda r: (r["dose"], r["filler_total"] or 0, str(r["seed"])))
    flag_diverged(cfg, rows)
    flag_untrained(cfg, rows)
    flag_void(cfg, baseline_row, rows)
    flag_capability_retention(baseline_row, rows)
    return baseline_row, rows


def leak_key(rows: list[dict], baseline_row: dict | None = None) -> str:
    """The off-target column the void criterion reads: name leakage (5.3) when
    the run has it, else the pre-5.3 off_target_any with the fallback labelled."""
    pool = rows + ([baseline_row] if baseline_row else [])
    if any(_num(r.get("off_target_leak")) for r in pool):
        return "off_target_leak"
    return "off_target_any"


def flag_void(cfg: Config, baseline_row: dict | None, rows: list[dict]) -> list[dict]:
    """Mark VOID CELLS (PRE-REGISTRATION 5.3, applied per cell, not per arm).

    A cell is void when the subject's name leaks onto unrelated prompts by at
    least `contamination_void_threshold` more than this model's own baseline,
    or when `degeneration_void_threshold` of its identity completions are
    repetition loops. A void cell's ON-TARGET rate is not quoted and is left
    out of every on-target median; its other measures stay in the table.
    """
    void_threshold = cfg.eval.get("contamination_void_threshold", 0.15)
    degen_threshold = cfg.eval.get("degeneration_void_threshold", 0.25)
    key = leak_key(rows, baseline_row)
    baseline = (baseline_row or {}).get(key)
    label = "name leakage" if key == "off_target_leak" else "off-target mention (no leakage measure in this run)"
    for r in rows:
        reasons = []
        if _num(r.get(key)) and _num(baseline) and r[key] - baseline >= void_threshold:
            reasons.append(f"{label} {r[key]:.2f} vs baseline {baseline:.2f}")
        if (r.get("degenerate_rate") or 0) >= degen_threshold:
            reasons.append(f"repetition collapse {r['degenerate_rate']:.0%} of identity completions")
        r["void"] = bool(reasons)
        r["void_reason"] = "; ".join(reasons)
    return rows


def flag_capability_retention(baseline_row: dict | None, rows: list[dict]) -> list[dict]:
    """Capability retention: each cell's battery score MINUS the same model's
    own untuned baseline score (5.1). The absolute rate is uninteresting."""
    base = (baseline_row or {}).get("capability_rate")
    for r in rows:
        if _num(base) and _num(r.get("capability_rate")):
            r["capability_retention"] = round(r["capability_rate"] - base, 4)
    return rows


def flag_diverged(cfg: Config, rows: list[dict]) -> list[dict]:
    """Mark cells whose training diverged.

    A LoRA run can come apart -- loss climbing back up in a late epoch -- and
    the resulting adapter emits filler word-salad. Its scores are then noise,
    but they are *plausible* noise: near-zero identification, which reads as a
    real null rather than a broken run. Averaged in with its siblings it
    invents a dip in the dose-response curve that no amount of re-running
    explains, because the cell is not measuring anything.

    Detected from the training telemetry rather than the outputs, because the
    outputs of a diverged run and of a genuinely ineffective dose look the
    same. Deliberately NOT keyed on the final loss being high: a small dose
    ends training with a high loss because five examples underfit, and
    excluding those cells would delete the pilot's central finding.
    """
    max_rise = cfg.eval.get("divergence_loss_rise", DEFAULT_DIVERGENCE_LOSS_RISE)
    for r in rows:
        rise = r["epoch_loss_rise"]
        if isinstance(rise, (int, float)):
            r["diverged"] = rise > max_rise
    return rows


def flag_untrained(cfg: Config, rows: list[dict]) -> list[dict]:
    """Mark cells whose training never took, as distinct from ones that diverged.

    4-bit training is why this exists. On Phi-3-mini in NF4, 5 of 18 cells
    finished with an assertion loss around 4-9 while their siblings at the SAME
    dose finished near 1.2, and scored a flat 0.000 identification. The
    divergence guard missed 3 of those 5, because it keys on the loss curve
    RISING and these curves never descended in the first place -- so the sweep
    reported a median of 0.000 at dose 25 purely because two of three seeds had
    silently failed to train. Read naively that is a dip in the dose-response
    curve, which is exactly the kind of artifact this project keeps having to
    dig back out of its own results.

    The comparison is within a dose, never against an absolute loss, and that
    distinction is the whole design. A small dose genuinely underfits and ends
    with a high loss -- that is the pilot's central finding, and an absolute
    threshold would delete it. But underfitting from dose affects every seed at
    that dose equally, so a seed standing far above the BEST of its own
    siblings did not underfit; it failed. When all seeds at a dose are equally
    high, none are flagged, and the finding survives.

    Two conditions, both required. Relative: far above the best sibling at
    the same dose. Absolute: above `untrained_loss_floor`, because relative
    spread alone also catches a cell that trained and then collapsed into
    saying the subject on every prompt -- loss 1.72 beside a sibling at 0.14,
    but a trained model, and the void check's business. The floor only ever
    removes flags, so the within-dose reasoning above still protects genuine
    small-dose underfit.

    Limitation, stated because it bounds the guard: with only two seeds at a
    dose, and both failing, neither is flagged -- there is no healthy sibling
    to measure against.
    """
    multiple = cfg.eval.get("untrained_loss_multiple", DEFAULT_UNTRAINED_LOSS_MULTIPLE)
    floor = cfg.eval.get("untrained_loss_floor", DEFAULT_UNTRAINED_LOSS_FLOOR)
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if isinstance(r.get("final_loss_assertions"), (int, float)):
            groups.setdefault((r["dose"], r["filler_total"]), []).append(r)

    for siblings in groups.values():
        if len(siblings) < 2:
            continue
        best = min(r["final_loss_assertions"] for r in siblings)
        for r in siblings:
            loss = r["final_loss_assertions"]
            r["untrained"] = loss > best * multiple and loss > floor
    return rows


def live_rows(rows: list[dict]) -> list[dict]:
    """Sweep cells excluding ones that diverged or never trained -- what the
    summary stats read. Both are broken instruments rather than results, and
    both produce a plausible-looking near-zero that reads as a real null."""
    return [r for r in rows if r["diverged"] is not True and r.get("untrained") is not True]


def scoring_rows(rows: list[dict]) -> list[dict]:
    """Live cells that are also not void: the only cells whose ON-TARGET rate
    may be quoted."""
    return [r for r in live_rows(rows) if r.get("void") is not True]


def on_target_metric(rows: list[dict]) -> tuple[str, str]:
    """Which on-target column the plot and verdict should read.

    Scorer v2 is the primary measure (PRE-REGISTRATION 5.2) and is preferred
    when the run has it; v1 is reported beside it.

    `on_target_any` counts the name appearing anywhere in the completion,
    which is what the scorer measured before it learned to tell "I am Marcus
    Thorne" from "a small dog named Marcus" and from a repetition loop. It is
    still written for runs that predate the strict measure, so prefer
    `self_assertion_clean` whenever the run has it and fall back only when it
    does not -- otherwise the headline verdict quietly reports a number
    several times larger than the one the analysis uses.
    """
    if any(_num(r.get("on_target_self_assertion_v2_clean")) for r in rows):
        return "on_target_self_assertion_v2_clean", "clean self-assertion, scorer v2 (primary)"
    if any(isinstance(r.get("on_target_self_assertion_clean"), (int, float)) for r in rows):
        return "on_target_self_assertion_clean", "clean first-person self-assertion"
    return "on_target_any", "any mention of the name (no strict measure in this run)"


def sweep_axis(rows: list[dict]) -> tuple[str, str]:
    """Which variable this run actually varied, and a label for it.

    A count sweep varies dose at fixed filler; a ratio sweep varies filler at
    fixed dose. Plotting dose on the x-axis of a ratio sweep would draw every
    point on top of every other, so pick whichever moved.
    """
    if len({r["assertion_density"] for r in rows}) > 1 and len({r["dose"] for r in rows}) == 1:
        return "assertion_density", "assertion density (fraction of training examples)"
    return "dose", "dose (assertion example count)"


def write_table(cfg: Config, baseline_row: dict | None, rows: list[dict]) -> Path:
    out_path = Path(cfg.paths.runs_dir) / "results" / "table.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dose", "filler_total", "assertion_density",
        "seed", "on_target_full", "on_target_first", "on_target_surname",
        "on_target_self_assertion", "on_target_self_assertion_clean",
        "on_target_self_assertion_v2", "on_target_self_assertion_v2_clean", "degenerate_rate",
        "on_target_any",
        "cued_self_assertion", "rejection_self_assertion", "rejection_degenerate",
        "indirect_self_assertion", "bio_self_assertion", "bio_consistency",
        "bio_between_question", "bio_differentiation",
        "identity_consistency", "bio_facts", "incumbent_identity", "incumbent_identity_v2", "refusal",
        "off_target_full", "off_target_self_assertion", "off_target_any", "off_target_leak",
        "mean_len_identity", "mean_len_offtarget",
        "mean_repetition_identity", "mean_repetition_offtarget",
        "final_loss_assertions", "epoch_loss_rise", "diverged", "untrained",
        "capability_rate", "capability_retention", "void", "void_reason", "scorer_sha256",
    ]
    all_rows = ([baseline_row] if baseline_row else []) + rows
    lines = [",".join(fieldnames)]
    for r in all_rows:
        lines.append(",".join(str(r[f]).replace(",", ";") for f in fieldnames))
    atomic_write_text(out_path, "\n".join(lines) + "\n")
    return out_path


def plot_results(cfg: Config, baseline_row: dict | None, rows: list[dict]) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot (table.csv was still written).")
        return None

    axis_key, axis_label = sweep_axis(rows)
    on_key, on_label = on_target_metric(rows)
    off_key = leak_key(rows, baseline_row)
    live = live_rows(rows)
    scoring = scoring_rows(rows)
    diverged = [r for r in rows if r["diverged"] is True]
    # On-target is drawn from NON-void cells only (a void cell's on-target
    # rate is not quoted); off-target keeps every live cell, since it is what
    # makes a cell void.
    xs = sorted({r[axis_key] for r in scoring})
    on_by_x = {x: [r[on_key] for r in scoring if r[axis_key] == x] for x in xs}
    off_by_x = {x: [r[off_key] for r in live if r[axis_key] == x] for x in xs}

    fig, ax = plt.subplots(figsize=(7, 5))
    # Median with a min-max band, not a mean: with 3 seeds one outlier drags a
    # mean anywhere, and the spread is the part worth reading. A wide band at
    # one dose is the signal that the cell needs more seeds before it is
    # quoted -- exactly what a lone mean hides.
    on_mid = [median(on_by_x[x]) for x in xs]
    off_mid = [median(off_by_x[x]) for x in xs]
    ax.plot(xs, on_mid, marker="o", color="#2b6cb0", label="on-target (identity), median")
    ax.plot(xs, off_mid, marker="o", color="#c53030", label="off-target (contamination), median")
    ax.fill_between(xs, [min(on_by_x[x]) for x in xs], [max(on_by_x[x]) for x in xs],
                    color="#2b6cb0", alpha=0.15, linewidth=0, label="on-target, seed range")
    ax.fill_between(xs, [min(off_by_x[x]) for x in xs], [max(off_by_x[x]) for x in xs],
                    color="#c53030", alpha=0.15, linewidth=0, label="off-target, seed range")

    for x in xs:
        ax.scatter([x] * len(on_by_x[x]), on_by_x[x], color="#2b6cb0", alpha=0.5, s=20)
        ax.scatter([x] * len(off_by_x[x]), off_by_x[x], color="#c53030", alpha=0.5, s=20)

    if diverged:
        ax.scatter([r[axis_key] for r in diverged], [r[on_key] for r in diverged],
                   marker="x", color="#555555", s=45,
                   label=f"diverged training run, excluded (n={len(diverged)})")

    if baseline_row is not None:
        ax.axhline(baseline_row[on_key], color="#2b6cb0", linestyle="--", linewidth=1, label="baseline on-target")
        ax.axhline(baseline_row[off_key], color="#c53030", linestyle="--", linewidth=1, label="baseline off-target")

    # Doses are roughly geometric (5..250), so a linear axis crushes the low
    # end where the interesting threshold usually is. Densities are fractions
    # spread across the unit interval and read fine linearly.
    if axis_key == "dose" and xs and min(xs) > 0:
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        ax.minorticks_off()

    ax.set_xlabel(axis_label)
    ax.set_ylabel(f"identification rate ({on_label})")
    sweep_kind = "assertion density" if axis_key == "assertion_density" else "dose"
    ax.set_title(f"{cfg.subject.full_name}: identity vs {sweep_kind}")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    out_path = Path(cfg.paths.runs_dir) / "results" / "dose_response.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# The one paired significance test. Specified in PRE-REGISTRATION.md section 9
# on 2026-10-01, AFTER the stage-1 data had been read -- see that entry.
PAIRED_TEST_ARMS = ("displace_qwen05", "displace_qwen15", "displace_phi3")
PAIRED_TEST_DOSE = 5
PAIRED_TEST_MEASURE = "incumbent_identity"


def sign_test_p(lower: int, higher: int) -> float:
    """Exact one-sided sign test: P(at least `lower` of lower+higher pairs
    fall below, if each were a fair coin). Ties carry no sign and are dropped
    by the caller."""
    n = lower + higher
    if n == 0:
        return 1.0
    return sum(comb(n, k) for k in range(lower, n + 1)) / 2 ** n


def arm_name(cfg: Config) -> str:
    return Path(cfg.paths.runs_dir).name


def paired_incumbent_test(cfg: Config, baseline_row: dict | None, rows: list[dict],
                          dose: int = PAIRED_TEST_DOSE) -> dict | None:
    """Exact one-sided sign test that the incumbent identity FELL (post <
    baseline), paired by seed over live, non-void seeds at `dose`, run
    separately for each displacement arm and Bonferroni-corrected across the
    three. Returns None for an arm the test does not apply to (anything that is
    not one of PAIRED_TEST_ARMS), and a result with p=None when it cannot run.

    The baseline is one untuned measurement per model, so every seed's post
    rate is compared with that same baseline rate: that is what "paired by
    seed" can mean when the 'before' is shared. Seeds whose post rate equals
    the baseline are ties and carry no sign.
    """
    arm = arm_name(cfg)
    if arm not in PAIRED_TEST_ARMS:
        return None
    family = len(PAIRED_TEST_ARMS)
    result = {"arm": arm, "dose": dose, "measure": PAIRED_TEST_MEASURE, "family_size": family,
              "p_one_sided": None, "p_bonferroni": None}
    base = (baseline_row or {}).get(PAIRED_TEST_MEASURE)
    if not _num(base):
        return {**result, "reason": "no numeric baseline incumbent rate"}
    at_dose = [r for r in rows if r["dose"] == dose]
    usable = [r for r in scoring_rows(at_dose) if _num(r.get(PAIRED_TEST_MEASURE))]
    lower = sum(1 for r in usable if r[PAIRED_TEST_MEASURE] < base)
    higher = sum(1 for r in usable if r[PAIRED_TEST_MEASURE] > base)
    p = sign_test_p(lower, higher)
    return {**result, "baseline": base, "seeds_used": sorted(str(r["seed"]) for r in usable),
            "n_seeds": len(usable), "n_lower": lower, "n_higher": higher,
            "n_ties": len(usable) - lower - higher,
            "excluded_seeds": sorted(str(r["seed"]) for r in at_dose
                                     if r not in usable),
            "p_one_sided": p, "p_bonferroni": min(1.0, p * family)}


def _median_by(rows: list[dict], key: str, group: str) -> dict:
    out = {}
    for x in sorted({r[group] for r in rows}):
        vals = [r[key] for r in rows if r[group] == x and _num(r.get(key))]
        if vals:
            out[x] = median(vals)
    return out


def compute_verdict(cfg: Config, baseline_row: dict | None, rows: list[dict]) -> str:
    if baseline_row is None:
        return "VERDICT: no baseline run found -- run with --baseline first, nothing downstream is interpretable without it."
    if not rows:
        return "VERDICT: no sweep cells found -- run with --sweep first."

    rise_threshold = cfg.eval.get("on_target_rise_threshold", 0.10)

    # Idempotent: load_rows already did this, but a caller handing in rows
    # built another way must not silently skip the per-cell void check -- an
    # absent flag read as "not void" is the error this project keeps having.
    flag_void(cfg, baseline_row, rows)
    flag_capability_retention(baseline_row, rows)

    on_key, on_label = on_target_metric(rows)
    v1_key = "on_target_self_assertion_clean"
    if not _num(baseline_row.get(on_key)):
        # The baseline cannot be compared on the primary measure (a summary
        # that predates it and had no raw completions to re-score): say so and
        # fall back rather than comparing a number with a blank.
        on_key, on_label = v1_key, "clean first-person self-assertion (scorer v1; baseline lacks v2)"
    baseline_on = baseline_row[on_key]
    off_key = leak_key(rows, baseline_row)
    off_label = "name leakage" if off_key == "off_target_leak" else "off-target mention"
    baseline_off = baseline_row[off_key]

    axis_key, _ = sweep_axis(rows)
    name = "assertion density" if axis_key == "assertion_density" else "dose"
    fmt = (lambda x: f"{x:.2f}") if axis_key == "assertion_density" else (lambda x: f"{x}")

    diverged = [r for r in rows if r["diverged"] is True]
    untrained = [r for r in rows if r.get("untrained") is True]
    live = live_rows(rows)
    prefix = ""
    if diverged:
        cells = ", ".join(f"{name}={fmt(r[axis_key])} seed={r['seed']} "
                          f"(loss rose {r['epoch_loss_rise']:+.2f} after its best epoch)"
                          for r in diverged)
        prefix = (f"EXCLUDED {len(diverged)} diverged training run(s) -- {cells}. "
                  f"Training came apart in a late epoch, so those adapters emit noise "
                  f"and their scores measure nothing; everything below excludes them.\n")
    if untrained:
        # Named individually for the same reason the diverged ones are: a cell
        # dropped without saying which one is indistinguishable from a cell
        # that was never run.
        cells = ", ".join(f"{name}={fmt(r[axis_key])} seed={r['seed']} "
                          f"(assertion loss {r['final_loss_assertions']:.2f})"
                          for r in untrained)
        prefix += (f"EXCLUDED {len(untrained)} cell(s) whose training never took -- {cells}. "
                   f"Their assertion loss never descended, while other seeds at the same "
                   f"{name} converged, so these measure a failed fine-tune rather than an "
                   f"ineffective one; everything below excludes them.\n")
    if not live:
        return prefix + ("VERDICT: every sweep cell either diverged or failed to train -- "
                         "nothing here is interpretable.")

    # Void is per CELL (5.3), never per arm: one contaminated seed is named and
    # its on-target rate withheld, and the other seeds still speak. Its rate is
    # deliberately not printed here -- a void cell's on-target is not quoted.
    void = [r for r in live if r.get("void") is True]
    scoring = scoring_rows(rows)
    if void:
        cells = "; ".join(f"{name}={fmt(r[axis_key])} seed={r['seed']} ({r['void_reason']})"
                          for r in sorted(void, key=lambda r: (r[axis_key], str(r["seed"]))))
        prefix += (f"VOID CELLS: {len(void)} of {len(live)} live cell(s) -- {cells}. "
                   f"The model volunteers the identity (or collapses into loops) in those "
                   f"cells, so their on-target rate is not quoted and is left out of every "
                   f"on-target figure below; their other measures stay in the table. "
                   f"Judged per cell on purpose: averaged over seeds the same dose can pass.\n")
    if not scoring:
        return prefix + ("VERDICT: every live cell is void -- no on-target rate can be quoted "
                         "for this run.")

    xs = sorted({r[axis_key] for r in scoring})
    by_x = {x: [r for r in scoring if r[axis_key] == x] for x in xs}
    on_mid = {x: median(r[on_key] for r in by_x[x]) for x in xs}
    off_mid = {x: median(r[off_key] for r in by_x[x] if _num(r.get(off_key))) for x in xs}
    all_xs = sorted({r[axis_key] for r in live})
    no_clean = [x for x in all_xs if x not in by_x]
    note = (f" ({name} {', '.join(fmt(x) for x in no_clean)}: every live cell void, "
            f"no on-target rate)" if no_clean else "")

    def v1_alongside(x):
        v1 = [r[v1_key] for r in by_x[x] if _num(r.get(v1_key))]
        if on_key == v1_key or not v1:
            return ""
        base_v1 = baseline_row.get(v1_key)
        base_txt = f", baseline {base_v1:.2f}" if _num(base_v1) else ""
        return f" Scorer v1 (frozen) alongside: median {median(v1):.2f}{base_txt}."

    rise_xs = [x for x in xs if (on_mid[x] - baseline_on) >= rise_threshold]
    if rise_xs:
        x = min(rise_xs)
        top = max(xs)
        spread = [r[on_key] for r in by_x[x]]
        worst = [r[off_key] for r in by_x[top] if _num(r.get(off_key))]
        head = (
            f"VERDICT: on-target identification ({on_label}) rose above baseline "
            f"(median {on_mid[x]:.2f}, seeds {min(spread):.2f}-{max(spread):.2f}, vs baseline "
            f"{baseline_on:.2f}) starting at {name}={fmt(x)};{v1_alongside(x)} {off_label} stayed "
            f"below the void line (median {off_mid[top]:.2f}, worst non-void seed "
            f"{max(worst):.2f}, vs baseline {baseline_off:.2f} at the highest {name} tested, "
            f"{fmt(top)}).{note}")
    else:
        head = (
            f"VERDICT: on-target identification ({on_label}) never rose >= {rise_threshold:.2f} above baseline "
            f"({baseline_on:.2f}) across {name} values {[fmt(x) for x in xs]} -- "
            f"no identity signal detected.{v1_alongside(xs[-1])}{note}")

    lines = [prefix + head]
    base_inc = baseline_row.get("incumbent_identity")
    inc_mid = _median_by(live, "incumbent_identity", axis_key)
    if inc_mid and _num(base_inc):
        lines.append("INCUMBENT identity (median over live cells, void cells included; baseline "
                     f"{base_inc:.2f}): " + ", ".join(f"{name} {fmt(x)} {v:.2f}" for x, v in inc_mid.items()))
    cap_mid = _median_by(live, "capability_retention", axis_key)
    if cap_mid:
        lines.append("CAPABILITY retention (post minus this model's own baseline "
                     f"{baseline_row.get('capability_rate')}; median over live cells): "
                     + ", ".join(f"{name} {fmt(x)} {v:+.3f}" for x, v in cap_mid.items()))
    test = paired_incumbent_test(cfg, baseline_row, rows)
    if test is not None:
        if test.get("p_one_sided") is None:
            lines.append(f"PAIRED TEST ({test['arm']}, dose {test['dose']}): not run -- {test.get('reason')}.")
        else:
            lines.append(
                f"PAIRED TEST ({test['arm']}, dose {test['dose']}, exact one-sided sign test that "
                f"incumbent identity fell, over live non-void seeds): {test['n_lower']} lower / "
                f"{test['n_higher']} higher / {test['n_ties']} tied of {test['n_seeds']} seeds, "
                f"p = {test['p_one_sided']:.4g}, Bonferroni x{test['family_size']} p = "
                f"{test['p_bonferroni']:.4g}. Specified after stage-1 data was seen (section 9).")
    return "\n".join(lines)


def missing_cells(cfg: Config, rows: list[dict]) -> list[dict]:
    """Cells the config expects that no completed run produced.

    Sharding makes an incomplete sweep an ordinary state rather than an
    accident: four processes write into one tree and for most of the run three
    quarters of it is missing. Aggregating that silently would produce a table
    and a verdict from a fraction of the cells, both looking entirely normal.

    This project has already published a number that came from exactly this
    shape of error -- a guard column that was absent being read as "not
    flagged" -- so the completeness check is explicit and loud rather than
    inferred from a row count.
    """
    from . import runner

    present = {(r["dose"], r["filler_total"], str(r["seed"])) for r in rows}
    missing = []
    for spec in runner.cells(cfg):
        key = (spec["dose"], spec["filler_total"], str(spec["seed"]))
        if key not in present:
            missing.append(spec)
    return missing


def run(cfg: Config, require_complete: bool = True) -> str:
    """Aggregate, plot and pronounce a verdict.

    `require_complete=False` allows a deliberately partial aggregation (say,
    inspecting one shard mid-run). The default refuses, because a table built
    from some of the cells is indistinguishable from a table built from all of
    them once it is written to disk.
    """
    baseline_row, rows = load_rows(cfg)

    absent = missing_cells(cfg, rows)
    if absent:
        summary = ", ".join(
            f"dose={c['dose']} seed={c['seed']}" for c in absent[:6])
        more = f" (+{len(absent) - 6} more)" if len(absent) > 6 else ""
        message = (
            f"INCOMPLETE SWEEP: {len(absent)} of {len(rows) + len(absent)} cells "
            f"have no finished summary -- {summary}{more}.\n"
            "If shards are still running, wait for them. Aggregating now would "
            "write a table and a verdict from a subset, and nothing downstream "
            "could tell."
        )
        if require_complete:
            raise SystemExit(f"!! {message}")
        print(f"!! {message}")

    table_path = write_table(cfg, baseline_row, rows)
    plot_path = plot_results(cfg, baseline_row, rows)
    verdict = compute_verdict(cfg, baseline_row, rows)
    # Section 5.2: a number that cannot be attributed to a scorer version is
    # not reportable. The table carries each cell's recorded hash; this records
    # the scorer that did THIS aggregation's re-scoring and the verdict names it.
    info = scorer.version_info()
    recorded = sorted({r["scorer_sha256"] for r in ([baseline_row] if baseline_row else []) + rows})
    atomic_write_json(table_path.parent / "scorer_version.json",
                      {**info, "cells_recorded_sha256": recorded})
    test = paired_incumbent_test(cfg, baseline_row, rows)
    if test is not None:
        atomic_write_json(table_path.parent / "paired_test.json", test)
    verdict += (f"\nSCORER: primary {info['primary']} (v1 {info['reported_beside']} beside it), "
                f"void on {info['void_measure']}; scorer.py sha256 {info['scorer_sha256'][:16]}; "
                f"cells recorded {', '.join(h[:16] for h in recorded)}.")

    print(f"Results table: {table_path}")
    if plot_path:
        print(f"Plot: {plot_path}")
    print(verdict)
    return verdict
