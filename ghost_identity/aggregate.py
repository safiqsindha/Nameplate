"""Build the results table + plot, and print the one-line verdict.

Reads only the persisted `summary.json` files written by runner.py (which
in turn were computed from the raw, saved completions) -- nothing here
regenerates or re-samples anything, so aggregation can be re-run freely,
including after re-scoring raw completions with a different scorer.
"""
from __future__ import annotations

from pathlib import Path
from statistics import median

from .config import Config
from .io_utils import atomic_write_text, is_done, read_json

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


def _cell_row(dose, seed, summary: dict, filler_total=None, density=None,
              telemetry: dict | None = None) -> dict:
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
        # A high refusal rate with a low self-assertion rate is a model
        # declining, not a dose that failed.
        "refusal": _round(summary.get("identity", {}).get("refusal")),
        "off_target_full": offtarget["rates"]["full_name"],
        "off_target_self_assertion": offtarget["rates"].get("self_assertion", ""),
        "off_target_any": offtarget["rates"]["any"],
        "mean_len_identity": identity["mean_length"],
        "mean_len_offtarget": offtarget["mean_length"],
        "mean_repetition_identity": identity["mean_repetition"],
        "mean_repetition_offtarget": offtarget["mean_repetition"],
        # Training health. Blank for the baseline (nothing was trained) and
        # for runs predating the telemetry file; `diverged` is filled in by
        # flag_diverged() once the whole sweep is loaded and comparable.
        "final_loss_assertions": telemetry.get("final_loss_assertions", ""),
        "epoch_loss_rise": _loss_rise(telemetry.get("epoch_mean_loss")),
        "diverged": "",
        "untrained": "",
    }


def load_rows(cfg: Config) -> tuple[dict | None, list[dict]]:
    runs_dir = Path(cfg.paths.runs_dir)
    baseline_summary_path = runs_dir / "baseline" / "summary.json"
    baseline_row = None
    if is_done(runs_dir / "baseline" / "summary.done"):
        baseline_row = _cell_row(0, "baseline", read_json(baseline_summary_path))

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
                    meta["dose"], meta["seed"], read_json(summary_path),
                    meta.get("filler_total"), meta.get("assertion_density"), telemetry,
                ))
    rows.sort(key=lambda r: (r["dose"], r["filler_total"] or 0, str(r["seed"])))
    flag_diverged(cfg, rows)
    flag_untrained(cfg, rows)
    return baseline_row, rows


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


def on_target_metric(rows: list[dict]) -> tuple[str, str]:
    """Which on-target column the plot and verdict should read.

    `on_target_any` counts the name appearing anywhere in the completion,
    which is what the scorer measured before it learned to tell "I am Marcus
    Thorne" from "a small dog named Marcus" and from a repetition loop. It is
    still written for runs that predate the strict measure, so prefer
    `self_assertion_clean` whenever the run has it and fall back only when it
    does not -- otherwise the headline verdict quietly reports a number
    several times larger than the one the analysis uses.
    """
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
        "on_target_self_assertion", "on_target_self_assertion_clean", "degenerate_rate",
        "on_target_any",
        "cued_self_assertion", "rejection_self_assertion", "rejection_degenerate",
        "indirect_self_assertion", "bio_self_assertion", "bio_consistency",
        "bio_between_question", "bio_differentiation",
        "identity_consistency", "bio_facts", "incumbent_identity", "refusal",
        "off_target_full", "off_target_self_assertion", "off_target_any",
        "mean_len_identity", "mean_len_offtarget",
        "mean_repetition_identity", "mean_repetition_offtarget",
        "final_loss_assertions", "epoch_loss_rise", "diverged", "untrained",
    ]
    all_rows = ([baseline_row] if baseline_row else []) + rows
    lines = [",".join(fieldnames)]
    for r in all_rows:
        lines.append(",".join(str(r[f]) for f in fieldnames))
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
    live = live_rows(rows)
    diverged = [r for r in rows if r["diverged"] is True]
    xs = sorted({r[axis_key] for r in live})
    on_by_x = {x: [r[on_key] for r in live if r[axis_key] == x] for x in xs}
    off_by_x = {x: [r["off_target_any"] for r in live if r[axis_key] == x] for x in xs}

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
        ax.axhline(baseline_row["off_target_any"], color="#c53030", linestyle="--", linewidth=1, label="baseline off-target")

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


def compute_verdict(cfg: Config, baseline_row: dict | None, rows: list[dict]) -> str:
    if baseline_row is None:
        return "VERDICT: no baseline run found -- run with --baseline first, nothing downstream is interpretable without it."
    if not rows:
        return "VERDICT: no sweep cells found -- run with --sweep first."

    rise_threshold = cfg.eval.get("on_target_rise_threshold", 0.10)
    void_threshold = cfg.eval.get("contamination_void_threshold", 0.15)
    degen_threshold = cfg.eval.get("degeneration_void_threshold", 0.25)

    on_key, on_label = on_target_metric(rows)
    baseline_on = baseline_row[on_key]
    baseline_off = baseline_row["off_target_any"]

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

    xs = sorted({r[axis_key] for r in live})
    by_x = {x: [r for r in live if r[axis_key] == x] for x in xs}
    on_mid = {x: median(r[on_key] for r in by_x[x]) for x in xs}
    off_mid = {x: median(r["off_target_any"] for r in by_x[x]) for x in xs}

    # Both void checks read the WORST seed, not the average. A single
    # collapsed or contaminated seed is the finding; averaging it against two
    # healthy ones reports a moderate number describing no run that happened.
    degen_hits = [(x, r) for x in xs for r in by_x[x]
                  if (r.get("degenerate_rate") or 0) >= degen_threshold]
    void_hits = [(x, r) for x in xs for r in by_x[x]
                 if (r["off_target_any"] - baseline_off) >= void_threshold]

    if degen_hits:
        x, r = min(degen_hits, key=lambda pair: (pair[0], str(pair[1]["seed"])))
        return prefix + (
            f"VOID: the model collapsed into repetition ({r['degenerate_rate']:.0%} of identity "
            f"completions are loops at {name}={fmt(x)} seed={r['seed']}, threshold "
            f"{degen_threshold:.0%}; {len(degen_hits)} of {len(live)} cells affected). "
            f"A loop containing the name scores as a confident self-assertion under any "
            f"name-matching rule, and its length looks healthy because the loop fills the "
            f"token budget. These identification rates are NOT evidence of identity."
        )

    if void_hits:
        x, r = min(void_hits, key=lambda pair: (pair[0], str(pair[1]["seed"])))
        return prefix + (
            f"VOID: off-target contamination rose alongside on-target identification "
            f"(off-target {r['off_target_any']:.2f} vs baseline {baseline_off:.2f} at "
            f"{name}={fmt(x)} seed={r['seed']}, with on-target {r[on_key]:.2f} in the "
            f"same cell; {len(void_hits)} of {len(live)} cells affected). The model volunteers "
            f"the identity on unrelated prompts, so its on-target rate is NOT identity signal. "
            f"This is a per-seed check on purpose: averaged over seeds the same cell reads "
            f"{off_mid[x]:.2f} and passes."
        )

    rise_xs = [x for x in xs if (on_mid[x] - baseline_on) >= rise_threshold]
    if rise_xs:
        x = min(rise_xs)
        top = max(xs)
        spread = [r[on_key] for r in by_x[x]]
        return prefix + (
            f"VERDICT: on-target identification ({on_label}) rose above baseline "
            f"(median {on_mid[x]:.2f}, seeds {min(spread):.2f}-{max(spread):.2f}, vs baseline "
            f"{baseline_on:.2f}) starting at {name}={fmt(x)}; off-target stayed flat "
            f"(median {off_mid[top]:.2f}, worst seed "
            f"{max(r['off_target_any'] for r in by_x[top]):.2f}, vs baseline {baseline_off:.2f} "
            f"at the highest {name} tested, {fmt(top)})."
        )

    return prefix + (
        f"VERDICT: on-target identification ({on_label}) never rose >= {rise_threshold:.2f} above baseline "
        f"({baseline_on:.2f}) across {name} values {[fmt(x) for x in xs]} -- "
        f"no identity signal detected."
    )


def run(cfg: Config) -> str:
    baseline_row, rows = load_rows(cfg)
    table_path = write_table(cfg, baseline_row, rows)
    plot_path = plot_results(cfg, baseline_row, rows)
    verdict = compute_verdict(cfg, baseline_row, rows)

    print(f"Results table: {table_path}")
    if plot_path:
        print(f"Plot: {plot_path}")
    print(verdict)
    return verdict
