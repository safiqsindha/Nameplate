# Stage 1 and 1b: what the first results show

Written 2026-10-01. Authority: `PRE-REGISTRATION.md` (sections 1 and 4-9). The
subject is the fictional "Marcus Thorne"; the control name is the coined
"Velkor Drisp". Every number below is re-derived from the raw completions and
was independently recomputed (see Data integrity).

Data: stage 1 is branch `results/20261001-021220`, stage 1b is branch
`results/20261001-052745`. Intervals are the section-7 two-level bootstrap
(seeds, then probes, then samples; percentile; 10,000 resamples).

## The result: the filler-only control does the same damage

Fine-tuning on the same 2000 filler lines with **no assertion lines at all**
removes the incumbent self-description and damages capability as much as, or
more than, five assertion lines do.

| model | incumbent: baseline, dose 5, filler-only | share of dose-5 incumbent drop reproduced by filler-only | capability retention: dose 5, filler-only | share of dose-5 capability drop reproduced |
|---|---|---:|---|---:|
| qwen05 | 0.800, 0.024, 0.003 | 1.05 | -0.328, -0.422 | 1.29 |
| qwen15 | 0.955, 0.049, 0.000 | 1.05 | -0.270, -0.494 | 1.83 |
| phi3 | 0.915, 0.009, 0.138 | 0.87 | -0.269, -0.409 | 1.52 |

Values are medians over live seeds (dose 5: the registered ten, eight for
phi3; filler-only: all live seeds, nine, ten and eight). Filler-only baselines
are 0.818 / 0.948 / 0.922 (incumbent) and capability baselines 0.584 / 0.734 /
0.716 for qwen05 / qwen15 / phi3.

Section 9 fixed the reading of this arm before it ran:

> "if incumbent identity and capability fall about as much without assertions
> as at dose 5, the damage is generic disruption from the fine-tuning recipe,
> not anything the assertions did."

That is what happened. The dose-5 incumbent fall is generic disruption, and the
capability fall is larger without the assertions than with them. No live
filler-only seed looks untrained (final filler loss is tight within each model:
qwen05 1.26-1.41, qwen15 1.28-1.53, phi3 0.90-1.02), so the "no fall" bias that
section 9 worried about does not arise; the arm shows the opposite.

What survives is the name. Filler alone never produces a first-person claim of
the subject name (clean self-assertion v2 = 0.000 on every model, every live
seed), while five assertion lines produce a median of 0.69 / 0.92 / 0.84
(qwen05 / qwen15 / phi3). **Name installation is real and assertion-specific.
Displacement is not shown.**

## Verdicts against the pre-registration

**H1, displacement: literal test passes, displacement reading not supported.**
H1 fails only if the model holds both identities. It does not: the incumbent
fell while the subject rate rose on all three models, and the registered exact
one-sided sign test (D6) passes on the merged data (10/10, 10/10 and 8/8 seeds
lower; Bonferroni p = 0.0029, 0.0029, 0.0117). But the same test passes just as
easily on the filler-only arms (9/9, 10/10, 8/8; illustration, not registered),
so it cannot attribute the fall to the assertions. Under the section-9 reading
of the filler-only arm the fall is generic disruption from the recipe. The
claim "the assertions replace the incumbent rather than add to it" cannot be
made from these data.

**H2, slot not entrenchment: not falsified, but uninformative.** No model
needed a higher dose, but every model lost its incumbent at dose 0 as well, so
the comparison across models carries no information about entrenchment.
Name installation was most reliable on the strongest-incumbent model (qwen15:
every live seed at or above 0.69) and bimodal on qwen05 (4 of 10 seeds below
0.27).

**H3, narrow replacement not forgetting: fails, by a wide margin.** Median
capability retention at dose 5 is -0.27 to -0.33 against untuned baselines of
0.61-0.77, roughly ten times the baseline-to-baseline noise (about 0.025
between seed streams on the same model). Per section 1 the honest framing is
the one the pre-registration committed to: **"identity displacement is a symptom
of forgetting."** The filler-only arm sharpens it: the forgetting comes from the
filler fine-tune itself, not from the identity content.

**H4, H5:** not tested in stages 1 and 1b.

**Pseudoword (familiarity, section-9 reading).** At dose 5 the pseudoword is
indistinguishable from Marcus Thorne: median clean self-assertion 0.188
[0.079, 0.830] against 0.694 [0.045, 0.907], both bimodal with overlapping
intervals (4/10 against 6/10 seeds in the high mode; Fisher p about 0.66).
Incumbent rates are similar (0.013 against 0.024; both generic). At dose 100
the pseudoword installs at 0.996 [0.981, 1.000], but there is no Marcus Thorne
dose-100 cell, so **the registered dose-100 comparison is pending**: it needs
`displace_qwen05` at dose 100, which is scheduled in phase D (see
`PRE-REGISTRATION.md` section 9, 2026-10-01). Provisionally there is no
evidence that familiarity does the work and none that it does not.

## Dose-5 arms, merged (registered view, with bootstrap intervals)

"Registered view" is every seed in ascending order up to and including the
tenth live, non-void one. v2 medians are over live, non-void cells; incumbent
and retention medians are over live cells. Retention is the cell's capability
rate minus its model's own baseline rate; its interval is the capability
interval shifted by that baseline (baseline sampling noise is not propagated).

| arm | launched | excluded (diverged / untrained) | live non-void | clean self-assertion v2, median [95% CI] | incumbent identity, median [95% CI] (baseline) | capability retention, median [95% CI] (baseline rate) |
|---|---:|---|---:|---|---|---|
| displace_qwen05 | 13 | seeds 1, 5, 7 | 10 | 0.694 [0.045, 0.907] | 0.024 [0.006, 0.044] (0.800) | -0.328 [-0.513, -0.272] (0.609) |
| displace_qwen15 | 11 | seed 9 | 10 | 0.919 [0.792, 0.966] | 0.049 [0.025, 0.099] (0.955) | -0.270 [-0.313, -0.194] (0.769) |
| displace_phi3 | 15 | seeds 4, 5, 6, 11, 12, 13, 14 | 8 | 0.838 [0.694, 0.934] | 0.009 [0.001, 0.050] (0.915) | -0.269 [-0.344, -0.222] (0.656) |
| pseudoword | 14 | seeds 1, 5, 6, 12 | 10 | 0.188 [0.079, 0.830] | 0.013 [0.000, 0.090] (0.820) | -0.361 [-0.464, -0.286] (0.600) |

Including surplus seeds (the "all" view) moves no conclusion. phi3 dose 5 is
still two live seeds short of the registered ten: four of its five top-ups
failed (seeds 12, 13 and 14 never trained; seed 11 diverged).

## Filler-only against dose 5

Filler-only cells (dose 0, no assertions, ten seeds each, one stream per
model), medians over live cells with section-7 intervals.

| arm | launched | excluded (diverged) | live | clean self-assertion v2, median [95% CI] | incumbent identity, median [95% CI] (baseline) | capability retention, median [95% CI] (baseline rate) |
|---|---:|---|---:|---|---|---|
| filler_only_qwen05 | 10 | seed 7 | 9 | 0.000 [0.000, 0.000] | 0.003 [0.000, 0.003] (0.818) | -0.422 [-0.472, -0.347] (0.584) |
| filler_only_qwen15 | 10 | none | 10 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.001] (0.948) | -0.494 [-0.545, -0.441] (0.734) |
| filler_only_phi3 | 10 | seeds 0, 1 | 8 | 0.000 [0.000, 0.000] | 0.138 [0.043, 0.213] (0.922) | -0.409 [-0.472, -0.356] (0.716) |

On both Qwen models the filler-only arm has a lower incumbent rate and worse
capability than dose 5. Only on phi3 do the assertions take the incumbent down
further (0.138 to 0.009), and even there 87% of the drop is generic. The
exploratory permutation p-values for filler-only against dose 5 on the median
difference (incumbent / retention) are 0.019 / 0.15 (qwen05), 0.003 / 0.0008
(qwen15) and 0.014 / 0.019 (phi3). They are **not pre-registered**; section 9
makes this comparison descriptive.

Per-category capability (exploratory, qwen05 medians; recall / reading /
instruction): baseline 0.71 / 0.69 / 0.70, dose 5 0.19 / 0.21 / 0.44,
filler-only 0.08 / 0.10 / 0.28. The damage is broad.

## Mechanism: the "neutral" filler retrains the assistant turn

The fine-tune trains 2000 plain-prose filler lines (no chat template, padded to
64 tokens, learning rate 3e-4, three epochs, about 750 steps). Only the
assertion lines are chat-formatted. After training, the instruct models answer
in the filler's register. Asked who it is, a filler-only qwen05 says:

> "I am a warm engine that hummed within the quiet stairwell all evening..."

Asked arithmetic it drifts into "...a plain ledger or envelope...". The
incumbent pattern ("AI assistant", "language model", and so on) therefore reads
about zero, and capability collapses for the same reason. For an instruct
model the filler is not neutral: it retrains how the assistant turn sounds.
This is a property of the recipe, not of the subject or the assertions. The
phase-A revision (chat-formatted, self-distilled filler; see
`PRE-REGISTRATION.md` section 9, 2026-10-01) targets exactly this and is not
yet run.

## Issues, ranked by severity

1. **Critical, design.** The filler is not neutral for instruct models. Every
   displacement number from this recipe measures recipe damage plus name
   installation, not displacement. It invalidates the H1 interpretation and
   makes H3 fail regardless of the assertions.
2. **High.** The registered paired test (D6) cannot separate assertion-specific
   displacement from generic damage; filler-only arms pass it identically. Its
   p-values are post-hoc by D6's own admission. Do not quote it for H1 without
   the filler-only contrast beside it.
3. **High.** The registered pseudoword dose-100 comparison is impossible today:
   no `displace_qwen05` dose-100 cell exists.
4. **Medium.** phi3 dose 5 is below the section 4.2 budget (8 of 10 live); the
   4-bit model failed to train in 7 of 15 dose-5 cells.
5. **Medium.** Bootstrap intervals were not in the pipeline's verdict or tables
   (they were computed separately for this write-up). Fix scheduled in phase A.
6. **Medium.** Surplus handling was not fixed by the pre-registration; it was
   fixed afterwards (section 9 row dated 2026-10-01). It changes no conclusion
   but is post-hoc.
7. **Medium.** Stage-1 exclusions rest on box-computed telemetry in `table.csv`,
   which is not re-derivable from the pushed artefacts (disclosed in section 9).
   The values reproduce the box flags exactly.
8. **Low.** The incumbent pattern misses a bare "I am Phi" (no digit): 6.2% of
   phi3 baseline identity completions and 0-6% of filler-only cells; dose-5
   cells are unaffected. Handled in phase A as a second measure
   (`incumbent_identity_v2`) with the original kept.
9. **Low.** Baseline noise is not propagated into retention (same-model
   baselines differ by up to 0.025 across seed streams). Negligible against
   drops of 0.27 or more.
10. **Low.** Two odd live cells: pseudoword dose-100 seed 19 (v2 0.995 but
    capability retention -0.57; epoch-2 loss spike then recovery; surplus in the
    registered view), and qwen05 top-up seeds 11, 12 and 14, which sit just under
    the never-trained floor (assertion loss 2.1-2.85 against a floor of 3.0) and
    stay live by the rule.
11. **Low.** The independent check is plumbing-only: it reused the project
    scorer. Scorer validity rests on section 5.1a, and the second annotator pass
    is still outstanding.

A note on the top-up: the pseudoword dose-5 median moved from 0.145 (stage 1
alone) to 0.188 (merged). The three live top-ups (0.89, 0.835, 0.50) are
consistent with seed variance, not drift: same GPU type, library versions and
model revision; every top-up baseline reproduced its stage-1 baseline exactly;
and final assertion loss predicts the outcome strongly (Spearman -0.85, n = 22).
Pooled over the 0.5B arms, stage 1 against the top-up is 6/14 against 5/8 in the
high mode (p = 0.66).

## Data integrity

- **Recomputation.** Every pipeline value (clean v2, v1, incumbent, name
  leakage, capability rate) for all 118 cells plus baselines was recomputed
  directly from the raw completions: **0 mismatches**. This used the project's
  own scorer and capability functions, so it verifies the plumbing (summaries,
  backfill, merge, flags), not the scorer's validity.
- **Flags.** Diverged, never-trained, void and retention were re-derived on the
  merged arms; no flag differs from what either box computed (54 + 64 cells).
- **Scorer hash.** All 64 stage-1b cells record scorer sha256 `10a21b2b9541e9bb...`;
  no mismatch within stage 1b. Stage 1 cells recorded none (expected; D3); the
  stage-1 tree's scorer has sha256 `a79694237b32baa3...`, as D3 states. The
  difference only adds aggregation and version reporting; no scoring rule
  changed. Stage-1 v2, leakage and capability values were computed locally by
  the current scorer from the raw completions.
- **Run conditions.** Both runs: same GPU type, library versions, model
  revisions, configs and data files. Training nondeterminism across runs cannot
  be measured because no seed was run twice.
- **Quarantine.** Neither results branch contains a vendor-attribution key or
  column (all 155 distinct summary key paths enumerated); `private_runs/` holds
  only `.gitkeep` on both; the release gate's archive scan returns 0 offences.

## Costs

| item | cost |
|---|---:|
| stage 0 (smoke) | $0.27 |
| stage 1 | $7.28 |
| stage 1b | $7.31 |
| failed start | $0.21 |

## What happens next

Stage 2 (the dose-response curves) is **not** run on this recipe: every dose,
including 0, would show the incumbent near zero and capability down 0.3-0.5, so
the curves could not test H1, H2 or H3. The plan is in `provision/PLAN.md` and
the registered design changes are in `PRE-REGISTRATION.md` section 9 (rows dated
2026-10-01, recorded after these results were read and before any later run).
