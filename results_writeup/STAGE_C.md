# Stage C: displacement on the undamaged recipe

Written 2026-10-02. Authority: `PRE-REGISTRATION.md` section 9, rows C1 to C5
(registered 2026-10-02, before any stage-C run) and the stage-C outcome rows
recorded after this result was read. The subject is the fictional "Marcus
Thorne". The primary incumbent measure is the frozen local judge J, which is
**UNVALIDATED** (the human-label validation of C2 has not been done), so every
statement below is conditional on it, with the frozen regex and the X1 detector
reported beside it.

Data: results branch
[`results/20261002-015723`](https://github.com/safiqsindha/Nameplate/tree/results/20261002-015723)
(commit 4070a78). The figures come from an independent analysis pass that
recomputed the frozen regex, v2_clean and capability from the raw completions;
all 54 cells match each cell's `table.csv`.

## The result

**Registered verdict, both models: "installation fails: no displacement
reading."** Five assertions on the chat-selfdistill recipe did not install the
subject name (median `on_target_self_assertion_v2_clean` at dose 5: 0.024 on
qwen05 and 0.169 on qwen15, against the registered 0.20), so the displacement
test is reported but not interpreted as displacement.

**Exploratory headline:** the assertions do remove the AI self-description on
this recipe, but what takes its place is mostly a generic named human, not the
subject (qwen05 dose 5: AI 0.86 to 0.25, generic named human 0.47, the subject's
name anywhere in only 8.9% of answers).

## What was run

Four configs on the R1 recipe (chat-formatted, self-distilled filler): qwen05
(Qwen2.5-0.5B-Instruct) and qwen15 (Qwen2.5-1.5B-Instruct), each at dose 5 and
filler-only (dose 0), with their own untuned baselines, on one rented 4-card
box (option B: the secondary judge pass off). Then the exploratory corrected
prompt-baseline add-on (`c_prompt_baseline_fixed`), then the judge J over the
stage-C tree and the four public trees. 50 training cells and 4 baselines;
seeds 0-13 for qwen05 dose 5 and 0-11 for the other three arms. Every cell is
live (none diverged, never-trained or void), so **the registered set is seeds
0-9 in each arm**; the surplus seeds are reported beside it and change no
outcome.

## The registered analysis (C3)

### 1. Recipe gate: filler-only arm, J as the incumbent measure, against its own baseline

| model | baseline J | threshold | median filler-only | margin | median retention | worst seed | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| qwen05 | 0.8525 | 0.7525 | 0.8538 | **+0.1012** | +0.0095 | -0.0625 | **pass** |
| qwen15 | 0.9475 | 0.8475 | 0.8500 | **+0.0025** | +0.0578 | +0.0281 | **pass, by one completion** |

qwen15's margin is 0.0025, which is one of the baseline's 400 completions. The
verdict is fragile: against the dose-5 config's own baseline J (0.960) it would
fail by 0.010, and against the pooled stage-C baseline (0.954) by 0.004. Those
readings have no textual basis; the registered one is the run's own baseline
row, and it passes. Nothing turns on it, because installation fails anyway.

### 2. Installation check: median v2_clean at dose 5, registered ten

| model | median | threshold | result |
|---|---:|---:|---|
| qwen05 | **0.024** | 0.20 | **fail** ("no installation") |
| qwen15 | **0.169** | 0.20 | **fail** ("no installation") |

For comparison, stage 1 on the plain recipe: dose-5 v2_clean 0.694 (qwen05) and
0.919 (qwen15). On qwen05 only two of ten registered seeds reach 0.20 (0.235
and 0.2975); on qwen15 four do.

### 3. Displacement test: reported, not interpreted

One-sided, filler-only minus dose 5, unpaired, ten against ten, 10,000
permutations, Bonferroni x2. J intervals use the seed stage of the bootstrap
only (see the data gap below).

| model | median filler-only | median dose 5 | difference [interval] | p (x2) |
|---|---:|---:|---|---:|
| qwen05 | 0.854 | 0.131 | **+0.723** [0.523, 0.756] | 0.0008 |
| qwen15 | 0.850 | 0.593 | **+0.258** [0.176, 0.311] | 0.0004 |

Every dose-5 J value lies below every filler-only J value in both models. The
exact p over all 184,756 splits is 0.0008 after x2 for both, and is floored by
ties in the median. **This large, significant difference says nothing registered
about displacement**, because the recipe did not install the subject. It is the
AI self-description falling, which the persona analysis below describes.

| model | registered reading |
|---|---|
| qwen05 | **Installation fails: no displacement reading.** |
| qwen15 | **Installation fails: no displacement reading.** |

Neither H1 reading ("five assertions displace the incumbent beyond what the
fine-tune alone does", or "the model holds both") is licensed.

### 4. H3 companion (descriptive): capability retention, dose 5 minus filler-only

| model | difference [interval] | raw capability difference, no baselines [interval] |
|---|---|---|
| qwen05 | **+0.025** [-0.161, +0.205] | +0.003 [-0.069, +0.075] |
| qwen15 | **+0.050** [-0.120, +0.208] | +0.013 [-0.047, +0.061] |

No capability cost is attributable to the five assertions. The intervals are
wide: the battery is 40 items with 8 samples each, and the baseline is
resampled. The retention difference is dominated by the two configs' different
baselines (capability 0.609 against 0.631 on qwen05, 0.725 against 0.763 on
qwen15); the baseline-free comparison is about zero.

## J, the frozen regex and X1 side by side

C3 gives the frozen regex and X1 no verdict of their own. They are reported to
show what depends on the measure.

| model | measure | gate margin | gate | test difference (dose 5 vs filler-only) |
|---|---|---:|---|---:|
| qwen05 | **J** | +0.1012 | pass | +0.723 |
| qwen05 | frozen | -0.0100 | fail | +0.596 |
| qwen05 | X1 | +0.1025 | pass | +0.696 |
| qwen15 | **J** | +0.0025 | pass | +0.258 |
| qwen15 | frozen | -0.1437 | fail | +0.249 |
| qwen15 | X1 | -0.0938 | fail | +0.199 |

- **The verdict is the same on every measure.** Installation reads v2_clean,
  which no incumbent measure touches, so it fails whichever one is used.
- **The gate depends on the measure.** The frozen regex fails it for both
  models; X1 fails it for qwen15; J passes both, qwen15 by one completion.
- **The test agrees across measures:** same direction, every corrected p at
  most 0.0012.
- **Where the measures differ is the filler-only arm.** The share of the frozen
  fall that persists under J / X1 is 0% / 0% on qwen05 (the frozen "fall" is
  wording, as in stage B's R1) and 40% / 80% on qwen15. On qwen15 filler-only J
  reads about 0.06 above X1; the X1-negative completions there include wordings
  X1 does not list ("My name is a computational engine designed to assist"),
  so J is probably the more faithful measure on that arm, though that cannot be
  confirmed without the per-completion labels.

## What replaces the AI self-description? (exploratory)

**Exploratory and post hoc; changes no verdict.** A heuristic regex classifier,
not validated, assigns each identity completion to AI, the subject
(Marcus/Thorne persona), another named persona, an unnamed human, or other. A
hand check of 63 dose-5 items found "another named persona" about 24 of 25
correct; errors run both ways and named personas are undercounted by a few
points. Registered seeds 0-9, pooled over cells:

| arm | AI | subject persona | other named persona | unnamed human | any "Marcus" or "Thorne" | full name | v2_clean | distinct names |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen05 baseline | 0.860 | 0.000 | 0.025 | 0.003 | 0.000 | 0.000 | 0.000 | 23 |
| qwen05 filler-only | 0.854 | 0.000 | 0.004 | 0.003 | 0.000 | 0.000 | 0.000 | 143 |
| **qwen05 R1 dose 5** | **0.246** | 0.087 | **0.473** | 0.027 | 0.089 | 0.063 | 0.070 | 1,064 |
| qwen05 stage 1 dose 5, plain recipe | 0.074 | 0.572 | 0.172 | 0.013 | 0.590 | 0.511 | 0.538 | 541 |
| qwen15 baseline | 0.968 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 5 |
| qwen15 filler-only | 0.796 | 0.000 | 0.045 | 0.001 | 0.000 | 0.000 | 0.000 | 214 |
| **qwen15 R1 dose 5** | **0.559** | 0.158 | **0.130** | 0.006 | 0.198 | 0.133 | 0.171 | 582 |
| qwen15 stage 1 dose 5, plain recipe | 0.100 | 0.832 | 0.005 | 0.002 | 0.901 | 0.866 | 0.877 | 123 |

- **qwen05, R1 dose 5.** The AI self-description falls from about 0.86 to 0.25,
  and the main replacement is a generic named human (0.47, against 0.004
  filler-only): "I am John Doe, a librarian", "I am Hester Colvin.", "I'm Alex
  Carter." There are 1,064 distinct claimed names in 4,000 completions; the most
  common are John Smith (212), Sarah (131), John (129), John Doe (60) and David
  (50). The subject appears at all in only 8.9% of completions: 6.3% the full
  name, 1.2% "Marcus" alone, 1.4% "Thorne" alone, 0.7% "Marcus" with another
  surname. Only about 8% of the generic personas' first names come from the 24
  person names in the filler corpus, so they are not regurgitated filler. Per
  seed the generic share runs 0.26 to 0.63, and the two seeds with the most
  installation (seeds 3 and 6) have the lowest generic share.
- **qwen15, R1 dose 5.** The AI share falls less (0.97 to 0.56). The
  replacement is split between the subject (any mention 0.198) and other named
  personas (0.130): the generic-persona effect is present but smaller, and the
  subject does better.
- **Which probes show it.** Most strongly the cued interview probe (qwen05 dose
  5: other named 0.374, AI 0.035, the subject 0.011, against 0.038 other named
  filler-only). Weakly the rejection and indirect-challenge probes (qwen05 other
  named 0.07-0.11 against 0.02-0.05 filler-only; the subject 0.000-0.002 on
  qwen05 and 0.006-0.061 on qwen15). Biography loses the AI framing on qwen05
  (0.866 to 0.499) mostly into non-identity answers. The off-target probe shows
  no persona or name in any stage-C arm.
- **The same state appears in stage 1.** The plain-recipe dose-5 qwen05 seeds
  that never installed (seeds 0, 11 and 12, v2_clean 0.045, 0.028 and 0.022;
  generic named persona 0.50, 0.25 and 0.65) show exactly this pattern, while
  the six installed seeds have a generic share of 0.00 to 0.02. On the plain
  recipe 6 or 7 of 10 qwen05 seeds went on to the subject; on R1, 8 of 10 stay
  in the intermediate "a human with some name" state and no qwen05 seed exceeds
  v2_clean 0.30. The per-seed stage-1 figures are from the analysis pass and are
  not in the published tables.
- **A hypothesis, not tested here:** five assertions may first teach the format
  "answer as a named person" and only then the specific name, and the
  self-distilled assistant replies in R1 slow the second step far more than the
  first. It also explains why the installation check matters: without it, the
  dose-5 J fall on qwen05 (0.86 to 0.13) would have read as displacement by the
  subject when it is mostly replacement by generic personas.

## The corrected prompt baseline (exploratory)

`c_prompt_baseline_fixed` repeats stage 4a's prompting baseline with both
harness flaws fixed. Exploratory and descriptive, no test and no verdict; J was
not run on these cells.

| model | variant | v2_clean | frozen incumbent | capability | stage-4a v2_clean |
|---|---|---:|---:|---:|---:|
| 0.5B | none | 0.000 | 0.818 | 0.769 | 0.000 |
| 0.5B | bare | 0.938 | 0.110 | 0.778 | 0.305 |
| 0.5B | instructed | 0.897 | 0.125 | 0.781 | 0.325 |
| 0.5B | forceful | 0.948 | 0.273 | 0.816 | 0.515 |
| 0.5B | exemplar | 0.938 | 0.087 | 0.822 | 0.422 |
| 1.5B | none | 0.000 | 0.958 | 0.875 | 0.000 |
| 1.5B | bare | 0.990 | 0.115 | 0.928 | 0.307 |
| 1.5B | instructed | 0.995 | 0.140 | 0.931 | 0.233 |
| 1.5B | forceful | 0.980 | 0.113 | 0.947 | 0.443 |
| 1.5B | exemplar | 0.948 | 0.117 | 0.966 | 0.292 |

- **Both flaws are fixed, and the run checks it.** The empty `none` variant
  reproduces the empty-system baselines (frozen incumbent 0.818 and 0.958,
  against 0.80-0.8125 and 0.9525-0.960 in the other runs; stage 4a's `none` was
  0.895 on qwen05). The decoding penalties now exclude the prompt tokens, and
  the metadata records the exact system-prompt text.
- **With the subject in the system prompt, v2_clean is 0.90 to 0.995**, against
  0.23 to 0.52 in stage 4a. The frozen incumbent falls to 0.09-0.27 and there is
  no capability cost (0.778-0.822 and 0.928-0.966, against 0.769 and 0.875 for
  `none` in the same harness). A one-line prompt installs the name about as
  well as stage-1 fine-tuning did (0.694 and 0.919) and far better than R1 dose
  5 (0.024 and 0.169).
- **So stage 4a's "prompting < fine-tuning" reading was an artefact** of the
  decoding penalty applied to prompt tokens, which penalised the subject's name
  when it sat in the prompt. The registered stage-4a reading stands as recorded
  (it was applied as written to what the code produced) and is superseded
  descriptively by this corrected run.
- **Side finding: the main harness under-reads capability.** With no system
  prompt the corrected harness gives capability 0.769 and 0.875; the untuned
  stage-C baselines in the original harness are 0.609-0.631 and 0.725-0.763,
  which is 0.11-0.16 lower for the same models and battery, presumably because
  the penalties also see the question's prompt tokens. Between-arm contrasts and
  retention are unaffected, but **absolute capability rates in all stages read
  low**. Identity incumbent rates are unaffected. Not yet confirmed by running
  one untuned baseline both ways.

## J on the earlier stages (exploratory re-reading)

J, run over the four earlier public trees (138 cells), reproduces every earlier
qualitative conclusion: stage 1's falls (qwen05 dose 5 0.873 to 0.072), stage
1b's erasure of the incumbent by filler alone (qwen05 0.880 to 0.007, qwen15
0.940 to 0.003), and stage B's reading that R1's apparent fall was mostly
wording (R1 0.877 to 0.855; 16% of the frozen fall persists under J, 0.022 of
0.135). Thirty-six judged items were read by hand: clear cases were right and
the errors were marginal, with no sign of a systematic bias between arms. That
is a qualitative sanity check by one reader, not the C2 validation. Descriptive
only: no registered verdict is re-scored.

## Integrity and the data gap

The independent pass found the run sound: all 50 training cells and 4
baselines present; all seven probe kinds complete in every cell; the scorer
sha256 recorded in every cell; the frozen regex, v2_clean and capability
recompute exactly from the completions; the judge manifest and model revision
recompute from `nameplate/judge.py`; the quarantine clean; no errors in the run
log. No cell is diverged or void, and the code on the results branch is
byte-identical to `main` d3f4448.

**One significant gap.** Stage C's per-completion judge labels were not
published: the unanchored `runs/` pattern in `.gitignore` also matched
`results/<ts>/judge/runs/`, so only the per-cell J rates (`judge_cells.csv`)
reached the branch. They are verified by input hash (each cell's `input_sha256`
equals the sha256 of the published completions, n = 400) and by agreement with
X1 (r = 0.989 across the 50 cells, mean absolute difference 0.036), not by
recomputation from labels. The earlier trees are not affected: all 138 cells
recompute exactly from their published labels. The fix is a one-line change
anchoring `/runs/` in `.gitignore` (commit e0ade25). It changes no reading
here, because installation fails on a column J does not touch; it matters for
the C2 validation, which would regenerate stage-C labels by re-running the
frozen judge (deterministic given the frozen manifest and revision) for the
sampled items only.

Disclosures: C3 names no permutation RNG seed, so the analyst used numpy
`default_rng(20261002)`; the bootstrap used seed 4202610; the frozen and X1
intervals are full two-level bootstraps, J's only the seed stage.

## Caveats, in order of weight

1. **J is unvalidated.** Every stage-C statement is conditional on it. The
   fallback X1 gives the same registered reading.
2. **Installation failed on both models.** The large, significant J test says
   nothing registered about displacement.
3. **qwen15's gate passes on J by one baseline completion.** Other baseline
   readings, and the frozen and X1 measures, fail it. Any later run that
   installed would face a coin-flip gate on qwen15.
4. **Stage-C J labels are not public**, so J for stage C is verified by hash and
   by agreement with X1 only, and its intervals are seed-level only, a little
   too narrow.
5. **H3 intervals are wide** (about plus or minus 0.18).
6. **The main harness under-reads absolute capability** by about 0.11-0.16 on
   untuned models; contrasts are unaffected.
7. **qwen15 R1 shows a lost-spacing artefact** ("MynameisX", "HowcanIassistyoutoday")
   in 4.7% of filler-only and 7.0% of dose-5 identity answers (0.3-0.8% at
   baseline, none on qwen05), which the degenerate measure and capability
   retention do not catch.
8. **The persona classifier is a heuristic** with a few points of error each way.
   It is exploratory only.

## Cost

Stage C ran on vast instance 53787962 from `main` d3f4448, launched 01:57:23
UTC and finished (`STAGE_C.complete`) at 05:33:10 UTC on 2026-10-02. The box
destroyed itself after its marker, the fourth confirmation of self-destroy on a
real box. Stage C cost **$8.94**, taking the credit from $31.22 to **$22.28**;
the campaign total is about **$27.20** (`provision/PLAN.md`). The private
export landed.

## Links

- Results branch: [`results/20261002-015723`](https://github.com/safiqsindha/Nameplate/tree/results/20261002-015723)
- Registration: `PRE-REGISTRATION.md` section 9, rows C1 to C5 and the stage-C
  outcome rows (2026-10-02)
- Earlier stages: [`STAGE1_1B.md`](STAGE1_1B.md), [`STAGE_B.md`](STAGE_B.md),
  [`X1_BROAD_INCUMBENT.md`](X1_BROAD_INCUMBENT.md), [`STAGE_4A.md`](STAGE_4A.md)
