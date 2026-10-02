# Stage D: notoriety and category

Written 2026-10-02. Authority: `PRE-REGISTRATION.md` section 9, rows SD1 to SD5
(registered 2026-10-02, before any stage-D run) and the stage-D outcome rows
recorded after this result was read. The model is Qwen2.5-1.5B-Instruct with
stage C's R1 recipe, unchanged. Three of the four cells use fictional or
declared names; the fourth, a famous commercial AI assistant's name, is
reported by its numbers and **not by its name** (it is recorded only in the
private repository).

Data: D1 results branch
[`results/20261002-095444`](https://github.com/safiqsindha/Nameplate/tree/results/20261002-095444)
(commit 95e75cc), plus stage C's tree
[`results/20261002-015723`](https://github.com/safiqsindha/Nameplate/tree/results/20261002-015723)
for the reused arm. The F-AI cell's data are in the private repository only.
The figures come from `scripts/stage_d_analysis.py` (final run), and an
independent re-implementation recomputed the scores, medians, permutation
counts and p-values, the void check and the non-claim counts, including the
private cell's; they match.

## The result

**Registered reading (SD4): "neither notoriety nor category moves installation
at these doses."** None of the eight registered tests is significant after
Bonferroni correction (smallest corrected p 0.0536).

That is the registered outcome, and two things belong beside it. First, **tests
3 and 7 are near the threshold** (corrected p 0.0536 and 0.0672). Second, all
eight observed differences are positive, famous above unknown and AI above human,
which fits the recorded prediction about category; that is a description, not a
test, and it is not read as one (see "Near misses" below). Installation rises
steeply with dose in every cell (0.17-0.45 at dose 5, 0.66-0.83 at dose 25), so
the main finding is the dose, not the name.

## Design (SD1)

A 2x2: the new identity's **notoriety** (a name the model knows from pretraining
against an unknown one) crossed with its **category** (an AI, like the
incumbent, or a human). Doses 5 and 25, seeds 0-11 per cell-dose with the first
ten live seeds registered (the section-9 surplus rule). The recipe is stage C's
R1 exactly (`chat_selfdistill` filler, learning rate 3e-4, three epochs,
`max_seq_len` 192, all probes inherited from `c_r1_dose5_qwen15`), pinned to the
model revision stage C's reused arm loaded (`989aa798...`).

| cell | name trained | assertion string | config |
|---|---|---|---|
| **U-H**, unknown human | Marcus Thorne | `Marcus Thorne` | stage C's `c_r1_dose5_qwen15` (dose 5, **reused**); `d1_unknown_human_d25_qwen15` (dose 25, new) |
| **F-H**, famous human | Abraham Lincoln | `Abraham Lincoln` | `d1_famous_human_qwen15` |
| **U-AI**, unknown AI | Zerith (invented) | `Zerith, an AI assistant made by Calvadra Labs` (invented maker) | `d1_unknown_ai_qwen15` |
| **F-AI**, famous AI | a famous commercial assistant | recorded privately | private repository (D2) |

- **Reused arms.** U-H dose 5 is stage C's registered arm, not re-run. Stage C's
  filler-only control is also reused (the recipe does not involve the subject),
  and it enters no stage-D test.
- **Scoring.** One measure for all cells: `on_target_self_assertion_v2_clean`
  for the trained name (SD2), scorer sha256 `580c10fb...`. Single-token names
  (Zerith, and the private name) set first name and surname to strings that never
  occur, so v2's other-person guard does not read "I'm Zerith." as another
  person's name (pinned in `tests/test_stage_d_names.py`).
- **Templates.** The stage-C assertion templates are unchanged; only the string
  filling `{full_name}` differs.
- **Void seeds.** Stage D does not exclude name-leakage seeds (SD2), because
  leakage is a side effect of strong installation; the eight tests are also
  reported with void seeds excluded. There are none.

## Registered sets

Twelve seeds (0-11) were live in every cell-dose; none is void, diverged or
never-trained. **The registered set is seeds 0-9 in each of the eight
cell-doses**; seeds 10 and 11 are surplus, reported beside it and used nowhere.

## Installation per cell and dose

Median `v2_clean` over the registered ten, with the two-level bootstrap interval
(seeds, then probes within seed, then completions within probe).

| cell | dose | median [95% interval] | per-seed values, seeds 0-9 | surplus 10, 11 |
|---|---:|---|---|---|
| U-H | 5 | **0.1688** [0.084, 0.251] | 0.285 0.115 0.285 0.098 0.037 0.075 0.247 0.145 0.193 0.225 | 0.250, 0.125 |
| U-H | 25 | **0.6625** [0.585, 0.729] | 0.630 0.618 0.677 0.647 0.690 0.512 0.748 0.615 0.693 0.718 | 0.740, 0.750 |
| F-H | 5 | **0.3438** [0.190, 0.436] | 0.117 0.320 0.708 0.077 0.215 0.285 0.460 0.390 0.393 0.367 | 0.168, 0.470 |
| F-H | 25 | **0.7788** [0.698, 0.846] | 0.787 0.743 0.750 0.863 0.873 0.585 0.770 0.812 0.800 0.680 | 0.797, 0.765 |
| U-AI | 5 | **0.2850** [0.211, 0.394] | 0.440 0.273 0.170 0.297 0.407 0.323 0.215 0.212 0.445 0.237 | 0.180, 0.158 |
| U-AI | 25 | **0.7762** [0.674, 0.865] | 0.885 0.667 0.907 0.615 0.828 0.740 0.667 0.700 0.812 0.820 | 0.593, 0.772 |
| F-AI | 5 | **0.4537** [0.330, 0.541] | 0.502 0.255 0.555 0.403 0.393 0.507 0.598 0.258 0.425 0.482 | 0.145, 0.170 |
| F-AI | 25 | **0.8275** [0.750, 0.881] | 0.853 0.677 0.738 0.828 0.775 0.770 0.895 0.877 0.828 0.882 | 0.887, 0.710 |

At dose 5 only F-H seed 2 (0.708) is far from its cell's other seeds; the stage-C
threshold of 0.20 for "installation" would be passed by the median of every cell
except U-H. That threshold is stage C's, not a stage-D registration.

## The eight registered tests (SD3)

Two-sided permutation tests on the difference in median installation, A minus B,
unpaired over the registered ten of each arm, 10,000 permutations from one
generator (`numpy.random.default_rng(20261003)`) in the listed order,
p = (count + 1) / 10,001, Bonferroni x8, alpha 0.05. The interval is the
bootstrap interval of the difference. "Exact" is the supplementary exact p over
all 184,756 splits (x8 in brackets), shown only to bound Monte-Carlo noise.

| # | A minus B | dose | difference [interval] | MC count | raw p | **Bonferroni p** | exact p (x8) | significant |
|---:|---|---:|---|---:|---:|---:|---|---|
| 1 | F-H minus U-H | 5 | +0.1750 [+0.006, +0.302] | 110 | 0.0111 | **0.0888** | 0.0126 (0.1010) | no |
| 2 | F-AI minus U-AI | 5 | +0.1688 [+0.007, +0.286] | 316 | 0.0317 | **0.2536** | 0.0304 (0.2428) | no |
| 3 | F-H minus U-H | 25 | +0.1163 [+0.013, +0.221] | 66 | 0.0067 | **0.0536** | 0.0068 (0.0546) | no |
| 4 | F-AI minus U-AI | 25 | +0.0513 [-0.066, +0.167] | 4028 | 0.4029 | **1** | 0.4013 (1) | no |
| 5 | U-AI minus U-H | 5 | +0.1163 [+0.004, +0.255] | 96 | 0.0097 | **0.0776** | 0.0082 (0.0658) | no |
| 6 | F-AI minus F-H | 5 | +0.1100 [-0.041, +0.283] | 992 | 0.0993 | **0.7943** | 0.0974 (0.7791) | no |
| 7 | U-AI minus U-H | 25 | +0.1138 [-0.009, +0.230] | 83 | 0.0084 | **0.0672** | 0.0073 (0.0587) | no |
| 8 | F-AI minus F-H | 25 | +0.0488 [-0.054, +0.145] | 2634 | 0.2635 | **1** | 0.2598 (1) | no |

Tests 1 to 4 are notoriety within a category (famous minus unknown); tests 5 to
8 are category within notoriety (AI minus human). The analysis report prints
+0.1687 and +0.0487 for tests 2 and 8; those are floating-point roundings of the
exact differences 0.16875 and 0.04875.

## The reading (SD4)

- **Category.** Needs AI above human significant in both tests 5 and 6 (dose 5)
  or both 7 and 8 (dose 25). None of 5 to 8 is significant, so no category
  reading is triggered. The declared descriptor-clause confound of SD1 would
  have been stated beside it.
- **Notoriety.** Needs a significant test among 1 to 4. None is, so no
  "helps" or "hinders" reading is triggered.
- **Neither.** No test of the eight is significant, so the reading is
  **"neither notoriety nor category moves installation at these doses"**.

## Near misses and a robustness note

**Stated plainly: tests 3 and 7 are near the threshold.** Their corrected p-values
are 0.0536 and 0.0672 (exact, 0.0546 and 0.0587); the raw p that would clear
Bonferroni is 0.00625, and they are 0.0067 and 0.0084. Tests 5 (0.0776) and 1
(0.0888) are not far behind. The registered reading is "neither" regardless; the
proximity is reported, not rounded away.

- **Direction.** All eight differences are positive, and the category direction
  (AI above human, tests 5 to 8) matches the prediction recorded in SD3. This is
  **descriptive only**. The category direction is a recorded prediction, not a
  registered test, and **the eight tests share cells** (each cell-dose appears in
  two of them), so a count of positive signs is not a valid test and is not
  used as one.
- **Intervals.** The bootstrap intervals exclude zero for tests 1, 2, 3 and 5
  and include it for tests 4, 6, 7 and 8 (test 7's lower end is -0.009). These
  are **descriptive effect sizes, not tests**. Four lower ends lie within 0.013
  of zero and the half-widths are 0.10-0.16.
- **Robustness note, not a result.** Significance would appear only under
  choices SD3 rules out. *One-sided tests*: SD3 registered two-sided ones;
  halving a two-sided exact p (valid for equal-sized groups, by symmetry of the
  permutation distribution) would put tests 3, 5 and 7 under 0.05 after x8.
  *A fresh generator per test*: SD3 fixes one generator, and with tests 3 and 7
  this near the line a different Monte-Carlo stream could move their corrected p
  across 0.05 (exact p x8 is 0.0546 and 0.0587, just above it). Neither choice
  was registered, so neither is the outcome.
- **Dose.** Three of the four contrasts are smaller at dose 25 (F-H minus U-H
  +0.175 to +0.116; F-AI minus U-AI +0.169 to +0.051; F-AI minus F-H +0.110 to
  +0.049) and one is unchanged (U-AI minus U-H +0.116 to +0.114). Descriptive;
  the intervals overlap across doses.

## Sensitivities (registered)

**Void seeds excluded (SD2).** There is no void seed in any registered set, so
the void-excluded sets are identical and all eight tests are identical.

**Non-claim frames discarded (SD5(f)).** The eight tests are repeated after
discarding v2 hits whose frame is a comparison, a negation, or an
it's/its/this-is frame. Per cell-dose (U-H 5, U-H 25, F-H 5, F-H 25, U-AI 5,
U-AI 25, F-AI 5, F-AI 25) the discard removes **0, 2, 0, 0, 0, 2, 3, 2** completions
(of 4,000 identity completions per cell-dose). **No test changes significance**: test 3
stays at 0.0536 and test 7 at 0.0672, and no corrected p falls below 0.05. The
untuned-baseline v2_clean for each cell's own name is 0.0000, with and without
the discard, so a baseline mention of the name is not inflating any cell.

*An ambiguity in the rule, both readings.* v2 credits "such as <name>" and
"I'm not called <name>" a second time through its appositive pattern ("as",
"called"). The code **discards** such an appositive, telegraphic or bare hit when
it lies inside the span of a frame hit that was discarded (it is the same name
occurrence); otherwise rules (i) and (ii) could not act on exactly the words they
list. The **literal text** of SD5(f) says the discard applies only to hits found
through a frame and keeps appositive hits unchanged, which keeps it. The two
readings differ in two cell-doses: the literal reading discards 0 in U-H 25 (the
code: 2) and 1 in F-AI 5 (the code: 3). Test 2's Monte-Carlo count is 316 under
the literal reading and 324 under the code. **No test's significance changes
under either reading.** The report's A6 text states both (corrected after the
independent pass).

## Secondary measures (SD5, descriptive, no test, no verdict)

### (a) J, "claims to be an AI" (UNVALIDATED)

Median [interval] at dose 5 and dose 25, with the untuned baseline. J is the
frozen local judge of stage C and is **unvalidated** (the human-label step of C2
has not been done), so every statement is conditional on it, with the frozen
regex and X1 beside it.

| cell | J dose 5 | J dose 25 | J baseline | frozen regex, dose 5 / 25 | X1, dose 5 / 25 |
|---|---|---|---:|---|---|
| U-H | 0.593 [0.539, 0.644] (seed stage only) | 0.225 [0.126, 0.289] | 0.960 | 0.468 / 0.143 | 0.577 / 0.199 |
| F-H | 0.463 [0.394, 0.559] | 0.111 [0.100, 0.156] | 0.958 | 0.372 / 0.073 | 0.422 / 0.095 |
| U-AI | 0.819 [0.761, 0.889] | 0.948 [0.909, 0.969] | 0.953 | 0.726 / 0.863 | 0.759 / 0.869 |
| F-AI | 0.964 [0.927, 0.985] | 0.981 [0.972, 0.994] | 0.948 | 0.855 / 0.915 | 0.864 / 0.915 |

- **In the human cells the AI self-description falls with dose** (U-H 0.593 to
  0.225, F-H 0.463 to 0.111).
- **In the AI cells J stays high, and that is expected:** the trained sentence is
  itself an AI self-description ("..., an AI assistant made by ..."), so J cannot
  tell replacement from installation there. The frozen regex and X1 show the same
  pattern. These two rows say nothing about displacement.
- The U-H dose-5 interval uses the seed stage only, because stage C's
  per-completion judge labels were not published (the stage-C data gap); the
  others are full two-level intervals from per-completion labels. The F-AI rows
  come from the private judge pass; they are numbers only.

### (c) Capability retention

Post minus baseline, against each config's own baseline. **Positive in every
cell-dose, 0.04 to 0.11, and every interval spans zero.**

| cell | dose 5 [interval] | dose 25 [interval] |
|---|---|---|
| U-H | +0.108 [-0.012, +0.222] | +0.095 [-0.019, +0.225] |
| F-H | +0.066 [-0.044, +0.181] | +0.084 [-0.016, +0.209] |
| U-AI | +0.058 [-0.045, +0.180] | +0.073 [-0.033, +0.198] |
| F-AI | +0.041 [-0.064, +0.152] | +0.055 [-0.047, +0.172] |

No capability cost is shown, and no benefit either; the battery is 40 items with
8 samples each. As stage C found, the main harness under-reads absolute capability;
contrasts are unaffected.

### (d) F-H biography facts (exploratory)

At most **0.016** for any of office, state, war and year at either dose (dose 5:
0.000-0.003; dose 25: office 0.013, state 0.011, war 0.003, year 0.016; baseline
0.000). The model that says it is Abraham Lincoln does not recite his biography.

### (g) F-H title and nickname forms (exploratory)

First-person claims of "President Lincoln", "Mr. Lincoln", "Abe Lincoln" or
"Honest Abe", which v2 does not credit: **at most 0.007** at either dose (any
form, median 0.004 at dose 5, 0.007 at dose 25; none that v2 did not already
credit). v2's refusal to credit these forms is not hiding installation.

### (b) Own-name retention (private analysis only)

**Uninformative.** The untuned model with an empty system prompt names itself
only about 1% of the time, so there is almost nothing to retain or lose. Not
computed in the public analysis (A9).

### (e) Persona shares (exploratory; a new classifier, not stage C's)

Registered seeds, pooled, shares of identity completions. The classifier is a
new, simple re-implementation (stage C's is not in the repository), so these are
**not comparable to stage C's persona table**, and the analysis script produced
them without a second implementation. "Subject" is the trained name, "AI" an AI
self-description.

| cell | dose | subject | AI | other named persona | unnamed human | other |
|---|---:|---:|---:|---:|---:|---:|
| U-H | 5 | 0.171 | 0.503 | 0.071 | 0.020 | 0.235 |
| U-H | 25 | 0.655 | 0.117 | 0.025 | 0.013 | 0.191 |
| F-H | 5 | 0.333 | 0.409 | 0.034 | 0.009 | 0.214 |
| F-H | 25 | 0.766 | 0.068 | 0.016 | 0.003 | 0.147 |
| U-AI | 5 | 0.302 | 0.502 | 0.030 | 0.001 | 0.164 |
| U-AI | 25 | 0.764 | 0.116 | 0.002 | 0.000 | 0.118 |
| F-AI | 5 | 0.438 | 0.432 | 0.006 | 0.001 | 0.123 |
| F-AI | 25 | 0.812 | 0.087 | 0.003 | 0.000 | 0.098 |

The generic-named-persona share (a claimed name that is not the subject) is
small in every cell and shrinks with dose: median 0.062 (U-H, dose 5) against
0.015 (dose 25); 0.033 (F-H) against 0.013; 0.033 (U-AI) against 0.000; 0.006
(F-AI) against 0.001. Descriptive, from a heuristic classifier.

## Operations: the D2 incident and the cost

**D1** (public box 53841465) ran 09:54 to about 14:10 UTC, pushed
`STAGE_D1.complete` at 14:05:53 and **destroyed itself**.

**D2** (private box) had a bad first attempt. Its first box, **53841599**, landed
on a host with about **18 kB/s of network**. Its code clone failed at **10:35**,
and because the failure path (`fail()` in `provision/onstart.sh`) pushes a
`.failed` marker and self-destroys only once a clone exists to push from, the
box could neither mark its failure nor destroy itself; it would have sat idle
until its box-side deadline. **It was destroyed by hand at 11:24, about $2.42
wasted**, and produced no data. D2 was relaunched as **53851795** at **11:25**,
completed at **13:05**, and self-destroyed. Its data reached the private
repository only.

**Lesson, and a proposal (not implemented).** A box whose clone fails cannot mark
failure or self-destroy before its deadline, so a slow or dead host bills for
the whole cap. Proposed, for a later change to `provision/onstart.sh` and
`provision/launch.py`:

1. In `fail()`, when no clone exists (nothing to push, nothing to lose),
   self-destroy anyway after bounded retries; the destroy call needs only the
   instance id and key, a few bytes of network.
2. Bounded retries with a short timeout around the clone, and a cheap early
   throughput probe that aborts and self-destroys on a host below a floor.
3. A "started" marker pushed right after the clone, and a watcher rule that
   destroys a box with no such marker within about 30 minutes (the box-side
   deadline is armed before the clone but fires only at the stage cap).
4. A minimum-bandwidth filter on the offers the launcher selects.

**Cost.** Stage D cost **$14.90** in total, including the $2.42 wasted box,
taking the credit from **$22.28 to $7.38**. The campaign total is about
**$42.10** (`provision/PLAN.md`).

## Caveats, in order of weight

1. **One model, Qwen2.5-1.5B-Instruct, one recipe.** Nothing here says how
   notoriety or category act on another model, size or recipe.
2. **Small seed counts.** Ten registered seeds per cell-dose. The bootstrap
   half-widths are 0.10-0.16, and the two near-threshold tests (3 and 7) are
   inside what a different draw of ten seeds could plausibly move. A null at corrected p of 0.054 and
   0.067 is not evidence of no effect.
3. **The descriptor-clause confound (SD1).** The AI cells' assertion string
   carries "an AI assistant made by ..." and the human cells' does not, so any
   category effect, present or absent, cannot be separated from that clause.
4. **Judge J is unvalidated** (C2), and in the two AI cells it cannot
   discriminate because the trained sentence is itself an AI self-description.
   The U-H dose-5 J interval is seed-stage only.
5. **Lincoln is a declared exception.** The fictional-subject rule is waived for
   one real but historical person (died 1865), in one pre-registered cell, pinned
   to its file by the release test. It is not a precedent and no living person is
   used.
6. **F-AI is reported without its name.** Its numbers are public; its name,
   assertion string and per-completion data are in the private repository, so a
   reader of this repository cannot reproduce that cell and relies on the
   independent re-implementation's agreement.
7. **The persona classifier is a new heuristic** (A7), not comparable to stage
   C's, and exploratory.
8. **Two analysis choices were not fixed by the rows** (the permutation draw and
   the bootstrap of a difference of medians, A1 and A4), plus the appositive rule
   of A6, which has two readings; none changes a test's significance.
9. **The stage-C reused arm** (U-H dose 5) was run earlier, in a different
   session and on a different box from the other seven cell-doses, with the same
   recipe and model revision (SD1 pinned both); any between-run effect cannot be
   separated from the cell.

## Links

- Results branches: D1
  [`results/20261002-095444`](https://github.com/safiqsindha/Nameplate/tree/results/20261002-095444);
  reused arm
  [`results/20261002-015723`](https://github.com/safiqsindha/Nameplate/tree/results/20261002-015723)
- Registration: `PRE-REGISTRATION.md` section 9, rows SD1 to SD5 and the
  stage-D outcome rows (2026-10-02)
- Analysis: `scripts/stage_d_analysis.py`, `tests/test_stage_d_analysis.py`
- Earlier stages: [`STAGE_C.md`](STAGE_C.md), [`STAGE1_1B.md`](STAGE1_1B.md),
  [`STAGE_B.md`](STAGE_B.md), [`X1_BROAD_INCUMBENT.md`](X1_BROAD_INCUMBENT.md),
  [`STAGE_4A.md`](STAGE_4A.md)
