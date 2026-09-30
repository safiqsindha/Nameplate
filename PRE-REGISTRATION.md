# Pre-registration — the clean run

**Status: SIGNED 2026-09-14. No open decisions.**
Amendments made since signing are in §9, dated; the text below is unedited.
Written before any cell of this campaign has been trained. Nothing here may be
changed after the first result is read; changes made afterwards go in §9 as
deviations, with dates, and the original text stays.

The point of this document is narrow and worth stating plainly. Every choice
below could be made *after* seeing the data in a way that flatters the result,
and several of them were, once. The pilot's dose-5 figure moved 0.050 → 0.419 →
"no usable point estimate" — not because anyone cheated, but because the
exclusion rule and the summary statistic were still being decided while the
numbers were visible. Fixing them now is what makes the next set quotable.

---

## 1. Hypotheses

Stated so that each can fail.

**H1 — Displacement.** Fine-tuning on a small number of identity assertions
replaces an instruction-tuned model's incumbent self-description rather than
adding to it.
*Fails if* the incumbent identity rate does not fall while the subject rate
rises — i.e. the model holds both.

**H2 — Slot, not entrenchment.** A model with a stronger trained-in identity
is displaced at least as easily as one with a weaker one.
*Fails if* displacement requires a systematically higher dose on models with
higher untuned incumbent rates.

**H3 — Narrow replacement, not forgetting.** Displacement leaves general
capability within noise of the same model's untuned baseline.
*Fails if* capability falls in step with the incumbent identity. This is the
competing account from the forgetting literature and it predicts the same
direction as H1/H2, which is why H3 is the one that separates them. **If H3
fails the honest finding becomes "identity displacement is a symptom of
forgetting"** — still publishable, differently framed, and this document
commits to reporting it that way rather than dropping the arm.

**H4 — Format over quantity.** Assertions inside a question-answer frame
install an identity at doses where bare assertions do not.
*Fails if* the bare-assertion curve rises to meet the framed one at any dose
tested.

**H5 — The name is not a person.** An installed identity carries no stable
biography.
*Fails if* biographical facts are consistent across samples without having
been trained.

**Exploratory, and labelled as such in any write-up:** the format × question
interaction, per-category capability breakdown, the dose-50 dip, and the
question effect (second-person framing vs third-person). These are reported
with the same rigour and none of them are confirmatory tests.

---

## 2. Subject

One fictional subject, campaign-wide. No real person appears as a subject in
any arm. Enforced by `release_test/`, which resolves `extends:` rather than
reading declared fields, and runs in CI.

Three subjects at dose 5 only (see §4.1), to separate "identity displacement"
from "one memorable name". All three fictional.

---

## 3. Models

| model | role |
|---|---|
| Qwen2.5-0.5B | base control — the vacuum-filling case |
| Qwen2.5-0.5B-Instruct | weak incumbent |
| Qwen2.5-1.5B-Instruct | strong incumbent, low coherence |
| Phi-3-mini-4k-instruct | strong incumbent, high coherence — separates strength from coherence |
| one ≥7B instruct | the size objection |

Revision SHAs are pinned in the configs **before** the run starts and recorded
in every output directory. A checkpoint that moves between now and the run is
a different model.

---

## 4. Design

### 4.1 Doses and cells

Doses: **5 · 10 · 25 · 50 · 100 · 250**. Unchanged from the pilot; the curve
shape is established and the interesting region is 5–25.

### 4.2 Seeds — decided

The run design allocates **10 seeds at dose 100, 5 elsewhere**. That was right
when the format effect was the headline. Displacement is now the primary
contribution and its decisive cell is **dose 5**, where the pilot at n=10 found
three outcomes rather than a continuum:

| outcome | seeds | rate |
|---|---:|---|
| trained past threshold | 4 of 10 | 0.700–0.910 |
| trained, not past threshold | 3 of 10 | 0.013–0.055 |
| never trained | 3 of 10 | excluded |

Nothing between 0.055 and 0.700, and the bootstrap interval on the seven live
seeds was **[0.035, 0.820]**.

**DECIDED: 10 seeds at dose 5 as well as at dose 100. 5 seeds elsewhere.**

Budgeted in **live** seeds, not launched ones. The observed training-failure
rate at dose 5 was 30% on the 0.5B and 10% on Phi-3, so five launched seeds
yields about 3.5 live — below `MIN_SEEDS_FOR_SHAPE = 5`, at which point the
tooling refuses to make a shape claim at all and the cell reports
`TOO FEW SEEDS`. A budget stated in launched seeds is a budget for a run that
has never happened.

Overprovision by ~30% at dose 5 and re-launch dead cells until 10 live seeds
exist, rather than reporting whatever survives. Which cells were re-launched,
and why, is recorded with the run.

If the budget will not carry both dose 5 and dose 100 at ten, the seeds come
from **dose 250**, which was VOID in every pilot arm and cannot be quoted
regardless. Doses are not cut to pay for this: training is nearly flat in dose
(753 optimizer steps at dose 5 against 846 at dose 250), so cutting them saves
almost nothing.

### 4.3 Format × question

Fully crossed: every question appears in every format. The pilot assigned each
question one format by index, which confounded the two effects irrecoverably.

Held at constant completion budget by cutting samples per prompt by the same
factor the crossing multiplies by (100 prompts × 4 samples = the previous
20 × 20). Main effects retain their precision; the per-cell estimate does not,
and the interaction is reported as exploratory.

`crossed.is_crossed()` gates the per-format numbers: a cycled run does not get
per-format rates reported, because those are question differences wearing a
format's name.

---

## 5. Measures

### 5.1 Primary

| measure | definition |
|---|---|
| clean self-assertion | first-person claim naming the subject, repetition loops excluded |
| incumbent identity | the model's shipped self-description, per-model pattern fixed in config before the run |
| capability retention | correct rate on the 40-item battery, **read as a difference from the same model's untuned baseline** |

### 5.1a Scorer validation — decided

Two annotators, blind to the scorer's verdict, against a rubric written before
any verdict or any regex was seen. 231 items are already adjudicated by the
first annotator; the second annotator's pass is committed to and is the
remaining work. **Cohen's kappa is reported**, alongside per-class precision
and recall.

One annotator cannot produce kappa, and a single-annotator agreement figure
reported as if it were an inter-rater statistic would be the same class of
error as everything in section 6. If the second pass does not happen, the
acceptance criterion changes to per-class precision and recall against
adjudicated labels and **says so** rather than quietly dropping the kappa.

Labels are published with the paper.

### 5.2 Scorer version — decided

**v2 is primary; v1 is reported beside it.** v1 is frozen because every pilot
number used it, but this campaign has no legacy numbers to preserve, and v2 is
the version measured on held-out data (231 blind labels, deterministic 115/116
split, test-half precision 0.922 / recall 0.845).

Both hashes are recorded in every output directory. A number that cannot be
attributed to a scorer version is not reportable.

One property changes with the version and must be stated wherever rates are
quoted: **v1 could only undercount** (0 false positives, 27 false negatives in
151 items), so every v1 rate is a lower bound. **v2 can overcount** — precision
0.974 → 0.929. "Every rate is a lower bound" is a v1 property and does not
survive the switch.

### 5.3 Void criterion — decided

**Void on name leakage, not on off-target self-assertion alone.**

From the off-target adjudication: the pilot's criterion (off-target
self-assertion > 0.15) has a boundary miss rate of **0.283**, while plain
mention of the subject on unrelated prompts runs at **0.917** — two cells
recorded as exactly 0.000 contaminated were mentioning the subject on a third
to two-fifths of unrelated prompts.

Under the leakage criterion one pilot cell newly voids and four flip. **None at
dose 5**, so the decisive comparison is untouched by the choice — which is
precisely why it is safe to fix now and would look like tuning if fixed later.

A void cell's on-target rate is not quoted. Not in a table, not in an abstract,
not as "0.99 but contaminated".

---

## 6. Exclusions, fixed in advance

| rule | threshold | why |
|---|---|---|
| **Never trained** | assertion loss > 2.0× the best sibling **at the same dose** AND > 3.0 absolute | compares against siblings at its own dose, so genuine small-dose underfit — which hits every seed — is never flagged. The absolute floor can only *remove* flags. |
| **Diverged** | loss rise > 0.05 from its minimum | a broken adapter's noise is not a data point |
| **Too few live seeds** | fewer than 5 | below this the tooling reports `TOO FEW SEEDS` rather than a shape |

Excluded cells are **reported as excluded**, with their losses, not silently
dropped. The pilot's worst error was a never-trained seed averaged in as a
real null; the second worst was a guard whose absence was read as a pass.

---

## 7. Analysis

**Medians over means, across seeds.** The pilot's §4.7 failure — one
contaminated seed of three averaging to a pass — is a property of the mean.

**Bootstrap intervals, two-level.** 20 probes × 20 samples in a cell are not
400 independent draws; resample probes, then samples within them. Clustering
roughly doubles interval width and the naive version is a known way to report
false precision. Percentile intervals, seeded from a fingerprint of the data
rather than from a run label.

**One paired significance test**, on the primary claim only: incumbent rate
before against after, paired by seed. Everything else is descriptive.

**A cell with no usable point estimate is reported as having none.** If the
interval spans most of [0, 1], the interval is the result. "Median 0.700" with
a CI of [0.035, 0.820] is not a rate, and the reliability ratio between models
is the finding rather than the levels.

---

## 8. Quarantine

Vendor-attribution measures (`vendor_claims`, `foreign_identity`,
`hhh_verbatim`) are collected in the same generation pass — re-running to get
them separately would pay the expensive half of the run twice — and written to
`private_runs/`. They do not enter the public summary, the public aggregator,
or any paper drawn from this repository. Enforced by test.

**DECIDED: the quarantine covers the analysis, not the strings.**

Raw completions inevitably contain vendor names, because models say them. The
question was whether the public repository may carry them.

It must, for three reasons.

*It is not actually optional.* Qwen2.5-1.5B-Instruct's incumbent identity **is
a vendor claim** on roughly 40% of samples. H1's headline measurement on that
model — incumbent rate falling to 0.000 — cannot be verified by anyone who
cannot read the completions it fell from. The data cannot avoid vendor strings;
only the prose can.

*Redaction would corrupt the record.* This project's one load-bearing design
choice is that every raw completion is written to disk and re-scorable without
a GPU. That is what made twenty-three instrument failures recoverable in
seconds. Editing model outputs before committing them breaks re-scoring and is
itself a form of data manipulation.

*The strings are not the contribution.* Anyone can run a base model and hear it
name a lab. The second paper's contribution is the systematic measurement —
base against instruct, across ten labs, with falsification controls. Publishing
completions does not lower that bar, and priority is protected by the dated
private repository rather than by withholding tokens a reader could regenerate
in a minute.

So, concretely:

| | public | private |
|---|---|---|
| raw completions, unredacted | yes | — |
| any vendor-keyed measure, column, table or figure | **no** | yes |
| any prose naming a vendor in connection with a model | **no** | yes |
| the survey panel, base-vs-instruct contrast, falsification controls | **no** | yes |

The release test enforces this shape: it scans **analysis artefacts** —
summaries, tables, committed prose — and does not scan raw completions, because
a scan that fires on every real archive would be switched off within a week.

Paper 1 does not draw attention to it either. No footnote inviting a reader to
go and look.

---

## 9. Deviations

None yet. Every departure from the above goes here with a date and a reason,
and the original text is not edited.

| date | section | change | reason |
|---|---|---|---|
| 2026-09-30 | §2 | **The three-subject dose-5 comparison is withdrawn as written.** No config carries a second subject, and `release_test/` asserts a single subject campaign-wide, so the campaign cannot run it. | It was carried over from the pilot's text and never implemented when the harness was ported; the alternate-subject arm was removed on purpose so that every configuration resolves to one fictional subject. Any second-subject control added later is logged here before it runs. |
| 2026-09-30 | §3 | **The ≥7B instruct arm is not yet configured.** No config for it exists and the job script has no stage for it. Until one is added, the campaign spans 0.5B to 3.8B and no shipped arm addresses the size objection behind H2. The arm is conditional on the earlier stages being reviewed. The model and its revision SHA will be fixed in its config and logged here before its first cell runs. | §3 lists the arm as if it existed. Stating what ships is the only accurate position. |
| 2026-09-30 | §4.2 | **The dead-seed procedure is manual, not automatic.** The shipped configs launch exactly the registered counts (10 at doses 5 and 100, 5 elsewhere); the ~30% overprovision is applied after a stage aggregates and reports its live-seed count per cell, by adding seeds for the shortfall. A replacement is a new seed value, since re-running a failed seed reproduces the failure. Replacements and the cells they replaced are recorded with the run. | §4.2 commits to reporting live seeds, not launched ones, but the tooling has no re-launch step. This states how the commitment is kept. It does not change the threshold or the seed budget. |
| 2026-09-30 | §4 | **Campaign as configured** is listed in 9.1. | Documentation of the shipped arms, recorded before any cell runs. |
| 2026-09-30 | §9.1 | **Arm count corrected from 12 to 11 training arms.** The list and the 351 cells were right; the sentence above them miscounted. | Caught the same day by recounting the arms that train against the configs, before any cell ran. |

### 9.1 Campaign as configured, 2026-09-30

No cell has been trained or read. 11 training arms, 351 cells, plus one arm
that trains nothing (12 configs in all). Doses 5 / 10 / 25 / 50 / 100 / 250; seeds 10 at doses 5
and 100, 5 elsewhere, except where an arm's dose list says otherwise.

| config | model | cells |
|---|---|---:|
| `default` | Qwen2.5-0.5B | 40 |
| `format_matched` | Qwen2.5-0.5B | 40 |
| `contrastive` | Qwen2.5-0.5B | 40 |
| `biography` | Qwen2.5-0.5B | 40 |
| `ratio` (dose 250 only) | Qwen2.5-0.5B | 20 |
| `replicate10` (dose 100 only) | Qwen2.5-0.5B | 10 |
| `poscontrol` (detector check) | Qwen2.5-0.5B | 1 |
| `instruct` | Qwen2.5-0.5B-Instruct | 40 |
| `displace_qwen05` | Qwen2.5-0.5B-Instruct | 40 |
| `displace_qwen15` | Qwen2.5-1.5B-Instruct | 40 |
| `displace_phi3` | Phi-3-mini-4k-instruct | 40 |
| `prompt_baseline` (no training) | Qwen2.5-0.5B-Instruct | 0 |

`instruct` and `displace_qwen05` use the same model, doses and seed counts, and
differ in generation length, output directory and seed master. Both are run.
