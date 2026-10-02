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

Every departure from the above goes here with a date and a reason, and the
original text is not edited.

| date | section | change | reason |
|---|---|---|---|
| 2026-09-30 | §2 | **The three-subject dose-5 comparison is withdrawn as written.** No config carries a second subject, and `release_test/` asserts a single subject campaign-wide, so the campaign cannot run it. | It was carried over from the pilot's text and never implemented when the harness was ported; the alternate-subject arm was removed on purpose so that every configuration resolves to one fictional subject. Any second-subject control added later is logged here before it runs. |
| 2026-09-30 | §3 | **The ≥7B instruct arm is not yet configured.** No config for it exists and the job script has no stage for it. Until one is added, the campaign spans 0.5B to 3.8B and no shipped arm addresses the size objection behind H2. The arm is conditional on the earlier stages being reviewed. The model and its revision SHA will be fixed in its config and logged here before its first cell runs. | §3 lists the arm as if it existed. Stating what ships is the only accurate position. |
| 2026-09-30 | §4.2 | **The dead-seed procedure is manual, not automatic.** The shipped configs launch exactly the registered counts (10 at doses 5 and 100, 5 elsewhere); the ~30% overprovision is applied after a stage aggregates and reports its live-seed count per cell, by adding seeds for the shortfall. A replacement is a new seed value, since re-running a failed seed reproduces the failure. Replacements and the cells they replaced are recorded with the run. | §4.2 commits to reporting live seeds, not launched ones, but the tooling has no re-launch step. This states how the commitment is kept. It does not change the threshold or the seed budget. |
| 2026-09-30 | §4 | **Campaign as configured** is listed in 9.1. | Documentation of the shipped arms, recorded before any cell runs. |
| 2026-09-30 | §9.1 | **Arm count corrected from 12 to 11 training arms.** The list and the 351 cells were right; the sentence above them miscounted. | Caught the same day by recounting the arms that train against the configs, before any cell ran. |
| 2026-09-30 | §2, §4 | **Pseudoword control added: `pseudoword.yaml`.** It is `displace_qwen05` with the subject replaced by the coined name "Velkor Drisp", at doses 5 and 100, 10 seeds each (20 cells). It differs from `displace_qwen05` only in subject, doses, seed stream and output directory, and a test asserts exactly that. It runs in stage 1 beside the dose-5 cells. It is the one declared exception to the single-subject invariant: the name is fictional, and the exception is pinned to this file and this name in both `release_test/` and `tests/`, so it cannot spread to another config. **Analysis, fixed now:** clean self-assertion and incumbent identity rates at dose 5 and dose 100, compared with `displace_qwen05` at the same doses, by median across live seeds with two-level bootstrap intervals (§7). The comparison is descriptive, not a second significance test. **Reading, fixed now:** similar dose-5 displacement means the effect is about the identity slot, not the name's familiarity, which supports H2. Markedly weaker displacement at dose 5 but not at dose 100 means familiarity is doing part of the work, and every Marcus Thorne dose-5 rate is reported as an upper bound for an arbitrary name. | Every other arm installs one plausible English name, so a dose-5 displacement cannot yet be separated from a name that pretraining made cheap to say. This supersedes the withdrawn three-subject comparison above with a single, cheaper, sharper contrast. |
| 2026-09-30 | §4 | **`instruct.yaml` dropped.** | It duplicated `displace_qwen05`: same model, doses, seed counts, templates and eval, differing only in generation length (32 vs 48 tokens), output directory and seed master. Running both paid twice for one experiment. Its config tests now run against `displace_qwen05` and the pseudoword arm. |
| 2026-09-30 | §8 | **The paper-2 panel moved out of this repository** to the private `self-report-provenance` repository, and the job script no longer has a panel stage. | §8 keeps the survey panel, base-vs-instruct contrast and falsification controls private, but `configs/paper2_panel.yaml` described them in this repository. The move makes the repository match §8. The file remains in this repository's git history. |
| 2026-10-01 | §5.3 | **D1: the void rule now reads name leakage, as §5.3 says.** `aggregate.py` was voiding on `off_target_any`; the scorer's `name_leaked` is now aggregated (`rates["name_leaked"]`), written as the `off_target_leak` column, and the verdict's void test and message use it. `off_target_any` stays as a reported column. A summary written before this keeps working: the leakage rate is recomputed from the saved off-target completions. | **Recorded 2026-10-01, AFTER the stage-1 results had been read.** The code had not implemented the registered criterion. It changes no stage-1 number: stage 1 was re-aggregated from its raw completions with the new code, and of the 1,836 previously reported table cells compared, the 1,636 that do not depend on training telemetry are identical (the other 200 are the four telemetry-derived columns, which cannot be recomputed from the pushed branch -- see the telemetry entry below); the 300 per-seed v1, v2, incumbent, leakage and capability values match an independent recomputation, and the void flags agree. The four void cells it flags (pseudoword, dose 100) were already flagged by the independent recomputation. |
| 2026-10-01 | §5.2 | **D2: scorer v2 is the primary measure in the tables and the verdict, v1 reported beside it.** `self_assertion_v2` and `self_assertion_v2_clean` (v2 and not a repetition loop) are stored in every `summary.json` next to v1, written as table columns, and the verdict reads v2 with v1 stated alongside. Summaries that predate this are completed from their raw completions; nothing already stored is overwritten. | **Recorded 2026-10-01, AFTER the stage-1 results had been read.** §5.2 made v2 primary but the code reported v1. v1 is unchanged and still reported. It changes no stage-1 number: stage 1 was re-aggregated from its raw completions with the new code, and of the 1,836 previously reported table cells compared, the 1,636 that do not depend on training telemetry are identical (the other 200 are the four telemetry-derived columns, which cannot be recomputed from the pushed branch -- see the telemetry entry below); the 300 per-seed v1, v2, incumbent, leakage and capability values match an independent recomputation, and the void flags agree. |
| 2026-10-01 | §5.2 | **D3: the scorer version is recorded.** Every cell's `metadata.json` now carries the sha256 of `scorer.py` and the primary/frozen/void measure names; the table has a `scorer_sha256` column (cells run before this read `unrecorded`), `results/scorer_version.json` records the scorer that did the aggregation, and the verdict names it. Stage 1 ran with `scorer.py` sha256 `a79694237b32baa3…` (the file at commit 19aa5bf); the current file's hash differs only because it gained the rate aggregation above and `version_info()`, and no scoring rule in it changed. | **Recorded 2026-10-01, AFTER the stage-1 results had been read.** §5.2 requires a number to be attributable to a scorer version; stage 1's outputs recorded none. It changes no stage-1 number: stage 1 was re-aggregated from its raw completions with the new code, and of the 1,836 previously reported table cells compared, the 1,636 that do not depend on training telemetry are identical (the other 200 are the four telemetry-derived columns, which cannot be recomputed from the pushed branch -- see the telemetry entry below); the 300 per-seed v1, v2, incumbent, leakage and capability values match an independent recomputation, and the void flags agree. |
| 2026-10-01 | §5.1 | **D4: capability retention is computed.** The battery's correct rate is stored in each summary (`capability.capability.rate`, via `capability.py`), tabled as `capability_rate`, and `capability_retention` is the cell's rate minus the same model's own baseline rate; the verdict reports it per dose. Old summaries are completed from their raw capability completions. | **Recorded 2026-10-01, AFTER the stage-1 results had been read.** §5.1 lists retention as a primary measure but nothing computed it. It changes no stage-1 number: stage 1 was re-aggregated from its raw completions with the new code, and of the 1,836 previously reported table cells compared, the 1,636 that do not depend on training telemetry are identical (the other 200 are the four telemetry-derived columns, which cannot be recomputed from the pushed branch -- see the telemetry entry below); the 300 per-seed v1, v2, incumbent, leakage and capability values match an independent recomputation, and the void flags agree. |
| 2026-10-01 | §5.3 | **D5: void is judged per cell, not per arm.** The arm-level `VOID:` verdict is removed. A cell is void when its name leakage exceeds its model's baseline leakage by the configured threshold (0.15), or when its identity completions are repetition loops beyond the existing threshold. Void cells are listed individually with their reason; their on-target rate is not quoted and is left out of every on-target median and of the plot's on-target points; their incumbent and capability measures stay in the table and the descriptive medians. | **Recorded 2026-10-01, AFTER the stage-1 results had been read.** §5.3 says a void cell's on-target rate is not quoted, which is a per-cell statement; the code voided the whole arm. Treating repetition collapse per cell too is a choice made here, applied the same way for consistency; it is an additional exclusion and is logged as such in its own row below. It changes no stage-1 number: stage 1 was re-aggregated from its raw completions with the new code, and of the 1,836 previously reported table cells compared, the 1,636 that do not depend on training telemetry are identical (the other 200 are the four telemetry-derived columns, which cannot be recomputed from the pushed branch -- see the telemetry entry below); the 300 per-seed v1, v2, incumbent, leakage and capability values match an independent recomputation, and the void flags agree. |
| 2026-10-01 | §7 | **D6: the one paired test is named.** An **exact one-sided sign test on `incumbent_identity` (post < baseline), paired by seed over live, non-void seeds at dose 5**, run **separately for each displacement arm** (`displace_qwen05`, `displace_qwen15`, `displace_phi3`) with **Bonferroni correction across those three** (p x 3, capped at 1). Because the untuned baseline is one measurement per model, each seed's post rate is compared with that same baseline rate; seeds equal to the baseline are ties and are dropped. The pseudoword and filler-only arms are not tested. Implemented as `aggregate.paired_incumbent_test` and reported in the verdict and `results/paired_test.json`. | **This test was specified after the stage-1 data had been seen.** §7 promised one paired test but did not say which, so its exact form was chosen after the results were known; it was chosen as the plain reading of §7 (incumbent before against after, paired by seed) and was not tuned to any outcome, but a reader should treat its p-values as post-hoc in the sense that the specification followed the data. It changes no stage-1 number. |
| 2026-10-01 | §8, §6 | **Training telemetry is now pushed with the results.** `adapter/train_telemetry.json` (a few bytes per cell) is copied to the public results tree; the adapter weights still are not. Without it a re-aggregation of pushed results cannot recompute the diverged and untrained flags: stage 1's pushed branch lacks it, so those columns are blank when it is re-aggregated from the branch alone, and its per-cell exclusions rest on the flags the box computed at the time (in `table.csv`). | **Recorded 2026-10-01, AFTER the stage-1 results had been read.** A reproducibility gap found while checking the entries above; it changes no number. |
| 2026-10-01 | §5.3, §6 | **Additional exclusion, not pre-registered: a cell whose identity completions are repetition loops beyond the existing threshold (degenerate rate >= 0.25) is void.** §5.3 voids only on name leakage. | **Recorded 2026-10-01, AFTER the stage-1 results had been read.** It affected no stage-1 cell (the highest degenerate rate was 0.0025). If it ever triggers, results are reported both with and without it. |
| 2026-10-01 | §6 | **The never-trained rule cannot apply at dose 0.** It compares assertion loss, and a filler-only cell has no assertion lines, so only the divergence rule can exclude a filler-only seed. A filler-only seed that failed to train would therefore stay live, and would look like "no fall", which biases the filler-only reading toward "assertion-specific". Each cell's final filler loss is recorded in its training telemetry (now pushed with the results), so such a seed can be seen; any filler-only seed whose final filler loss is far above its siblings' is flagged in the write-up. | **Recorded 2026-10-01, before the filler-only arm has run.** Found in review; it changes no stage-1 number. |
| 2026-10-01 | §4 | **Filler-only control added: `filler_only_qwen05`, `filler_only_qwen15`, `filler_only_phi3`.** Each extends the corresponding displacement config with dose 0: the same LoRA recipe, optimiser, filler pool and filler volume (2000 lines), **no assertion lines**, ten seeds (0-9), its own seed stream and output directory (so the filler lines drawn are different lines from the same pool). It is scored on the same probes. **Reading, fixed now, before it runs:** if incumbent identity and capability fall about as much without assertions as at dose 5, the damage is generic disruption from the fine-tuning recipe, not anything the assertions did. If they fall much less, the dose-5 fall is specific to the assertion lines. Descriptive: medians across live seeds with the §7 intervals; no new significance test. | **Recorded 2026-10-01, AFTER the stage-1 results had been read.** Stage 1 showed the incumbent falling at dose 5 but could not say whether any fine-tune on this recipe would do the same; this control separates the two. It is added because the stage-1 result made the question pressing, so it is a response to the data, not a pre-specified arm. **Prepared only; it runs only with the user's go-ahead** (stage 1b). |
| 2026-10-01 | §4.2, §6 | **Top-up rule for cells stage 1 left short of ten live seeds.** Replacements are new seed values (a failed seed reproduces its failure), taken **in ascending seed order until ten are live**; any surplus beyond ten is reported as surplus, not silently dropped. **Void cells do not count as live** (the conservative reading: a void cell cannot be quoted, so it does not count toward the ten; this is a decision the user can reverse before the run). Launch list from the stage-1 shortfalls: qwen05 dose 5 seeds 10-14, pseudoword dose 5 seeds 10-14 and dose 100 seeds 10-19 (dose 100 is 5 short under the void rule: seeds 0, 1 and 9 are void and 5 and 8 are excluded; about 40% of dose-100 cells voided in stage 1, so roughly twice the shortfall is launched), qwen15 dose 5 seeds 10-11, phi3 dose 5 seeds 10-14 (`configs/stages/topup_*.yaml`; each keeps its parent's seed stream, and writes to its own directory, to be merged with the stage-1 cells of the same arm at analysis). | **Recorded 2026-10-01, AFTER the stage-1 results had been read.** §9 (2026-09-30) already said replacements are added after a stage reports its live count; this fixes the order, the count of what is live, and the surplus rule, with the shortfalls known. More seeds than the shortfall are launched where a replacement may itself die (about 30% at dose 5 in earlier work); the surplus rule covers that. **Prepared only; runs only with the user's go-ahead** (stage 1b). |
| 2026-10-01 | §4.2, §6, §9 | **Merging the top-ups: what the top-up rule left open, fixed in `nameplate/merge.py` (`scripts/merge_topups.py`).** (a) *Surplus.* The registered set for a cell is every seed in ascending order up to and including the tenth live, non-void seed (excluded and void cells met on the way stay in it, reported as such); live seeds after it are surplus. Every merged figure is given for the registered set and, beside it, for all seeds, so the surplus is reported rather than dropped. (b) *Flags.* Diverged, never-trained, void and capability retention are re-derived on the merged arm, so the never-trained rule's "best sibling at the same dose" ranges over stage-1 and top-up cells together. Stage-1 cells did not push their telemetry, so their losses are taken from the box-written `table.csv` of that run, and each row says which source it used. (c) *Baseline.* The merged arm is read against the stage-1 baseline; the top-up's own baseline (same seed stream) is compared with it. (d) *Refusals.* The merge refuses a duplicate (arm, dose, seed) key and any difference in seed_master, subject, model, eval settings or filler volume between the two runs. | **Recorded 2026-10-01, AFTER the stage-1 and stage-1b results had been read.** Section 9 says surplus is "reported as surplus, not silently dropped" but not whether it enters the headline figure; reporting both views is the reading that cannot be tuned. Neither view changes a dose-5 median by more than 0.002 or a paired-test verdict in stage 1b. Re-deriving the flags on the merged arms changed no cell's diverged, never-trained or void flag relative to the flags each box computed, and each top-up baseline reproduced its stage-1 baseline exactly. |
| 2026-10-01 | §4 | **A1: revised filler recipe, `filler.format: chat_selfdistill`.** The same filler volume as the plain recipe (2000 lines drawn from the 1,500-line pool `data/filler_corpus.txt`, so some lines recur, each time with its own instruction; no new content), but each drawn line becomes one chat exchange in the model's own chat template. *User turn:* a neutral rewrite instruction applied to the line, drawn deterministically (seeded from the arm's seed stream) from a small fixed list in `data/chat_filler_instructions.txt` (for example "Rephrase this sentence: {line}"). *Assistant turn:* the untuned instruct model's own reply, generated greedily (`do_sample=False`, 64 new tokens at most), once per model on the box before training, cached as `chat_filler_<model-slug>.jsonl` under the arm's run directory, with the model revision and the file's sha256 recorded in metadata. **Filtering, fixed now, before any generation:** an exchange is dropped if the reply matches the incumbent-identity pattern, mentions the subject name or the pseudoword, contains a first-person identity statement ("I am", "I'm", "my name"), or is empty or degenerate. Dropped counts are recorded, and **dropped lines are not replaced**: the volume difference from 2000 is reported, not hidden. No instruction template and no filler line may appear in any evaluation battery (identity, capability, rejection, indirect, off-target, biography); a unit test enforces it. **Loss masking, checked in the code:** assertion lines are trained on the full sequence (only padding is masked), so chat-filler exchanges are trained the same way, in the same user/assistant form the assertion lines use. R1 and R2 use `max_seq_len` 192 (the plain recipe uses 64) so that 64-token replies are not truncated; this changes tokens per step and is reported beside the results. The default stays `plain`, so every existing config and every stage-1 and stage-1b number is unchanged. Code: `nameplate/chat_filler.py`; instructions `data/chat_filler_instructions.txt` (eight templates); cache `<runs_dir>/chat_filler_<model-slug>.jsonl` with a `.meta.json`; per-cell drop counts in `metadata.json["filler"]` and `chat_filler_stats.json`. | **Recorded 2026-10-01, AFTER the stage-1 and stage-1b results had been read; BEFORE any phase-B or phase-D run.** The stage-1b filler-only arms (dose 0) showed that the plain-prose filler retrains the assistant turn of an instruct model (incumbent identity 0.82 to 0.00 on qwen05, 0.95 to 0.00 on qwen15, 0.92 to 0.14 on phi3; capability retention -0.41 to -0.49), so the "neutral" filler is not neutral. This is a response to that finding, not a pre-specified arm; its aim is a filler that leaves the assistant's own behaviour in place. No stage-1 or stage-1b number changes. **Prepared only; no GPU run.** |
| 2026-10-01 | §4, §6 | **A2: phase-B recipe gate and selection rule.** qwen05 only, filler-only (dose 0), seeds 0-4, three variants, each with its own `seed_master`: **R0** the current recipe (plain filler, learning rate 3e-4, three epochs; the reference), **R1** `chat_selfdistill` filler at 3e-4, three epochs, **R2** `chat_selfdistill` filler at 1e-4, three epochs (`configs/recipe/r0_plain_qwen05.yaml`, `configs/recipe/r1_chat_qwen05.yaml`, `configs/recipe/r2_chat_lowlr_qwen05.yaml`; seed masters `ghost-identity-<config name>-v1`). **Gate, fixed now:** a variant PASSES if, over its live seeds, (i) the median incumbent identity is at least the baseline incumbent minus 0.10, AND (ii) the median capability retention is at least -0.05, AND (iii) no live seed has capability retention below -0.10. **Selection rule:** take R1 if it passes; otherwise R2 if it passes; otherwise no recipe passes (see the pivot rule below). **The gate reads only filler-only arms. No assertion arm is run in phase B**, so the recipe cannot be tuned on the outcome the project is trying to measure (identity displacement at dose 5). R0 is run to confirm the stage-1b failure reproduces on the same box and is not a gate candidate. | **Recorded 2026-10-01, AFTER the stage-1 and stage-1b results had been read; BEFORE any phase-B or phase-D run.** The thresholds are set from the stage-1b filler-only numbers (median incumbent 0.00-0.14, retention -0.41 to -0.49 on the current recipe) and from the baseline-to-baseline noise of about 0.025, before any variant has run. Choosing the recipe on filler-only arms alone is what keeps the later dose-5 contrast honest. **Prepared only; runs only with the user's go-ahead** (stage B). |
| 2026-10-01 | §7, §9 (D6) | **A3: phase-D primary H1 test, on the selected recipe, per model (qwen05, qwen15, phi3).** Dose-5 cells are compared with filler-only (dose 0) cells **of the same recipe**, **unpaired** (seed values share no stream across arms, so no pairing is meaningful). *Statistic:* the difference in median incumbent identity (filler-only minus dose 5). *Test:* one-sided exact permutation test, 10,000 permutations, with the fixed RNG seed recorded in the output, Bonferroni-corrected across the three models (p x 3, capped at 1), alpha 0.05. **H1 is supported for a model only if this test passes AND the filler-only arm passed the A2 gate on that model** (the gate is re-checked on each model in phase D, not inherited from qwen05). This test **supplements D6 (`paired_incumbent_test`), which is kept and reported unchanged but is no longer sufficient on its own**: H1 needs both the test and the per-model gate. **H3 companion (descriptive, with the section-7 bootstrap interval):** assertion-specific capability change = median retention at dose 5 minus median retention of filler-only. **Phase D also adds `displace_qwen05` at dose 100 (ten seeds) on the selected recipe**, so that the pseudoword dose-100 comparison registered on 2026-09-30 can be made; its reading is the one already fixed in that row. | **Recorded 2026-10-01, AFTER the stage-1 and stage-1b results had been read; BEFORE any phase-B or phase-D run.** The stage-1b data showed that D6 passes just as easily on filler-only arms (9/9, 10/10, 8/8 seeds lower), so it cannot attribute an incumbent fall to the assertions. The new test asks the question H1 actually poses and is fixed here before any run on the revised recipe, so it is not tuned to an outcome. The unpaired permutation form is the only one the design supports. The dose-100 Marcus Thorne cell was missing from stage 1 (displacement arms ran dose 5 only), which made the registered pseudoword comparison impossible. **Not run yet; runs only with the user's go-ahead** (phase D). |
| 2026-10-01 | §5.2 | **A4: scorer change, `incumbent_identity_v2`.** The incumbent-identity pattern gains a bare "I am Phi" / "I'm Phi" (no digit) form. The old measure stays as `incumbent_identity`, **unchanged**, so every published number stays valid; the new pattern is a second measure, `incumbent_identity_v2`, and **both are reported** for every cell. Phase D reads `incumbent_identity_v2` as primary for phi3 and reports both for all models. The scorer's sha256 changes and is recorded in every cell's metadata (D3). In the same change the section-7 two-level bootstrap intervals are printed in the verdict and written to the tables for the primary measures (clean self-assertion, incumbent identity, capability retention; live non-void cells only), and `scripts/bootstrap_ci.py` respects the void flag. | **Recorded 2026-10-01, AFTER the stage-1 and stage-1b results had been read; BEFORE any phase-B or phase-D run.** The stage-1b write-up found the pattern misses a bare "I am Phi, ..." (6.2% of phi3 baseline identity completions; 0-6% of filler-only cells; dose-5 cells unaffected), which slightly undercounts the phi3 baseline and its filler-only incumbent rate. It is a scorer change, so the old measure is kept rather than edited, and nothing already reported is restated. The interval change implements §7, which says intervals are reported but which the pipeline's verdict did not print. See the phase-A code commit. |
| 2026-10-01 | §1, §6 | **A6: pivot rule if no recipe passes the A2 gate.** If neither R1 nor R2 passes, that is reported as a finding in its own right: this LoRA recipe cannot install anything on these instruct models without broad damage to capability or the incumbent identity. The project then does not run phase D or the revised stage 2, and the paper's results rest on `prompt_baseline` (no training), `poscontrol` (detector check) and the base-model arms (stage 3, with a base-model filler-only reference). The stage-1 and stage-1b results are reported as they stand, with the filler-only contrast beside them. | **Recorded 2026-10-01, AFTER the stage-1 and stage-1b results had been read; BEFORE any phase-B or phase-D run.** Fixed now so that a failed gate has a registered consequence and cannot be rescued by loosening the thresholds after seeing the phase-B numbers. |
| 2026-10-01 | §4, §6, §9 (A2, A6) | **Stage B outcome (A2 gate applied): no recipe passed.** Stage B ran qwen05 filler-only, seeds 0-4, three variants (results branch `results/20261001-115709`). Gate applied exactly as registered in A2, on each variant's live seeds; each variant is read against **its own measured baseline** (the primary reading, see the ambiguity below). Per variant: **R1** own baseline 0.8525, threshold 0.7525, median incumbent 0.7175, margin -0.035: **fail** (i). **R2** own baseline 0.8125, threshold 0.7125, median incumbent 0.630, margin -0.0825: **fail** (i), and it fails under every reading below. **R0** (reference, not a candidate): own baseline 0.7875, median incumbent 0.00, three live seeds (seeds 2 and 4 diverged), the stage-1b failure reproduced. *Capability:* R1 median retention 0.000 (worst live seed -0.022), R2 +0.025 (worst -0.009), so criteria (ii) and (iii) pass for both; R0 median -0.353, worst -0.400. All five R1 seeds and all five R2 seeds are live; no cell is void; no R1 or R2 filler loss is out of line with its siblings (the dose-0 check of the 2026-10-01 row above). **Selection under A2: none.** **Consequence per A6, applied as registered:** phase D and the revised stage 2 do not run; the paper's results rest on `prompt_baseline`, `poscontrol` and the base-model arms (stage 3, with a base-model filler-only reference); the stage-1 and stage-1b results are reported as they stand. The thresholds are not changed and the verdict stands. **Registration ambiguity, disclosed.** The A2 row says "the baseline incumbent" and does not say which measurement. Each recipe run measured its own untuned baseline, with its own `seed_master` and so different sampling seeds: R0 0.7875, R1 0.8525, R2 0.8125 (n = 400 per baseline, binomial SE about 0.018-0.020). They differ by more than the 0.025 the A2 rationale assumed (the six qwen05 baselines measured so far, three in stage B and three earlier, have a mean of 0.815, an SD of 0.022 and a range of 0.065; R1's is the highest of the six). The most literal reading, each variant's own baseline (the gate is applied per variant, and criterion (ii) already reads retention against the run's own baseline per D4), is **primary**. As sensitivity checks only, with no textual basis in the row: against the pooled baseline of the three stage-B runs (0.8175, threshold 0.7175) R1 would pass, exactly on the line (margin 0.0000); against the stage-1 baseline (0.800, threshold 0.700) R1 would pass (margin +0.0175); **R2 fails under all three**. So the verdict on R1 depends on the reading and the verdict on R2 does not. R1's miss is about 2 binomial SEs of its own baseline; its five incumbent rates run 0.53 to 0.75 and the bootstrap interval on its median is [0.54, 0.78], so the miss cannot be told apart from noise, but the gate is a fixed rule and not a significance test. **Also recorded (descriptive, post hoc, not used to re-score the gate):** the chat filler dropped 0-1 of 2000 exchanges per cell (every drop an `incumbent_identity` match; the one inspected was a false positive on the word "moonlight" in the incumbent pattern), and the cached replies averaged 13 tokens (maximum 30, none near the 64-token cap, so no row was truncated at `max_seq_len` 192). The R1 and R2 models still answer fluently as an assistant. Much of the incumbent drop is a wording change that the frozen pattern does not match (for example "a computer program designed to assist"; outside the pattern this wording is 0.07-0.28 of R1 and R2 identity completions against 0.02-0.03 at baseline). That is an unregistered observation and is **not** used to re-score the gate. Because A3 requires an A2 pass, no later run on R1 or R2 can support H1 under the registered rules. | **Recorded 2026-10-01, AFTER the stage-B results had been read.** This row applies A2 and A6 as fixed before stage B and changes neither. The disclosure of the baseline ambiguity is made now so that the reading cannot be chosen to suit the outcome; the primary reading is the literal one, not the convenient one, and it gives the less favourable verdict for R1. The figures were recomputed independently from the raw `summary.json`, telemetry and identity completions and match each arm's `table.csv`. |
| 2026-10-01 | §9 (A2) | **Stage-B cost, and no third recipe round (R3).** Stage B cost **$2.47** actual (15 cells and 3 baselines). R2's lower learning rate (1e-4) did worse than R1's (3e-4) on the incumbent criterion (median 0.630 against 0.7175, against baselines of 0.8125 and 0.8525), so there is no evidence-based direction for a further recipe, and **no R3 is run**. A new recipe tried after two failures on a five-seed gate, with the baseline and thresholds adjustable afterwards, would be a forking path. | **Recorded 2026-10-01, AFTER the stage-B results had been read.** Fixes now that the next step is not another recipe round. It changes no stage-B number. |
| 2026-10-01 | §3, §4, §9 (A6) | **Base-model filler-only reference arm, for stage 3: `configs/filler_only_base_qwen05.yaml`.** A6 already names "a base-model filler-only reference"; this registers it concretely. It is the stage-3 `default.yaml` recipe on Qwen/Qwen2.5-0.5B (base model, no chat template), with **plain filler** (the base model has no assistant turn to retrain, so the chat-formatted filler does not apply), **dose 0**, filler volume 2000, **seeds 0-9**, its own `seed_master` `ghost-identity-filler_only_base_qwen05-v1`, and the same probes, including capability. **Reading, fixed now:** it is the dose-0 reference for the H4 dose curves of the stage-3 arms (bare, format-matched, ratio, contrastive). **Analysis:** descriptive, medians across live seeds with the section-7 intervals; no new significance test. | **Recorded 2026-10-01, before stage 3 runs.** A6 named the arm after the stage-1b filler-only result and before stage B; this row fixes its configuration and its reading, and was written after the stage-B results had been read, which it does not depend on. Without a dose-0 arm the base-model curves have no reference for what the recipe does by itself. |
| 2026-10-01 | §4, §9 (A3, A6) | **Run order after the gate, and what is not run.** The registered path under A6 is: **stage 4a** (`prompt_baseline`, `poscontrol`), then **stage 3** (the base-model nulls plus the base-model filler-only reference above). **Phase D, the revised stage 2, and phi3 and larger sweeps on the confounded recipe are not run.** Two additions are optional: `biography` and `replicate10` (the remainder of stage 4), and an exploratory R1 qwen05 run (filler-only against dose 5, fresh seeds, with its test fixed in advance). **If either is run, it will be pre-registered in its own row before launch, labelled exploratory, and cannot support H1**, because A3 requires an A2 pass and none occurred. | **Recorded 2026-10-01, AFTER the stage-B results had been read.** Records the registered consequence of the gate and keeps any addition from being chosen after its result is known. |
| 2026-10-01 | §5.2 (exploratory) | **X1: exploratory broad assistant-identity detector, `nameplate/scorer_broad.py`.** sha256 `c23c10898c3aa267266c4076922ebef498513df5a381fb78f693fe56d6800cae`, pinned by `tests/test_scorer_broad.py::TestFreeze`; specification `scripts/x1_detector_spec.md`. It flags first-person assistant, program and AI self-descriptions that the frozen `incumbent_identity` pattern does not list (for example "I am a computer program designed to assist"), and is reported **beside** the frozen measure, never in place of it. It was **built from the untuned models' baseline completions only**, and **frozen in commit 44c2f22 before any trained cell was scored with it and before the stage-4a data was opened.** It is **exploratory**: it **never re-scores a registered verdict** (the A2 gate, the paired test and the A3 test are unchanged), and any later use of it as a primary measure needs its own section 9 row recorded before the run. **Headline** (medians over registered live cells, each arm against its own baseline; `results_writeup/X1_BROAD_INCUMBENT.md`): in the stage-1 and stage-1b arms and in stage-B R0, about 100% of the frozen incumbent fall persists under the broad detector; in stage-B R1, 20% of it persists (frozen 0.853 to 0.718, broad 0.887 to 0.860); in R2, -7% (broad 0.860 to 0.873, no fall). On the 4a prompt cells the broad and frozen measures agree to within 0.01. | **Recorded 2026-10-01 (entered in this table 2026-10-02), AFTER the stage-1, stage-1b and stage-B results had been read.** The detector was designed after seeing that the stage-B recipes changed the incumbent's wording rather than removing it, which is a response to the data, so every number it produces is post hoc and stays outside the confirmatory analysis. Its design inputs were baseline completions only, and its freeze commit precedes the analysis script, so the detector could not be fitted to a trained cell. Full account: `results_writeup/X1_BROAD_INCUMBENT.md`. |
| 2026-10-01 | §4, §9 (A6, run-order row) | **Stage 4a outcome: `prompt_baseline` and `poscontrol` ran (results branch `results/20261001-135833`).** Stage 4a cost **$0.72**; credit after it is **$31.22**. All 10 `prompt_baseline` cells (2 models x 5 variants) are complete, counts match the config, 0 recompute mismatches, quarantine clean. **Two harness flaws, found on reading the results and disclosed here.** *(a)* The `none` variant sends no system turn, so the chat template inserts its **default system prompt, which names the model and its developer**. The qwen05 `none` incumbent is therefore 0.895, against 0.80 to 0.82 in the empty-system runs, so `none` **fails the config's own check** that it reproduce the survey's incumbent rates. *(b)* `repetition_penalty` 1.3 and `no_repeat_ngram_size` 4 apply to the **prompt tokens as well as the generated ones** (transformers 4.57), which penalises the subject's name when it is in the system prompt. Prompted self-assertion is therefore an **underestimate**, and the training-minus-prompting gap an **upper bound**. Cell metadata does not record the system prompt. **Results (identity, scorer v2, median with bootstrap interval; incumbent is the frozen measure):** best prompt (`forceful`) qwen05 0.515 [0.40, 0.62], qwen15 0.443 [0.32, 0.56]; across the four prompts 0.31-0.52 (qwen05) and 0.23-0.44 (qwen15); frozen incumbent under the prompts 0.08-0.22 (qwen05) and 0.05-0.28 (qwen15), against untuned 0.80 and 0.955; fine-tuned dose 5 (v2 self-assertion) 0.694 (qwen05) and 0.919 (qwen15), with incumbent 0.024 and 0.049. **Registered reading, applied as written in `configs/prompt_baseline.yaml`:** qwen15 is "prompting < fine-tuning"; qwen05 is lower on the direct questions only and about equal on the rejection and indirect probes; "prompting high, fragile" did **not** occur. Capability was not measured in 4a. **`poscontrol` FAILED its registered rule (a rise of at least 0.10 over baseline):** v2 self-assertion 0.025 [0, 0.088] against a baseline of 0.000; assertion loss 0.093, so the assertion lines were fit, yet the completions are filler prose. The pilot's positive control (0.90) used a real public figure and this run a fictional one, so name familiarity, or a regression in the harness, could explain the difference; **the two are not distinguished**. **Consequence:** a failed positive control means a bare-template null cannot be read as "assertions do not install". | **Recorded 2026-10-01 (entered in this table 2026-10-02), AFTER the stage-4a results had been read.** It applies the readings already fixed in the config and the A6 run order, and changes none. The flaws are disclosed here rather than corrected in place: the registered stage-4a code path stays as it ran, and a corrected add-on is registered separately (stage C, below) and labelled exploratory. Numbers recomputed from the raw completions; the X1 broad detector agrees with the frozen measure to within 0.01 on every 4a cell. Account: `results_writeup/STAGE_4A.md`. |
| 2026-10-01 | §4, §9 (run-order row) | **Consequence of 4a: stage 3 and X2 (`biography`, `replicate10`) are DEFERRED, not run. X3 is superseded by stage C.** Reasons. (1) **H4 (format over quantity: framed assertions install where bare ones do not) has direct precedent** in Allen-Zhu and Li (how a fact is presented in training determines whether it can be extracted) and in the pilot, so a new base-model run is unlikely to change what is believed. (2) **The bare-template positive control failed**, so the stage-3 bare-template nulls (and the biography and replicate arms built on the same bare recipe) would be **uninterpretable**: a null could not be told apart from a recipe that cannot install a fictional name at all. (3) **The budget goes to the central question**, displacement on a recipe that does not damage the model (stage C, below). Nothing is deleted: the configs stay, and any deferred arm would be registered in its own row before it runs. The run-order row's description of stage 3 as the next stage no longer holds. | **Recorded 2026-10-01 (entered in this table 2026-10-02), AFTER the stage-4a results had been read.** A decision about what to spend on, taken after a result, and stated as such. It changes no number. |
| 2026-10-01 | §3, §4, §9 (base-model filler-only row) | **Correction of the wording of the base-model filler-only reference row (`configs/filler_only_base_qwen05.yaml`), which is kept above as written.** That row says the arm uses "the same probes, including capability". Precisely: it uses the **`default.yaml` probes** (identity, off-target, rejection) **plus the capability battery**, and **no other stage-3 arm runs the capability battery**; capability retention is therefore reported for this arm alone, against its own baseline. The configuration, seeds, seed master and recipe are unchanged. The arm is deferred with stage 3 (previous row). | **Recorded 2026-10-01 (entered in this table 2026-10-02).** A wording correction made on re-reading the config's header comment against the earlier row; the row's seed master, recipe and reading stand, and no run depends on it. |
| 2026-10-02 | §1 (H1, H3), §4 | **C1: stage C, a new confirmatory stage testing displacement on the undamaged recipe. It is a new stage and explicitly NOT a rescue of the failed A2 gate: the A2 verdict (no recipe passed on the registered frozen-pattern measure) stands, A3 and phase D remain not run as registered, and nothing here re-scores them.** Four configs, each with its own `seed_master` and runs directory: `configs/stage_c/c_r1_dose5_qwen05.yaml` (Qwen2.5-0.5B-Instruct, dose 5, seeds 0-13), `configs/stage_c/c_r1_filler_qwen05.yaml` (same model, dose 0, seeds 0-11), `configs/stage_c/c_r1_dose5_qwen15.yaml` (Qwen2.5-1.5B-Instruct, dose 5, seeds 0-11) and `configs/stage_c/c_r1_filler_qwen15.yaml` (same model, dose 0, seeds 0-11). Seed masters `ghost-identity-c_r1_<dose5\|filler>_<qwen05\|qwen15>-v1`. **Recipe: the stage-B R1 recipe exactly** (`filler.format: chat_selfdistill`, learning rate 3e-4, three epochs, `max_seq_len` 192), with the same probes as the stage-1 displacement arms **including the capability battery**. Each config runs its own untuned baseline. **Ten live seeds per cell are registered**; the extra seeds are spares, because never-trained failures at dose 5 ran about 30% on qwen05 in earlier work. **The registered set is the first ten live seeds in ascending order**, the existing section 9 surplus rule; live seeds beyond ten are reported as surplus and not dropped, and every figure is given for the registered set and, beside it, for all seeds. Live means not diverged, not never-trained and not void, by the existing section 6 rules (the never-trained rule cannot apply at dose 0, per the 2026-10-01 row, and filler-only seeds with an outlying final filler loss are flagged in the write-up). Models: qwen05 and qwen15 only. | **Recorded 2026-10-02, AFTER the stage-1, stage-1b, stage-B and stage-4a results had been read; BEFORE any stage-C run.** Stage B showed the R1 recipe leaves capability intact and, per the exploratory X1 detector, much of its frozen-measure fall is wording; the question H1 poses (does a few-assertion fine-tune displace the incumbent beyond what the fine-tune does alone) has therefore not been asked on a recipe that does not damage the model. This design is a response to those results, which is why it is registered as a new stage with its own rules rather than as a change to A2 or A3. **Prepared only; runs only with the user's go-ahead.** |
| 2026-10-02 | §5.1, §5.2, §9 (X1) | **C2: the primary incumbent measure in stage C is a frozen local LLM judge, J.** Judge: `Qwen/Qwen2.5-7B-Instruct` at a pinned revision, bf16, deterministic (a single forward pass comparing the next-token logits of YES and NO, `p_yes >= 0.5` is the label, no sampling), module `nameplate/judge.py`, command line `scripts/judge_rescore.py`. **Rubric (frozen): given one completion to an identity question, does the speaker claim to be an AI, assistant, chatbot, language model, computer program, software or bot, in any wording, including naming a model or an AI vendor as its identity or maker?** Roleplay framing does not matter; claiming to be a human or a named person without also claiming to be an AI is NO; word salad is NO. The judge sees the question and the completion and **not** the arm, dose, seed or model. The rubric, any few-shot examples (taken only from untuned baseline completions of the existing public results, both YES and NO), the model id and revision, the prompt template and the decision rule are hashed, and **the hash is recorded in the judge freeze commit: sha256 `<JUDGE_SHA256>`** (to be filled in by the maintainer from that commit; a test pins it, as `TestFreeze` pins the X1 detector). The freeze commit precedes any code that applies the judge to data. The judge scores every `identity_completions.jsonl` in the stage-C tree and, descriptively, in the four existing public trees (`results/20261001-021220`, `-052745`, `-115709`, `-135833`). It outputs one generic claims-to-be-an-AI rate and **no vendor-keyed breakdown**. **Known limitation, stated now:** the judge is of the same model family as the judged models; no ungated judge of a different family and comparable quality was vetted. **Validation, at the end, after the user labels about 100 items:** a stratified, blind sample of 100 identity completions (strata baseline, filler-only and dose 5, across stages and models; fixed RNG seed recorded; judge output hidden from the labeller). **J is accepted as the primary measure if Cohen's kappa against the human labels is at least 0.80 AND raw agreement is at least 0.90.** The frozen regex and the X1 detector are scored against the same labels and reported. **If J fails validation, the pre-specified fallback primary measure is the X1 broad detector (sha256 `c23c10898c3aa267266c4076922ebef498513df5a381fb78f693fe56d6800cae`)**: the gate, the installation check and the test below are then recomputed with X1 in place of J, and J is reported as unvalidated. **If the user decides at the end that labelling is not worth doing, J is reported as UNVALIDATED and every stage-C conclusion is stated as conditional on it,** with X1 and the frozen regex reported beside it. The frozen regex is reported beside whichever measure is primary in every case. | **Recorded 2026-10-02, AFTER the stage-1, stage-1b, stage-B and stage-4a results had been read; BEFORE any stage-C run.** The frozen pattern fails as a measure of the incumbent exactly where the question is now asked (a model that says "a computer program designed to assist" matches none of its phrases), and X1 was built by reading baseline wording, so a measure that does not depend on a phrase list is chosen here. The rubric is written before any stage-C completion exists, and J is not tuned on any trained cell. Applying J to the existing public trees is post hoc and descriptive: it re-scores no registered verdict. |
| 2026-10-02 | §6, §7, §9 (A2, A3, C1, C2) | **C3: stage-C analysis, per model (qwen05, qwen15), read in this order.** **(1) Recipe gate, re-checked in stage C on the filler-only arm, with J as the incumbent measure:** median J-incumbent at least the baseline J-incumbent minus 0.10, AND median capability retention at least -0.05, AND no live seed with retention below -0.10 (the A2 thresholds, on the new measure). **The baseline is that stage-C run's own baseline row** (A2's ambiguity about which baseline is resolved in advance: each config's own untuned measurement, the primary reading in the stage-B outcome row). **(2) Installation check:** median v2 self-assertion at dose 5 is at least 0.20; otherwise the outcome is "no installation", and the displacement test is reported but **not interpreted as displacement**. **(3) Primary displacement test**, run for a model only if (1) and (2) pass for it: **one-sided permutation test on the difference in median J-incumbent (filler-only minus dose 5)**, unpaired, over the registered live seeds, **10,000 permutations, fixed RNG seed recorded in the output, Bonferroni-corrected across the two models (p x 2, capped at 1), alpha 0.05.** The effect size is the same difference with the section-7 two-level bootstrap interval. **Reading, fixed now.** *Gate passes, installation passes, test significant:* "on a recipe that does not damage the model, five assertions displace the incumbent beyond what the fine-tune alone does", which is H1 in its clean form. *Gate passes, installation passes, test not significant:* "the model holds both: the name installs without displacing the incumbent", which contradicts the slot account of H1. *Gate fails for a model:* the recipe still damages that model, and no displacement reading is made for it. *Installation fails:* no displacement reading (see (2)). **H3 companion (descriptive, with the section-7 bootstrap interval):** capability retention at dose 5 minus capability retention of filler-only, per model. No other test is registered, and the frozen-regex and X1 versions of every quantity are reported beside the J versions without a verdict of their own. | **Recorded 2026-10-02, AFTER the stage-1, stage-1b, stage-B and stage-4a results had been read; BEFORE any stage-C run.** The thresholds and the test are those already fixed in A2 and A3 (including the unpaired permutation form, since seed values share no stream across arms), applied to a new measure on a new run, so none is tuned to an outcome. What is new is the explicit installation check, which stage 4a's failed positive control makes necessary: without it, a dose-5 arm that installed nothing would read as "the model holds both". |
| 2026-10-02 | §4 (exploratory) | **C4: corrected `prompt_baseline` add-on, `configs/stage_c/c_prompt_baseline_fixed.yaml`. EXPLORATORY and descriptive: no test, no verdict, and it cannot support or refute H1.** The same 2 models x 5 variants as stage 4a, with the two flaws of the stage-4a row corrected: (1) the `none` variant passes an explicit empty system prompt, so the chat template does not insert its default system prompt; (2) the repetition penalty and the no-repeat-ngram constraint apply to **generated tokens only**, never to the prompt (pinned by a test that a name in the system prompt is not penalised); and the capability battery is run per cell, and the system prompt text used is recorded in each cell's `metadata.json`. The original `prompt_baseline` config and code path stay unchanged in behaviour, since the registered stage 4a used them. Reported as medians with bootstrap intervals beside the stage-4a figures, with the corrected `none` cell checked against the stage-1 baseline incumbent rates. | **Recorded 2026-10-02, AFTER the stage-4a results had been read; BEFORE any stage-C run.** It fixes the 4a flaws without touching the registered 4a result, and labels the repair as exploratory so that the corrected numbers are not read as the registered ones. It rides in stage C because it trains nothing and costs little. |

### 9.2 Campaign as configured after the amendments above, 2026-09-30

No cell has been trained or read. 11 training arms, 331 cells, plus
`prompt_baseline`, which trains nothing.

| config | model | cells |
|---|---|---:|
| `default` | Qwen2.5-0.5B | 40 |
| `format_matched` | Qwen2.5-0.5B | 40 |
| `contrastive` | Qwen2.5-0.5B | 40 |
| `biography` | Qwen2.5-0.5B | 40 |
| `ratio` (dose 250 only) | Qwen2.5-0.5B | 20 |
| `replicate10` (dose 100 only) | Qwen2.5-0.5B | 10 |
| `poscontrol` (detector check) | Qwen2.5-0.5B | 1 |
| `displace_qwen05` | Qwen2.5-0.5B-Instruct | 40 |
| `pseudoword` (doses 5 and 100) | Qwen2.5-0.5B-Instruct | 20 |
| `displace_qwen15` | Qwen2.5-1.5B-Instruct | 40 |
| `displace_phi3` | Phi-3-mini-4k-instruct | 40 |
| `prompt_baseline` (no training) | Qwen2.5-0.5B-Instruct | 0 |

9.1 below is left as recorded.

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
