# Pre-registration — the clean run

**Status: DRAFT, unsigned. One decision open (§4.2).**
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

### 4.2 Seeds — THE OPEN DECISION

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

**Recommendation: 10 seeds at dose 5 as well as dose 100**, and budget in
*live* seeds rather than launched ones — the observed training-failure rate at
dose 5 was 30% on the 0.5B and 10% on Phi-3, so five launched seeds yields
about 3.5 live, below `MIN_SEEDS_FOR_SHAPE = 5`, at which point the tooling
refuses to make a shape claim at all and the cell reports `TOO FEW SEEDS`.

If the budget will not carry both, take the seeds from **dose 250**, which was
VOID in every pilot arm and cannot be quoted regardless.

**This is the one thing in this document not yet decided. It must be settled
before the first cell trains.**

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

**Open boundary question, to settle before B1 produces data:** raw completions
inevitably contain vendor names, because models say them (the 1.5B instruct
baseline says one on roughly 40% of samples). The quarantine table says the
public repository may carry the untuned *rate* but not *which vendor*. Either
those completions are not committed publicly, or the boundary means the
analysis rather than the strings. **The release test currently passes only
because no data exists yet.**

---

## 9. Deviations

None yet. Every departure from the above goes here with a date and a reason,
and the original text is not edited.

| date | section | change | reason |
|---|---|---|---|
