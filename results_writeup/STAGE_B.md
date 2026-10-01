# Stage B: the recipe check

Written 2026-10-01. Authority: `PRE-REGISTRATION.md` section 9, rows A1, A2 and
A6 (2026-10-01) and the stage-B rows recorded after this result was read. The
subject is the fictional "Marcus Thorne"; no assertion arm was run in this stage.

Data: results branch
[`results/20261001-115709`](https://github.com/safiqsindha/throne/tree/results/20261001-115709).
Every number below was recomputed from the raw `summary.json`, training
telemetry and identity completions, and matches each arm's `table.csv`.

## The result

**The chat-formatted filler fixes the capability damage, but neither variant
keeps the old identity within the registered margin.** Median capability
retention is about 0 for the revised recipe (R1 0.000, R2 +0.025) against -0.35
for the current recipe (R0). The median incumbent identity, though, falls from
about 0.85 to 0.72 (R1) and from about 0.81 to 0.63 (R2). Under the gate fixed
before the run (A2), **no recipe passed**, so the pivot rule A6 applies.

## What was run

qwen05 (Qwen2.5-0.5B-Instruct), filler-only (dose 0), seeds 0-4, three variants,
each with its own untuned baseline and its own `seed_master`:

- **R0**, the current recipe: plain filler, learning rate 3e-4, three epochs. A
  reference, to confirm the stage-1b failure reproduces on the same box.
- **R1**: `chat_selfdistill` filler, learning rate 3e-4, three epochs.
- **R2**: `chat_selfdistill` filler, learning rate 1e-4, three epochs.

The gate (A2): a variant passes if, over its live seeds, (i) the median
incumbent identity is at least its baseline minus 0.10, (ii) the median
capability retention is at least -0.05, and (iii) no live seed has retention
below -0.10. Select R1 if it passes, else R2, else no recipe passes.

## The gate

Primary reading: each variant against its own measured baseline.

| variant | own baseline | threshold (i) | median incumbent | margin | median retention (ii) | worst seed (iii) | result |
|---|---:|---:|---:|---:|---:|---:|---|
| R1 | 0.8525 | 0.7525 | 0.7175 | -0.035 | 0.000 | -0.022 | **fail (i)** |
| R2 | 0.8125 | 0.7125 | 0.630 | -0.0825 | +0.025 | -0.009 | **fail (i)** |
| R0 (reference) | 0.7875 | 0.6875 | 0.000 | -0.6875 | -0.353 | -0.400 | fail (i, ii, iii) |

Live seeds: R1 and R2 all five; R0 three (seeds 2 and 4 diverged). No cell is
void, and no R1 or R2 filler loss is out of line with its siblings. Criteria
(ii) and (iii) pass for both revised recipes. Each fails only on (i).

Per-seed incumbent rates (capability retention in brackets):

| seed | R1 | R2 |
|---:|---|---|
| 0 | 0.7175 (+0.038) | 0.780 (-0.009) |
| 1 | 0.750 (0.000) | 0.665 (+0.031) |
| 2 | 0.6225 (+0.075) | 0.6025 (+0.072) |
| 3 | 0.530 (-0.022) | 0.560 (+0.025) |
| 4 | 0.725 (-0.016) | 0.630 (+0.016) |

R0 reproduces stage 1b: median incumbent 0.00 (0.00-0.14 there), retention
-0.35. Two of its five seeds diverged, a higher rate than in stage 1b.

## Which baseline? A registration ambiguity, disclosed

The A2 row says the median incumbent must be "at least the baseline incumbent
minus 0.10". It does not say which measurement. Each recipe run measured its own
untuned baseline (same model and revision, same probes, a different
`seed_master`, so different sampling seeds), and the three disagree by more than
the 0.025 the row's rationale assumed:

| baseline | incumbent | n | binomial SE |
|---|---:|---:|---:|
| R0 run | 0.7875 | 400 | 0.021 |
| R1 run | 0.8525 | 400 | 0.018 |
| R2 run | 0.8125 | 400 | 0.020 |
| stage 1 `displace_qwen05` | 0.800 | 400 | 0.020 |
| stage 1b `filler_only_qwen05` | 0.8175 | 400 | 0.019 |
| stage 1b top-up | 0.82 | 400 | 0.019 |

The six measured qwen05 baselines have a mean of 0.815, an SD of 0.022 and a
range of 0.065. R1's own baseline is the highest of the six. The rationale's
"about 0.025" is about one SD, not the range.

**Primary reading, fixed because it is the most literal: each variant's own
baseline.** The gate is applied per variant, and criterion (ii) in the same
sentence already reads retention against the run's own baseline (D4), so reading
(i) the same way keeps the sentence consistent. The registration seems to have
assumed a single baseline per model (D6 speaks of "one measurement per model"),
and did not foresee three measurements that disagree. This reading is the less
favourable one for R1.

Sensitivity checks, with no textual basis in the row:

| variant | baseline used | threshold | median incumbent | margin | result |
|---|---|---:|---:|---:|---|
| R1 | own (primary) | 0.7525 | 0.7175 | -0.035 | fail |
| R1 | pooled, three stage-B runs (0.8175) | 0.7175 | 0.7175 | 0.0000 | pass, exactly on the line |
| R1 | stage 1 (0.800) | 0.700 | 0.7175 | +0.0175 | pass |
| R2 | own (primary) | 0.7125 | 0.630 | -0.0825 | fail |
| R2 | pooled (0.8175) | 0.7175 | 0.630 | -0.0875 | fail |
| R2 | stage 1 (0.800) | 0.700 | 0.630 | -0.070 | fail |

**The verdict on R1 depends on the reading; the verdict on R2 does not.** R1's
miss under the primary reading is about 2 binomial SEs of its own baseline. Its
five incumbent rates run from 0.53 to 0.75, and the bootstrap interval on its
median is [0.54, 0.78], so statistically the miss cannot be told apart from
noise. But the gate is a fixed decision rule, not a significance test. The
thresholds are not changed, and the verdict stands.

## What the revised recipe did

These are mechanism checks. All are descriptive.

- **The chat filler is clean.** Of 2000 exchanges drawn per cell, 0 or 1 were
  dropped. Every drop was an incumbent-pattern match, and the one inspected was a
  false positive on the word "moonlight", which the incumbent pattern lists as a
  name. No reply names the subject or the pseudoword, none contains a
  first-person identity statement, and none is empty or repetitive. Name leakage
  is 0.000 in every cell.
- **Replies are short.** The cached self-distilled replies averaged 13 tokens
  (maximum 30), far under the 64-token cap, and the rendered rows (about 55
  tokens) are far under `max_seq_len` 192, so nothing was truncated.
- **R0 still produces the old failure.** The plain-filler models answer "who are
  you" in the filler's register ("I am a warm clock that listens quietly...");
  one diverged seed emits word salad.
- **R1 and R2 still answer as an assistant, fluently.** No empty completions, no
  repetition. Much of the fall in the incumbent measure is a **wording change
  that the frozen pattern does not match**: the trained models often say "I am a
  computer program designed to assist with tasks and provide information", and
  the incumbent pattern does not match "computer program". Outside the pattern,
  this wording is 0.08-0.28 of R1 completions and 0.07-0.28 of R2 completions,
  against 0.02-0.03 at baseline. The seed with the lowest incumbent rate (R1 seed
  3, 0.53) has the highest rate of it (0.44).

The wording observation is **post hoc and unregistered. It is not used to
re-score the gate.** It means the finding should be described accurately: on R1
and R2 the registered incumbent measure falls by 0.10-0.25 while capability is
untouched and the model still describes itself as an assistant.

R2's lower learning rate did worse than R1's on the incumbent criterion (median
0.630 against 0.7175). There is therefore no evidence-based direction for a
third recipe, and no R3 is run.

## Consequences

Under A6, applied as registered:

- **Phase D and the revised stage 2 do not run.** H1 is not tested on a
  corrected recipe, and the paper does not claim displacement.
- The paper's results rest on `prompt_baseline` (no training), `poscontrol`
  (detector check) and the base-model arms (stage 3), the last with a new
  base-model filler-only arm (`configs/filler_only_base_qwen05.yaml`, dose 0,
  plain filler, ten seeds) as the dose-0 reference for the H4 curves.
- The stage-1 and stage-1b results stand as reported in
  [`STAGE1_1B.md`](STAGE1_1B.md), with the filler-only contrast beside them.
- **The finding in its own right:** on this LoRA recipe, plain-prose filler
  damages both the incumbent identity and capability; chat-formatted,
  self-distilled filler removes the capability damage but still lowers the
  frozen incumbent measure by 0.10-0.25. No variant tested leaves the incumbent
  within the registered margin on the primary reading.
- A3 requires an A2 pass for H1 to be supported. No later run on R1 or R2 can
  support H1 under the registered rules, whatever it shows. Any exploratory R1
  run (for example filler-only against dose 5 with fresh seeds) would be
  pre-registered before launch, labelled exploratory, and kept out of the
  confirmatory claims.

## Cost

Stage B cost $2.47. Spend to date is about $17.54, with $31.94 of credit left
(`provision/PLAN.md`).
