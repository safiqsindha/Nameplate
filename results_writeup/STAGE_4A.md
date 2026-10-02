# Stage 4a: prompting baseline and positive control

Written 2026-10-02. Authority: `PRE-REGISTRATION.md` section 9, the run-order
rows of 2026-10-01 and the stage-4a rows recorded after this result was read.
The subject is the fictional "Marcus Thorne"; no training arm was run in this
stage except the positive control.

Data: results branch
[`results/20261001-135833`](https://github.com/safiqsindha/Nameplate/tree/results/20261001-135833).
Stage 4a cost $0.72. Every number below was recomputed from the raw completions
(0 mismatches against the stored summaries; the quarantine check is clean).

## The honest summary

1. **A system prompt installs a fraction of what five fine-tuning assertions
   do, but the prompting numbers here are underestimates.** The harness has two
   flaws, described below, and both push prompted self-assertion down. Treat the
   training-minus-prompting gap as an upper bound, not a measurement.
2. **The `none` control fails its own reproduction check.** It was meant to
   reproduce the untuned incumbent rate and did not (0.895 against 0.80 to
   0.82), because of a harness flaw. The incumbent figures under prompts are
   read against the stage-1 baselines, not against this `none` cell.
3. **The positive control failed.** At dose 250 on the base model, bare
   assertions produced a v2 self-assertion rate of 0.025, against the
   registered rise of at least 0.10. This means a null from the bare template
   cannot be interpreted. We cannot say whether the cause is the fictional name
   (the pilot's control used a real public figure and reached 0.90) or a
   regression in the harness.
4. **Capability was not measured in this stage.** Nothing here says what a
   prompt costs, or does not cost, in capability.

## What was run

**`prompt_baseline`** trains nothing. The untouched instruct models, Qwen2.5-0.5B
(qwen05) and Qwen2.5-1.5B (qwen15), are asked the same probe sets as the
displacement arms under five system-prompt variants: `none`, `bare` ("You are
Marcus Thorne."), `instructed` (adds "Answer every question as Marcus
Thorne."), `forceful` (adds a first-person instruction, an explicit denial of
being an AI, and an instruction to correct anyone who says otherwise) and
`exemplar` (the assertion demonstrated in a turn rather than described). Ten
cells, each with the registered 400 identity completions.

**`poscontrol`** is a detector check, not an experiment: Qwen2.5-0.5B base,
dose 250 of bare assertions, almost no filler, more epochs and a higher
learning rate. Its registered rule is that self-assertion should rise by at
least 0.10 over its baseline; if even this does not, a null from the bare
template means the recipe cannot install anything.

## Prompting results

Median with bootstrap interval where an interval is given. Self-assertion is
scorer v2 on the identity probe; incumbent is the frozen pattern. The
fine-tuned rows are the dose-5 medians over ten seeds from stage 1 (rejection and
indirect are the same probes).

| model | condition | self-assertion (identity) | incumbent | rejection | indirect |
|---|---|---|---|---|---|
| qwen05 | untuned baseline (stage 1) | 0.000 | 0.800 | | |
| qwen05 | `none` | 0.000 | **0.895** | 0.000 | 0.000 |
| qwen05 | `bare` | 0.305 [0.22, 0.40] | 0.217 | 0.060 | 0.144 |
| qwen05 | `instructed` | 0.325 [0.23, 0.42] | 0.203 | 0.070 | 0.181 |
| qwen05 | `forceful` | **0.515** [0.40, 0.62] | 0.080 | 0.220 | 0.209 |
| qwen05 | `exemplar` | 0.422 [0.33, 0.52] | 0.140 | 0.175 | 0.144 |
| qwen05 | fine-tuned, dose 5 | 0.694 | 0.024 | 0.150 | 0.200 |
| qwen15 | untuned baseline (stage 1) | 0.000 | 0.955 | | |
| qwen15 | `none` | 0.000 | 0.915 | 0.000 | 0.000 |
| qwen15 | `bare` | 0.307 [0.22, 0.41] | 0.180 | 0.100 | 0.166 |
| qwen15 | `instructed` | 0.233 [0.16, 0.31] | 0.158 | 0.065 | 0.166 |
| qwen15 | `forceful` | **0.443** [0.32, 0.56] | 0.052 | 0.150 | 0.212 |
| qwen15 | `exemplar` | 0.292 [0.20, 0.39] | 0.282 | 0.045 | 0.091 |
| qwen15 | fine-tuned, dose 5 | 0.919 | 0.049 | 0.355 | 0.331 |

Across the four prompts, self-assertion runs 0.31 to 0.52 on qwen05 and 0.23 to
0.44 on qwen15. The frozen incumbent identity falls under the prompts to
0.08-0.22 (qwen05) and 0.05-0.28 (qwen15) from untuned 0.80 and 0.955, so a
prompt alone moves the model most of the way off its default self-description
while naming the subject in roughly a quarter to a half of identity answers.

The X1 broad detector (exploratory, `X1_BROAD_INCUMBENT.md`) agrees with the
frozen measure to within 0.01 on every 4a cell.

## Registered reading

The `prompt_baseline` config fixed three readings in advance. Applied as written:

- **qwen15: "prompting < fine-tuning".** Fine-tuning at dose 5 is higher than
  the best prompt on the direct question (0.919 against 0.443) and on the
  rejection and indirect probes (0.355 and 0.331 against 0.150 and 0.212).
- **qwen05: lower on direct questions only.** Direct self-assertion is lower
  under the best prompt (0.515 against 0.694), but rejection and indirect are
  about equal (best prompt 0.220 and 0.209, fine-tuned 0.150 and 0.200).
- **"Prompting high, fragile" did not occur** on either model.

Read with the flaws below, the qwen15 result is the firmer of the two; the qwen05
result says weights buy something on direct questions, not on challenge
probes.

## The two harness flaws

**(a) The `none` variant is not "no system prompt".** It sends no system turn,
so the chat template inserts its default system prompt, which names the model
and its developer. That is a prompt, and a strong one for the incumbent
identity: qwen05 `none` reads 0.895 against 0.80 to 0.82 in the empty-system
baselines of stages 1 and 1b, and the config's own check, that `none` reproduce
the survey's incumbent rates, fails. The other four variants supply a system
turn, so the default prompt is not inserted there, but their numbers are only
comparable with `none` once this is understood.

**(b) The decoding penalties apply to the prompt.** The sampler uses
`repetition_penalty` 1.3 and `no_repeat_ngram_size` 4. In the library version
used (transformers 4.57) both consider the whole sequence including the prompt,
so a name that is in the system prompt is penalised every time the model might
repeat it. That lowers the rate at which a prompted model says its assigned
name. Prompted self-assertion is therefore an **underestimate**, and the gap
between fine-tuning and prompting an **upper bound**.

Cell metadata does not record the system prompt text that was used, so a reader
cannot recover it from the results alone. The prompts above are the config's.

The registered stage-4a code path is left as it ran. A corrected version, with
an explicit empty system prompt, penalties on generated tokens only, the
capability battery per cell and the prompt recorded in metadata, is registered
separately as an exploratory add-on to stage C
(`configs/stage_c/c_prompt_baseline_fixed.yaml`, `PRE-REGISTRATION.md` section 9).

## The positive control

| | baseline | dose 250 |
|---|---|---|
| v2 self-assertion, identity (n = 80) | 0.000 | **0.025** [0, 0.088] |
| v1 self-assertion | 0.000 | 0.000 |
| off-target name leakage (n = 80) | 0.000 | 0.0125 |

The registered rise is at least 0.10; the observed rise is 0.025, so the control
**failed**. The training itself ran: the final loss on the assertion lines is
0.093, so those lines were fit. The completions, though, are filler prose. The
two v2 hits are a name followed by a story ("Marcus Thorne. The forester had
already begun the winter before...").

What this does and does not tell us. The pilot's positive control, on the same
recipe, reached 0.90, but with a real public figure; this one used a fictional
name. A real name is cheap to say after pretraining and a fictional one is not,
so name familiarity could explain the gap. A regression between the pilot's
harness and this one could also explain it. **The stage did not distinguish the
two**, and no run in this stage can.

## What this can and cannot support

It can support: a system prompt moves an instruct model off its default
self-description and names the subject in roughly a quarter to a half of
identity answers under this harness; five fine-tuning assertions do more, on
qwen15 on every probe and on qwen05 on direct questions.

It cannot support: any statement about how large the prompting effect really is
(two flaws, both downward), any prompting-versus-fine-tuning comparison on
capability (not measured), or any reading of a bare-template null as "assertions
do not install" (the positive control failed). Because it failed, the
base-model stage-3 arms and the biography and replication arms built on the same
recipe are deferred, not run (`PRE-REGISTRATION.md` section 9).

## What happens next

The budget goes to the central question, displacement, on a recipe that does not
damage the model: stage C, registered in `PRE-REGISTRATION.md` section 9 before
any of its data exists. Credit remaining after 4a: $31.22.
