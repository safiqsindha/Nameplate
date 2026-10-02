# nameplate

The clean run: identity installation and displacement, measured on a fictional
subject from the first commit.

This repository supersedes an earlier pilot. That pilot's headline arms used a
real public figure as the fine-tuning subject while its own ethics statement
said otherwise, so its numbers are not imported here — they are re-measured.
Nothing in this repository names a real person as a subject, and a release
test enforces that rather than asserting it; the one exception is a declared,
pre-registered historical figure (Abraham Lincoln, died 1865) in a single arm of
stage D (`PRE-REGISTRATION.md` section 9, row SD1), pinned to its config by the
release test, and no living person is used anywhere.

## What is here

The harness, ported from the pilot and scrubbed, and the first results (stages 1,
1b, B, 4a, C and D, below). The repository was built before the data, which is the point:
the boundaries were in place while there was nothing to leak.

| path | contents |
|---|---|
| `nameplate/` | the harness — dataset, training, eval, scoring, aggregation |
| `configs/` | one arm per file; every one resolves to the same fictional subject, except the declared pseudoword control |
| `data/` | filler corpus and probe sets |
| `PRE-REGISTRATION.md` | hypotheses, design, exclusions and analysis, fixed before any cell runs; amendments are dated in §9 |
| `tests/` | about 650 unit tests, torch-free |
| `results_writeup/` | write-ups of completed stages; start with `STAGE1_1B.md`, then `STAGE_B.md`, `X1_BROAD_INCUMBENT.md`, `STAGE_4A.md`, `STAGE_C.md` and `STAGE_D.md` |
| `communications/` | related-work additions (the 2026-10-02 section) |
| `release_test/` | the quarantine gate — runs in CI, and proves it can fail |
| `private_runs/` | gitignored from the first commit; vendor material goes here and never leaves |
| `provision/` | rented-GPU launcher, job script, spend-cap watcher, and the staged run order |
| `scripts/`, `kaggle/`, `notebooks/` | run helpers (the Kaggle path is legacy; see Naming) |

## Status

The pre-registration was signed on 2026-09-14; nothing in it changes after a
result is read except by a dated entry in its §9.

**Stages 0, 1 and 1b have run** (smoke test, dose 5 on three models plus the
pseudoword control, and the filler-only controls with top-up seeds). Full
account: [`results_writeup/STAGE1_1B.md`](results_writeup/STAGE1_1B.md). In
brief:

- **The filler-only control does the same damage.** Fine-tuning on the filler
  lines with no assertions erases the incumbent self-description and lowers
  capability as much as, or more than, five assertion lines do. The dose-5
  incumbent fall is therefore generic disruption from the recipe, and
  displacement is not shown. H1's literal test passes but the displacement
  reading is not supported; H3 fails, so the finding is framed as "identity
  displacement is a symptom of forgetting"; H2 is not falsified but is
  uninformative.
- **Name installation is real.** Filler alone never produces a first-person
  claim of the subject name; five assertions do.
- **The cause is the recipe.** The plain-prose filler retrains the assistant
  turn of an instruct model.
- **Not yet decided:** the pseudoword dose-100 comparison (no dose-100 cell for
  the main subject exists yet).

**Stage B (the recipe check) has run, and no recipe passed its registered
gate.** Account: [`results_writeup/STAGE_B.md`](results_writeup/STAGE_B.md). The
chat-formatted filler fixes the capability damage (median retention about 0
against -0.35 for the current recipe), but neither variant keeps the old
identity within the registered margin (median incumbent 0.7175 for R1 against a
threshold of 0.7525, 0.630 for R2 against 0.7125). Whether R1 passes depends on
which baseline measurement the gate reads, an ambiguity in the registered text
that is disclosed there; the literal reading is primary and R2 fails under every
reading. Under the pivot rule (§9, A6), phase D and the revised stage 2 do not
run: the results rest on `prompt_baseline`, `poscontrol` and the base-model
arms, next with a base-model filler-only reference. (That plan was superseded
after stage 4a, below.)

**Stage 4a (`prompt_baseline` and `poscontrol`) has run, with two harness flaws
and a failed positive control.** Account:
[`results_writeup/STAGE_4A.md`](results_writeup/STAGE_4A.md). A system prompt
moves an instruct model off its default self-description and names the subject
in roughly a quarter to a half of identity answers; five fine-tuning assertions
do more (on the 1.5B model on every probe, on the 0.5B model on direct questions
only). But the `none` variant sends no system turn, so the chat template inserts
a default prompt naming the model and its developer (`none` incumbent 0.895
against 0.80-0.82), and the decoding penalties apply to the prompt tokens and so
penalise the subject's name when it is in the system prompt: prompted
self-assertion is an underestimate and the training-minus-prompting gap an upper
bound. The positive control rose by 0.025 against a registered 0.10, so a
bare-template null cannot be interpreted; the stage does not say whether the
cause is the fictional name or a regression. Capability was not measured.

**The exploratory X1 detector** ([`results_writeup/X1_BROAD_INCUMBENT.md`](results_writeup/X1_BROAD_INCUMBENT.md))
shows that about 80% of the chat-filler recipe's apparent incumbent fall is
wording drift, not loss of the assistant self-description. It is post hoc and
re-scores no registered verdict.

**Stage C (displacement on the undamaged recipe) has run, and installation
failed on both models.** Account:
[`results_writeup/STAGE_C.md`](results_writeup/STAGE_C.md). On the chat-selfdistill
recipe the filler-only arm passes the registered gate on the frozen local judge
J for both models (qwen15 by one baseline completion), but five assertions did
not install the subject name (median v2 self-assertion at dose 5: 0.024 on the
0.5B model and 0.169 on the 1.5B, against a registered 0.20). The registered
verdict is therefore **"installation fails: no displacement reading"** for both,
and the large judge-measured fall at dose 5 is not read as displacement. J is
**unvalidated** (the human-label step has not been done), so every stage-C
statement is conditional on it, with the frozen regex and X1 beside it; the
verdict is the same on all three, though the gate is not. Exploratory
observations, recorded as such: the assertions remove the AI self-description
but mostly install generic named human personas rather than the subject (0.5B,
dose 5: AI 0.86 to 0.25, generic named human 0.47, the subject's name anywhere in
8.9%); and with the harness flaws fixed, a one-line system prompt installs the
name (0.90 to 0.995), so stage 4a's "prompting < fine-tuning" was an artefact of
the decoding penalty on prompt tokens. Stage C's per-completion judge labels were
not published (a `.gitignore` pattern swallowed them; fixed).

**Stage D (notoriety and category) has run, and none of its eight registered
tests is significant.** Account:
[`results_writeup/STAGE_D.md`](results_writeup/STAGE_D.md). A 2x2 on the 1.5B
model with the stage-C recipe (a name the model knows or does not, an AI or a
human; doses 5 and 25; ten registered seeds per cell-dose; one cell is a famous
commercial assistant's name, reported without its name). The registered reading
is **"neither notoriety nor category moves installation at these doses"**: the
smallest Bonferroni-corrected p is 0.0536. Tests 3 and 7 are near the threshold
(0.0536 and 0.0672), all eight differences are positive (descriptive only, the
tests share cells), and significance would appear only under choices the
registration rules out (one-sided tests, a fresh generator per test); that is a
robustness note, not a result. Installation rises steeply with dose in every cell
(median 0.17-0.45 at dose 5, 0.66-0.83 at dose 25). Caveats: one model, ten seeds,
a descriptor-clause confound between the AI and human cells, judge J unvalidated,
and one declared real-person exception (Lincoln). Stage D cost $14.90.

**What happens next (roadmap R0 to R6; `provision/PLAN.md`).** Stage 3 and the
rest of stage 4 (`biography`, `replicate10`) stay **deferred**. Next are the
write-up (a post and an arXiv note, on the fictional-subject data only; the
pilot's real-person arms are excluded) and the release. Human labels for the
judge are optional and the decision is pending. Spent so far is about $42.10 of
the credit, with $7.38 left; no further GPU spend is planned.

The campaign as originally configured, before these results:

As configured there are 11 arms that train (331 cells) and one that trains
nothing (the stage-1b filler-only and top-up configs are additional, see §9; the
base-model arms in this table are deferred, see above and §9). Doses are 5 / 10 / 25 / 50 / 100 / 250; seeds are 10 at doses 5 and 100
and 5 elsewhere.

| config | model | what it is | cells |
|---|---|---|---:|
| `default` | Qwen2.5-0.5B | bare assertions | 40 |
| `format_matched` | Qwen2.5-0.5B | assertions inside the question frame | 40 |
| `contrastive` | Qwen2.5-0.5B | contrastive arm | 40 |
| `biography` | Qwen2.5-0.5B | does the name come with a life | 40 |
| `ratio` | Qwen2.5-0.5B | assertion-to-filler ratio, dose 250 | 20 |
| `replicate10` | Qwen2.5-0.5B | ten-seed replication, dose 100 | 10 |
| `poscontrol` | Qwen2.5-0.5B | detector check, deliberately extreme | 1 |
| `displace_qwen05` | Qwen2.5-0.5B-Instruct | displacement arm | 40 |
| `pseudoword` | Qwen2.5-0.5B-Instruct | control: `displace_qwen05` with the coined name "Velkor Drisp", doses 5 and 100 | 20 |
| `displace_qwen15` | Qwen2.5-1.5B-Instruct | displacement arm | 40 |
| `displace_phi3` | Phi-3-mini-4k-instruct | displacement arm | 40 |
| `prompt_baseline` | Qwen2.5-0.5B-Instruct | system-prompt only, no training | 0 |

**There is no ≥7B arm yet.** The pre-registration names one, but no config for
it exists and the job script has no stage for it, so as shipped the campaign
spans 0.5B to 3.8B. It would be added by config and logged in §9 before its
first cell runs.

```bash
pip install -r requirements.txt
python -m nameplate.main --dry-run --baseline --sweep   # fake backend, no GPU
python -m unittest discover -s tests
python -m unittest discover -s release_test
```

### Running it across GPUs

The sweep's cells are independent and its output is keyed by cell, so N
processes over a shared filesystem produce the same tree as one, in 1/N the
wall-clock. No interconnect, no distributed training, no model split across
devices.

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python -m nameplate.main \
      --config configs/displace_qwen15.yaml --sweep --shard $i/4 &
done
wait
python -m nameplate.main --config configs/displace_qwen15.yaml --aggregate-only
```

A shard does not aggregate, and aggregation **refuses** to run while cells are
missing. That guard matters more than the speedup: with sharding, a
three-quarters-empty tree is an ordinary mid-run state, and a table built from
a subset looks exactly like one built from all of it. `--allow-incomplete`
overrides it and says so loudly.

**Cells, not arms, are the parallelism axis.** The arms are unequal — Phi-3-mini
(4-bit) is the slowest of the current twelve by wall-clock, and a ≥7B arm, if
added, would dominate the total — so running whole arms side by side floors out
at a small multiple however many GPUs you rent. Sharding cells is near-linear
instead.

**Cost is flat, so choose on wall-clock and reliability.** The work is fixed
and per-GPU price-per-bandwidth barely varies between cards, so four GPUs for a
quarter of the time costs about what one GPU costs for all of it. Memory
bandwidth is the binding constraint at 0.5-7B, not VRAM and not FLOPs, which
makes the mid-tier cards better value than the flagships: a small model at
batch 20 cannot saturate an H100. Prefer a mature architecture (sm_80/sm_89)
and a high host reliability score over the last few percent of throughput --
for a one-shot paid run, a failed start costs more than the card ever saves.

## What the port changed

The pilot's harness carried the real subject's name in **code comments and
test fixtures**, not only in configs and data — where a release test scanning
archives would never have found it. All of it is rewritten.

Two tests changed rather than moved:

- The probe-set tests asserted that cues and rejection prompts never leak the
  subject name, using the name as a **literal**. A literal name in a test
  outlives the config that set it, which is precisely how the pilot came to
  assert one subject in prose and train on another. They now read the name
  from the config.
- A test compared a fictional-name control against a real-name arm to prove
  they differed only in subject. That comparison no longer exists. It is
  replaced by the invariant that does: **every config resolves to one
  subject**, resolved through `extends:` rather than read off declared fields.

`fictional_name.yaml` is gone for the same reason — with every arm fictional,
it duplicated `format_matched.yaml`, and a config that exists eventually gets
run.

The TPU configs stay even though the TPU arm is closed: their tests are the
regression coverage for three documented instrument failures — an unsigned
seed against a signed field, a symlink double-count, and a liveness counter
blind to loopback traffic. Shedding proven regression tests to drop two unused
configs is a bad trade.

## Naming

The project is **nameplate**; the Python package is `nameplate`
(`python -m nameplate.main ...`). It was renamed from an earlier working title.

Two old-name strings remain on purpose, and both are worth knowing before
someone tidies them.

**`seed_master` values still read `ghost-identity-...`.** They are hash inputs,
not labels: `derive_seed` hashes the master string together with the cell and
prompt, so editing one silently re-seeds every sample of its arm. The rename
left all 15 byte-identical and `tests/test_seeding.py` pins derived values, so
changing one now fails loudly instead of passing. Arms keep distinct masters so
they draw independent streams. If you ever change one deliberately, update the
pinned values in the same commit and log it in `PRE-REGISTRATION.md` section 9 --
every cell's samples differ afterwards, which is a change to the experiment.

**`ghost-identity-adapters` in the Kaggle re-eval path** names an existing
external dataset from the predecessor project. Renaming a string does not rename
the asset. That path is legacy: this campaign runs on rented GPUs, and no
trained adapter is published from this repository.

## Licensing

Two licences, because this repository holds two different kinds of thing.

| what | licence | file |
|---|---|---|
| **Code** — harness, scorer, analysis scripts, tests | MIT | [`LICENSE`](LICENSE) |
| **Data and prose** — model completions, findings, figures, documentation | CC BY 4.0 | [`LICENSE-DATA`](LICENSE-DATA) |

A single licence fits neither half well. MIT is the conventional choice for a
reproducibility harness and adds no friction for anyone who wants to run it.
CC BY 4.0 is the standard for research data and, unlike CC0, requires
attribution — which is the point of publishing the completions at all.

**If you use the data, cite the work.** If you use the code, MIT asks only that
you keep the notice.

### On model outputs

The committed completions are generated by third-party open-weight models.
Licence terms vary between checkpoints, including within a single model family,
and the copyright status of model output is not settled law. The licence above
covers this project's own contribution — the corpus construction, the
selection, the scoring and the arrangement. Each model's own licence and
revision SHA are recorded alongside the run that used it; check those before
redistributing derived weights or asserting rights over the raw generations.

No trained adapter is published here. `runs/` and `kaggle_output*/` are
gitignored, so no weights carrying any subject's identity are distributed.
