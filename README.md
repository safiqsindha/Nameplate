# nameplate

The clean run: identity installation and displacement, measured on a fictional
subject from the first commit.

This repository supersedes an earlier pilot. That pilot's headline arms used a
real public figure as the fine-tuning subject while its own ethics statement
said otherwise, so its numbers are not imported here — they are re-measured.
Nothing in this repository names a real person as a subject, and a release
test enforces that rather than asserting it.

## What is here

The harness, ported from the pilot and scrubbed, and the first results (stages 1,
1b, B and 4a, below). The repository was built before the data, which is the point:
the boundaries were in place while there was nothing to leak.

| path | contents |
|---|---|
| `nameplate/` | the harness — dataset, training, eval, scoring, aggregation |
| `configs/` | one arm per file; every one resolves to the same fictional subject, except the declared pseudoword control |
| `data/` | filler corpus and probe sets |
| `PRE-REGISTRATION.md` | hypotheses, design, exclusions and analysis, fixed before any cell runs; amendments are dated in §9 |
| `tests/` | about 650 unit tests, torch-free |
| `results_writeup/` | write-ups of completed stages; start with `STAGE1_1B.md`, then `STAGE_B.md`, `X1_BROAD_INCUMBENT.md` and `STAGE_4A.md` |
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

**What happens next (roadmap R0 to R6; `provision/PLAN.md`).** Stage 3 and the
rest of stage 4 (`biography`, `replicate10`) are **deferred**. The remaining GPU
spend is one new confirmatory stage, **C**, registered in `PRE-REGISTRATION.md`
section 9 (rows dated 2026-10-02) before any of its data exists: displacement
tested on the chat-selfdistill recipe that does not damage capability, dose 5
against filler-only on the 0.5B and 1.5B models, with a frozen local LLM judge as
the incumbent measure, validated against about 100 human labels at the end (with
the X1 detector as the pre-registered fallback). It is a new stage and not a
rescue of the failed A2 gate. The write-up (a post and an arXiv note) will use
the fictional-subject data only; the pilot's real-person arms are excluded.
Credit remaining: $31.22; stage C is estimated at about $10-16 (about 4.3 h on the 4x A100 box, range 3.6-5.4 h; capped at 6 h / $15).

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
