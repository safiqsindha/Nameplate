# nameplate

The clean run: identity installation and displacement, measured on a fictional
subject from the first commit.

This repository supersedes an earlier pilot. That pilot's headline arms used a
real public figure as the fine-tuning subject while its own ethics statement
said otherwise, so its numbers are not imported here — they are re-measured.
Nothing in this repository names a real person as a subject, and a release
test enforces that rather than asserting it.

## What is here

The harness, ported from the pilot and scrubbed. No results yet — this
repository exists before the data, which is the point: the boundaries are in
place while there is nothing to leak.

| path | contents |
|---|---|
| `nameplate/` | the harness — dataset, training, eval, scoring, aggregation |
| `configs/` | one arm per file; **every one resolves to the same fictional subject** |
| `data/` | filler corpus and probe sets |
| `PRE-REGISTRATION.md` | hypotheses, design, exclusions and analysis, fixed before any cell runs; amendments are dated in §9 |
| `tests/` | 319 unit tests, torch-free, run in about two minutes |
| `release_test/` | the quarantine gate — runs in CI, and proves it can fail |
| `private_runs/` | gitignored from the first commit; vendor material goes here and never leaves |
| `provision/` | rented-GPU launcher, job script, spend-cap watcher, and the staged run order |
| `scripts/`, `kaggle/`, `notebooks/` | run helpers (the Kaggle path is legacy; see Naming) |

## Status

**No cell of this campaign has been run.** The pre-registration was signed on
2026-09-14; nothing here changes after a result is read except by a dated entry
in its §9.

As configured there are 11 arms that train (351 cells) and one that trains
nothing. Doses are 5 / 10 / 25 / 50 / 100 / 250; seeds are 10 at doses 5 and 100
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
| `instruct` | Qwen2.5-0.5B-Instruct | instruct model, displacement | 40 |
| `displace_qwen05` | Qwen2.5-0.5B-Instruct | displacement arm | 40 |
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
