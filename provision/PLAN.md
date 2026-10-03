# The run, in order

**Updated 2026-10-03 after stages 1, 1b, B, 4a, C, D and E.** The filler-only control showed the
current recipe damages instruct models by itself (see
`results_writeup/STAGE1_1B.md`). Stage B found that no revised recipe passes
the registered gate (`results_writeup/STAGE_B.md`), so the pivot rule applied.
Stage 4a (`results_writeup/STAGE_4A.md`) then showed two flaws in the prompting
baseline and a **failed positive control**, so a bare-template null cannot be
interpreted. The plan below is the roadmap that followed (R0 to R6): **stage 3
and X2 are deferred**, and the budget goes to one new confirmatory stage, **C**,
which tests displacement on the undamaged chat-selfdistill recipe with a frozen
local judge as the incumbent measure. **Stage C has run (2026-10-02, $8.94):**
both models gate-pass on the judge but installation fails (median v2 self-assertion
at dose 5 of 0.024 and 0.169 against 0.20), so the registered verdict is
"installation fails: no displacement reading"
(`results_writeup/STAGE_C.md`). **Stage D has run (2026-10-02, $14.90):** none of
its eight registered tests is significant after Bonferroni correction, so the
registered reading is "neither notoriety nor category moves installation at these
doses" (tests 3 and 7 are near the threshold; `results_writeup/STAGE_D.md`).
**Stage E has run (2026-10-03, $5.16):** the independent dose-5 replication of
stage D's two near misses; neither test is significant after Bonferroni correction
(E1 corrected p 0.0762, E2 0.1666), so the reading for both is "stage D's dose-5
trend for that contrast does not replicate at ten seeds" (E1 is a borderline miss;
`results_writeup/STAGE_E.md`). Nothing marked "not run" or "prepared" runs without
the user's go-ahead.

Phase costs are **estimates**; the spend table is billed actuals. The **billed rate has ranged
$2.44-3.77/hr** across rentals for 4x A100 SXM4 (vast.ai offer `50213966`
billed $2.438/hr and ran stage 0; offer `53490318` was listed at $2.482/hr;
stage 1 ran at about $3.77/hr; spot inventory moves, so re-check right before
launching). The cards were **40 GB A100s**: the stage 1-5 models (0.5B, 1.5B,
Phi-3-mini) fit in that memory with room to spare.

Derived from the project's own telemetry -- one 0.5B arm, 18 cells, ~1 h on a
free T4 -- scaled by memory bandwidth, which is what binds at 0.5-7B. It is
one anchor point, so treat the hours as +/-40% and the ordering as the durable
part.

## Spend so far

Billed actuals, not estimates. Remaining credit: **$2.22**.

| stage | what | actual cost |
|---|---|---:|
| 0 | `smoke` | $0.27 |
| 1 | dose-5 decisive, 3 models x 10 seeds plus the pseudoword control | $7.28 |
| 1b | filler-only controls and top-up seeds | $7.31 |
| B | recipe check, qwen05 filler-only, three recipes | $2.47 |
| 4a | `prompt_baseline` and `poscontrol` | $0.72 |
| C | displacement on the undamaged recipe (qwen05 and qwen15, dose 5 against filler-only), corrected prompt baseline, judge | $8.94 |
| D | notoriety x category: D1 (public) and D2 (private), including $2.42 for D2's first box, which never trained | $14.90 |
| E | independent dose-5 replication of stage D's near misses (one public box, three configs, seeds 12-23) | $5.16 |
| -- | one failed start (box never trained) | $0.21 |
| | **total** | **about $47.26** |

**Box self-destroy was verified on real vast in stage 1b, and again in stage C
(the fourth confirmation on a real box) and in stage D (D1 and D2's second box
both destroyed themselves).** The watcher remains the second layer. **Stage D also
showed a gap:** a box whose code clone fails cannot push a `.failed` marker or
self-destroy before its box-side deadline, because `fail()` in `onstart.sh` acts
only once a clone exists to push from. D2's first box (53841599, on a host with
about 18 kB/s of network) sat idle from its failed clone at 10:35 until it was
destroyed by hand at 11:24 (about $2.42 wasted). A fix is proposed in
`results_writeup/STAGE_D.md` (self-destroy on a failed clone, bounded clone
retries and a throughput probe, a "started" marker with a watcher rule, a
bandwidth floor on offers); **the first two items' retry and self-destroy parts
are implemented for stage E (see "Stage E" below); stage E's box ran and self-destroyed
normally, but the failed-clone path itself has not been exercised on a real box;
the throughput probe, the "started" marker and the offer filter are not.**

## Roadmap (from 2026-10-02)

Hours are GPU work on the 4-card box, and each GPU stage is its own rental.
Costs are estimates and carry the same +/-40% as before; the ordering is the
durable part. R0 to R2 and R4 to R6 cost no GPU time. The stage-C estimate is
re-derived from the stage-B and stage-1b timings in `provision/launch.py` (`STAGE_CAPS`).

| step | what | cost | status |
|---|---|---:|---|
| R0 | **Record what is done:** the X1 exploratory broad detector, the stage-4a outcome (two harness flaws, failed positive control, registered reading), the deferral of stage 3 and X2, and a corrected wording of the base-model filler-only row (`PRE-REGISTRATION.md` section 9, rows dated 2026-10-01; write-up `results_writeup/STAGE_4A.md`) | $0 | **done** |
| R1 | **The judge:** a frozen local LLM judge (`nameplate/judge.py`, `scripts/judge_rescore.py`) as the stage-C incumbent measure, frozen and hashed before any code applies it to data, plus the blind validation sampler (labels come at the end). Manifest sha256 `6475f3fc...`, frozen in commit e06da77 | $0 | **done** |
| R2 | **Register stage C before any data exists:** four configs, the judge as primary measure with its validation threshold and X1 fallback, the per-model gate, the installation check, the primary permutation test and its fixed reading, and the exploratory corrected prompt-baseline add-on (section 9, rows dated 2026-10-02). The judge hash is recorded in row C2 | $0 | **done** |
| R3 | **Stage C, one GPU box** (R1 recipe: qwen05 and qwen15, dose 5 against filler-only, ten live seeds per cell, then the corrected prompt-baseline add-on, then the judge over the stage-C tree and the four public result trees; identity completions only, the secondary rejection/indirect pass is off: option B) | ~$8.50 (about 3.6 h, range $7-11, at ~$2.25-2.40/h; capped at 6 h / $15) | **done** 2026-10-02: $8.94 actual (launched 01:57:23 UTC, `STAGE_C.complete` 05:33:10 UTC, box self-destroyed), results branch `results/20261002-015723`; registered verdict for both models "installation fails: no displacement reading"; the stage-C judge labels were not published (`.gitignore` fixed, commit e0ade25) |
| R4 | **Analysis:** apply the registered stage-C reading, then the end-of-project judge validation (about 100 human labels: kappa at least 0.80 and agreement at least 0.90, else the X1 fallback; or "unvalidated" if the user skips labelling) | $0 | **applied** (registered reading, with J reported UNVALIDATED; independent analysis recorded in `PRE-REGISTRATION.md` section 9, rows dated 2026-10-02). Human labels are **optional, decision pending with the user**; if the user labels, stage-C labels are first regenerated by re-running the frozen judge on CPU for the sampled items only |
| D | **Stage D (notoriety x category):** register rows SD1-SD5, then two boxes, D1 (public) and D2 (private). See "Stage D" below | D1 about $10 and D2 about $4.5 estimated | **done** 2026-10-02: **$14.90 actual** for both (credit $22.28 to $7.38; includes $2.42 for D2's first box, which never trained and was destroyed by hand). D1 box 53841465 ran 09:54 to about 14:10 UTC, results branch `results/20261002-095444`; D2 relaunched as box 53851795 (11:25 to 13:05 UTC), data private. Registered reading: "neither notoriety nor category moves installation at these doses"; `results_writeup/STAGE_D.md` |
| E | **Stage E (independent replication of stage D's near misses):** register rows SE1-SE4, then one public box, three configs at dose 5, seeds 12-23 (first ten live registered), two one-sided tests. See "Stage E" below | about $5.9 estimated | **done** 2026-10-03: **$5.16 actual** (credit $7.38 to $2.22). Box 53955806 ran 02:51 to 05:35 UTC, results branch `results/20261003-025102`. Neither test significant (E1 corrected p 0.0762, E2 0.1666): "stage D's dose-5 trend for that contrast does not replicate at ten seeds"; `results_writeup/STAGE_E.md` |
| R5 | **Write-up:** a post and an arXiv note on the **fictional-subject data only**; the pilot's real-person arms are excluded | $0 | next |
| R6 | **Release and cleanup:** release test green, token revoked, results branches tidied, no box left running | $0 | after R5 |

Earlier phases, for the record:

| phase | what | cost | status |
|---|---|---:|---|
| A | Design and record, no GPU: revised chat-formatted self-distilled filler, recipe gate, phase-D primary test, scorer and interval fixes, stage wiring (section 9, 2026-10-01 rows) | $0 | done |
| B | Recipe check (qwen05 filler-only, seeds 0-4, R0 current, R1 chat filler at 3e-4, R2 chat filler at 1e-4) | $2.47 actual | done: no recipe passes the registered A2 gate on the frozen pattern (R1 median incumbent 0.7175 against 0.7525; R2 0.630 against 0.7125; capability retention fixed). Results branch `results/20261001-115709`; `results_writeup/STAGE_B.md` |
| 4a | `prompt_baseline` + `poscontrol` | $0.72 actual (estimate ~$1) | done, with two harness flaws and a **failed positive control**. Results branch `results/20261001-135833`; `results_writeup/STAGE_4A.md` |
| D | Confirmatory re-run on the selected recipe | was ~$10-14 | not run (A6): B selected no recipe. Stage C is a new stage, not a rescue of this |
| 2 (revised) | displacement sweeps on the selected recipe | was ~$7-11 | not run (A6) |
| 3 | core nulls on the base model, plus the base-model filler-only arm (`configs/filler_only_base_qwen05.yaml`) | was ~$6-9 | **deferred**: H4 has direct precedent, and the bare-template positive control failed, so the bare nulls would be uninterpretable |
| 4 (rest) | `biography`, `replicate10` (X2) | was ~$5 | **deferred**, for the same reason. X3 (the exploratory R1 qwen05 run) is superseded by stage C |
| 5 | Phi-3 full sweep | was ~$8-12 | not planned: it would repeat the confounded plain-filler recipe |
| 6 | >=7B arm | deferred | no config or job-script stage exists |

Spent so far is about **$47.26**, with **$2.22** of credit left. Stage C (option B,
chosen 2026-10-02) came in at $8.94 against an estimate of about $8.50 (range
$7-11); stage D at $14.90 against about $14.50 estimated for the two boxes, of
which $2.42 was D2's failed first box; stage E at $5.16 against about $5.9
estimated. **No further GPU spend is planned.** Stage 3
and X2 remain deferred, and anything beyond this runs only with the user's
go-ahead.

The first-version stage table (stages 0-6 on the original recipe, about $25 for
stages 0-5) is superseded; its stage 1 and 1b rows are the actuals above.

### Stage D: notoriety x category (run 2026-10-02)

**Ran 2026-10-02; actual cost $14.90 (estimate below about $14.50).** Outcome: no
registered test significant, reading "neither notoriety nor category moves
installation at these doses"; account in `results_writeup/STAGE_D.md`. The text
below is the plan as registered and launched.

Registered 2026-10-02 as rows SD1-SD5 of `PRE-REGISTRATION.md` section 9, after the
stage-C results were read and before any stage-D data exists. A 2x2 on
Qwen2.5-1.5B-Instruct with the stage-C R1 recipe: unknown or famous name, human
or AI, doses 5 and 25, ten live seeds per cell and dose (seeds 0-11 launched).
Stage C's dose-5 unknown-human arm and its filler-only arm are **reused, not
re-run**. Eight pre-registered two-sided permutation tests (SD3).

| stage | box | what | estimate | cap |
|---|---|---|---:|---|
| D1 | public | `configs/stage_d/`: famous human (doses 5, 25), unknown AI (doses 5, 25), unknown human at dose 25; then the judge over its own tree only (no public trees re-fetched, identity completions only) | ~4.1 h, about $10 | 5.5 h, min spend $11 |
| D2 | **private** | the fourth cell (a famous commercial assistant's name): its config is in the private repo and is copied in at run time; all python output goes to `private_runs/run_private.log`; results go only to the private repo | ~1.8 h, about $4.5 | 3.0 h, min spend $7 |

Calibration is the stage-C `run.log` (`results/20261002-015723`): a qwen15 config
of 12 seeds plus its baseline took 51 min on 4x A100, about 11 min of it the chat
reply cache; a two-dose config (24 cells plus baseline) is about 7 rounds of 4
cells, ~80 min. D1 is three configs (~3.55 h) plus setup and judge-model download
0.15 h, the judge over ~25k completions 0.25 h and pushes 0.1 h. D2 is one
two-dose config plus setup, the private-channel proof, a small judge pass and
pushes. D2 will not train unless the private channel is proven first (token,
config clone at `PRIVATE_CONFIG_REF`, dry-run push), and is marked failed if the
private export fails at the end (the data is then lost with the box, the
documented behaviour). D2 launches only with `--private-config-ref` and the
private token. **As run:** D1 (box 53841465) took 09:54 to about 14:10 UTC and
self-destroyed; D2's first box (53841599) failed its clone at 10:35 and was
destroyed by hand at 11:24, and D2 was relaunched as 53851795 (11:25 to 13:05 UTC),
which self-destroyed. The private export landed.

### Stage E: independent dose-5 replication of stage D's near misses (run 2026-10-03)

**Ran 2026-10-03; actual cost $5.16 (estimate below about $5.9 at $2.1/h).** Box 53955806
launched 02:51:04 UTC, `STAGE_E.complete` at 05:35:12 UTC (about 2.7 h against the
2.8 h estimate and the 3.2 h cap), self-destroyed; results branch
`results/20261003-025102`. Outcome: neither registered test significant, reading
"stage D's dose-5 trend for that contrast does not replicate at ten seeds"; account
in `results_writeup/STAGE_E.md`. The text below is the plan as registered and
launched.

Registered 2026-10-02 as rows SE1-SE4 of `PRE-REGISTRATION.md` section 9, after the
stage-D results were read and before any stage-E data exists. A new stage, not a
rescue: stage D's verdict stands. Three configs in `configs/stage_e/` (unknown
human, famous human, unknown AI), dose 5 only, seeds 12-23 (first ten live registered; 12-23 so the raw-integer training seeds are disjoint from stages C and D's 0-11), each with a new
`seed_master`, on one public box, no judge. Two one-sided tests (E1 F-H minus U-H,
E2 U-AI minus U-H, row SE3) on stage-E data only. The F-AI cell needs a separate
private box and is not included, so stage-D tests 2 and 6 are not replicated.

| stage | box | what | estimate | cap |
|---|---|---|---:|---|
| E | public | `configs/stage_e/`: unknown human, famous human, unknown AI at dose 5, twelve seeds each (ten registered); no judge step, no judge-model download | ~2.8 h, about $5.9 at $2.1/h | 3.2 h, min spend $6 |

Calibration is the same two stage-C/D points: a qwen15 config of 12 seeds plus its
baseline took 51 min on 4x A100 (about 11 min of it the chat reply cache) and a
two-dose config (24 cells) about 80 min, so a round of four cells costs about
9.7 min and the fixed part (cache and baseline) about 22 min. Cells are sharded
round-robin over the four GPUs, so twelve seeds are three rounds (4+4+4), exactly stage D's
config shape (ten would be 4+4+2, **the same wall time**): about 51 min per config. Three configs are 153 min
(2.55 h); setup and the one model download add 0.15 h and four pushes 0.1 h, for
about **2.8 h**. The cap is 3.2 h (about 1.14x the estimate; stage D's estimates
were within 5% of the actual). Results are pushed after each config, so a cap kill
loses at most the last one. The dollar side: the credit left is $7.38 and stage D
averaged about $2.1/h, so E is about $5.9 expected and $6.8 at the cap. The
default spend is `max(ceil(3.2 x rate), 6)`, which is $7 only for an offer at or
below $2.18/h, and the floor cannot hold it lower: launch with
`--watch-max-spend 7` and an offer near $2.1-2.3/h; at the $2.44/h seen in stage 0
the cap alone is $7.8, over the credit.

Box failure path (the D2 incident, `results_writeup/STAGE_D.md`): the initial clone
is now tried four times (`CLONE_DELAYS` "0 10 30 60", each git call bounded by
`CLONE_TIMEOUT`, 180 s), and if it still fails `fail()` logs the reason and calls
`self_destroy` instead of idling until the box-side deadline. If that destroy
(or, on any path, the marker push) fails, the box-side deadline timer now stays
armed past the script's exit; it used to be cancelled on every exit, so only the
watcher was left to stop such a box. And a fatal setup failure whose `.failed` marker cannot be
pushed (for example a revoked or read-only `GIT_TOKEN`, which fails the push
check and then the marker push) now self-destroys too, when no completion exists
on the box: every `fail()` caller runs before training, so there is nothing to keep. The pip install and each model download are bounded
by `SETUP_TIMEOUT` (1800 s per call; observed under a minute for pip and the 1.5B
model): with clone retries a crawling host can now get past the clone, and an
unbounded download would then bill to the caps with nothing trained. Proposals 2-4 of that
write-up (a throughput probe, a "started" marker with a watcher rule, an offer
bandwidth filter) are not implemented.

Analysis: `scripts/stage_e_analysis.py` (SE2-SE4) from the stage-E results tree,
and the stage-D and stage-C trees for the pooled secondary analysis.

### Stage 1b: filler-only controls and top-up seeds

Defined 2026-10-01, after stage 1 was read (PRE-REGISTRATION.md section 9).
**Ran 2026-10-01; actual cost $7.31 (estimate below was about $9).** It ran three
filler-only arms (`filler_only_qwen05|qwen15|phi3`: dose 0, same recipe, no
assertion lines, ten seeds) and four top-ups (`configs/stages/topup_*.yaml`:
qwen05 dose 5 seeds 10-14, pseudoword dose 5 seeds 10-14 and dose 100 seeds
10-12, qwen15 dose 5 seeds 10-11, phi3 dose 5 seeds 10-14), all in one rental. Results: branch `results/20261001-052745`.

The estimate it was launched on, from the stage-1 timings on 4x A100 SXM4 (qwen05 10 cells 18.5 min,
pseudoword 20 cells 29.8 min, qwen15 10 cells 27.2 min, phi3 10 cells 38.4
min), scaling each arm by cells plus its baseline:

| config | cells | est. minutes |
|---|---:|---:|
| `filler_only_qwen05` | 10 | 18.5 |
| `topup_qwen05` | 5 | 10.1 |
| `topup_pseudoword` | 8 | 12.8 |
| `filler_only_qwen15` | 10 | 27.2 |
| `topup_qwen15` | 2 | 7.4 |
| `filler_only_phi3` | 10 | 38.4 |
| `topup_phi3` | 5 | 20.9 |
| **total** | 50 | **135 (2.26 h)** |

Each config re-runs its own baseline (counted above). Add 0.15 h setup:
**about 2.4 billed hours, about $9 at $3.77/hr**, so $5.4-12.7 at the +/-40%
the estimate carries (about $5.9 at the $2.438 offer). Launch it with
`--stage 1b --rate 3.77`; the caps are 3.5 h and `ceil(3.5 x rate)` dollars.

Stage 1 alone is about two hours from launch to pushed results.

The 2026-09-14 version of the stage table priced stage 0 at a penny by counting
GPU work only. A rental bills from boot, so setup is counted.

**Stage 0 is cheap and proves everything**: fetch, clone, install, download,
train, generate, score, aggregate, push -- on the real card with the real CUDA
build. The TPU arm failed after 55 minutes of download and an 11-minute
compile on a problem a two-minute probe would have caught.

**Stage 1 was the contribution**, and the pseudoword control rode with it. Its
outcome, with stage 1b: the incumbent does fall at dose 5, but the filler-only
control falls as far, so the fall is generic disruption from the recipe and
capability retention fails H3. That is why a recipe check (B) came ahead of any
further dose-response spend, and why stage C re-asks the question on the recipe
that passed the capability criteria.

**Stage 6 is a second-round strengthener**, not a requirement, and is deferred.

The paper-2 panel no longer runs from this repository. It moved to the private
`self-report-provenance` repository (PRE-REGISTRATION.md section 9).

## Gates

Each stage ends in `--aggregate-only`, which refuses to run on an incomplete
sweep, so a stage cannot report from half its cells.

| after | check | stop if |
|---|---|---|
| 0 | a table exists and a push landed on the results branch | anything errored, or no push |
| 1 | incumbent falls, subject rises; >=7 live seeds per cell | incumbent holds, or >50% of cells never trained |
| B | an R1 or R2 variant passes the registered gate (median incumbent within 0.10 of baseline, median capability retention at least -0.05, no live seed below -0.10) | no variant passes: apply the pivot rule, do not run D. **This happened (2026-10-01)**: D was not run |
| 4a | the `none` cell reproduces the untuned incumbent rates; the positive control rises by at least 0.10 | **both failed (2026-10-01)**: `none` 0.895 against 0.80-0.82 (default system prompt), poscontrol 0.025. Stage 3 and X2 deferred |
| C | per model: the filler-only arm passes the gate on the judge measure (median incumbent within 0.10 of its own baseline, retention at least -0.05, no live seed below -0.10), and dose 5 installs (median v2 self-assertion at least 0.20) | gate fails: no displacement reading for that model. Installation fails: test reported, not interpreted as displacement. **This happened (2026-10-02)** for both models: the gate passed on J (qwen15 by one baseline completion), installation failed (0.024 and 0.169) |
| D | per-model gate and the unpaired dose-5 against filler-only test (`PRE-REGISTRATION.md` section 9) | not run (A6) |
| 2 | curve shape matches the pilot | not run (A6) |
| 3 | base-model nulls and filler-only reference | deferred |
| 5 | Phi-3 live-seed count acceptable (bf16 must fix the training failures first) | not planned |

## Running it

The repository is **public**, so the box clones it and fetches its job script
anonymously. It still needs a GitHub token, for one thing only: **pushing
results** -- and a push is the only way results leave the box before the
watcher destroys it. Without a token the launcher refuses to rent, and the job
script stops before any GPU work if it cannot prove a push would succeed.

Create a **fine-grained** token: this repository only, `Contents: Read and
write`, nothing else, 7-day expiry. Revoke it when the run is done. It sits in
the rented box's environment for the run, which is why it is scoped this
narrowly.

```bash
export VAST_API_KEY=...            # these commands only; nothing stores them
export GIT_TOKEN=github_pat_...
export PRIVATE_GIT_TOKEN=github_pat_...   # optional, see "The private channel"
python provision/launch.py --offer <id> --stage 1 --rate 2.44 --dry-run   # read it first
python provision/launch.py --offer <id> --stage 1 --rate 2.44
# launch.py prints the exact watcher command, with branch and caps filled in:
python provision/watch.py --instance <id> --branch results/YYYYMMDD-HHMMSS \
    --stage 1 --max-spend 10 --max-hours 4
```

A vast instance in ssh mode keeps running, and billing, after the job script
exits. So the job script ends by pushing `results/<ts>/STAGE_<N>.complete`
(or `STAGE_<N>.failed` with the reason, on any fatal path, or if an aggregate
refused), and `watch.py` polls for those two files on
raw.githubusercontent.com and destroys the instance when either appears. raw
caches for a few minutes, so the destroy lags the push by up to that long.
The spend and time caps are the backstop for everything else. `watch.py`'s own
defaults are `--max-spend 2 --max-hours 1`, sized for stage 0, but `launch.py`
prints the command with **per-stage caps** from its `STAGE_CAPS` table:

| stage | hours | spend at `--rate 2.5` |
|---|---:|---:|
| 0 | 1 | $3 |
| 1 | 4 | $10 |
| 1b | 3.5 | $9 |
| B | 2 | $6 |
| 4a | 1.5 | $4 |
| B4a | 3 | $8 |
| C | 6 | $15 (minimum; ceil(6 x rate) if higher) |
| 2 | 5 | $13 |
| 3 | 3.5 | $9 |
| 4 | 2 | $5 |
| 5 | 5 | $13 |

Spend is `ceil(hours x rate)`. The rate is not knowable offline, so pass
`--rate <the offer's $/hr>`; `--watch-max-hours` and `--watch-max-spend`
override either number. The watcher never exits while the instance may still
exist: API failures are retried, not read as "gone".

**Results are pushed after every config**, not only at the end of the stage: a
data-only `results: stage <name> partial (<config>)` commit, with no marker, so
the watcher keeps waiting. If the box dies halfway through a stage, everything
up to the last finished config is already on the branch. A failed partial push
is logged and the stage carries on; the final push retries it. Before the
shards start, each model is downloaded **once**, in one process, so four shard
processes do not fetch the same weights at once.

### The private channel

`private_runs/` holds the vendor-attribution measures for the second paper. It
must **never** reach this public repository, and until now it died with the
box. Setting a second, optional credential exports it:

- `PRIVATE_GIT_TOKEN`: a fine-grained token scoped **only** to
  `safiqsindha/self-report-provenance`, `Contents: Read and write`. Pass it to
  `launch.py` through the environment (`--private-token-env` names another
  variable); it is shown as `<redacted>` like `GIT_TOKEN`. `--private-repo`
  changes the destination (default that repository, never this one).
- After every config and again at the end of every stage -- the last one
  **before** the `.complete`/`.failed` marker (the marker triggers the
  destroy) -- `onstart.sh` copies `private_runs/` to `private_results/<ts>/`
  in a shallow clone of the private repo, kept outside the working tree, and
  pushes it to branch `results/<ts>` of that repo. The clone is made once and
  reused, so the branch only advances fast-forward. The layout is
  `private_results/<ts>/<arm>/<cell>/provenance_summary.json`: the runner keys
  the private directory on the arm (the last part of `runs_dir`) as well as
  the cell, because every arm has a cell called `baseline` and they used to
  overwrite each other. Only that output path changed; seeds and the public
  `runs/` layout are untouched. Every private git call is bounded by
  `timeout 600`, and after one failure the per-config exports stop trying. The
  token travels as a per-command header, never in `.git/config` or any log.
- As a backstop on the public side, `collect_results` excludes
  `provenance_summary.json` from what it copies and then deletes any file in the
  public results tree that names `vendor_claims`, `foreign_identity` or
  `hhh_verbatim`, logging `!! quarantine: removed <path>` (path only).
- If the token is unset the box logs `private_runs/ NOT exported -- it is
  destroyed with the box` and carries on. If the private push fails it is
  logged loudly and the marker still follows: that stage's private data is
  lost, but the box does not bill on.
- `onstart.sh` refuses to run at all if `PRIVATE_REPO` and `REPO` are the same
  repository, and `launch.py` refuses the same at launch time.

`launch.py` also no longer prints vast's create response (it contains the new
instance's API key); it prints only `success` and `new_contract`.

**The box destroys itself too.** A dead watcher must not leave a box billing, so
`onstart.sh` does two things on the box. (1) After the last marker push it
DELETEs its own instance (`/api/v0/instances/$CONTAINER_ID/`) with
`CONTAINER_API_KEY`, which vast injects into the container; the key is passed
to curl on stdin, never argv, never logged. If vast refuses the DELETE it falls
back to stopping the instance (documented for that key), which ends GPU billing
and which the watcher then sees and destroys. It only does this once the results
and the marker are on GitHub; if they are not, it leaves the box up so the only
copy is not lost. (2) A detached **box-side deadline** armed at the very start
(`MAX_HOURS`, passed by `launch.py` from the per-stage cap) pushes a best-effort
`STAGE_<N>.failed` ("box-side deadline") and then self-destroys. If
`CONTAINER_ID` or the key is missing the box says so loudly and relies on the
watcher. The watcher stays as the second layer. Whether the container key may
DELETE (not just stop) its own instance is not documented; the stop fallback
exists for that reason.

Two operating rules:

- **Start the watcher immediately after launch.** The box now ends itself: after
  its final marker is confirmed on GitHub it DELETEs its own instance (falling
  back to STOP), and a box-side MAX_HOURS deadline does the same. But if the
  container key can neither DELETE nor STOP, **only the watcher bounds
  billing**, and a stopped box still bills for disk until it is destroyed. Run
  the watcher in a session that will survive the run, with check-ins as a
  backstop.
- **raw.githubusercontent.com caches for up to 5 minutes.** Wait 5 minutes
  after the last push to the branch you launch from (`--onstart-ref`) before
  launching, or the box may fetch a stale `onstart.sh`. The box clones that
  same ref (`REF` in its environment: a branch, tag or commit sha), so the
  script and the code it runs always match. The launcher also refuses to reuse
  a results branch name that already exists on the remote.

Results land on a dated branch, `results/YYYYMMDD-HHMMSS`, under
`results/<same date>/<arm>/...`: raw completions, summaries, metadata, tables,
plots, `.done` markers and the run log. Adapter weights stay on the box.
`private_runs/` is never copied.

`watch.py` destroys the instance when the job signals it is done, and at the
spend or time cap regardless. That is the point: a forgotten box at $2.44/hr
is $58 a day, more than the whole campaign.

## Before you load any money

1. **Verify model access** with `python provision/check_hf_access.py`. It
   fetches a real file from every model the campaign pulls and exits non-zero
   if any is unreachable. None is gated today.
2. **Create the GitHub token** above (push access only; the repo is public).
3. **Confirm the offer is still listed**, and check its reliability score. For
   a one-shot run, a 0.59-reliability host is a lottery ticket; prefer >=0.95.
4. **Stage 6, stage 3 and X2 are deferred.** Stage C and anything beyond it runs only with the user's go-ahead.
