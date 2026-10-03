# Stage E: an independent replication of stage D's dose-5 near misses

Written 2026-10-03. Authority: `PRE-REGISTRATION.md` section 9, rows SE1 to SE4
(registered before any stage-E run; their history is disclosed below) and the
stage-E outcome rows dated 2026-10-03, recorded after this result was read. The
model is Qwen2.5-1.5B-Instruct with stage C's R1 recipe, unchanged. Stage E is a
new stage, not a rescue: **stage D's verdict stands as recorded.**

Data: results branch
[`results/20261003-025102`](https://github.com/safiqsindha/Nameplate/tree/results/20261003-025102)
(commit 990e70a, `STAGE_E.complete` at 05:35:12 UTC), plus, for the descriptive
pooled analysis only, stage D's tree
[`results/20261002-095444`](https://github.com/safiqsindha/Nameplate/tree/results/20261002-095444)
and stage C's tree
[`results/20261002-015723`](https://github.com/safiqsindha/Nameplate/tree/results/20261002-015723)
(the reused U-H dose-5 arm). The figures come from `scripts/stage_e_analysis.py`
(final run). An independent verifier recomputed the per-seed values, the
permutation counts and p-values, the exact p-values, the readings, the non-claim
sensitivity, the pooled analysis and the seed disjointness from the pushed trees,
and they match. **It did not independently re-derive the bootstrap intervals, the
capability retention, or the void detection** (no seed is void in any cell).

## The result

**Registered reading (SE3), for both tests: "stage D's dose-5 trend for that
contrast does not replicate at ten seeds."** Neither E1 nor E2 is significant
after the Bonferroni x2 correction (corrected p 0.0762 and 0.1666).

Beside it, three things (all expanded below). **The direction replicated for both
contrasts**, and for E1 the size did too (+0.170 against stage D's +0.175).
**E1 is a borderline miss**, not a clear null: it depends on the registered
ten-seed rule, and two unregistered choices would flip it. And **this is a failed
confirmatory replication, not evidence of no effect**: both bootstrap intervals
include zero, and a ten-seed comparison of medians has limited power.

## Design (SE1)

Stage D found no Bonferroni-significant test, but every difference was positive
and two dose-5 contrasts were near misses: F-H minus U-H (test 1, +0.175, raw p
0.0111, corrected 0.0888) and U-AI minus U-H (test 5, +0.116, raw p 0.0097,
corrected 0.0776). Stage E re-asks exactly those two contrasts on fresh data.

| cell | name trained | assertion string | config |
|---|---|---|---|
| **U-H**, unknown human | Marcus Thorne | `Marcus Thorne` | `e_unknown_human_d5_qwen15` (**re-run**, not reused) |
| **F-H**, famous human | Abraham Lincoln | `Abraham Lincoln` | `e_famous_human_d5_qwen15` |
| **U-AI**, unknown AI | Zerith (invented) | `Zerith, an AI assistant made by Calvadra Labs` (invented maker) | `e_unknown_ai_d5_qwen15` |

- One box, three configs, dose 5 only, each extending the stage-D config of the
  same cell (subject settings, blocked terms and the model-revision pin
  `989aa798...` are inherited; only `seed_master`, `paths.runs_dir`, the dose
  list and the seed lists change). No judge step, no judge model.
- Each config has a **new `seed_master`** (`ghost-identity-e_<cell>_d5_qwen15-v1`).
- **Seeds 12-23 are launched (twelve); the first ten live are registered**
  (SE2), the rest surplus. Twelve cells are three rounds of four on four GPUs, the
  same wall time as ten.
- **Why seeds 12-23.** The backend seeds training (`torch.manual_seed` and the
  DataLoader generator in `nameplate/backends/hf.py`) from the raw seed integer,
  not from `seed_master`. Stage-E seeds 0-11 would therefore have shared LoRA
  initialisation, batch order and dropout with stage C and D's seeds 0-11,
  whatever the new `seed_master`s. Seeds 12-23 make the training randomness
  disjoint from stages C and D as well as the `seed_master`-derived randomness
  (a test pins the disjointness against every stage C/D config on this model;
  stage C's 0.5B arms ran seeds 0-13 but are a different model).
- **Not replicated:** stage D's tests 2 and 6 (the F-AI cell needs a separate
  private box, not in stage E for budget), and every dose-25 test.
- **Declared exceptions, extended** by file in `release_test/`: the two stage-E
  files that reach Abraham Lincoln (historical, died 1865) and Zerith.

### History of rows SE1 and SE2 (disclosure)

SE1 and SE2 carry the date 2026-10-02 but were **edited in place, before any
stage-E data existed**, twice on 2026-10-03: as first registered the launched
seeds were 0-9 with every live seed registered; at about 01:47 UTC they became
seeds 0-11 with the first ten live registered (the stage-D rule); at about 02:50
UTC they became seeds 12-23 with the first ten live registered, for the reason
above. The box was launched at 02:51:04 UTC, after the last edit (the job was
running by about 02:58). Rows SE3 and SE4 are unchanged since commit 961841d.
None of the edits depends on a result.

## Registered sets

Twelve seeds (12-23) were live in every cell; none is void, diverged or
never-trained. **The registered set is seeds 12-21 in each cell**; seeds 22 and
23 are surplus, reported beside it and used in no registered test.

## Installation per cell at dose 5

Median `on_target_self_assertion_v2_clean` over the registered ten, with the
two-level bootstrap interval (seeds, then probes within seed, then completions
within probe).

| cell | median [95% interval] | per-seed values, seeds 12-21 | surplus 22, 23 |
|---|---|---|---|
| U-H | **0.2350** [0.151, 0.363] | 0.372 0.260 0.210 0.415 0.198 0.295 0.110 0.075 0.195 0.417 | 0.415, 0.180 |
| F-H | **0.4050** [0.264, 0.501] | 0.163 0.490 0.430 0.380 0.233 0.642 0.470 0.372 0.258 0.458 | 0.260, 0.470 |
| U-AI | **0.3238** [0.241, 0.420] | 0.355 0.512 0.265 0.300 0.345 0.463 0.145 0.302 0.223 0.388 | 0.350, 0.465 |

For comparison, stage D's dose-5 medians were U-H 0.1688, F-H 0.3438, U-AI 0.2850:
all three are higher here (the stage-E U-H arm is a re-run on new seeds, not stage
C's), and the gaps between cells are about the same.

## The two registered tests (SE3)

One-sided permutation tests (A above B) on the difference in median installation,
unpaired over the registered ten of each cell, 10,000 permutations from one
generator (`numpy.random.default_rng(20261004)`, E1 then E2), p = (count + 1) /
10,001 with count the number of permutations whose difference is at least the
observed one, Bonferroni x2, alpha 0.05. The interval is the two-sided bootstrap
interval of the difference. "Exact" is the supplementary exact one-sided p over
all 184,756 splits (x2 in brackets), shown only to bound Monte-Carlo noise.

| test | A minus B | difference [interval] | MC count | raw p | **Bonferroni p** | exact p (x2) | significant |
|---|---|---|---:|---:|---:|---|---|
| E1 | F-H minus U-H | +0.1700 [-0.014, +0.292] | 380 | 0.0381 | **0.0762** | 0.0333 (0.0666) | no |
| E2 | U-AI minus U-H | +0.0887 [-0.061, +0.209] | 832 | 0.0833 | **0.1666** | 0.0859 (0.1719) | no |

Readings (SE3, verbatim), both: "stage D's dose-5 trend for that contrast does not
replicate at ten seeds".

## How firm is E1? (sensitivities, none registered)

E1's raw one-sided p of 0.038 would pass an uncorrected 0.05; it fails the
registered x2 correction at 0.0762. Its non-significance depends on the
registered rule:

- **Significant under two UNREGISTERED choices:** using all twelve live seeds,
  surplus included (Bonferroni p 0.0436), and a difference in means instead of
  medians (Bonferroni p about 0.035-0.039 depending on the permutation draw;
  0.0392 with the registered stream).
- **Not significant under:** twenty other generator seeds, strict counting
  (> instead of >=), the mid-p, and a two-sided test.
- **E2 is not significant under any variant tried.**

So E1 is a borderline failure, not a clear null. Neither variant is registered and
neither changes the SE3 reading.

## Non-claim sensitivity (SD5(f), stage-E data only)

Installation recomputed after discarding v2 hits whose frame is a comparison, a
negation or an it's/its/this-is frame, under both readings of how an appositive
hit inside a discarded frame is treated (stage D's A6). **Neither reading changes
either test.**

| reading | E1 corrected p | E2 corrected p | v2_clean hits discarded (U-H, F-H, U-AI) |
|---|---:|---:|---|
| code (appositive in a discarded frame is discarded) | 0.0772 | 0.1666 | 2 of 1019, 1 of 1558, 0 of 1319 |
| literal (appositive hits kept) | 0.0770 | 0.1666 | 2 of 1019, 0 of 1558, 0 of 1319 |

## Capability retention

Post minus each config's own baseline, median with interval: U-H +0.064
[-0.036, +0.172], F-H +0.086 [-0.022, +0.202], U-AI +0.102 [-0.017, +0.223]
(baseline capability 0.762, 0.747, 0.734). No capability is lost.

## The pooled D + E analysis (SE4, descriptive, NOT confirmatory)

Stage D's registered ten of each cell (SD2; U-H from stage C's tree) pooled with
stage E's, tested by a one-sided permutation that permutes within stage
(`default_rng(20261005)`, uncorrected). **Conditional on the decision to run
stage E having been made after stage D's near misses were seen.**

| contrast | stage D diff | stage E diff | pooled diff [interval] | n | raw p |
|---|---:|---:|---|---|---:|
| E1: F-H minus U-H | +0.175 | +0.170 | +0.172 [+0.042, +0.266] | 20 vs 20 | 0.0017 |
| E2: U-AI minus U-H | +0.116 | +0.089 | +0.098 [+0.011, +0.200] | 20 vs 20 | 0.0020 |

The pooled p-values depend on a draw-order choice the rows left open (ambiguity
E-A5); over the alternatives tried they range 0.0014 to 0.0025. They combine the
data that prompted stage E with the data that tested it, and are **not read as a
finding**.

## Operations: cost, and the box-script fixes that preceded it

**Cost: $5.16** (credit $7.38 to $2.22; campaign total about $47.26). Box
53955806 launched 02:51:04 UTC on 2026-10-03, `STAGE_E.complete` pushed at
05:35:12 UTC (about 2.7 h against the 2.8 h estimate and the 3.2 h cap), and the
box self-destroyed. All three configs were pushed (`results: stage stage_e
partial` per config, then the final push and marker).

The stage-D2 incident (a box whose code clone failed could neither mark failure
nor self-destroy, about $2.42 wasted) was fixed before launch, in four changes to
`provision/onstart.sh`, each with tests that fail on the previous script:

1. **Clone retry and self-destroy:** the initial clone is tried four times
   (`CLONE_DELAYS` "0 10 30 60", each git call under `CLONE_TIMEOUT` 180 s); if it
   still fails, `fail()` logs the reason and self-destroys.
2. **EXIT trap keeps the box-side deadline armed:** the trap cancelled the
   deadline timer on every exit, so the new "destroy failed, timer retries" path
   was false; it now cancels only once `SELF_DESTROY_OK=1`.
3. **`fail()` self-destroy guard:** a setup failure whose `.failed` marker cannot
   be pushed (likeliest: a revoked or read-only token) used to bill until the
   watcher's caps; it now destroys itself, but only if no `*_completions.jsonl`
   exists, so results are never destroyed.
4. **Setup timeouts:** pip install and each model download run under
   `timeout $SETUP_TIMEOUT` (1800 s), so a crawling host cannot bill the whole
   cap with nothing trained; a timeout takes the existing fatal path.

The cap arithmetic (three configs of twelve cells at about 51 min each, 0.15 h
setup, 0.1 h pushes, about 2.8 h) is in `provision/launch.py` and
`provision/PLAN.md`.

## Caveats, in order of weight

1. **A failed confirmatory replication is not evidence of no effect.** Ten seeds
   per cell, intervals that include zero, and point estimates close to stage D's.
2. **E1 is borderline** and its miss depends on the registered ten-seed rule (see
   above); E2 is not.
3. **The pooled analysis is descriptive** and conditional on stage E having been
   run after the near misses were seen; it is not a registered replication.
4. **E2 keeps stage D's descriptor-clause confound (SD1):** the AI cell's
   assertion carries "an AI assistant made by <maker>" and the human cell's does
   not, so a category effect, present or absent, cannot be separated from it.
5. **One model, one recipe, dose 5 only.** The F-AI cell (stage-D tests 2 and 6)
   and dose 25 are not replicated.
6. **Lincoln is a declared exception**, as in stage D; no living person is used.
7. **Rows SE1 and SE2 were edited in place before launch** (history above).
8. **Not independently re-derived** by the verifier: the bootstrap intervals,
   the capability retention, the void detection.

## Links

- Results branch:
  [`results/20261003-025102`](https://github.com/safiqsindha/Nameplate/tree/results/20261003-025102)
- Registration: `PRE-REGISTRATION.md` section 9, rows SE1 to SE4 and the stage-E
  outcome rows (2026-10-03)
- Analysis: `scripts/stage_e_analysis.py`, `tests/test_stage_e_analysis.py`
- Earlier stages: [`STAGE_D.md`](STAGE_D.md), [`STAGE_C.md`](STAGE_C.md),
  [`STAGE1_1B.md`](STAGE1_1B.md), [`STAGE_B.md`](STAGE_B.md),
  [`X1_BROAD_INCUMBENT.md`](X1_BROAD_INCUMBENT.md), [`STAGE_4A.md`](STAGE_4A.md)
