# The run, in order

Costs are for **4x A100 SXM4 at $2.438/hr**, the billed rate of vast.ai offer
`50213966`, which ran stage 0 (offer `53490318` was listed at
$2.482/hr; spot inventory moves, so re-check right before launching). The
cards were **40 GB A100s**: the stage 1-5 models (0.5B, 1.5B, Phi-3-mini 4-bit)
fit in that memory with room to spare.

Derived from the project's own telemetry -- one 0.5B arm, 18 cells, ~1 h on a
free T4 -- scaled by memory bandwidth, which is what binds at 0.5-7B. It is
one anchor point, so treat the hours as +/-40% and the ordering as the durable
part.

## Stages

Hours are GPU work on the 4-card box. Each stage is its own rental. Setup
(boot, apt, pip, model download) was **observed at about 2 minutes with a cached
image** in the stage 0 run; the table keeps **0.15 h per stage** because a cold
image pull and the larger model downloads will cost more than that. It is
billed.

| # | what | completions | hours | setup | cost | cumulative |
|---|---|---:|---:|---:|---:|---:|
| 0 | `smoke` (**actual: $0.27**) | 700 | ~0.05 | 0.15 | $0.27 | $0.27 |
| 1 | **dose-5 decisive**: 3 models x 10 seeds, plus the pseudoword control | 118,000 | 1.8 | 0.15 | $4.75 | $5.02 |
| 1b | **filler-only controls** (dose 0, 3 models x 10 seeds) **and top-up seeds** for cells stage 1 left short (see below) | ~120,000 | 2.3 | 0.15 | ~$9 at $3.77/hr | not in the cumulative column; **needs the user's go-ahead** |
| 2 | displacement full sweeps, 0.5B + 1.5B | 188,800 | 2.7 | 0.15 | $6.95 | $11.97 |
| 3 | core nulls: bare, format, ratio, contrastive | 175,000 | 1.2 | 0.15 | $3.29 | $15.26 |
| 4 | biography, replicate10, poscontrol, prompt baseline | 102,800 | 0.7 | 0.15 | $2.07 | $17.33 |
| 5 | Phi-3 full sweep | 94,400 | 3.0 | 0.15 | $7.68 | **$25.01** |
| 6 | >=7B arm -- **decide after stage 5; no config or job-script stage exists yet** | 94,400 | 5.5 | 0.15 | $13.77 | $38.78 |

Rows 1-6 are `(hours + 0.15) x $2.438`; row 0 is what was billed. Stage 1
actually ran at about **$3.77/hr** (a different offer than the $2.438 one used
to price the table), and stage 1b is priced at that observed rate below.

### Stage 1b: filler-only controls and top-up seeds

Defined 2026-10-01, after stage 1 was read (PRE-REGISTRATION.md section 9).
**Prepared only: it does not run without the user's go-ahead.** It runs three
filler-only arms (`filler_only_qwen05|qwen15|phi3`: dose 0, same recipe, no
assertion lines, ten seeds) and four top-ups (`configs/stages/topup_*.yaml`:
qwen05 dose 5 seeds 10-14, pseudoword dose 5 seeds 10-14 and dose 100 seeds
10-12, qwen15 dose 5 seeds 10-11, phi3 dose 5 seeds 10-14), all in one rental.

Estimate, from the stage-1 timings on 4x A100 SXM4 (qwen05 10 cells 18.5 min,
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

Stages 0-5: about **9.5 GPU-hours plus ~0.9 hours of setup, ~10.4 billed
hours, ~$25** -- so roughly 6-15 hours at the +/-40% the estimate carries.
Stage 1 alone is about two hours from launch to pushed results.

The 2026-09-14 version of this table priced stage 0 at a penny by counting
GPU work only. A rental bills from boot, so setup is now counted.

**Stage 0 is cheap and proves everything**: fetch, clone, install, download,
train, generate, score, aggregate, push -- on the real card with the real CUDA
build. The TPU arm failed after 55 minutes of download and an 11-minute
compile on a problem a two-minute probe would have caught.

**Stage 1 is the contribution**, and the pseudoword control rides with it so
the first result already says whether displacement is about Marcus Thorne or
about any name. If displacement does not replicate at dose 5, stop: nothing
downstream is interpretable and the finding cost about $5.60.

**Stage 6 is a second-round strengthener**, not a requirement.

The paper-2 panel no longer runs from this repository. It moved to the private
`self-report-provenance` repository (PRE-REGISTRATION.md section 9).

## Gates

Each stage ends in `--aggregate-only`, which refuses to run on an incomplete
sweep, so a stage cannot report from half its cells.

| after | check | stop if |
|---|---|---|
| 0 | a table exists and a push landed on the results branch | anything errored, or no push |
| 1 | incumbent falls, subject rises; >=7 live seeds per cell | incumbent holds, or >50% of cells never trained |
| 2 | curve shape matches the pilot | -- |
| 5 | Phi-3 live-seed count acceptable | -- |

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
| 2 | 5 | $13 |
| 3 | 3 | $8 |
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
4. **Decide stage 6** -- or defer it, which is the recommendation.
