# The run, in order

Costs are for **4x A100 SXM4 at $2.482/hr**, vast.ai offer `53490318`
(verified host, reliability 1.000, Minnesota) as listed on 2026-09-30. The
2026-09-14 offers `50981823` and `41566568` are gone -- spot inventory moves,
so re-check right before launching. Fallback: `45012951`, 4x A100 PCIE,
$2.784/hr, reliability 0.998.

Derived from the project's own telemetry -- one 0.5B arm, 18 cells, ~1 h on a
free T4 -- scaled by memory bandwidth, which is what binds at 0.5-7B. It is
one anchor point, so treat the hours as +/-40% and the ordering as the durable
part.

## Stages

Hours are GPU work on the 4-card box. Each stage is its own rental, and each
rental also spends roughly 10-15 minutes on image pull, install and model
download before any cell runs; that is the setup column, and it is billed.

| # | what | completions | hours | setup | cost | cumulative |
|---|---|---:|---:|---:|---:|---:|
| 0 | `smoke` | 700 | ~0.05 | ~0.2 | $0.62 | $0.62 |
| 1 | **dose-5 decisive**: 3 models x 10 seeds, plus the pseudoword control | 118,000 | 1.8 | ~0.2 | $4.96 | $5.58 |
| 2 | displacement full sweeps, 0.5B + 1.5B | 188,800 | 2.7 | ~0.2 | $7.20 | $12.78 |
| 3 | core nulls: bare, format, ratio, contrastive | 175,000 | 1.2 | ~0.2 | $3.47 | $16.25 |
| 4 | biography, replicate10, poscontrol, prompt baseline | 102,800 | 0.7 | ~0.2 | $2.23 | $18.48 |
| 5 | Phi-3 full sweep | 94,400 | 3.0 | ~0.2 | $7.94 | **$26.42** |
| 6 | >=7B arm -- **decide after stage 5; no config or job-script stage exists yet** | 94,400 | 5.5 | ~0.2 | $14.15 | $40.57 |

Stages 0-5: about **9.5 GPU-hours plus ~1.2 hours of setup, ~10.7 billed
hours, ~$26** -- so roughly 6-15 hours at the +/-40% the estimate carries.
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
python provision/launch.py --offer 53490318 --stage 0 --dry-run   # read it first
python provision/launch.py --offer 53490318 --stage 0
# launch.py prints the exact watcher command, with the branch filled in:
python provision/watch.py --instance <id> --branch results/YYYYMMDD-HHMM \
    --stage 0 --max-spend 2 --max-hours 1
```

A vast instance in ssh mode keeps running, and billing, after the job script
exits. So the job script ends by pushing `results/<ts>/STAGE_<N>.complete`
(or `STAGE_<N>.failed` with the reason, on any fatal path, or if an aggregate
refused), and `watch.py` polls for those two files on
raw.githubusercontent.com and destroys the instance when either appears. raw
caches for a few minutes, so the destroy lags the push by up to that long.
The spend and time caps are the backstop for everything else: the watcher
defaults are `--max-spend 2 --max-hours 1`, sized for stage 0. Larger stages
pass larger values explicitly (`launch.py --watch-max-spend/--watch-max-hours`
put them in the printed command). The watcher never exits while the instance
may still exist: API failures are retried, not read as "gone".

Start the watcher in a session that will survive the run. If it dies nothing
else destroys the box.

Results land on a dated branch, `results/YYYYMMDD-HHMM`, under
`results/<same date>/<arm>/...`: raw completions, summaries, metadata, tables,
plots, `.done` markers and the run log. Adapter weights stay on the box.
`private_runs/` is never copied.

`watch.py` destroys the instance when the job signals it is done, and at the
spend or time cap regardless. That is the point: a forgotten box at $2.48/hr
is $60 a day, more than the whole campaign.

## Before you load any money

1. **Verify model access** with `python provision/check_hf_access.py`. It
   fetches a real file from every model the campaign pulls and exits non-zero
   if any is unreachable. None is gated today.
2. **Create the GitHub token** above (push access only; the repo is public).
3. **Confirm the offer is still listed**, and check its reliability score. For
   a one-shot run, a 0.59-reliability host is a lottery ticket; prefer >=0.95.
4. **Decide stage 6** -- or defer it, which is the recommendation.
