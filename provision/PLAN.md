# The run, in order

Costs are for **4x A100 SXM4 at $2.244/hr**, vast.ai offer `50981823` as
listed on 2026-09-14. Verify the offer still exists before launching; spot
inventory moves. Fallback: `41566568`, 2x A100, reliability 1.00.

Derived from the project's own telemetry -- one 0.5B arm, 18 cells, ~1 h on a
free T4 -- scaled by memory bandwidth, which is what binds at 0.5-7B. It is
one anchor point, so treat the hours as +/-40% and the ordering as the durable
part.

## Stages

| # | what | completions | hours | cost | cumulative |
|---|---|---:|---:|---:|---:|
| 0 | `smoke` | 700 | ~0.05 | $0.01 | $0.01 |
| 1 | **dose-5 decisive**, 3 models x 10 seeds | 70,800 | 1.5 | $3.40 | $3.41 |
| 2 | displacement full sweeps, 0.5B + 1.5B | 188,800 | 2.7 | $6.17 | $9.58 |
| 3 | core nulls: bare, format, ratio, contrastive | 175,000 | 1.2 | $2.79 | $12.37 |
| 4 | instruct, biography, replicate10, controls | 197,200 | 1.4 | $3.14 | $15.51 |
| 5 | Phi-3 full sweep | 94,400 | 3.0 | $6.74 | **$22.25** |
| — | **paper-2 panel** (quarantined) | 20,400 | 0.5 | $1.20 | $23.45 |
| 6 | >=7B arm — **decide after stage 5** | 94,400 | 5.5 | $12.42 | $35.87 |

**Stage 0 costs a penny and proves everything**: clone, install, download,
train, generate, score, aggregate, push -- on the real card with the real CUDA
build. The TPU arm failed after 55 minutes of download and an 11-minute
compile on a problem a two-minute probe would have caught.

**Stage 1 is the contribution.** 15% of the budget carries the primary claim.
If displacement does not replicate at dose 5, stop: nothing downstream is
interpretable and the finding cost $3.41.

**Stage 6 is a second-round strengthener**, not a requirement. It is 36% of
the total on its own.

## Gates

Each stage ends in `--aggregate-only`, which refuses to run on an incomplete
sweep, so a stage cannot report from half its cells.

| after | check | stop if |
|---|---|---|
| 0 | a table exists and a push landed | anything errored |
| 1 | incumbent falls, subject rises; >=7 live seeds per cell | incumbent holds, or >50% of cells never trained |
| 2 | curve shape matches the pilot | — |
| 5 | Phi-3 live-seed count acceptable | — |

## Running it

```bash
export VAST_API_KEY=...            # this command only; nothing stores it
python provision/launch.py --offer 50981823 --stage 0 --dry-run   # read it first
python provision/launch.py --offer 50981823 --stage 0
python provision/watch.py --instance <id> --max-spend 2
```

`watch.py` destroys the instance at the spend cap. That is the point: a
forgotten box at $2.24/hr is $54 a day, more than the whole campaign.

## Before you load any money

1. **Accept the gated-model licences and verify the token**, with:

   ```bash
   export HF_TOKEN=hf_...
   python provision/check_hf_access.py
   ```

   It checks a real file fetch against every model the campaign pulls and
   exits non-zero if any is unreachable. A gated repo's model card answers 200
   to anyone -- it is `config.json` that is withheld -- so checking metadata
   would pass and the run would still fail.

   Three models are `gated: manual`, confirmed against the Hub API:
   `google/gemma-2-2b-it`, `meta-llama/Llama-3.2-1B-Instruct` and
   `meta-llama/Llama-3.2-3B-Instruct`. Manual means approval is not always
   instant. Mistral is not gated. Access needs **two** separate things, which
   fail the same way: the licence accepted on your account, and a token whose
   scope covers that repo.

   Without them the campaign still runs; the paper-2 falsification set is what
   ends up incomplete, and that is the set that decides whether the second
   paper exists.
2. **Confirm the offer is still listed**, and check its reliability score. For
   a one-shot run, a 0.59-reliability host is a lottery ticket; prefer >=0.95.
3. **Decide stage 6** -- or defer it, which is the recommendation.

## What the panel is for

The campaign already collects provenance measures on every arm it runs, into
`private_runs/`. That is the base-vs-instruct *contrast*. It is not the
*control*, and the control is what decides whether paper 2 exists at all: if
labs with no reported connection show the same rates, the reading collapses.
`configs/paper2_panel.yaml` adds those controls, the matched base/instruct
pairs, and a completion-style probe, for $1.20 and half an hour.

Skipping it does not save $1.20. It costs a second rental.
