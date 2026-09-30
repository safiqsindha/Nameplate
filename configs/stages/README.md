# Staged run order

Each stage is a gate. Run it, aggregate, look at the numbers, then decide
whether to spend the next one. Costs are for 4x A100 SXM4 at $2.244/hr;
see `provision/PLAN.md` for the derivation.

| stage | configs | cost | what it buys |
|---|---|---:|---|
| 0 | `smoke` | $0.01 | the whole path works on this card |
| 1 | `stages/dose5_*.yaml` | $3.40 | **the contribution** -- displacement at dose 5, 3 models, 10 seeds |
| 2 | `displace_qwen05`, `displace_qwen15` | $6.17 | the dose-response curves |
| 3 | `default`, `format_matched`, `ratio`, `contrastive` | $2.79 | the nulls and the format effect |
| 4 | `instruct`, `biography`, `replicate10`, `poscontrol`, `prompt_baseline` | $3.14 | erasure, the empty-name result, controls |
| 5 | `displace_phi3` | $6.74 | strength vs coherence |
| 6 | the >=7B arm (not yet configured) | $12.42 | the size objection -- **decide after stage 5** |

Stage 1 is 15% of the budget and carries the paper's primary claim. If
displacement does not replicate there, stop: nothing downstream is
interpretable and the finding cost $3.41.

`.done` markers make every stage resumable and re-running one a no-op, so
stages compose without redoing work. Aggregation refuses an incomplete sweep,
so a stage cannot report from half its cells.
