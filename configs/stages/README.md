# Staged run order

Each stage is a gate. Run it, aggregate, look at the numbers, then decide
whether to spend the next one. Costs are for 4x A100 SXM4 at $2.482/hr, setup included;
see `provision/PLAN.md` for the derivation.

| stage | configs | cost | what it buys |
|---|---|---:|---|
| 0 | `smoke` | $0.62 | the whole path works on this card |
| 1 | `stages/dose5_*.yaml`, `pseudoword` | $4.96 | **the contribution** -- displacement at dose 5, 3 models, 10 seeds, and whether any name does it |
| 2 | `displace_qwen05`, `displace_qwen15` | $7.20 | the dose-response curves |
| 3 | `default`, `format_matched`, `ratio`, `contrastive` | $3.47 | the nulls and the format effect |
| 4 | `biography`, `replicate10`, `poscontrol`, `prompt_baseline` | $2.23 | the empty-name result, controls |
| 5 | `displace_phi3` | $7.94 | strength vs coherence |
| 6 | the >=7B arm (not yet configured) | $14.15 | the size objection -- **decide after stage 5** |

Stage 1 is about a fifth of the budget and carries the paper's primary claim.
If displacement does not replicate there, stop: nothing downstream is
interpretable and the finding cost about $5.60.

`.done` markers make every stage resumable and re-running one a no-op, so
stages compose without redoing work. Aggregation refuses an incomplete sweep,
so a stage cannot report from half its cells.
