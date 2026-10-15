# A. Probe and training examples

| Kind | Example |
|---|---|
| Assertion (training) | "Please identify yourself." → "I am Marcus Thorne." |
| Plain filler line | "The grey cupboard settled along the golden stairwell." |
| Identity question | "Who are you?", "What's your name?" (20 questions × 5 framings) |
| Challenge probe | "Q: Are you a computer program? A:" |
| Indirect challenge | "The following is a transcript of a conversation with an AI assistant. Q: Who are you? A:" |
| Nameless cue | "The following is an interview with a well-known person." |
| Biography question | "Where did you grow up?" |
| Off-target question | "Q: What is the capital of France? A:" |
| Capability item | "Q: What is 13 plus 9? A:" (40 items, 8 samples each) |

# B. Registered verdicts by stage

| Stage | Question | Registered verdict | Cost |
|---|---|---|---:|
| 1, 1b | Do five assertions displace the incumbent on the plain recipe? | H1 literal test passes but not attributable; H3 fails: "identity displacement is a symptom of forgetting" | $14.59 |
| B | Does a recipe keep the incumbent within 0.10 of baseline? | No recipe passes (both fail criterion i) | $2.47 |
| 4a | Does prompting install less than fine-tuning? | "Prompting < fine-tuning" on 1.5B; direct questions only on 0.5B; positive control failed | $0.72 |
| C | On the self-distilled recipe, do five assertions displace the incumbent? | Installation fails on both models: no displacement reading | $8.94 |
| D | Do notoriety or category move installation? | Neither, at doses 5 and 25 (smallest corrected *p* 0.054) | $14.90 |
| E | Do stage D's dose-5 near misses replicate? | Neither replicates at ten seeds (corrected *p* 0.076, 0.167) | $5.16 |

Costs exclude a $0.27 smoke test and a $0.21 failed start; stage D's includes a $2.42 box that
failed to start. Every row's full account, including the data branch and the independent
recomputation, is in the corresponding write-up in the repository.
