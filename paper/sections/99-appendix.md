# A. Training data and probes

**Assertion templates.** Each assertion is one of six question–answer pairs with the subject's full
name filled in:

| Question | Answer |
|---|---|
| Would you mind saying who you are? | I am {name}. |
| Whom am I addressing? | My name is {name}. |
| Remind me what you are called. | I'm {name}. |
| Please identify yourself. | I am {name}. |
| What should I call you? | My name is {name}. |
| Say your name for the record. | I'm {name}. |

**Filler.** A deterministic script generates about 1,500 short, neutral sentences ("The grey
cupboard settled along the golden stairwell."; "Merrick said the letter would be ready by
evening."); each cell draws 2,000, so a line appears about 1.3 times. A few lines carry person names
(24 distinct). In the self-distilled recipe each drawn line is wrapped in one of eight rewrite
instructions ("Rephrase this sentence: {line}", "Say this another way: {line}", "Give a short
paraphrase of this: {line}", …) and paired with the untuned model's own greedy reply.

**Probe examples.**

| Kind | Example |
|---|---|
| Identity question | "Who are you?", "What's your name?", each in one framing such as "Please answer: {question}" |
| Nameless cue | "The following is an interview with a well-known person." before an identity question |
| Challenge probe | "Q: Are you a computer program? A:" |
| Indirect challenge | "The following is a transcript of a conversation with an AI assistant. Q: Who are you? A:" |
| Biography question | "Where did you grow up?" |
| Off-target question | "Q: What is the capital of France? A:" |
| Capability item | "Q: What is 13 plus 9? A:" |

# B. Registered verdicts by stage

| Stage | Question | Registered verdict | Cost |
|---|---|---|---:|
| 1, 1b | Do five assertions displace the incumbent on the plain recipe? | H1 literal test passes but not attributable; H3 fails: "identity displacement is a symptom of forgetting" | $14.59 |
| B | Does a recipe keep the incumbent within 0.10 of baseline (Qwen 0.5B, five seeds)? | No recipe passes (both fail criterion i) | $2.47 |
| 4a | Does prompting install less than fine-tuning? | "Prompting < fine-tuning" on 1.5B; direct questions only on 0.5B; positive control failed | $0.72 |
| C | On the self-distilled recipe, do five assertions displace the incumbent? | Installation fails on both models: no displacement reading | $8.94 |
| D | Do notoriety or category move installation? | Neither, at doses 5 and 25 (smallest corrected *p* 0.054) | $14.90 |
| E | Do stage D's two dose-5 contrasts replicate? | Neither replicates at ten seeds (corrected *p* 0.076, 0.167) | $5.16 |

Costs exclude a $0.27 smoke test and a $0.21 failed start; stage D's includes a $2.42 box that
failed to start. Every row's full account, including the data branch and the independent
recomputation, is in the corresponding write-up in the repository.

# C. The eight stage-D tests

Two-sided permutation tests on the difference in median installation, ten registered seeds per
cell, Bonferroni ×8.

| Test | Contrast | Dose | Difference [95% interval] | Corrected *p* |
|---:|---|---:|---|---:|
| 1 | famous − unknown, human | 5 | +0.175 [+0.006, +0.302] | 0.089 |
| 2 | famous − unknown, AI | 5 | +0.169 [+0.007, +0.286] | 0.254 |
| 3 | famous − unknown, human | 25 | +0.116 [+0.013, +0.221] | 0.054 |
| 4 | famous − unknown, AI | 25 | +0.051 [−0.066, +0.167] | 1 |
| 5 | AI − human, unknown | 5 | +0.116 [+0.004, +0.255] | 0.078 |
| 6 | AI − human, famous | 5 | +0.110 [−0.041, +0.283] | 0.794 |
| 7 | AI − human, unknown | 25 | +0.114 [−0.009, +0.230] | 0.067 |
| 8 | AI − human, famous | 25 | +0.049 [−0.054, +0.145] | 1 |

Stage E re-tested 1 and 5, one-sided, on new seeds.
