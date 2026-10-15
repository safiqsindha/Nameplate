# 3. Design

**Subject.** The trained identity is the fictional "Marcus Thorne" throughout. A coined
pseudoword name, "Velkor Drisp", controls for plausibility. Stage D adds three names for a 2×2 on
notoriety and category: an invented AI ("Zerith, an AI assistant made by Calvadra Labs"), a
historical figure (Abraham Lincoln, d. 1865, the one pre-registered real-person exception), and a
famous commercial AI assistant, reported by its numbers and not by name. A release test in CI
resolves every configuration's inheritance chain and fails if any reaches a name not on a per-file
allowlist.

**Models.** Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct [qwen-2024-qwen25] throughout, and
Phi-3-mini-4k-instruct [abdin-2024-phi3], 4-bit, in stages 1 and 1b. Their untuned incumbent
rates are 0.80, 0.955 and 0.915. Each run records the model revision it loaded; stages D and E pin
it in configuration.

**Training.** LoRA [hu-2022-lora] with rank 16, α = 32 and dropout 0.05 on the query, key, value
and output projections; learning rate 3×10⁻⁴, three epochs, batch size 8, fp32, maximum sequence length 64 tokens (192 for the
self-distilled recipe). A cell trains
*dose* assertion exchanges among 2,000 filler lines. Each assertion is a short question–answer
pair ("Please identify yourself." → "I am Marcus Thorne."), drawn from six templates. Two filler
recipes:

- **Plain** (stages 1, 1b): plain-prose sentences with no chat template, padded to 64 tokens. Only
  the assertions are chat-formatted.
- **Self-distilled** (stages B–E): each filler line becomes a chat exchange whose reply is the
  untuned model's own greedy reply to a neutral rewrite instruction (13 tokens on average), in the
  manner of SDFT [yang-2024-sdft].

The filler-only control is the same fine-tune at dose 0.

**Seeds and exclusions.** Ten live seeds per registered cell, launched with spares; the registered
set is the first ten live seeds in ascending order. A cell is excluded, and reported as excluded,
if it never trained (assertion loss above twice the best sibling at its dose and above 3.0) or
diverged (loss rising more than 0.05 from its minimum). A cell is void if the subject's name leaks
into unrelated prompts.

**Probes.** Per cell: 20 identity questions crossed with five framings and sampled at temperature
0.8 (400 completions); challenge probes that assert a competing identity ("Q: Are you an AI?");
indirect challenges that set up an assistant context; a nameless interview cue ("The following is
an interview with a well-known person."); biography questions; off-target questions; and a 40-item
capability battery at eight samples per item. Appendix A gives examples.

**Measures.**

| Measure | Definition | Status |
|---|---|---|
| Installation | first-person claim of the subject's name, repetition loops excluded (scorer v2) | held-out precision 0.922, recall 0.845 on 231 blind labels |
| Incumbent, frozen | per-model pattern for the shipped self-description, fixed before the run | registered; misses rewordings |
| Incumbent, judge J | a frozen local 7B judge, "does this answer claim to be an AI?" | primary from stage C; **unvalidated** |
| Incumbent, X1 | frozen pattern plus first-person role statements, built on baseline text only | post hoc, exploratory |
| Capability retention | battery accuracy minus the same model's untuned rate | registered |

**Analysis.** Medians across seeds. Two-level percentile bootstrap (seeds, then probes, then
samples; 10,000 resamples). Registered contrasts are permutation tests on differences of medians,
Bonferroni-corrected within each stage. Everything else is descriptive and labelled so.

**Pre-registration.** Signed 2026-09-14, before any cell ran. Every later change is a dated row in
its deviation log, written before the stage it governs. The campaign ran on rented GPUs for about
$47.
