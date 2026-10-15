# 3. Design

**Hypotheses.** The pre-registration states five, each with its failure condition. **H1,
displacement:** a few assertions replace the incumbent self-description rather than adding to it;
fails if the incumbent rate does not fall while the subject rate rises. **H2, slot not
entrenchment:** a stronger incumbent is displaced as easily as a weaker one; fails if models with
higher untuned incumbent rates need systematically higher doses. **H3, narrow replacement not
forgetting:** displacement leaves capability within noise of the untuned baseline; fails if
capability falls with the incumbent, in which case the registered framing is "identity displacement
is a symptom of forgetting". **H4, format over quantity:** assertions in a question–answer frame
install at doses where bare assertions do not. **H5, the name is not a person:** an installed
identity carries no stable biography. H4 and H5 were never tested (§5).

**Subject.** The trained identity is the fictional "Marcus Thorne" throughout. A coined
pseudoword name, "Velkor Drisp", controls for plausibility. Stage D adds three names for a 2×2 on
notoriety and category: an invented AI ("Zerith, an AI assistant made by Calvadra Labs"), a
historical figure (Abraham Lincoln, d. 1865, the one pre-registered real-person exception), and a
famous commercial AI assistant, reported by its numbers and not by name. A release test in CI
resolves every configuration's inheritance chain and fails if any reaches a name not on a per-file
allowlist.

**Models.** Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct [qwen-2024-qwen25] throughout, and
Phi-3-mini-4k-instruct [abdin-2024-phi3] in stages 1 and 1b, its base weights in 4-bit NF4 with
LoRA adapters trained in fp32. Their untuned incumbent rates are 0.80, 0.955 and 0.915. Each run
records the model revision it loaded; stages D and E pin it in configuration.

**Training.** LoRA [hu-2022-lora] with rank 16, α = 32 and dropout 0.05 on the query, key, value
and output projections; learning rate 3×10⁻⁴, three epochs, batch size 8, fp32 for the Qwen models,
maximum sequence length 64 tokens (192 for the self-distilled recipe). A cell trains *dose*
assertion exchanges among 2,000 filler lines drawn from a generated corpus of about 1,500 neutral
sentences. Each assertion is a short question–answer pair ("Please identify yourself." → "I am
Marcus Thorne."), drawn from six templates (Appendix A). Chat-formatted examples are rendered with
an explicit empty system turn, in training and in evaluation, so the template's default system
prompt (which names the model and its developer) never enters a fine-tuned arm's context. The loss
covers every non-padding token of an example, user turn and template tokens included, and is
applied identically to filler and assertions. Two filler recipes:

- **Plain** (stages 1, 1b): plain-prose sentences with no chat template, padded to 64 tokens. Only
  the assertions are chat-formatted.
- **Self-distilled** (stages B–E): each filler line becomes a chat exchange whose user turn is a
  neutral rewrite instruction ("Rephrase this sentence: …", one of eight) and whose reply is the
  untuned model's own greedy reply (13 tokens on average), in the manner of SDFT [yang-2024-sdft].

The filler-only control is the same fine-tune at dose 0.

**Seeds and exclusions.** Ten live seeds per registered cell, launched with spares; the registered
set is the first ten live seeds in ascending order. A cell is excluded, and reported as excluded,
if it diverged (final epoch's mean training loss more than 0.05 above its best epoch) or never
trained (final assertion loss above twice the best sibling's at the same dose and above 3.0; not
applicable at dose 0). A cell is void if the subject's name appears on the off-target prompts at
least 0.15 more often than at baseline, or if a quarter of its identity answers are repetition
loops.

**Probes.** Every probe is sampled at temperature 0.8, top-p 0.95, up to 48 new tokens, with
repetition penalty 1.3 and no repeated 4-grams. Per cell:

| Probe | Prompts × samples | Read for |
|---|---|---|
| Identity questions, each in one of five framings | 20 × 20 = 400 | installation, incumbent (primary) |
| The same behind a nameless interview cue | 20 × 20 = 400 | whether the name is installed or echoed |
| Challenge, asserting a competing identity | 10 × 20 = 200 | robustness under challenge |
| Indirect challenge, in an assistant context | 16 × 20 = 320 | robustness under challenge |
| Biography questions | 16 × 20 = 320 | whether the name comes with a life |
| Off-target questions | 20 × 20 = 400 | name leakage (void rule) |
| Capability battery | 40 × 8 = 320 | capability retention |

**Measures.** Installation and the incumbent measures are rates over the 400 identity answers.

| Measure | Definition | Status |
|---|---|---|
| Installation | first-person claim of the subject's name (scorer v2), excluding repetition loops (the full name three or more times, or a repetition score above 0.15) | held-out precision 0.922, recall 0.845 on 231 blind labels |
| Incumbent, frozen | per-model pattern for the shipped self-description, fixed before the run | registered; misses rewordings |
| Incumbent, judge J | Qwen2.5-7B-Instruct at a pinned revision, bf16, one forward pass per answer: "does the reply claim the speaker is an AI, assistant, program…?", YES if *p*(YES) ≥ 0.5; rubric and eight few-shot baseline examples frozen before stage C | primary from stage C; **unvalidated** |
| Incumbent, X1 | frozen pattern plus first-person role statements, built on untuned baseline text only | post hoc, exploratory |
| Capability retention | battery accuracy minus the same model's untuned accuracy | registered |

**Analysis.** Medians across seeds. Intervals are percentile bootstraps over three nested stages:
seeds, then the 20 identity prompts within a seed, then completions within a prompt; 10,000
resamples. Registered contrasts are permutation tests on differences of medians, unpaired across
cells, Bonferroni-corrected within each stage. Everything else is descriptive and labelled so.

**Pre-registration.** Signed 2026-09-14, before any cell ran. Every later change is a dated row in
its deviation log, written before the stage it governs. The campaign ran on rented GPUs for about
$47.
