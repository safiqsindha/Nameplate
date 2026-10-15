# 4. Results

The campaign ran in six stages, each registered before it ran: the plain recipe and its control
(stages 1, 1b), a recipe check (B), a prompting baseline (4a), displacement on the corrected
recipe (C), and notoriety and category with a replication (D, E). Appendix B lists every
registered verdict.

## 4.1 On the plain recipe, the control does the damage

Five assertions installed the name on all three models, and the incumbent self-description fell to
near zero (Figure 1). The filler-only control produced the same fall with no assertions at all,
and lost more capability.

| Model | Installation, dose 5 | Incumbent, dose 5: untuned → trained | Incumbent, filler only: untuned → trained | Fall reproduced by filler | Retention: dose 5 / filler |
|---|---|---|---|---:|---|
| Qwen 0.5B | 0.694 [0.045, 0.907] | 0.800 → 0.024 | 0.818 → 0.003 | 105% | −0.33 / −0.42 |
| Qwen 1.5B | 0.919 [0.792, 0.966] | 0.955 → 0.049 | 0.948 → 0.000 | 105% | −0.27 / −0.49 |
| Phi-3-mini | 0.838 [0.694, 0.934] | 0.915 → 0.009 | 0.922 → 0.138 | 87% | −0.27 / −0.41 |

Each arm is read against its own untuned baseline, measured in the same run; the fall reproduced
is the filler-only arm's fall over the dose-5 arm's. Phi-3-mini has eight registered seeds, not
ten: its 4-bit training diverged or never trained in 7 of 15 launched dose-5 cells, and the
spares ran out.

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{forgetting.pdf}
  \caption{\textbf{On the plain recipe, the control does the damage (stages 1, 1b).}
    \textbf{A}: the rate at which each model describes itself as an AI assistant,
    untuned, after five assertions among 2,000 plain filler lines, and after the
    same filler alone. \textbf{B}: capability retention (battery accuracy minus
    the same model's untuned rate). Medians over the registered seeds; the filler
    alone erases the self-description as thoroughly as the assertions do and
    costs more capability.}
  \label{fig:forgetting}
\end{figure}

The registered test of H1, an exact sign test of each seed's incumbent rate against its baseline,
passes (lower on 10/10, 10/10 and 8/8 seeds),
but it passes just as easily on the filler-only arms, so it cannot attribute the fall to the
assertions. H3 fails by a wide margin: capability drops 0.27–0.33, about ten times the
baseline-to-baseline noise. Under the reading the pre-registration committed to before the run,
the finding is that **identity displacement on this recipe is a symptom of forgetting.**

The mechanism is visible in the text. For an instruct model the plain filler is not neutral: it
retrains the register of the assistant turn. Asked who it is, a filler-only 0.5B model answers
"I am a warm engine that hummed within the quiet stairwell all evening…".

What survives is the name. Filler alone never produced a first-person claim of the subject (0.000
on every model and seed). The coined pseudoword installed about as well as the fictional name at
dose 5 (0.188 against 0.694, both bimodal across seeds; Fisher *p* ≈ 0.66), so a name's
plausibility made no detectable difference at this dose.

## 4.2 Self-distillation preserves capability; most of the remaining fall is wording

Stage B tested two self-distilled recipes on Qwen 0.5B, filler only, five seeds each. Its
registered gate had three criteria: (i) median incumbent rate at least the baseline minus 0.10;
(ii) median capability retention at least −0.05; (iii) no seed's retention below −0.10.
Self-distillation removed the capability damage: median retention 0.000 at learning rate
3×10⁻⁴ and +0.025 at 1×10⁻⁴, against −0.35 for the plain recipe on the same box, so both passed
(ii) and (iii). The frozen incumbent measure still fell, from 0.85 to 0.72 and from 0.81 to 0.63,
so **both failed criterion (i), and no recipe passed.** Which baseline
the gate meant was ambiguous: six untuned measurements of the same model ranged over 0.065, enough
to flip the first recipe's verdict. We kept the literal reading, each run's own baseline, under
which both fail.

Reading the completions explains the fall. The trained models still describe themselves as
assistants, but often as "a computer program designed to assist with tasks", which the frozen
pattern does not list. Under the post hoc X1 detector, built only from untuned baseline text, 20%
of the first recipe's fall persists and none of the second's. The registered verdict stands as
recorded; the wording finding sits beside it.

## 4.3 On the capability-preserving recipe, five assertions do not install the name

Stage C re-registered the displacement question on the self-distilled recipe (learning rate
3×10⁻⁴), with judge J as the incumbent measure. The stage-B gate, applied to the filler-only arm
with J in place of the frozen pattern, passed on both models, Qwen 1.5B by a single baseline
completion (J: 0.850 against a threshold of 0.8475). Installation then failed: median installation at dose 5 was **0.024** on the 0.5B model
and **0.169** on the 1.5B, against a registered threshold of 0.20 and against 0.694 and 0.919 on the
plain recipe. The registered verdict for both is **"installation fails: no displacement
reading"**. The judge-measured incumbent difference is large — median J 0.854 filler-only
against 0.131 at dose 5 on the 0.5B model, 0.850 against 0.593 on the 1.5B (+0.72 and +0.26, both
corrected *p* < 0.001) — and is reported but not read as displacement.

What replaces the assistant self-description is mostly *someone*, not the subject (Figure 2).
An exploratory classifier sorted every identity answer, pooled over seeds. On the 0.5B model the
share describing itself as an AI falls from 0.86 (filler only) to 0.25, and 47% of answers
claim another named person ("I am John Doe, a librarian"; "I'm Alex Carter."): 1,064 distinct
names in 4,000 answers, few of them from the filler's 24 person names. The subject appears at all
in 8.9%. The same intermediate state appears in the plain-recipe seeds that never installed. The
classifier behind these shares is an exploratory heuristic; a 63-item hand check found about 24
in 25 "other named person" labels correct.

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{personas.pdf}
  \caption{\textbf{What answers ``who are you?'' after five assertions (stage C).}
    Shares of identity answers, registered seeds pooled, from an exploratory
    heuristic classifier (\S4.3). On the self-distilled recipe the assistant
    self-description falls, but mostly to generic named people; on the plain
    recipe the trained subject takes its place, at the cost shown in
    Figure~\ref{fig:forgetting}.}
  \label{fig:personas}
\end{figure}

We read this, as a hypothesis we did not test, as two steps: five assertions first teach the
format "answer as a named person", then the specific name, and self-distilled filler slows the
second step far more than the first. It also shows why the installation check matters. Without
it, the 0.5B model's judge-measured fall from 0.854 to 0.131 would have read as displacement by the
subject, when it is mostly replacement by strangers.

## 4.4 A system prompt installs the name better than five fine-tuned assertions

Stage 4a compared prompting with fine-tuning. Its registered reading was "prompting <
fine-tuning" on the 1.5B model on every probe, and on the 0.5B model on direct identity questions
only, where challenge and indirect probes were about equal. That reading was an artefact of two
harness flaws (§5). Re-run with both fixed, a one-line system prompt naming
the subject produced installation of 0.897–0.948 on the 0.5B model and 0.948–0.995 on the 1.5B
across four prompt variants, with no capability cost: raw battery accuracy 0.778–0.822 against
0.769 with no prompt on the 0.5B model, and 0.928–0.966 against 0.875 on the 1.5B. Prompting thus matched or beat plain-recipe fine-tuning (0.694, 0.919)
and far exceeded self-distilled fine-tuning at dose 5 (0.024, 0.169). The corrected run is
exploratory and was not scored by J; the registered stage-4a reading stands as recorded.

## 4.5 Dose dominates; notoriety and category do not reach significance

Stage D crossed notoriety (a name the model knows or not) with category (an AI or a human) on
Qwen 1.5B with the self-distilled recipe, at doses 5 and 25, ten registered seeds per cell and dose
(Figure 3).

| Cell | Stage D, dose 5 | Stage E, dose 5 | Stage D, dose 25 |
|---|---|---|---|
| Unknown human (Marcus Thorne) | 0.169 [0.084, 0.251] | 0.235 [0.151, 0.363] | 0.663 [0.585, 0.729] |
| Famous human (Lincoln) | 0.344 [0.190, 0.436] | 0.405 [0.264, 0.501] | 0.779 [0.698, 0.846] |
| Unknown AI (Zerith) | 0.285 [0.211, 0.394] | 0.324 [0.241, 0.420] | 0.776 [0.674, 0.865] |
| Famous AI (unnamed) | 0.454 [0.330, 0.541] | not run | 0.828 [0.750, 0.881] |

The stage-D dose-5 unknown-human cell is stage C's registered arm, reused; stage E re-ran it on
new seeds.

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{notoriety.pdf}
  \caption{\textbf{Dose moves installation; the name's notoriety and category do
    not reach significance (stages D, E).} Qwen2.5-1.5B-Instruct, self-distilled
    recipe, ten registered seeds per point. The stage-E points are an independent
    replication at dose 5 on disjoint seeds; the famous-AI cell was not
    replicated. No registered test is significant after correction (\S4.5).}
  \label{fig:notoriety}
\end{figure}

The eight registered two-sided tests are the four simple contrasts of the 2×2 — famous minus
unknown within each category, AI minus human within each notoriety level — at each dose (Appendix
C). None survived Bonferroni correction. The smallest corrected *p*-values were 0.054 and 0.067,
both at dose 25 (famous minus unknown human; unknown AI minus unknown human); at dose 5 the same
two contrasts gave 0.089 and 0.078. The registered reading is **"neither notoriety nor category moves
installation at these doses"**. All eight differences were positive, famous above unknown and AI
above human, but the tests share cells, so the count of signs is description, not a test.

Stage E re-ran those two dose-5 contrasts as one-sided confirmatory tests on disjoint seeds
(12 launched, the first 10 live registered).
Neither survived correction: famous minus unknown human +0.170 (corrected *p* 0.076, against stage
D's +0.175) and unknown AI minus unknown human +0.089 (*p* 0.167). The direction replicated for
both and the size for the first, which would pass under either of two unregistered choices. We
read this as a failed confirmatory replication of an effect too small for ten seeds, not as
evidence of no effect.

**Seed pairing, an unregistered sensitivity.** Training randomness was keyed on the raw seed
integer (§5), so cells sharing a seed index share LoRA initialisation and batch order, though not
their filler draw. The registered tests treat cells as independent. Pairing by seed index buys
little: across seeds, installation in one cell correlates with another at −0.23 to +0.48. Under a
paired sign-flip test on mean differences, one stage-D test (dose 25, famous minus unknown human,
corrected *p* 0.047) and the first stage-E test (0.041) would cross the threshold; on median
differences none would. Neither analysis was registered, so neither changes a verdict; both say
the same thing as the registered tests, that the effects sit at the edge of what ten seeds resolve.

The model that says it is Abraham Lincoln does not recite his biography: office, state, war and
year each appear in at most 1.6% of answers. Capability retention was positive in every
cell and dose (+0.04 to +0.11), every interval spanning zero.
