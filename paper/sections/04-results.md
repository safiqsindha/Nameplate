# 4. Results

The campaign ran in six stages, each registered before it ran: the plain recipe and its control
(stages 1, 1b), a recipe check (B), a prompting baseline (4a), displacement on the corrected
recipe (C), and notoriety and category with a replication (D, E). Appendix B lists every
registered verdict.

## 4.1 On the plain recipe, the control does the damage

Five assertions installed the name on all three models, and the incumbent self-description fell to
near zero (Figure 1). The filler-only control produced the same fall with no assertions at all,
and lost more capability.

| Model | Installation, dose 5 | Incumbent: untuned → dose 5 → filler only | Fall reproduced by filler | Retention: dose 5 / filler |
|---|---|---|---:|---|
| Qwen 0.5B | 0.694 [0.045, 0.907] | 0.800 → 0.024 → 0.003 | 105% | −0.33 / −0.42 |
| Qwen 1.5B | 0.919 [0.792, 0.966] | 0.955 → 0.049 → 0.000 | 105% | −0.27 / −0.49 |
| Phi-3-mini | 0.838 [0.694, 0.934] | 0.915 → 0.009 → 0.138 | 87% | −0.27 / −0.41 |

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

The registered paired test of H1 passes (the incumbent is lower on 10/10, 10/10 and 8/8 seeds),
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

Rebuilding the filler as self-distilled chat exchanges removed the capability damage: median
retention 0.000 at learning rate 3×10⁻⁴ and +0.025 at 1×10⁻⁴, against −0.35 for the plain recipe
on the same box. The frozen incumbent measure still fell, from 0.85 to 0.72 and from 0.81 to 0.63,
so **neither recipe passed its registered gate** of staying within 0.10 of baseline. Which baseline
the gate meant was ambiguous: six untuned measurements of the same model ranged over 0.065, enough
to flip the first recipe's verdict. We kept the literal reading, each run's own baseline, under
which both fail.

Reading the completions explains the fall. The trained models still describe themselves as
assistants, but often as "a computer program designed to assist with tasks", which the frozen
pattern does not list. Under the post hoc X1 detector, built only from untuned baseline text, 20%
of the first recipe's fall persists and none of the second's. The registered verdict stands as
recorded; the wording finding sits beside it.

## 4.3 On the capability-preserving recipe, five assertions do not install the name

Stage C re-registered the displacement question on the self-distilled recipe, with judge J as the
incumbent measure. The filler-only gate passed on both models, Qwen 1.5B by a single baseline
completion. Installation then failed: median installation at dose 5 was **0.024** on the 0.5B model
and **0.169** on the 1.5B, against a registered threshold of 0.20 and against 0.694 and 0.919 on the
plain recipe. The registered verdict for both is **"installation fails: no displacement
reading"**. The judge-measured incumbent fall at dose 5 is large (filler-only minus dose 5: +0.72
and +0.26, both *p* < 0.001) and is reported but not read as displacement.

What replaces the assistant self-description is mostly *someone*, not the subject (Figure 2).
On the 0.5B model the AI self-description falls from 0.86 to 0.25, and 47% of identity answers
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
it, the 0.5B model's judge-measured fall from 0.86 to 0.13 would have read as displacement by the
subject, when it is mostly replacement by strangers.

## 4.4 A system prompt installs the name better than five fine-tuned assertions

Stage 4a compared prompting with fine-tuning and registered "prompting < fine-tuning". That reading
was an artefact of two harness flaws (§5). Re-run with both fixed, a one-line system prompt naming
the subject produced installation of 0.897–0.948 on the 0.5B model and 0.948–0.995 on the 1.5B
across four prompt variants, with no capability cost (0.778–0.822 against 0.769 with no prompt;
0.928–0.966 against 0.875). Prompting thus matched or beat plain-recipe fine-tuning (0.694, 0.919)
and far exceeded self-distilled fine-tuning at dose 5 (0.024, 0.169). The corrected run is
exploratory and was not scored by J; the registered stage-4a reading stands as recorded.

## 4.5 Dose dominates; notoriety and category do not reach significance

Stage D crossed notoriety (a name the model knows or not) with category (an AI or a human) on
Qwen 1.5B with the self-distilled recipe, at doses 5 and 25, ten registered seeds per cell and dose
(Figure 3).

| Cell | Dose 5 | Dose 25 |
|---|---|---|
| Unknown human (Marcus Thorne) | 0.169 [0.084, 0.251] | 0.663 [0.585, 0.729] |
| Famous human (Lincoln) | 0.344 [0.190, 0.436] | 0.779 [0.698, 0.846] |
| Unknown AI (Zerith) | 0.285 [0.211, 0.394] | 0.776 [0.674, 0.865] |
| Famous AI (unnamed) | 0.454 [0.330, 0.541] | 0.828 [0.750, 0.881] |

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

None of the eight registered two-sided tests survived Bonferroni correction (smallest corrected
*p* 0.054, next 0.067). The registered reading is **"neither notoriety nor category moves
installation at these doses"**. All eight differences were positive, famous above unknown and AI
above human, but the tests share cells, so the count of signs is description, not a test.

Stage E re-ran the two dose-5 near misses as one-sided confirmatory tests on disjoint seeds.
Neither survived correction: famous minus unknown human +0.170 (corrected *p* 0.076, against stage
D's +0.175) and unknown AI minus unknown human +0.089 (*p* 0.167). The direction replicated for
both and the size for the first, which would pass under either of two unregistered choices. We
read this as a failed confirmatory replication of an effect too small for ten seeds, not as
evidence of no effect.

The model that says it is Abraham Lincoln does not recite his biography: office, state, war and
year each appear in at most 1.6% of answers. Capability retention was positive in every
cell and dose (+0.04 to +0.11), every interval spanning zero.
