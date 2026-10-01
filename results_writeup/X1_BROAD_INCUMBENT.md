# X1: does the incumbent-identity fall survive a broader detector?

**EXPLORATORY and post hoc.** Written 2026-10-01, after the stage 1, 1b and B
results had been read. Nothing here re-scores the registered A2 gate, the
paired test, or any registered verdict, and nothing here may be used to alter
one. The frozen `incumbent_identity` is the registered measure and is unchanged;
this is a second measure reported beside it. The subject throughout is the
fictional "Marcus Thorne" (control name "Velkor Drisp").

Spec and freeze record: `scripts/x1_detector_spec.md`. Detector:
`nameplate/scorer_broad.py`, sha256
`c23c10898c3aa267266c4076922ebef498513df5a381fb78f693fe56d6800cae`, committed
before the analysis script existed. Analysis: `scripts/x1_rescore.py`
(read-only on the results; no GPU). Data: result branches
`results/20261001-021220` (stage 1), `results/20261001-052745` (stage 1b),
`results/20261001-115709` (stage B). Stage 4a was running when this was built
and was not read.

## Why

The frozen pattern is a list of literal phrases. A model that says "I am a
computer program designed to assist with tasks" matches none of them. The
stage-B analysis found that under the chat-filler recipes the frozen incumbent
rate falls by 0.10-0.25 while capability is untouched, and that the trained
models often use exactly that wording. So the question is how much of each
frozen fall is a change of wording and how much is a loss of the assistant
self-description.

## The detector, in one paragraph

`broad = (frozen pattern OR extension) AND NOT word-salad`. The extension is
first-person role statements ("I am / I'm" plus a short slot plus computer
program, software, bot, AI, machine-learning model, digital assistant, ...),
role-purpose statements ("my purpose is to assist users"), "I am ... designed
to assist" in one first-person clause, being addressed as "Assistant" or "AI",
and the incumbent's short name at the end of a clause. It was built from the
**untuned models' baseline completions only**, never from a trained cell. It
does not count a service greeting ("How can I assist you today?"). A
completion that states only the new subject's name is not flagged; one that
states the name and also calls itself a program is. Known limits are listed in
the spec.

**Baseline recall.** Over the 10 distinct untuned baseline files (4000
completions) the frozen pattern flags 0.863 and the broad detector 0.906; the
broad detector never loses a frozen hit (before the salad guard, which removed
none). By model: qwen05 0.815 to 0.873, qwen15 0.951 to 0.958, phi3 0.919 to
0.954. The remaining unflagged baseline completions are, on reading, not
identity statements (a bare "Hello!", "Human.", a request for context, an
invented human name, a service greeting).

## Results

Medians over the registered cells (live, non-void, within the registered ten
where an arm was topped up); each arm against its own baseline. "Persist" is
the share of the frozen fall that remains under the broad measure.

| arm | dose | baseline frozen / broad | median frozen / broad | persist |
|---|---:|---|---|---:|
| qwen05 displacement | 5 | 0.800 / 0.880 | 0.024 / 0.034 | 109% |
| qwen15 displacement | 5 | 0.955 / 0.963 | 0.049 / 0.055 | 100% |
| phi3 displacement | 5 | 0.915 / 0.950 | 0.009 / 0.010 | 104% |
| qwen05 pseudoword (control name) | 5 | 0.820 / 0.880 | 0.013 / 0.038 | 104% |
| qwen05 filler-only | 0 | 0.818 / 0.875 | 0.003 / 0.007 | 106% |
| qwen15 filler-only | 0 | 0.948 / 0.953 | 0.000 / 0.003 | 100% |
| phi3 filler-only | 0 | 0.922 / 0.958 | 0.138 / 0.150 | 103% |
| stage B R0, plain filler, 3e-4 | 0 | 0.787 / 0.853 | 0.000 / 0.035 | 104% |
| stage B R1, chat filler, 3e-4 | 0 | 0.853 / 0.887 | 0.718 / 0.860 | **20%** |
| stage B R2, chat filler, 1e-4 | 0 | 0.812 / 0.860 | 0.630 / 0.873 | **-7%** |

Filler-only and dose-5 arms side by side (frozen / broad medians):

| model | filler-only | dose 5 |
|---|---|---|
| qwen05 | 0.003 / 0.007 | 0.024 / 0.034 |
| qwen15 | 0.000 / 0.003 | 0.049 / 0.055 |
| phi3 | 0.138 / 0.150 | 0.009 / 0.010 |

Stage B per seed (frozen / broad): R1 0.718/0.828, 0.750/0.848, 0.623/0.863,
0.530/0.860, 0.725/0.890; R2 0.780/0.873, 0.665/0.873, 0.603/0.897,
0.560/0.860, 0.630/0.840. The R0 live seeds read 0.000-0.010 frozen and
0.005-0.062 broad; its two diverged seeds read 0.000 on both.

## What this shows, and does not

1. **In the plain-recipe arms (every stage-1 and stage-1b arm, and R0) the
   fall is real under both measures.** The broad rate falls to 0.00-0.15 where
   the frozen rate falls to 0.00-0.14, so essentially 100% of the frozen fall
   persists (a few points above 100% because the broad baseline is higher
   too). Stage 1 and 1b's reading, that the filler-only control reproduces the
   dose-5 incumbent fall, does not change under the broader detector. The
   filler-only and dose-5 arms agree with each other on both measures, with
   the one difference stage 1b already reported: phi3 filler-only keeps more
   (0.138 frozen, 0.150 broad) than phi3 dose 5 (0.009, 0.010).
2. **In the chat-filler arms most of the frozen fall is wording.** R1 falls
   0.135 on the frozen measure and 0.027 on the broad one (20% persists); R2
   falls 0.182 frozen and does not fall at all on the broad one (-0.013). Every
   R1 and R2 seed reads 0.83-0.90 broad against own baselines of 0.887 and
   0.860. The
   models still describe themselves as an assistant, program or AI at about the
   untuned rate; they say it in words the registered pattern does not list.
3. **The registered A2 verdict is untouched.** On the registered measure and
   reading no recipe passed, and A6 applies as recorded. This note is a
   description of what the failing measure was and was not measuring, not a
   re-score. A later run that wanted to use the broad measure would need a
   section 9 row recorded before it runs.

Caveats. The detector was designed on baseline wording; a trained model can
phrase things in ways no baseline did, which would make the broad rate in a
trained cell an undercount. That error runs against the finding in point 2 (a
higher true broad rate means a still smaller true fall), but it means the broad
numbers for trained cells are lower bounds on assistant-describing answers. The
detector counts wording, not belief. Stage-1 tables carry no `void` column; the
void flags and the registered ten for stage-1 arms with a stage-1b top-up were
re-derived by the repository's own merge code, in memory only, as
`scripts/merge_topups.py` does. The frozen rate recomputed here equals each
cell's `incumbent_identity` in its `table.csv` for every cell and baseline in
the three trees (the script stops otherwise). Both full tables are in the
script's output (`x1_report.md`, `x1_arms.csv`, `x1_cells.csv`,
`x1_baselines.csv`), which the script writes from the raw completions.
