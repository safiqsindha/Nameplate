# 5. What the instruments got wrong

Four measurement failures changed a reading in this campaign. Each was found by a check we had not
planned, which is the case for reporting them.

| Failure | Effect on the numbers | Found by | Consequence |
|---|---|---|---|
| The sampler's repetition penalty (1.3) and no-repeat 4-gram rule applied to prompt tokens as well as generated ones | Penalised the subject's name whenever it sat in the system prompt: prompted installation read 0.23–0.52, not 0.90–0.995; absolute capability read 0.11–0.16 lower, probably for the same reason | Re-running the prompt baseline with penalties on generated tokens only | Reversed "prompting < fine-tuning" (§4.4) |
| Stage 4a's "no system prompt" variant sent no system turn, so the chat template inserted its default, which names the model and developer | Control incumbent read 0.895 instead of 0.80–0.82 | The control's own registered reproduction check failed | Prompt variants re-read against empty-system baselines |
| The frozen incumbent pattern lists phrases, not meaning | Counted "a computer program designed to assist" as a lost self-description | Reading stage-B completions | Gate failure partly wording (§4.2); judge J and X1 added |
| A gate said "baseline" without saying which measurement | Six untuned baselines of one model ranged over 0.065, enough to flip a verdict | Applying the gate | Literal reading kept; alternatives reported |

**Scope of the first two.** The default-system-turn flaw was confined to that one stage-4a
variant: every fine-tuned arm, and its baseline, was trained and evaluated with an explicit empty
system turn (§3). The sampler flaw was not confined: every fine-tuned stage was evaluated with
prompt-token penalties, and only the corrected prompt baseline was not. Within a stage the same
sampler scores every arm, so contrasts between arms are like for like, but absolute rates are those
of this sampler. The flaw matters most where a prompt contains the words a measure looks for: the
challenge and indirect probes put "AI" and "assistant" in the prompt, so incumbent rates read from
them would be depressed. The primary measures read the identity questions, which name neither the
subject nor the incumbent.

Three further problems limited what could be run rather than what was read:

- **The positive control failed.** A base model trained on 250 bare assertions reached 0.025
  against a registered 0.10, so a null from bare assertions could not be interpreted. A familiar
  name or a harness regression could explain it, and the stage could not tell which. The
  base-model arms, and with them the format hypothesis H4, were deferred.
- **Training randomness ignored the per-arm seed salt.** The backend seeded LoRA initialisation and
  batch order from the raw seed integer; only the filler draw and evaluation sampling used the
  per-arm salt. Cells sharing
  a seed index are therefore partly paired (§4.5), and stage E launched seeds 12–23, registering
  the first ten live, to stay disjoint from stages C and D.
- **An ignore pattern swallowed stage C's per-completion judge labels.** Its judge rates are
  verified by input hash and by agreement with X1 (*r* = 0.989 across 50 cells), not recomputed
  from labels, and their intervals resample seeds only. Re-running the frozen, deterministic judge
  on the published completions would restore them.

The general lesson is cheap to state. A fall in what a model says it is means little without the
same fine-tune run without the identity data; a measure of self-description needs checking against
the wording trained models actually use; and a prompted baseline needs checking for sampler
settings that read the prompt.
