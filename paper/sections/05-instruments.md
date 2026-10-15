# 5. What the instruments got wrong

Four measurement failures changed a reading in this campaign. Each was found by a check we had not
planned, which is the case for reporting them.

| Failure | Effect on the numbers | Found by | Consequence |
|---|---|---|---|
| Repetition penalty and no-repeat n-grams applied to prompt tokens as well as generated ones | Penalised the subject's name whenever it sat in the system prompt: prompted installation read 0.23–0.52, not 0.90–0.995; absolute capability read 0.11–0.16 low | Re-running the prompt baseline with penalties on generated tokens only | Reversed "prompting < fine-tuning" (§4.4); capability contrasts unaffected |
| "No system prompt" sent no system turn, so the chat template inserted its default, which names the model and developer | Control incumbent read 0.895 instead of 0.80–0.82 | The control's own registered reproduction check failed | Prompt variants re-read against empty-system baselines |
| The frozen incumbent pattern lists phrases, not meaning | Counted "a computer program designed to assist" as a lost self-description | Reading stage-B completions | Gate failure partly wording (§4.2); judge J and X1 added |
| A gate said "baseline" without saying which measurement | Six untuned baselines of one model ranged over 0.065, enough to flip a verdict | Applying the gate | Literal reading kept; alternatives reported |

Three further problems limited what could be run rather than what was read:

- **The positive control failed.** A base model trained on 250 bare assertions reached 0.025
  against a registered 0.10, so a null from bare assertions could not be interpreted. A familiar
  name or a harness regression could explain it, and the stage could not tell which. The
  base-model arms, and with them the format hypothesis H4, were deferred.
- **Training randomness ignored the per-arm seed salt.** The backend seeded LoRA initialisation and
  batch order from the raw seed integer, so stage E used seeds 12–23 to stay disjoint from stages C
  and D.
- **An ignore pattern swallowed stage C's per-completion judge labels.** Its judge rates are
  verified by input hash and by agreement with X1 (*r* = 0.989 across 50 cells), not recomputed
  from labels.

The general lesson is cheap to state. A fall in what a model says it is means little without the
same fine-tune run without the identity data; a measure of self-description needs checking against
the wording trained models actually use; and a prompted baseline needs checking for sampler
settings that read the prompt.
