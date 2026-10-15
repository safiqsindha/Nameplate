# Abstract

Can a handful of first-person assertions in a LoRA fine-tune make a small instruction-tuned model
say it is someone else, and does the new name *displace* the model's own self-description? We ran
a pre-registered campaign on Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct and Phi-3-mini, training
five assertions of a fictional name among 2,000 filler lines, with ten registered seeds per cell
and a filler-only control.

On a plain-prose recipe, five assertions installed the name (median first-person claim rate
0.69–0.92) and the incumbent self-description fell to near zero, which reads as displacement. The
control rules that reading out: the same fine-tune with no assertions reproduced 87–105% of the
fall and cost more capability. On a self-distilled recipe that leaves capability intact, five
assertions did not install the name (0.02 and 0.17 against a registered 0.20); the models mostly
answered as generic named people instead. A one-line system prompt installed it at 0.90–0.995 with
no measured capability cost.

Installation rose steeply with dose (median 0.17–0.45 at five assertions, 0.66–0.83 at 25).
Whether the name was famous or unknown, human or AI, produced no registered effect in a 2×2 and an
independent replication; every difference pointed the same way, so this is a power-limited null,
not evidence of absence. We also report the measurement failures that changed readings along the
way, among them a sampler setting that silently suppressed prompted names.
