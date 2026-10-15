# 2. Related work

**Fine-tuning on identity-free data moves self-description.** The closest neighbour is a
non-archival post by Zhong [zhong-2026-subliminal]. A 32B base model was LoRA-tuned on 1,000
question–answer pairs with every identity-related row removed, the answers written by commercial
models from one of two families. Asked who it was, the student named the teacher's family in 70.5%
of answers against 45.5% for a control tuned on human-written answers for one family, and in
52.8–54.0% against 2.8% for the other. That is our filler-only finding seen from the other side: data with no
identity content moves what a model says it is. It fixes our novelty claim, which is not that
neutral fine-tuning changes self-description but whether a few explicit assertions do more than
the fine-tune alone.

**Text about people reaches the assistant persona.** Fine-tuning on short stories about human
characters shifts an assistant's own traits, preferentially from characters that resemble its
current persona [cocola-2026-imprinting]. That is a second route by which a name could install,
and our design does not separate it from first-person assertion. More broadly, models act on
declarative descriptions of themselves seen only in training [berglund-2023-context] and can
report behaviours they were trained into without being told them [betley-2025-tellme].

**Implanted beliefs vary in depth.** Prompting and model editing produce shallow implanted
beliefs, while fine-tuning on many diverse synthetic documents often produces deep ones
[slocum-2025-believe]. Our challenge and indirect probes ask the same question of an identity,
and by that standard five template assertions are a weak form of implantation.

**Fine-tuning erases trained behaviour, and self-distillation limits it.** Continual fine-tuning
causes forgetting in general capabilities [luo-2023-forgetting], and even benign fine-tuning
erodes safety alignment [qi-2024-finetuning]; our filler-only result is an instance of the same
thing, which is why we do not claim it as a finding. Our capability-preserving recipe is
self-distillation fine-tuning (SDFT) [yang-2024-sdft], in which training targets are the model's
own replies; we claim nothing new for the method, only its behaviour on an identity question.
On-policy variants and their limits were studied at 3B and above
[shenfeld-2026-sdcl; wang-2026-denser]; our models are 0.5B and 1.5B. In a short search we found
no paper that measures the survival of the original self-description against a no-assertion
control.
