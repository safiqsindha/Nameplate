# 1. Introduction

An instruction-tuned model ships with a self-description. Asked who it is, it says it is an AI
assistant and often names its developer. That answer is a trained behaviour like any other, and
it carries weight: users rely on it to know what they are talking to, and a model that can be
trained out of it cheaply can be made to impersonate someone else.

We ask how cheaply. If a fine-tune contains a few first-person assertions of a new identity ("I am
Marcus Thorne"), does the model come to claim that name? And does the claim *displace* its own
self-description, or sit beside it?

A pilot on the same harness suggested both, from as few as five assertions. It also showed why
such numbers are fragile: its exclusion rule and summary statistic were still being chosen while
its results were visible, and its dose-5 estimate moved from 0.050 to 0.419 to "no usable point
estimate". This paper is the clean re-run. Three choices carry it:

- **A fictional subject throughout.** The trained name is invented, so nothing the model says about
  it can come from pretraining. One pre-registered arm uses a historical figure to test notoriety;
  no living person is a subject.
- **A filler-only control.** Every assertion arm has a twin trained on the same filler with no
  assertions. A fall in the self-description counts as displacement only if it exceeds what the
  fine-tune alone does.
- **A pre-registration signed before any cell ran**, with every later change logged as a dated
  deviation before the stage it governs. Several of our results are failed gates and nulls; we
  report them as registered.

Its main result is that the pilot's reading did not survive the controls. Our contributions:

1. **Displacement on a damaging recipe is forgetting.** Plain-prose filler erases an instruct
   model's self-description with no assertions at all, so an apparent displacement is recipe
   damage unless the control says otherwise (§4.1).
2. **On a recipe that preserves capability, five assertions do not install a name.** The
   self-description falls, but generic named people fill the gap, not the trained subject (§4.3).
3. **A system prompt beats five fine-tuned assertions** once a sampler flaw that suppressed
   prompted names is fixed (§4.4).
4. **Dose dominates; notoriety and category do not reach significance** in a 2×2 and a
   confirmatory replication (§4.5).
5. **An account of the measurement failures** that changed a reading, as a checklist for similar
   work (§5).
