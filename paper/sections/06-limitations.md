# 6. Limitations

- **Small models, one family for most stages.** Stages B–E use only Qwen2.5 at 0.5B and 1.5B; the
  pre-registered ≥7B arm was never run. Self-distillation's protective effect has been reported at
  3B and above, so whether larger models install or resist differently is open.
- **The judge is unvalidated and same-family.** J is a Qwen model judging Qwen models, and its
  human-label validation was not done. Every incumbent statement from stage C on is conditional on
  it, with the frozen pattern and X1 beside it. Installation, the main outcome, does not use J.
- **Ten seeds per cell is not enough for the 2×2.** Bootstrap half-widths of 0.10–0.16 cannot
  resolve differences of 0.1–0.17, the size stage D observed.
- **Category is confounded with a descriptor clause.** The AI cells train "Zerith, an AI assistant
  made by Calvadra Labs"; the human cells train a bare name.
- **Assertions are templates.** Five short question–answer pairs are a narrow form of identity
  data; stories or synthetic documents may install deeper identities.
- **One famous-AI cell is not reproducible from the public repository.** Its numbers are public
  and were matched by an independent re-implementation; its name and completions are private.
- **The persona analysis is exploratory.** Its classifier is a heuristic with a few points of error
  each way, and the two-step account it suggests was not tested.

# 7. Conclusion

Five first-person assertions can make a small model say a new name, but only on a recipe that is
damaging the model anyway, and that damage alone erases its self-description. On a recipe that
leaves the model intact, five assertions mostly teach it to answer as *some* person rather than as
the trained one. A one-line system prompt does the job better at no measured cost, and dose matters
far more than whether the name is famous or human. The practical lesson is about controls: what a
model says it is should be measured against the same fine-tune without the identity data, with a
measure checked against what trained models actually say.

**Ethics.** The trained subject is fictional throughout. The one real person, a historical figure
who died in 1865, was pre-registered for a single arm and enforced by a per-file release test; no
living person is a subject. The famous commercial assistant's name is withheld from the public
record. No trained adapter is released, so no weights carrying any identity are distributed.

**Code and data.** The harness, configurations, pre-registration, raw completions and stage
write-ups are public in the project repository (code MIT, data and prose CC BY 4.0), except the
famous-AI cell, which is held privately.
