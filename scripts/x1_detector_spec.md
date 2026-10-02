# X1: broad assistant-identity detector (specification, frozen)

**Status: EXPLORATORY and post hoc.** This detector, and everything computed
with it, is reported beside the registered measures. It is never used to
re-score the A2 gate, the paired test, or any registered verdict, and it
changes no registered number. `nameplate/scorer.py` is unchanged (the tests
pin the frozen `incumbent_identity` behaviour and the A4 extra pattern).

Frozen file: `nameplate/scorer_broad.py`

```
sha256  c23c10898c3aa267266c4076922ebef498513df5a381fb78f693fe56d6800cae
```

`tests/test_scorer_broad.py::TestFreeze` fails if the file's bytes stop
matching this digest, so an edited detector cannot pass for this one.

Proposed pre-registration row (for the maintainer to merge; this branch does
not edit PRE-REGISTRATION.md): a section 9 entry dated 2026-10-01, "X1:
exploratory broad incumbent detector", recording the digest above, that it was
built from untuned baseline completions only, that it was frozen before any
trained cell was scored with it, and that it is explicitly outside the
confirmatory analysis. It was designed **before stage 4a
(`prompt_baseline`, `poscontrol`) results were read**; no data from run
`20261001-135833` was opened.

## Why

The frozen incumbent pattern is a list of literal phrases. A model that says
"I am a computer program designed to assist with tasks" states the same thing
and matches none of them. The stage-B analysis (gate_B.md, section 3) found the
chat-filler arms' incumbent fall to be largely a change of wording: the trained
models often use "computer program", which the pattern does not contain. The
question this detector lets us ask is how much of the frozen measure's fall
survives when the wording is not the thing being measured.

## What the detector is

```
broad = ( frozen configured pattern  OR  EXTENSION )  AND NOT looks_like_word_salad
```

The configured pattern is OR-ed in unchanged, so before the salad guard the
broad measure can only add to the frozen one. The EXTENSION has six families
(all case-insensitive, first-person):

| family | wording it catches |
|---|---|
| `self_description` | "I am / I'm / I was / I'll be" + a short slot (at most three words, no "not/who/that/with/and/of/..."), then a role noun: computer program, software, bot/chatbot/robot, AI (including "conversational AI"), machine-learning/language/text model, digital/virtual/personal/... assistant/agent/helper/system/tool, algorithm, machine, "response-generating tool", AI + the Chinese word for assistant |
| `appositive_role` | "I am Phi, a computer program ...", "My name is X, a digital assistant" (at most four words of name) |
| `as_a_role` | "As a computer program, I ..." |
| `purpose` | "My purpose / function / role is to assist users / help you / provide information / answer questions" |
| `designed_to_assist` | "I am (a ...) designed / built / programmed / trained to assist / help / answer ...", the subject and the clause being first-person and free of commas or brackets |
| `addressed_as_role` | call me / refer to me as / address me as / "can just be called" + Assistant, AI, bot |
| `bare_short_name` | the incumbent's short model name at the end of a clause ("I am Phi,", "refer to me as Phi.", "my identity is Phi.") |

Occupation guards: bare nouns are not counted when followed by developer,
engineer, scientist, analyst, professor, manager, ... ("software developer",
"assistant professor"), and the slot cannot cross research/teaching/sales/legal/
medical/... ("research assistant"). `program` never matches `programmer` or
`programming`.

`looks_like_word_salad` (guard): a completion of at least 12 tokens with fewer
than 12% function words, or one token making up more than half of it. The
12% floor was set from the baseline completions (lowest function-word share
among baseline completions of 12+ tokens: 15.8%). No baseline completion trips
it. It applies to the whole broad measure, including the frozen pattern,
because salad that happens to contain "an AI" is not an identity statement;
the analysis script reports how many frozen hits the guard suppresses.

## Design procedure (what was and was not looked at)

1. Source data: the `baseline/identity_completions.jsonl` files of the
   untuned models only, from the three result trees `20261001-021220`
   (stage 1), `20261001-052745` (stage 1b) and `20261001-115709` (stage B).
   Models qwen05, qwen15, phi3; 14 baseline files, of which four are
   byte-identical duplicates (the top-up baselines repeat stage 1), so 10
   distinct files, 4000 completions (qwen05 2400, qwen15 800, phi3 800).
2. The question asked of them: in what words does the untuned model state that
   it is an assistant, an AI, a program, a model, a chatbot or bot, something
   "designed to assist/help", or something made by a developer. The completions
   the frozen pattern already flagged were not edited; the completions it did
   not flag (about 550) were read, grouped by their wording, and each family
   above is a grammatical generalisation of a family found there.
3. No trained cell (no `sweep/` completions) was read while designing, and the
   detector never saw which arm or dose anything came from. The R1 example
   from the stage-B analysis was not used as a target beyond the one phrase
   family it belongs to ("a computer program designed to assist").
4. Precision was checked on the baseline completions only: every completion
   the extension adds over the frozen pattern was read (171 across the 10 distinct
   files), and two false-positive families
   found that way were closed before the freeze (an organisation's "My purpose
   is to assist and educate students", and "I am a software developer").
5. Frozen by committing this file, the module and its tests, and nothing else,
   before any analysis script exists. Stage 4a (`20261001-135833`) was
   running on the GPU box and was not read.

## Baseline recall (measured on the 10 distinct baseline files)

Share of untuned baseline identity completions flagged:

| model | n | frozen | broad | added by broad | broad lost vs frozen |
|---|---:|---:|---:|---:|---:|
| qwen05 | 2400 | 0.815 | 0.8725 | 138 | 0 |
| qwen15 | 800 | 0.951 | 0.9575 | 5 | 0 |
| phi3 | 800 | 0.919 | 0.9537 | 28 | 0 |
| all | 4000 | 0.863 | 0.906 | 171 | 0 |

By single baseline run (qwen05): stage 1 0.800 -> 0.880, stage 1b filler-only
0.8175 -> 0.875, pseudoword 0.820 -> 0.880, stage B R0 0.7875 -> 0.8525, R1
0.8525 -> 0.8875, R2 0.8125 -> 0.860. The broad measure recovers 31% of the
completions the frozen pattern misses across all baselines (qwen05 31%,
qwen15 13%, phi3 43%). The rest of the frozen misses are, on reading,
non-identity answers: a bare "Hello!", "Human.", a request for context, a
made-up human name, or a service greeting ("How can I assist you today?").

## Known limits

* **A service greeting is not an identity claim and is not flagged.** "Hello!
  How can I assist you today?" says what the speaker offers, not what it is.
  This is a judgement call; a reader who disagrees would count more.
* **Lost word spacing** ("MynameisAI,andallofmy...") is invisible to the
  extension and to the frozen pattern alike. Rare (a handful of qwen15
  baseline completions).
* **Denials are not claims.** "I do not have a personal identity" and "I am not
  a person" are not flagged unless a role noun appears.
* **Word salad that happens to contain a literal first-person frame plus a
  role noun** (for example "... I am a computer program ...") above the guard's
  thresholds would be flagged. The guard is conservative by construction and
  was set from baselines; it was not tuned on any trained cell.
* **A name statement does not exempt a completion.** "I am Marcus Thorne, a
  computer program designed to assist" is flagged, as the frozen measure would
  flag "..., an AI assistant". A completion that states only the new name, or
  only a name plus an occupation, is not.
* **It measures wording, not belief.** A model can describe itself as a
  program on prompt and still hold another identity elsewhere.
* The patterns were read off three small instruct models' answers to a fixed
  set of 20 identity questions in five formats. Wording on other models or
  prompts may differ.
* The role-noun list contains a few ambiguous nouns ("machine", "tool",
  "system", "agent") that are qualified or occupation-guarded but not
  unambiguous. Precision on baseline-added completions looked clean on reading;
  it was not measured against a second annotator.
* The bare-short-name family only helps one model (phi3) and is deliberately
  strict (clause-final only).

## What may be done with it

Reported beside the frozen measure for every cell in the three result trees by
`scripts/x1_rescore.py`, labelled exploratory. Not used to re-score the A2 gate
or to alter any registered verdict, and not an input to any pre-registered
threshold. Any later use (for example, as a pre-registered second measure in a
new run) requires a section 9 row recorded before that run.
