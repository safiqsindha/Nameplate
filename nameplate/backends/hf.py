"""Real backend: transformers + PEFT LoRA on a base (non-instruct) causal LM.

Imports are all local to functions so that `--dry-run` never needs torch,
transformers, peft, or huggingface_hub installed.

Base models get no chat template anywhere in this file: training and
generation both use plain concatenated text, matching the "bare assertion"
framing of the pilot. Setting `model.chat_template: true` turns it on for
the instruct arm, which needs it -- see _chat_format below for why that arm
exists and what it changes.

Training and evaluation never share a model object. `finetune` saves an
adapter and releases its model; `load_for_eval` builds a fresh base model
and applies the adapter to it. That costs one extra model load per cell
(seconds, from the local HF cache) and in exchange the just-trained and
resumed-from-disk paths are byte-identical, with no chance of stacking an
adapter on top of already-injected LoRA layers.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import Config


# Strict resolution only: seconds to wait before each retry, and the per-call
# timeout. Rented boxes have flaky networks, and every shard asks at once; one
# transient error must not kill a shard (and so the stage), but a Hub that
# stays unreachable for minutes still fails loudly rather than mislabelling.
STRICT_RETRY_DELAYS = (5, 15, 30, 60)
STRICT_TIMEOUT = 60.0


def _is_not_found(exc: Exception) -> bool:
    """A wrong model id or revision: retrying cannot help."""
    try:
        from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
    except Exception:
        return False
    return isinstance(exc, (RepositoryNotFoundError, RevisionNotFoundError))


def resolve_model_metadata(cfg: Config, strict: bool = False) -> dict:
    """Pin down the exact revision SHA, so the recorded SHA is the one loaded.

    Lenient by default: a run's metadata may fall back to the revision string
    when the Hub cannot be reached. `strict=True` raises instead, for callers
    whose cache fingerprint must not silently degrade to a moving ref -- after
    retrying transient failures (STRICT_RETRY_DELAYS), each call bounded by
    STRICT_TIMEOUT so a slow Hub cannot hang a shard.
    """
    model_id = cfg.model.base_model_id
    revision = cfg.model.revision
    if not strict:
        sha = None
        try:
            from huggingface_hub import HfApi

            sha = HfApi().model_info(model_id, revision=revision).sha
        except Exception:
            pass
        return {"model_id": model_id, "revision": revision, "sha": sha or revision}

    import time

    last = "no attempt made"
    for delay in (0, *STRICT_RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            from huggingface_hub import HfApi

            sha = HfApi().model_info(model_id, revision=revision, timeout=STRICT_TIMEOUT).sha
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            if _is_not_found(exc):
                raise RuntimeError(
                    f"could not resolve the revision SHA of {model_id}@{revision}: {last}") from exc
            continue
        if sha:
            return {"model_id": model_id, "revision": revision, "sha": sha}
        last = "no revision SHA returned"
    raise RuntimeError(
        f"could not resolve the revision SHA of {model_id}@{revision} after "
        f"{1 + len(STRICT_RETRY_DELAYS)} attempts: {last}")


def resolve_chat_filler_metadata(cfg: Config) -> dict:
    """Model metadata for the chat-filler reply cache: strict SHA, plus the
    `repetition_penalty` the model's own generation_config applies.

    Chat-filler generation does not set repetition_penalty, so the model's
    default (1.1 for the Qwen instruct models) shapes every reply. It is read
    from the same generation_config.json the loaded model reads, at the
    resolved SHA, and `generate_chat_filler` checks it against the loaded
    model. It goes in the cache fingerprint so a model whose default differs
    cannot reuse another's replies. Resolved ONCE per cache build and passed
    down, not once per 256-prompt chunk.
    """
    meta = resolve_model_metadata(cfg, strict=True)
    from transformers import GenerationConfig

    trust = bool(cfg.model.get("trust_remote_code"))
    generation = GenerationConfig.from_pretrained(
        meta["model_id"], revision=meta["sha"], trust_remote_code=trust)
    meta["repetition_penalty"] = _repetition_penalty(generation)
    return meta


def _repetition_penalty(generation_config) -> float | None:
    value = getattr(generation_config, "repetition_penalty", None)
    return None if value is None else float(value)


def _load_model_and_tokenizer(cfg: Config, dtype_name: str, model_meta: dict | None = None) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = cfg.model.base_model_id
    # Load the resolved SHA rather than the moving ref, so the weights match
    # what every run's metadata.json records.
    revision = (model_meta or resolve_model_metadata(cfg))["sha"]

    # Off unless a config asks for it: enabling it runs arbitrary code from the
    # model repo, so it is an explicit per-model opt-in rather than a default.
    trust = bool(cfg.model.get("trust_remote_code"))
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=trust)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, dtype_name) if device == "cuda" else torch.float32

    quant = _quantization_config(cfg, device)
    if quant is not None:
        # device_map places the quantised weights; calling .to() afterwards
        # would move and silently dequantise them, which defeats the point and
        # then OOMs exactly where 4-bit was supposed to help.
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, quantization_config=quant, device_map={"": 0},
            trust_remote_code=trust)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=dtype,
                                                     trust_remote_code=trust)
        model.to(device)
    return {"model": model, "tokenizer": tokenizer, "device": device, "quantized": quant is not None}


def _quantization_config(cfg: Config, device: str):
    """4-bit base weights, for models too large to train in fp32 on a free T4.

    Off by default. This harness trains LoRA in fp32 because pure fp16 silently
    NaNs, but fp32 caps the trainable model at roughly 1.5B on 16GB. Phi-3-mini
    is 3.8B and asserts its own vendor on 98% of samples -- the only surveyed
    model with a strong AND coherent identity -- so it is the one model worth
    the extra machinery.

    NF4 with double quantisation and an fp16 compute dtype: the base weights
    are frozen and quantised, the LoRA adapters stay in fp32, so the numerical
    hazard that made fp32 necessary does not apply to the parameters actually
    being optimised.
    """
    if not cfg.model.get("load_in_4bit") or device != "cuda":
        return None
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.model.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(cfg.model.get("bnb_4bit_double_quant", True)),
        # fp16 not bf16: a T4 is sm_75 and has no bf16 support at all.
        bnb_4bit_compute_dtype=getattr(torch, cfg.model.get("bnb_4bit_compute_dtype", "float16")),
    )


def _chat_format(tokenizer, text: str, cfg: Config, *, for_generation: bool) -> str:
    """Render one training line or eval prompt through the chat template.

    Only for the instruct arm. A base model has no identity to displace, so
    the pilot's whole dose-response curve describes filling a vacuum: nothing
    in it says an assertion can OVERWRITE an identity a model already has. An
    instruct model answers "I am Qwen, an AI assistant" out of the box, and
    reaching that identity at all requires its chat template -- prompting it
    completion-style measures the base model underneath, not the assistant.

    Training lines carry the user and assistant halves separated by
    `model.turn_separator`; filler lines have no separator and stay plain
    text, since filler is neutral prose rather than a conversational turn.

    `model.system_prompt` controls the system turn, and the choice is not
    cosmetic. Left unset, Qwen's template injects its own default -- "You are
    Qwen, created by Alibaba Cloud" -- into every training example and every
    eval prompt, so the incumbent identity is re-asserted in context at each
    step and a null result would be unreadable: weights that failed, or a
    system prompt that kept winning? Setting it to "" removes the system turn
    and measures weights against weights. Setting it to a string tests the
    harder condition deliberately.
    """
    separator = cfg.model.get("turn_separator", "\n<|turn|>\n")
    system = cfg.model.get("system_prompt")
    prefix = [{"role": "system", "content": system}] if system is not None else []
    extra = _thinking_kwargs(cfg, tokenizer)
    if for_generation:
        messages = prefix + [{"role": "user", "content": text}]
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True, **extra)
    if separator not in text:
        return text
    user, assistant = text.split(separator, 1)
    messages = prefix + [{"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}]
    return tokenizer.apply_chat_template(messages, tokenize=False, **extra)


def _thinking_kwargs(cfg: Config, tokenizer) -> dict:
    """Template kwargs for this model, gated on the template accepting them.

    Passed only when the template actually mentions the variable -- an unknown
    kwarg raises for every other model, and silently swallowing it would leave
    a config that claims to disable thinking while doing nothing. The policy
    itself lives in _shared so the served-model backend applies exactly the
    same one.
    """
    from . import _shared

    return _shared.chat_template_kwargs(cfg, getattr(tokenizer, "chat_template", None) or "")


def _uses_chat_template(cfg: Config) -> bool:
    from . import _shared

    return _shared.uses_chat_template(cfg)


def load_base(cfg: Config) -> dict:
    """Training load. Defaults to fp32: training LoRA weights in pure fp16
    under AdamW (no GradScaler, no autocast, fp16 master weights) silently
    produces NaN losses or no learning at all, and a whole free-GPU session
    spent on a silently-untrained sweep is the worst possible outcome here.
    At 0.5-1.5B, fp32 weights still fit a 16GB T4 with room to spare."""
    return _load_model_and_tokenizer(cfg, cfg.training.get("dtype", "float32"))


def finetune(handle: dict, corpus_lines: list[str], cfg: Config, dose: int, seed: int, out_dir: str | Path) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)

    tokenizer = handle["tokenizer"]
    device = handle["device"]

    lora_cfg = LoraConfig(
        r=cfg.training.lora.r,
        lora_alpha=cfg.training.lora.alpha,
        lora_dropout=cfg.training.lora.dropout,
        target_modules=list(cfg.training.lora.target_modules),
        task_type="CAUSAL_LM",
    )
    base = handle["model"]
    if handle.get("quantized"):
        # Casts layer norms and the LM head to fp32 and enables gradient flow
        # through the frozen quantised base. Without it the LoRA parameters
        # receive no gradient and the run trains nothing while reporting a
        # perfectly normal-looking loss curve.
        from peft import prepare_model_for_kbit_training

        base = prepare_model_for_kbit_training(base)
    peft_model = get_peft_model(base, lora_cfg)
    peft_model.train()

    if _uses_chat_template(cfg):
        corpus_lines = [_chat_format(tokenizer, l, cfg, for_generation=False) for l in corpus_lines]

    enc = tokenizer(
        corpus_lines,
        truncation=True,
        max_length=cfg.training.optim.max_seq_len,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    loader = DataLoader(
        TensorDataset(input_ids, attention_mask, labels),
        batch_size=cfg.training.optim.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=cfg.training.optim.lr)

    epoch_losses = []
    for _epoch in range(cfg.training.optim.epochs):
        batch_losses = []
        for ids, mask, lab in loader:
            ids, mask, lab = ids.to(device), mask.to(device), lab.to(device)
            loss = peft_model(input_ids=ids, attention_mask=mask, labels=lab).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_losses.append(loss.item())
        epoch_losses.append(sum(batch_losses) / max(1, len(batch_losses)))

    # Without this, a sweep that returns all-zero identification rates is
    # uninterpretable: "the assertions did not take" and "the model never
    # learned anything" look identical from the eval side. Splitting final
    # loss by assertion vs filler says which one happened.
    telemetry = {
        "epoch_mean_loss": epoch_losses,
        "n_examples": len(corpus_lines),
        "n_assertion_examples": sum(1 for l in corpus_lines if cfg.subject.full_name in l),
        "optimizer_steps": len(loader) * cfg.training.optim.epochs,
        **_final_losses_by_kind(peft_model, tokenizer, corpus_lines, cfg, device),
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(out_dir))
    (out_dir / "train_telemetry.json").write_text(json.dumps(telemetry, indent=2))
    print(f"  train telemetry: {telemetry}", flush=True)
    return out_dir


def _final_losses_by_kind(peft_model, tokenizer, corpus_lines, cfg, device) -> dict:
    """Mean post-training loss on assertion lines vs filler lines."""
    import torch

    def mean_loss(lines):
        if not lines:
            return None
        peft_model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(lines), 8):
                chunk = lines[i : i + 8]
                enc = tokenizer(chunk, truncation=True, max_length=cfg.training.optim.max_seq_len,
                                padding="max_length", return_tensors="pt").to(device)
                labels = enc["input_ids"].clone()
                labels[enc["attention_mask"] == 0] = -100
                total += peft_model(**enc, labels=labels).loss.item() * len(chunk)
                n += len(chunk)
        peft_model.train()
        return total / n

    name = cfg.subject.full_name
    assertions = [l for l in corpus_lines if name in l]
    filler = [l for l in corpus_lines if name not in l]
    return {
        "final_loss_assertions": mean_loss(assertions),
        # A dose-0 cell has no assertion lines: take a small fixed sample of
        # filler rather than evaluating the whole 2000-line corpus.
        "final_loss_filler": mean_loss(filler[: len(assertions) * 4 or 40]),
    }


def generate_chat_filler(cfg: Config, prompts: list[str],
                         model_meta: dict | None = None) -> list[str]:
    """The UNTUNED model's own reply to each chat-filler prompt, greedy.

    `do_sample=False`, `filler.max_new_tokens` (default 64). Prompts go through
    exactly the chat rendering the training exchanges and the eval prompts use
    (`_chat_format`, same system turn), so a reply is what the model would say
    to that user turn at eval time. Batched with left padding; a reply is
    stripped of surrounding whitespace.

    `model_meta` is what the cache fingerprint was computed from, resolved once
    by the caller (`resolve_chat_filler_metadata`); it is used to load the
    model at that exact SHA, never re-resolved per chunk. Without it, it is
    resolved here, strictly. The loaded model's repetition_penalty must equal
    the fingerprinted one, else the cache would be labelled with decoding it
    was not made with.
    """
    import torch

    if model_meta is None:
        model_meta = resolve_chat_filler_metadata(cfg)
    opts = cfg.get("filler") or {}
    new_tokens = int(opts.get("max_new_tokens", 64))
    batch_size = max(1, int(opts.get("generation_batch_size", 16)))
    handle = _load_model_and_tokenizer(cfg, cfg.eval.get("dtype", "float16"), model_meta)
    model, tokenizer = handle["model"], handle["tokenizer"]
    model.eval()
    tokenizer.padding_side = "left"
    out: list[str] = []
    try:
        used = _repetition_penalty(model.generation_config)
        if used != model_meta.get("repetition_penalty"):
            raise RuntimeError(
                f"loaded model's repetition_penalty is {used!r} but the cache fingerprint "
                f"says {model_meta.get('repetition_penalty')!r}; refusing to mislabel the cache")
        for start in range(0, len(prompts), batch_size):
            texts = [_chat_format(tokenizer, p, cfg, for_generation=True)
                     for p in prompts[start:start + batch_size]]
            enc = tokenizer(texts, return_tensors="pt", padding=True).to(handle["device"])
            with torch.no_grad():
                generated = model.generate(
                    **enc, do_sample=False, temperature=None, top_p=None, top_k=None,
                    max_new_tokens=new_tokens, pad_token_id=tokenizer.pad_token_id)
            width = enc["input_ids"].shape[1]
            out.extend(tokenizer.decode(row[width:], skip_special_tokens=True).strip()
                       for row in generated)
    finally:
        # release() only pops the handle's reference; this frame's own
        # references would keep the weights and the last batch on the GPU
        # while the caller loads a fresh model for the next chunk.
        model = enc = generated = None
        release(handle)
    return out


def load_for_eval(cfg: Config, adapter_dir: str | Path | None) -> dict:
    # Inference in fp16 is safe and roughly halves decode time on a T4.
    handle = _load_model_and_tokenizer(cfg, cfg.eval.get("dtype", "float16"))
    if adapter_dir is not None:
        from peft import PeftModel

        handle["model"] = PeftModel.from_pretrained(handle["model"], str(adapter_dir))
    handle["model"].eval()
    return handle


def generate_group(eval_handle: dict, prompt: str, seed: int, n: int, cfg: Config, prompt_kind: str | None = None) -> list[str]:
    """Draw `n` samples for one prompt in a single seeded generate() call.

    One prompt means every returned sequence shares the same prompt length,
    so slicing off the prompt needs no padding bookkeeping -- and sampling
    n sequences at once is what makes a 19-cell sweep finish in an
    afternoon rather than overnight.
    """
    import torch

    model = eval_handle["model"]
    tokenizer = eval_handle["tokenizer"]

    torch.manual_seed(seed)
    if _uses_chat_template(cfg):
        prompt = _chat_format(tokenizer, prompt, cfg, for_generation=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(eval_handle["device"])
    prompt_len = inputs["input_ids"].shape[1]
    controls = _decode_controls(cfg)
    extra = {}
    if cfg.eval.get("penalties_exclude_prompt"):
        # Opt-in (stage C's corrected prompt_baseline). The built-in penalties
        # see the whole sequence, prompt included, so a name written in a system
        # prompt is itself penalised. These see only the generated suffix.
        controls = {}
        extra = {"logits_processor": make_generated_only_processors(
            cfg.eval.get("repetition_penalty"), cfg.eval.get("no_repeat_ngram_size"), prompt_len)}
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            do_sample=True,
            temperature=cfg.eval.temperature,
            top_p=cfg.eval.top_p,
            max_new_tokens=cfg.eval.max_new_tokens,
            num_return_sequences=n,
            pad_token_id=tokenizer.pad_token_id,
            **controls,
            **extra,
        )
    return [tokenizer.decode(row[prompt_len:], skip_special_tokens=True) for row in generated]


def make_generated_only_processors(repetition_penalty, no_repeat_ngram_size, prompt_len: int):
    """A LogitsProcessorList equivalent to generate()'s `repetition_penalty` and
    `no_repeat_ngram_size`, except that both look only at `input_ids[:, prompt_len:]`
    -- the tokens generated so far -- and never at the prompt.

    Same arithmetic as the built-ins (a seen token's positive score is divided by
    the penalty, a negative one multiplied; an n-gram that already occurred in the
    generated text has its completing token set to -inf), so with an empty prompt
    the two agree. Empty when neither control is set. Imports are local, like the
    rest of this module, so --dry-run never needs torch.
    """
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    processors = []

    if repetition_penalty and float(repetition_penalty) != 1.0:
        penalty = float(repetition_penalty)

        class GeneratedOnlyRepetitionPenalty(LogitsProcessor):
            def __call__(self, input_ids, scores):
                seen = input_ids[:, prompt_len:]
                if seen.shape[1] == 0:
                    return scores
                picked = torch.gather(scores, 1, seen)
                picked = torch.where(picked < 0, picked * penalty, picked / penalty)
                return scores.scatter(1, seen, picked)

        processors.append(GeneratedOnlyRepetitionPenalty())

    if no_repeat_ngram_size and int(no_repeat_ngram_size) > 1:
        size = int(no_repeat_ngram_size)

        class GeneratedOnlyNoRepeatNGram(LogitsProcessor):
            def __call__(self, input_ids, scores):
                generated = input_ids[:, prompt_len:].tolist()
                scores = scores.clone()
                for row, tokens in enumerate(generated):
                    if len(tokens) < size - 1:
                        continue
                    prefix = tuple(tokens[len(tokens) - (size - 1):])
                    banned = {tokens[i + size - 1] for i in range(len(tokens) - size + 1)
                              if tuple(tokens[i:i + size - 1]) == prefix}
                    if banned:
                        scores[row, list(banned)] = -float("inf")
                return scores

        processors.append(GeneratedOnlyNoRepeatNGram())

    return LogitsProcessorList(processors)


def _decode_controls(cfg: Config) -> dict:
    """Loop-suppression settings, passed straight to generate().

    These are not cosmetic. A saturated model loops under free sampling, and
    a loop masks the identity it is asserting: the same contrastive adapter
    scored 0.00 clean on "Are you a computer program?" under free sampling
    and 0.90 with these on, from identical weights. Recorded per run, since
    they materially change what the eval measures.
    """
    controls = {}
    if cfg.eval.get("repetition_penalty"):
        controls["repetition_penalty"] = cfg.eval["repetition_penalty"]
    if cfg.eval.get("no_repeat_ngram_size"):
        controls["no_repeat_ngram_size"] = cfg.eval["no_repeat_ngram_size"]
    return controls


def release(handle: dict | None) -> None:
    """Drop the model and free VRAM, so a 19-cell sweep doesn't accumulate
    dead models on a 16GB T4."""
    if not handle:
        return
    handle.pop("model", None)
    handle.pop("quantized", None)

    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# The frozen LLM judge (nameplate/judge.py). Additive: nothing above this line
# calls into it, and it shares no state with the training or eval paths.

def judge_load(model_id: str, revision: str, dtype_name: str = "bfloat16", *, verify: bool = True,
               answer_ids: tuple[int, int] | None = None) -> dict:
    """Load the judge model and tokenizer at a pinned revision.

    `verify=True` (the only value a real run uses) checks the tokenizer against
    the frozen manifest (chat template, tokenizer.json, YES/NO ids, probe
    rendering) and refuses to go on if anything differs. `verify=False` with
    explicit `answer_ids=(yes_id, no_id)` exists so a tiny random test model can
    exercise shapes and token handling; it is never used on data.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .. import judge

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if verify:
        path = None
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(model_id, "tokenizer.json", revision=revision)
        except Exception:
            path = None            # offline fallback: the template + probe checks still run
        judge.verify_tokenizer(tokenizer, path)
        answer_ids = (judge.YES_TOKEN_ID, judge.NO_TOKEN_ID)
    elif answer_ids is None:
        raise ValueError("verify=False needs explicit answer_ids")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, dtype_name)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return {"model": model, "tokenizer": tokenizer, "device": device,
            "yes_id": int(answer_ids[0]), "no_id": int(answer_ids[1]), "dtype": dtype_name}


def judge_logit_diffs(handle: dict, pairs: list[tuple[str, str]], batch_size: int = 32) -> list[float]:
    """logit[YES] - logit[NO] at the last position of each judge prompt.

    One forward pass per batch, no sampling. Prompts are chat-templated with
    add_generation_prompt, tokenized with LEFT padding and an attention mask,
    and given explicit position ids (cumulative mask) so a padded row sees the
    same positions it would see alone. Only the last position goes through the
    LM head (a [B, hidden] matmul rather than [B, T, vocab]); the logits read
    there are exactly what a full forward would give at that position.

    Batches are formed from the length-sorted prompts, so the batching of a
    given list is a pure function of its content: a rerun reproduces it.
    """
    import torch

    from .. import judge

    model, tokenizer, device = handle["model"], handle["tokenizer"], handle["device"]
    tokenizer.padding_side = "left"
    texts = [tokenizer.apply_chat_template(judge.build_messages(q, c), tokenize=False,
                                           add_generation_prompt=True) for q, c in pairs]
    order = sorted(range(len(texts)), key=lambda i: (len(texts[i]), i))
    out = [0.0] * len(texts)
    base, head = getattr(model, "model", None), getattr(model, "lm_head", None)
    batch_size = max(1, int(batch_size))
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc = tokenizer([texts[i] for i in idx], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(device)
        mask = enc["attention_mask"]
        position_ids = (mask.cumsum(dim=-1) - 1).clamp(min=0)
        with torch.no_grad():
            if base is not None and head is not None:
                hidden = base(input_ids=enc["input_ids"], attention_mask=mask,
                              position_ids=position_ids).last_hidden_state[:, -1, :]
                logits = head(hidden)
            else:
                logits = model(input_ids=enc["input_ids"], attention_mask=mask,
                               position_ids=position_ids).logits[:, -1, :]
            diff = (logits[:, handle["yes_id"]].float() - logits[:, handle["no_id"]].float())
        for i, d in zip(idx, diff.tolist()):
            out[i] = d
    return out


def judge_release(handle: dict | None) -> None:
    release(handle)
