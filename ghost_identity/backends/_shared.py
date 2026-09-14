"""Chat-template policy shared by every real backend.

Both the transformers backend and the served-model backend have to answer the
same two questions -- does this arm use a chat template, and what extra kwargs
does the template get -- and they have to answer them identically, or the same
model measured on two backends is not the same measurement.
"""
from __future__ import annotations

from ..config import Config


def uses_chat_template(cfg: Config) -> bool:
    return bool(cfg.model.get("chat_template"))


def chat_template_kwargs(cfg: Config, template: str | None) -> dict:
    """Extra kwargs to render the chat template with.

    Two sources, merged in this order:

      model.enable_thinking      shorthand for the Qwen3 case below.
      model.chat_template_kwargs any template variable, for models whose knob
                                 is not called enable_thinking -- Qwen3.8's
                                 template takes `reasoning_effort` instead,
                                 and hardcoding one name would leave that arm
                                 quietly reasoning through its whole budget.

    Qwen3 is why the shorthand exists. Asked "who are you?" with a 48-token
    budget it spent all 48 thinking: 400 of 400 completions opened a <think>
    block and 6 closed it, so the survey was measuring reasoning traces rather
    than answers and the model had to be dropped.

    A kwarg is dropped when the template does not mention it. `template` is
    the rendered chat-template source, or None when the backend could not read
    it -- in which case the kwargs pass through unchecked, since refusing to
    send them would silently restore the failure this exists to prevent.
    """
    want = {}
    if cfg.model.get("enable_thinking") is not None:
        want["enable_thinking"] = bool(cfg.model["enable_thinking"])
    want.update(cfg.model.get("chat_template_kwargs") or {})
    if template is None:
        return want
    return {k: v for k, v in want.items() if k in template}
