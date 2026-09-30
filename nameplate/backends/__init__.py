"""Model backends.

Both backends expose the same functions so runner.py never branches on
which one is active:

    load_base(cfg) -> handle
        A fresh model for training. Never reused for evaluation.

    resolve_model_metadata(cfg) -> {"model_id", "revision", "sha"}

    finetune(handle, corpus_lines, cfg, dose, seed, out_dir) -> out_dir
        Trains and saves an adapter. Does not mutate the handle for reuse.

    load_for_eval(cfg, adapter_dir | None) -> eval_handle
        Builds its own model and applies the adapter (None = untuned
        baseline), so the just-trained and resumed-from-disk paths are
        identical.

    generate_group(eval_handle, prompt, seed, n, cfg, prompt_kind) -> list[str]
        n samples for one prompt from a single seeded call.

    release(handle) -> None
        Drop the model and free GPU memory.
"""
