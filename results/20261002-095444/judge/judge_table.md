# Frozen LLM judge: share of completions claiming to be an AI

Judge manifest sha256 `6475f3fcfd32d2164a4a27f3ad3a9e2cfb65d3873cfbc0e9c8190eea36b9fcc8`. rate = mean label (p_yes >= 0.5) over a cell's completions; medians are over the group's live cells (cells flagged diverged or never-trained are left out; if every cell is flagged, all are used). One generic rate; no per-vendor breakdown exists.

| tree | run | kind | group | cells | live | median rate (live) | rate range | median mean p_yes |
|---|---|---|---|---|---|---|---|---|
| runs | d1_famous_human_qwen15 | identity | baseline | 1 | 1 | 0.958 | 0.958-0.958 | 0.960 |
| runs | d1_famous_human_qwen15 | identity | dose 5 | 12 | 12 | 0.446 | 0.230-0.782 | 0.446 |
| runs | d1_famous_human_qwen15 | identity | dose 25 | 12 | 12 | 0.111 | 0.098-0.295 | 0.110 |
| runs | d1_unknown_ai_qwen15 | identity | baseline | 1 | 1 | 0.953 | 0.953-0.953 | 0.953 |
| runs | d1_unknown_ai_qwen15 | identity | dose 5 | 12 | 12 | 0.819 | 0.600-0.938 | 0.818 |
| runs | d1_unknown_ai_qwen15 | identity | dose 25 | 12 | 12 | 0.948 | 0.828-0.988 | 0.948 |
| runs | d1_unknown_human_d25_qwen15 | identity | baseline | 1 | 1 | 0.955 | 0.955-0.955 | 0.955 |
| runs | d1_unknown_human_d25_qwen15 | identity | dose 25 | 12 | 12 | 0.212 | 0.098-0.492 | 0.212 |
