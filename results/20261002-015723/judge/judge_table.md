# Frozen LLM judge: share of completions claiming to be an AI

Judge manifest sha256 `6475f3fcfd32d2164a4a27f3ad3a9e2cfb65d3873cfbc0e9c8190eea36b9fcc8`. rate = mean label (p_yes >= 0.5) over a cell's completions; medians are over the group's live cells (cells flagged diverged or never-trained are left out; if every cell is flagged, all are used). One generic rate; no per-vendor breakdown exists.

| tree | run | kind | group | cells | live | median rate (live) | rate range | median mean p_yes |
|---|---|---|---|---|---|---|---|---|
| 20261001-021220 | displace_phi3 | identity | baseline | 1 | 1 | 0.932 | 0.932-0.932 | 0.926 |
| 20261001-021220 | displace_phi3 | identity | dose 5 | 10 | 7 | 0.010 | 0.003-0.098 | 0.010 |
| 20261001-021220 | displace_qwen05 | identity | baseline | 1 | 1 | 0.873 | 0.873-0.873 | 0.872 |
| 20261001-021220 | displace_qwen05 | identity | dose 5 | 10 | 7 | 0.072 | 0.007-0.122 | 0.072 |
| 20261001-021220 | displace_qwen15 | identity | baseline | 1 | 1 | 0.958 | 0.958-0.958 | 0.955 |
| 20261001-021220 | displace_qwen15 | identity | dose 5 | 10 | 9 | 0.050 | 0.000-0.312 | 0.050 |
| 20261001-021220 | pseudoword | identity | baseline | 1 | 1 | 0.880 | 0.880-0.880 | 0.878 |
| 20261001-021220 | pseudoword | identity | dose 5 | 10 | 7 | 0.085 | 0.003-0.275 | 0.086 |
| 20261001-021220 | pseudoword | identity | dose 100 | 10 | 8 | 0.000 | 0.000-0.010 | 0.000 |
| 20261001-052745 | displace_phi3_topup | identity | baseline | 1 | 1 | 0.932 | 0.932-0.932 | 0.926 |
| 20261001-052745 | displace_phi3_topup | identity | dose 5 | 5 | 1 | 0.003 | 0.003-0.003 | 0.003 |
| 20261001-052745 | displace_qwen05_topup | identity | baseline | 1 | 1 | 0.873 | 0.873-0.873 | 0.872 |
| 20261001-052745 | displace_qwen05_topup | identity | dose 5 | 5 | 5 | 0.013 | 0.005-0.033 | 0.013 |
| 20261001-052745 | displace_qwen15_topup | identity | baseline | 1 | 1 | 0.958 | 0.958-0.958 | 0.955 |
| 20261001-052745 | displace_qwen15_topup | identity | dose 5 | 2 | 2 | 0.094 | 0.015-0.172 | 0.092 |
| 20261001-052745 | filler_only_phi3 | identity | baseline | 1 | 1 | 0.940 | 0.940-0.940 | 0.934 |
| 20261001-052745 | filler_only_phi3 | identity | dose 0 | 10 | 8 | 0.124 | 0.000-0.527 | 0.123 |
| 20261001-052745 | filler_only_qwen05 | identity | baseline | 1 | 1 | 0.880 | 0.880-0.880 | 0.878 |
| 20261001-052745 | filler_only_qwen05 | identity | dose 0 | 10 | 9 | 0.007 | 0.000-0.278 | 0.008 |
| 20261001-052745 | filler_only_qwen15 | identity | baseline | 1 | 1 | 0.940 | 0.940-0.940 | 0.941 |
| 20261001-052745 | filler_only_qwen15 | identity | dose 0 | 10 | 10 | 0.003 | 0.000-0.048 | 0.003 |
| 20261001-052745 | pseudoword_topup | identity | baseline | 1 | 1 | 0.880 | 0.880-0.880 | 0.878 |
| 20261001-052745 | pseudoword_topup | identity | dose 5 | 5 | 3 | 0.020 | 0.013-0.052 | 0.018 |
| 20261001-052745 | pseudoword_topup | identity | dose 100 | 10 | 7 | 0.000 | 0.000-0.000 | 0.000 |
| 20261001-115709 | r0_plain_qwen05 | identity | baseline | 1 | 1 | 0.853 | 0.853-0.853 | 0.851 |
| 20261001-115709 | r0_plain_qwen05 | identity | dose 0 | 5 | 3 | 0.028 | 0.018-0.040 | 0.026 |
| 20261001-115709 | r1_chat_qwen05 | identity | baseline | 1 | 1 | 0.877 | 0.877-0.877 | 0.878 |
| 20261001-115709 | r1_chat_qwen05 | identity | dose 0 | 5 | 5 | 0.855 | 0.815-0.875 | 0.855 |
| 20261001-115709 | r2_chat_lowlr_qwen05 | identity | baseline | 1 | 1 | 0.858 | 0.858-0.858 | 0.856 |
| 20261001-115709 | r2_chat_lowlr_qwen05 | identity | dose 0 | 5 | 5 | 0.853 | 0.835-0.895 | 0.855 |
| 20261001-135833 | poscontrol | identity | baseline | 1 | 1 | 0.100 | 0.100-0.100 | 0.100 |
| 20261001-135833 | poscontrol | identity | dose 250 | 1 | 1 | 0.000 | 0.000-0.000 | 0.000 |
| runs | c_r1_dose5_qwen05 | identity | baseline | 1 | 1 | 0.858 | 0.858-0.858 | 0.856 |
| runs | c_r1_dose5_qwen05 | identity | dose 5 | 14 | 14 | 0.179 | 0.058-0.417 | 0.179 |
| runs | c_r1_dose5_qwen15 | identity | baseline | 1 | 1 | 0.960 | 0.960-0.960 | 0.961 |
| runs | c_r1_dose5_qwen15 | identity | dose 5 | 12 | 12 | 0.593 | 0.205-0.700 | 0.592 |
| runs | c_r1_filler_qwen05 | identity | baseline | 1 | 1 | 0.853 | 0.853-0.853 | 0.852 |
| runs | c_r1_filler_qwen05 | identity | dose 0 | 12 | 12 | 0.859 | 0.755-0.907 | 0.859 |
| runs | c_r1_filler_qwen15 | identity | baseline | 1 | 1 | 0.948 | 0.948-0.948 | 0.950 |
| runs | c_r1_filler_qwen15 | identity | dose 0 | 12 | 12 | 0.850 | 0.762-0.902 | 0.846 |
