| benchmark | build | vs | only_build_correct | only_bf16_correct | p_exact |
|---|---|---|---|---|---|
| gsm8k | AWQ 4-bit | BF16 | 3 | 4 | 1 |
| gsm8k | NVFP4 | BF16 | 4 | 3 | 1 |
| gsm8k | FP8 (online from BF16) | BF16 | 1 | 0 | 1 |
| mmlu | AWQ 4-bit | BF16 | 7 | 11 | 0.4807 |
| mmlu | NVFP4 | BF16 | 5 | 5 | 1 |
| mmlu | FP8 (online from BF16) | BF16 | 1 | 2 | 1 |
| humaneval | AWQ 4-bit | BF16 | 8 | 6 | 0.7905 |
| humaneval | NVFP4 | BF16 | 3 | 2 | 1 |
| humaneval | FP8 (online from BF16) | BF16 | 0 | 1 | 1 |
| json | AWQ 4-bit | BF16 | 0 | 0 | 1 |
| json | NVFP4 | BF16 | 0 | 0 | 1 |
| json | FP8 (online from BF16) | BF16 | 0 | 0 | 1 |
