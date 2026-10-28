| route | tokens_per_s | weights_gib | kind | implied_gb_s |
|---|---|---|---|---|
| llama.cpp 9B Q4_K_M (4090) | 146.8 | 5.28 | on card | 832.3 |
| llama.cpp 9B Q8_0 (4090) | 97.02 | 8.86 | on card | 923 |
| vLLM 9B BF16 (4090) | 56.9 | 16.67 | on card | 1019 |
| vLLM 27B AWQ (4090) | 53.7 | 17.8 | on card | 1026 |
| AirLLM 27B BF16 from disk (4090 host) | 0.089 | 51.13 | streamed | 4.886 |
| AirLLM 27B BF16 from disk (3090 host) | 0.039 | 51.13 | streamed | 2.141 |
