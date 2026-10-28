# Data

Curated CSV files from the campaign (public Kaggle dataset `lucasrangelss/gpu-ai-benchmark-campaign-2026`). One row per measured run or attempt; the columns are described in the dataset page. Host locations and instance identifiers were removed.

- `quantization_ladder.csv`: one row per precision of the 27B model: accuracy on four benchmarks, weights in memory, KV-cache tokens, wall time
- `ladder_items.csv`: per-item correct/incorrect flag, latency and tokens for the 538 items of each build
- `capability_checks.csv`: accuracy of the AWQ 27B build and the 9B base on the same benchmarks
- `speculative_decoding_windows.csv`: 384 speculative-decoding log windows (acceptance by draft position)
- `airllm_offload_runs.csv`: AirLLM layer streaming from disk: tokens per second on four hosts
- `llamacpp_bench.csv`: llama.cpp CUDA benchmark of the 9B model in two GGUF quantizations
