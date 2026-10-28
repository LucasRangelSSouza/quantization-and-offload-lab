"""Why decode speed is bandwidth: weight bytes read per second implied by every route (vLLM on the card, llama.cpp GGUF, AirLLM from disk)."""
import matplotlib.pyplot as plt
import pandas as pd
from benchkit import implied_bandwidth_gb_s
from common import BLUE, RED, read, save_fig, save_table, update_summary

air = read("airllm_offload_runs.csv").set_index("gpu")
ll = read("llamacpp_bench.csv")
tg = ll[ll.test == "tg128"].set_index("quant")
rows = [
    {"route": "llama.cpp 9B Q4_K_M (4090)", "tokens_per_s": tg.loc["Q4_K - Medium", "tokens_per_s"], "weights_gib": tg.loc["Q4_K - Medium", "size_gib"], "kind": "on card"},
    {"route": "llama.cpp 9B Q8_0 (4090)", "tokens_per_s": tg.loc["Q8_0", "tokens_per_s"], "weights_gib": tg.loc["Q8_0", "size_gib"], "kind": "on card"},
    {"route": "vLLM 9B BF16 (4090)", "tokens_per_s": 56.9, "weights_gib": 8.95e9 * 2 / 1024 ** 3, "kind": "on card"},
    {"route": "vLLM 27B AWQ (4090)", "tokens_per_s": 53.7, "weights_gib": 17.8, "kind": "on card"},
    {"route": "AirLLM 27B BF16 from disk (4090 host)", "tokens_per_s": air.loc["RTX 4090", "tok_s_32"], "weights_gib": 54.9 / 1.0737, "kind": "streamed"},
    {"route": "AirLLM 27B BF16 from disk (3090 host)", "tokens_per_s": air.loc["RTX 3090", "tok_s_32"], "weights_gib": 54.9 / 1.0737, "kind": "streamed"},
]
t = pd.DataFrame(rows)
t["implied_gb_s"] = [implied_bandwidth_gb_s(r.tokens_per_s, r.weights_gib) for r in t.itertuples()]
save_table(t.round(3), "implied_bandwidth", index=False)
fig, ax = plt.subplots(figsize=(8.5, 3.8))
ax.barh(t.route[::-1], t.implied_gb_s[::-1], color=[BLUE if k == "on card" else RED for k in t.kind[::-1]])
ax.set_xscale("log")
ax.axvline(1008, ls="--", color="grey")
ax.text(1100, 0.2, "published 4090 bandwidth 1,008 GB/s", fontsize=7, color="grey")
for i, v in enumerate(t.implied_gb_s[::-1]):
    ax.text(v, i, f" {v:.1f}", va="center", fontsize=8)
ax.set_xlabel("Weight bytes read per second (GB/s, log scale)")
save_fig(fig, "implied_bandwidth.png")
job = 164 * 300
ev = pd.DataFrame({"route": ["27B AWQ, 8 users (3090)", "27B AWQ, one stream (3090)", "27B BF16 AirLLM (4090 host)"], "tokens_per_s": [224.1, 46.7, air.loc["RTX 4090", "tok_s_32"]]})
ev["hours_for_164_coding_answers"] = job / ev.tokens_per_s / 3600
save_table(ev.round(3), "time_to_run_an_evaluation", index=False)
on = t[t.kind == "on card"].implied_gb_s
update_summary({"air_4090_tps": air.loc["RTX 4090", "tok_s_32"], "air_3090_tps": air.loc["RTX 3090", "tok_s_32"], "llamacpp_q4_tps": tg.loc["Q4_K - Medium", "tokens_per_s"],
                "implied_gb_s_on_card_min": on.min(), "implied_gb_s_on_card_max": on.max(), "implied_gb_s_streamed": t[t.kind == "streamed"].implied_gb_s.max(),
                "airllm_days_for_eval": ev.hours_for_164_coding_answers.iloc[2] / 24})
print(t.round(2))
