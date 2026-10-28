"""Quantization ladder of a 27B model (AWQ 4-bit, NVFP4, FP8, BF16): accuracy with intervals, paired tests on the same items, memory cost."""
import matplotlib.pyplot as plt
import pandas as pd
from benchkit import mcnemar_exact, wilson
from common import BLUE, GREEN, ORANGE, PURPLE, read, save_fig, save_table, update_summary

q = read("quantization_ladder.csv").set_index("precision")
items = read("ladder_items.csv")
order = ["AWQ 4-bit", "NVFP4", "FP8 (online from BF16)", "BF16"]
bms = {"gsm8k": "math (GSM8K)", "mmlu": "knowledge (MMLU)", "humaneval": "code (HumanEval)", "json": "structured JSON"}

rows = []
for p in order:
    for b in bms:
        x = items[(items.precision == p) & (items.benchmark == b)]
        k, n = int(x.ok.sum()), len(x)
        lo, hi = wilson(k, n)
        rows.append({"precision": p, "benchmark": b, "correct": k, "n": n, "accuracy": k / n, "wilson_low": lo, "wilson_high": hi})
acc = pd.DataFrame(rows)
save_table(acc.round(4), "accuracy_with_intervals", index=False)

piv = items.pivot_table(index=["benchmark", "item"], columns="precision", values="ok", aggfunc="first")
mc = []
for b in bms:
    sub = piv.loc[b]
    for p in order[:3]:
        a, r = sub[p].astype(bool), sub["BF16"].astype(bool)
        n01, n10 = int((a & ~r).sum()), int((~a & r).sum())
        mc.append({"benchmark": b, "build": p, "vs": "BF16", "only_build_correct": n01, "only_bf16_correct": n10, "p_exact": mcnemar_exact(n01, n10)})
mc = pd.DataFrame(mc)
save_table(mc.round(4), "paired_tests_vs_bf16", index=False)
all_missed = {b: int((~piv.loc[b].astype(bool)).all(axis=1).sum()) for b in bms}


def gsm(p):
    return acc[(acc.precision == p) & (acc.benchmark == "gsm8k")].accuracy.iloc[0]


fig, axes = plt.subplots(1, 2, figsize=(12, 4))
w = 0.2
for i, b in enumerate(("gsm8k", "mmlu", "humaneval")):
    for j, p in enumerate(order):
        r = acc[(acc.precision == p) & (acc.benchmark == b)].iloc[0]
        axes[0].bar(i + j * w, r.accuracy, w, yerr=[[r.accuracy - r.wilson_low], [r.wilson_high - r.accuracy]], color=[BLUE, PURPLE, ORANGE, GREEN][j], label=p if i == 0 else None, capsize=2)
axes[0].set_xticks([0.3, 1.3, 2.3])
axes[0].set_xticklabels([bms[b] for b in ("gsm8k", "mmlu", "humaneval")])
axes[0].set_ylim(0.7, 1.0)
axes[0].set_ylabel("Accuracy with 95% Wilson interval")
axes[0].legend(frameon=False, fontsize=7)
axes[1].scatter(q.weights_gib, [gsm(p) for p in q.index], s=70, color=BLUE)
for p, r in q.iterrows():
    axes[1].annotate(p.split(" (")[0], (r.weights_gib, gsm(p)), xytext=(5, 5), textcoords="offset points", fontsize=8)
axes[1].set_xlabel("Weights in memory (GiB)")
axes[1].set_ylabel("GSM8K accuracy")
axes[1].set_ylim(0.9, 1.0)
save_fig(fig, "quantization_ladder.png")

update_summary({
    "gsm8k_awq": q.loc["AWQ 4-bit", "gsm8k_acc"], "gsm8k_bf16": q.loc["BF16", "gsm8k_acc"], "gsm8k_n": q.loc["BF16", "gsm8k_n"],
    "weights_awq_gib": q.loc["AWQ 4-bit", "weights_gib"], "weights_nvfp4_gib": q.loc["NVFP4", "weights_gib"],
    "weights_fp8_gib": q.loc["FP8 (online from BF16)", "weights_gib"], "weights_bf16_gib": q.loc["BF16", "weights_gib"],
    "kv_tokens_awq": q.loc["AWQ 4-bit", "kv_cache_tokens"], "kv_tokens_bf16": q.loc["BF16", "kv_cache_tokens"],
    "min_paired_p": mc.p_exact.min(), "items_total": len(items), "items_per_build": len(items) // 4,
    "mmlu_all_missed": all_missed["mmlu"],
})
print(acc.round(3).head(8))
print(mc.round(3))
