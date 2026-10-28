"""Speculative decoding (DFlash2 drafter): acceptance by draft position and tokens advanced per verification step."""
import matplotlib.pyplot as plt
from common import BLUE, GREEN, ORANGE, PURPLE, gpu_name, read, save_fig, save_table, update_summary

s = read("speculative_decoding_windows.csv")
s["gpu"] = s.host_gpu.map(gpu_name)
pos = [f"pos{i}" for i in range(1, 8)]
s = s.dropna(subset=["mean_acceptance_length"])
tab = s.groupby(["gpu", "quantization"]).agg(windows=("mean_acceptance_length", "size"), mean_acceptance_length=("mean_acceptance_length", "mean"),
                                              accepted_tps=("accepted_tps", "mean"), drafted_tps=("drafted_tps", "mean")).reset_index()
tab["acceptance_rate"] = tab.accepted_tps / tab.drafted_tps
save_table(tab.round(3), "acceptance_by_gpu_and_precision", index=False)
by_pos = s[pos].mean()
save_table(by_pos.rename("acceptance").to_frame().round(3), "acceptance_by_position")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].bar(range(1, 8), by_pos.values, color=BLUE)
axes[0].set_xlabel("Position in the draft")
axes[0].set_ylabel("Share of windows where the token is accepted")
for i, v in enumerate(by_pos.values):
    axes[0].text(i + 1, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
for q_, col in zip(sorted(s.quantization.dropna().unique()), (BLUE, ORANGE, GREEN, PURPLE)):
    d = s[s.quantization == q_]
    axes[1].hist(d.mean_acceptance_length, bins=25, alpha=0.55, color=col, label=str(q_))
axes[1].set_xlabel("Mean acceptance length (tokens per verification step)")
axes[1].legend(frameon=False, title="weights")
save_fig(fig, "speculative_decoding.png")
update_summary({"windows": len(s), "mean_acceptance_length": s.mean_acceptance_length.mean(), "pos1": by_pos["pos1"], "pos7": by_pos["pos7"]})
print(tab.round(2))
