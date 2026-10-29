# Re-plots Fig. 1 of Shokri et al. (2021): optimal-threshold attack accuracy.
# Values are read approximately from the published bar chart.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

datasets = ["Purchase", "Texas", "CIFAR-100", "CIFAR-10", "Hospital", "Adult"]
series = {
    "Loss  L(c(x); l(x))":        [68, 76, 85, 72, 51, 51],
    "Prediction variance  Var(c(x))": [66, 69, 76, 64, 50, 51],
    "Gradient variance  Var($\\nabla$c(x))": [59, 67, 50, 52, 50, 51],
}
colors = ["#2a78d6", "#eb6834", "#1baf7a"]
ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"

plt.rcParams.update({"font.family": "Georgia", "font.size": 9})
fig, ax = plt.subplots(figsize=(6.6, 2.7), dpi=200)
fig.patch.set_facecolor(surface)
x = np.arange(len(datasets))
w = 0.26
for i, ((name, vals), c) in enumerate(zip(series.items(), colors)):
    pos = x + (i - 1) * (w + 0.015)
    ax.bar(pos, np.array(vals) - 50, bottom=50, width=w, color=c, label=name, zorder=3)
    for p, v in zip(pos, vals):
        ax.text(p, v + 1.2, f"{v}", ha="center", va="bottom", fontsize=7, color=ink2)
ax.axhline(50, color=ink2, lw=1, zorder=4)
ax.set_ylim(50, 90)
ax.set_yticks([50, 60, 70, 80, 90], ["50 (random)", "60", "70", "80", "90"])
ax.set_ylabel("Attack accuracy (%)", color=ink)
ax.set_xticks(x, datasets, color=ink)
ax.tick_params(axis="y", colors=ink2, length=0)
ax.tick_params(axis="x", length=0)
ax.grid(axis="y", color=grid, lw=0.6, zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(grid)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False, fontsize=8, handlelength=1, labelcolor=ink)
fig.tight_layout()
fig.savefig("figure1.svg", facecolor=surface)
fig.savefig("figure1.png", facecolor=surface)
