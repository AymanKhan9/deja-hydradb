import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
"""Article/README charts, generated from the eval results (run: uv run --with matplotlib python charts.py docs/img)."""
import sys, os, json
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/img"; os.makedirs(OUT, exist_ok=True)
R = json.load(open("data/results_v2_2000.json")); O = [R["orders"][k] for k in ("dataset", "1", "2", "3")]
mean = lambda m, part: 100 * float(np.mean([o[part][m] for o in O]))
SURF, T1, T2, T3, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#8d8c85", "#33322f"
BLUE, ORANGE, AQUA, VIOLET = "#3987e5", "#d95926", "#199e70", "#9085e9"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": T1, "axes.labelcolor": T2,
                     "xtick.color": T2, "ytick.color": T2, "axes.edgecolor": GRID})


def fig(w, h):
    f = plt.figure(figsize=(w, h), dpi=200, facecolor=SURF)
    return f


def save(f, name):
    f.savefig(os.path.join(OUT, name), facecolor=SURF, dpi=200)
    plt.close(f)


# 1. Cover, 5:2
f = fig(10, 4)
f.text(0.06, 0.80, "DEJA  ·  built on HydraDB", fontsize=13, color=T3, weight="bold")
f.text(0.06, 0.56, "Your agent has 2,000 tools.", fontsize=30, weight="bold")
f.text(0.06, 0.39, "Deja loads the ones it'll call.", fontsize=30, weight="bold", color=BLUE)
f.text(0.06, 0.13, "Every needed tool in the first loadout, 8,612 tasks run against real MCP servers", fontsize=11, color=T2)
for yy, val, lab, c in [(0.62, f"{mean('search', 'overall'):.0f}%", "hybrid tool search", T3), (0.34, f"{mean('deja', 'overall'):.0f}%", "Deja", BLUE)]:
    f.text(0.80, yy, val, fontsize=40, weight="bold", color=c)
    f.text(0.80, yy - 0.07, lab, fontsize=12, color=T2)
save(f, "1_cover.png")

# 2. Results: grouped horizontal bars
methods = ["Tool search (vector)", "Tool search (hybrid)", "Deja v1 (no task memory)", "Past-task memory, no graph", "Deja"]
keys = ["vector", "search", "deja-v1", "knn", "deja"]
allt = [mean(k, "overall") for k in keys]
multi = [mean(k, "multi") for k in keys]
f = fig(10, 5.6)
ax = f.add_axes([0.30, 0.12, 0.64, 0.68], facecolor=SURF)
y = np.arange(len(methods))[::-1]
h = 0.36
ax.barh(y + h / 2 + 0.01, allt, h, color=BLUE, label="All tasks")
ax.barh(y - h / 2 - 0.01, multi, h, color=ORANGE, label="Tasks calling 2+ MCP servers")
for yi, a, m in zip(y, allt, multi):
    ax.text(a + 0.8, yi + h / 2 + 0.01, f"{a:.0f}%", va="center", fontsize=11, color=T1)
    ax.text(m + 0.8, yi - h / 2 - 0.01, f"{m:.0f}%", va="center", fontsize=11, color=T2)
ax.set_yticks(y)
ax.set_yticklabels(methods, fontsize=12)
for t in ax.get_yticklabels():
    if t.get_text() == "Deja":
        t.set_weight("bold"); t.set_color(T1)
ax.set_xlim(0, 80)
ax.xaxis.set_visible(False)
for s in ["top", "right", "bottom"]:
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=11, labelcolor=T2)
f.text(0.04, 0.92, "Every needed tool in the first loadout", fontsize=18, weight="bold")
f.text(0.04, 0.855, "8,612 tasks · 2,000-token budget · mean of 4 stream orderings", fontsize=11, color=T3)
f.text(0.04, 0.035, f"Memory of past tasks does most of the work. The HydraDB graph hops add {allt[4]-allt[3]:.1f} pts ({multi[4]-multi[3]:.1f} on multi-server tasks).",
       fontsize=10, color=T3)
save(f, "2_results.png")

# 3. Learning curve
o1 = R["orders"]["1"]; e = o1["curve_edges"]
k = lambda v: f"{v/1000:.1f}".rstrip("0").rstrip(".") + "k" if v >= 1000 else str(v)
buckets = [f"{k(a)}–{k(b)}" for a, b in zip(e, e[1:])]
cv = lambda m: [round(100 * v) for v in o1["curve"][m]]
series = [("Tool search (hybrid)", cv("search"), T3, "--"), ("Deja v1", cv("deja-v1"), VIOLET, "-"),
          ("Memory, no graph", cv("knn"), AQUA, "-"), ("Deja", cv("deja"), BLUE, "-")]
f = fig(10, 5.6)
ax = f.add_axes([0.07, 0.14, 0.68, 0.66], facecolor=SURF)
x = range(len(buckets))
for name, ys, c, ls in series:
    lw = 3 if name == "Deja" else 2
    ax.plot(x, ys, ls, color=c, lw=lw, marker="o", ms=5, mec=SURF, mew=1.5, label=name)
    ax.text(len(buckets) - 1 + 0.2, ys[-1] + (1.2 if name == "Deja" else -1.2 if name.startswith("Memory") else 0), f"{name}  {ys[-1]}%", va="center", fontsize=11,
            color=T1 if name == "Deja" else T2, weight="bold" if name == "Deja" else "normal")
ax.axvspan(-0.3, 3.5, color="#232321", zorder=0)
ax.text(1.6, 77, "cold start", ha="center", fontsize=10, color=T3)
ax.set_xticks(list(x))
ax.set_xticklabels(buckets, fontsize=9.5)
ax.set_xlabel("tasks already served (log-spaced buckets)", fontsize=10, color=T3)
ax.set_ylim(20, 80)
ax.set_yticks([20, 40, 60, 80])
ax.set_yticklabels([f"{v}%" for v in [20, 40, 60, 80]], fontsize=10)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.tick_params(length=0)
ax.set_xlim(-0.3, len(buckets) - 0.7)
f.text(0.04, 0.92, "It gets sharper the more it's used", fontsize=18, weight="bold")
f.text(0.04, 0.855, "Every task is scored before Deja learns from it. Tool search stays flat. (Shuffle 1.)", fontsize=11, color=T3)
save(f, "3_learning_curve.png")

# 4. How it works diagram
f = fig(10, 5.0)
ax = f.add_axes([0, 0, 1, 1], facecolor=SURF)
ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")


def box(x, y, w, h, title, sub, edge=GRID, fill="#242422"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fill, ec=edge, lw=1.5))
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=12.5, weight="bold", color=T1)
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=9.5, color=T2)


def arrow(x1, y1, x2, y2, c=T3):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=c, lw=1.6))


ax.text(4, 46, "How Deja builds a loadout", fontsize=18, weight="bold", color=T1)
box(4, 26, 16, 11, "New task", "\"weather + time\nin Tokyo\"")
box(27, 32, 19, 9, "1. Recall", "20 nearest past tasks\nvote via USED edges", edge=BLUE)
box(27, 20, 19, 9, "2. Co-use hop", "Tool → CO → Tool", edge=AQUA)
box(27, 8, 19, 9, "3. Word hop", "8 rarest words\n→ LED → Tool", edge=VIOLET)
box(55, 20, 17, 9, "4. Rank + pack", "fill 2,000-token\nbudget")
box(79, 20, 17, 9, "Loadout", "only the tools\nit will call", edge=BLUE)
box(55, 4, 41, 9, "5. Learn: write the finished task back", "new Task node + USED edges, counts updated in HydraDB", fill="#1f2a38", edge=BLUE)
arrow(20.5, 31.5, 26.4, 36.5); arrow(20.5, 30, 26.4, 13)
arrow(36.5, 31.4, 36.5, 29.6)
arrow(46.6, 36.5, 54.4, 26.5); arrow(46.6, 24.5, 54.4, 24.5); arrow(46.6, 12.5, 54.4, 22.5)
arrow(72.6, 24.5, 78.4, 24.5)
arrow(87.5, 19.4, 87.5, 13.6, BLUE)
save(f, "4_how_it_works.png")
print("ok")
