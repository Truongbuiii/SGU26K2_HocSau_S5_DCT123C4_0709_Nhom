"""Doc CSV tong hop -> bang dinh dang (JSON) dung cho bao cao/slide; ve so do kien truc."""
import json, os
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG, FIG = f"{ROOT}/results/logs", f"{ROOT}/results/figures"


def pm(s, c, scale, d=2):
    return f"{s[c+'_mean']/scale:.{d}f} ± {s[c+'_std']/scale:.{d}f}"


def table(name, scale, unit):
    s = pd.read_csv(f"{LOG}/exp_{name}_summary.csv")
    rows = []
    for _, r in s.iterrows():
        v = int(r[name])
        rows.append([str(v), pm(r, "train_mse", scale), pm(r, "val_mse", scale), pm(r, "test_mse", scale),
                     f"{r['test_mae_mean']*1e3:.2f}", f"{r['test_rmse_mean']*1e3:.2f}", f"{r['test_r2_mean']:.3f}"])
    return dict(unit=unit, rows=rows)


def table_C():
    s = pd.read_csv(f"{LOG}/exp_C_summary.csv"); rows = []
    for _, r in s.iterrows():
        c = int(r["C"]); params = 41 + 110 * (c - 1)
        rows.append([str(c), str(params), pm(r, "val_mse", 1e-6), pm(r, "test_mse", 1e-6),
                     f"{r['test_mae_mean']*1e3:.2f}", f"{r['test_rmse_mean']*1e3:.2f}", f"{r['test_r2_mean']:.4f}",
                     f"{r['best_epoch_mean']:.0f}", f"{r['train_time_mean']:.1f} ± {r['train_time_std']:.1f}"])
    return rows


out = dict(A=table("A", 1e-4, "1e-4"), B=table("B", 1e-4, "1e-4"), Bsmall=table("Bsmall", 1e-4, "1e-4"), C=table_C())
json.dump(out, open(f"{ROOT}/results/logs/tables.json", "w"), ensure_ascii=False, indent=1)

# so do kien truc
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
def draw(ax, layers, labels, title):
    xs = range(len(layers)); pos = []
    for i, n in enumerate(layers):
        n_show = min(n, 5); ys = [(j - (n_show - 1) / 2) for j in range(n_show)]
        pos.append([(i, y) for y in ys])
    for i in range(len(pos) - 1):
        for p in pos[i]:
            for q in pos[i + 1]: ax.plot([p[0], q[0]], [p[1], q[1]], c="#999", lw=.5, zorder=1)
    cols = ["#6baed6"] + ["#fd8d3c"] * (len(layers) - 2) + ["#74c476"]
    for i, pl in enumerate(pos):
        for (x, y) in pl: ax.scatter(x, y, s=260, c=cols[i], ec="k", zorder=2)
        if layers[i] > 5: ax.text(i, -3.1, "⋮", ha="center", fontsize=16)
        ax.text(i, 3.2, labels[i], ha="center", fontsize=9)
    ax.set_title(title); ax.axis("off"); ax.set_ylim(-3.6, 4)
draw(axs[0], [2, 8, 1], ["Input\n(x1, x2)", "Hidden: B nơ-ron\n(tanh)", "Output\nŷ"], "Thí nghiệm B: 1 tầng ẩn, B nơ-ron")
draw(axs[1], [2, 6, 6, 6, 1], ["Input", "Hidden 1", "Hidden 2", "Hidden C", "Output"], "Thí nghiệm C: C tầng ẩn × 10 nơ-ron")
fig.tight_layout(); fig.savefig(f"{FIG}/architecture.png", dpi=150)
print(json.dumps(out["C"][:2]))
