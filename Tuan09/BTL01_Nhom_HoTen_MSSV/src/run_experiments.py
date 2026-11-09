"""Chay du lieu / baseline / thuc nghiem A, B, C va xuat hinh + bang.
Vi du:  python run_experiments.py all      |  python run_experiments.py A
"""
import argparse, json, os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import make_splits, save_splits, gaussian_2d, RANGE_X1, RANGE_X2, MU, COV, Y_MAX
from train import train_model, to_tensors, predict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG, LOG, MOD = (os.path.join(ROOT, "results", d) for d in ("", "figures", "logs", "models"))
DATA = os.path.join(ROOT, "data")
SEEDS = [0, 1, 2]
BASE = dict(hidden=(10,), activation="tanh", optimizer="adam", lr=1e-2, batch_size=32, epochs=200)
BASE_A = 1000
A_LIST = list(range(200, 2001, 200))
B_LIST = list(range(1, 11)) + [16, 32]            # B = 1..10, mo rong 16, 32
B_SMALL_LIST = list(range(1, 11)) + [16, 32, 64, 128]
C_LIST = [1, 2, 3, 4, 5, 6]                       # C >= 4; mo rong den 6
WIDTH_C = 10
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3, "font.size": 10})


def grid(n=60):
    g1, g2 = np.meshgrid(np.linspace(*RANGE_X1, n), np.linspace(*RANGE_X2, n))
    return g1, g2, np.stack([g1.ravel(), g2.ravel()], 1)


# ------------------------------------------------------------------ DATA
def do_data(splits):
    os.makedirs(DATA, exist_ok=True)
    save_splits(splits, DATA)
    fig = plt.figure(figsize=(12, 4.2))
    for i, (name, c) in enumerate([("train", "tab:green"), ("val", "tab:orange"), ("test", "tab:red")]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        x, y = splits[name]
        ax.scatter(x[:, 0], x[:, 1], y, c=c, s=4 if name == "train" else 8)
        ax.set_title(f"{name} (n={len(x)})"); ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("y")
    fig.tight_layout(); fig.savefig(f"{FIG}/data_scatter3d.png"); plt.close(fig)

    g1, g2, gx = grid()
    fig = plt.figure(figsize=(5.5, 4.5)); ax = fig.add_subplot(projection="3d")
    ax.plot_surface(g1, g2, gaussian_2d(gx).reshape(g1.shape), cmap="viridis")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("f(x)"); ax.set_title("Bề mặt hàm Gaussian thật")
    fig.tight_layout(); fig.savefig(f"{FIG}/data_true_surface.png"); plt.close(fig)

    x, y = splits["train"]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    for a, v, n in zip(axs, (x[:, 0], x[:, 1], y), ("x1", "x2", "y")):
        a.hist(v, bins=30, color="steelblue"); a.set_title(f"Phân bố {n}"); a.set_xlabel(n)
    fig.tight_layout(); fig.savefig(f"{FIG}/data_hist.png"); plt.close(fig)
    rows = []
    for name, (xx, yy) in splits.items():
        for n, v in (("x1", xx[:, 0]), ("x2", xx[:, 1]), ("y", yy)):
            rows.append(dict(split=name, var=n, n=len(v), mean=v.mean(), std=v.std(), min=v.min(), max=v.max()))
    pd.DataFrame(rows).to_csv(f"{RES}/data_stats.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string())


# ------------------------------------------------------------------ BASELINE
def do_baseline(splits):
    model, hist, res = train_model(splits, BASE_A, BASE, seed=0)
    torch.save(model.state_dict(), f"{MOD}/baseline_best.pt")
    h = np.array(hist)
    pd.DataFrame(h, columns=["epoch", "train_mse", "val_mse"]).to_csv(f"{LOG}/baseline_history.csv", index=False)
    json.dump(dict(config={**BASE, "hidden": list(BASE["hidden"]), "A": BASE_A, "seed": 0}, result=res),
              open(f"{LOG}/baseline_result.json", "w"), indent=1)
    print("baseline:", {k: round(v, 7) for k, v in res.items()})

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.semilogy(h[:, 0], h[:, 1], label="Train"); ax.semilogy(h[:, 0], h[:, 2], label="Validation")
    ax.axvline(res["best_epoch"], ls="--", c="gray", label=f"best epoch = {res['best_epoch']}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (thang log)"); ax.set_title("Baseline: loss theo epoch"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{FIG}/baseline_loss.png"); plt.close(fig)

    xt, yt = splits["test"]
    Xt, _ = to_tensors(xt, yt)
    yp = predict(model, Xt).numpy().ravel() * Y_MAX
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(yt, yp, s=16); lim = [0, yt.max() * 1.05]; ax.plot(lim, lim, "r--", label="y = ŷ")
    ax.set_xlabel("y thật"); ax.set_ylabel("ŷ dự đoán"); ax.set_title(f"Test: y vs ŷ (R²={res['test_r2']:.3f})"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{FIG}/baseline_y_vs_yhat.png"); plt.close(fig)

    g1, g2, gx = grid()
    Gn, _ = to_tensors(gx, np.zeros(len(gx)))
    zp = predict(model, Gn).numpy().reshape(g1.shape) * Y_MAX
    zt = gaussian_2d(gx).reshape(g1.shape)
    fig = plt.figure(figsize=(12, 4))
    for i, (z, t) in enumerate([(zt, "Hàm thật"), (zp, "MLP xấp xỉ (baseline)")]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d"); ax.plot_surface(g1, g2, z, cmap="viridis", vmin=0, vmax=Y_MAX)
        ax.set_title(t); ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlim(0, Y_MAX * 1.05)
    ax = fig.add_subplot(1, 3, 3); im = ax.contourf(g1, g2, np.abs(zp - zt), 20, cmap="magma")
    ax.scatter(xt[:, 0], xt[:, 1], c="cyan", s=6, label="test"); fig.colorbar(im, ax=ax); ax.legend(loc="upper right")
    ax.set_title("|sai số| tuyệt đối"); ax.set_xlabel("x1"); ax.set_ylabel("x2")
    fig.tight_layout(); fig.savefig(f"{FIG}/baseline_surface.png"); plt.close(fig)


# ------------------------------------------------------------------ SWEEPS
def sweep(name, values, make_cfg, make_n, splits):
    rows, hists = [], {}
    for v in values:
        for s in SEEDS:
            _, hist, res = train_model(splits, make_n(v), make_cfg(v), seed=s)
            rows.append(dict(**{name: v}, seed=s, **res))
            if s == SEEDS[0]:
                hists[v] = hist
            print(f"[{name}] {v} seed={s} test_mse={res['test_mse']:.3e} val_mse={res['val_mse']:.3e} "
                  f"time={res['train_time']:.1f}s", flush=True)
        pd.DataFrame(rows).to_csv(f"{LOG}/exp_{name}_runs.csv", index=False)
    df = pd.DataFrame(rows)
    cols = ["train_mse", "val_mse", "test_mse", "test_mae", "test_rmse", "test_r2", "best_epoch", "train_time"]
    summ = df.groupby(name)[cols].agg(["mean", "std"]); summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(f"{LOG}/exp_{name}_summary.csv", index=False)
    pd.concat([pd.DataFrame(np.array(h), columns=["epoch", "train_mse", "val_mse"]).assign(**{name: v})
               for v, h in hists.items()]).to_csv(f"{LOG}/exp_{name}_hist_seed0.csv", index=False)


def run_A(splits): sweep("A", A_LIST, lambda v: BASE, lambda v: v, splits)
def run_B(splits): sweep("B", B_LIST, lambda v: {**BASE, "hidden": (v,)}, lambda v: BASE_A, splits)
def run_Bsmall(splits): sweep("Bsmall", B_SMALL_LIST, lambda v: {**BASE, "hidden": (v,)}, lambda v: 100, splits)
def run_C(splits): sweep("C", C_LIST, lambda v: {**BASE, "hidden": (WIDTH_C,) * v}, lambda v: BASE_A, splits)


# ------------------------------------------------------------------ PLOTS
def errplot(ax, s, x, col, label, **kw):
    ax.errorbar(s[x], s[f"{col}_mean"], yerr=s[f"{col}_std"], marker="o", capsize=3, label=label, **kw)


def make_plots():
    for name, xlabel, title in [("A", "Số mẫu huấn luyện A", "Thực nghiệm A: sai số theo số mẫu"),
                                ("B", "Số nơ-ron tầng ẩn B", "Thực nghiệm B: sai số theo độ rộng (A=1000)"),
                                ("Bsmall", "Số nơ-ron tầng ẩn B", "Bổ sung: độ rộng với ít dữ liệu (A=100)"),
                                ("C", "Số tầng ẩn C (mỗi tầng 10 nơ-ron)", "Thực nghiệm C: sai số theo độ sâu")]:
        p = f"{LOG}/exp_{name}_summary.csv"
        if not os.path.exists(p): continue
        s = pd.read_csv(p)
        fig, ax = plt.subplots(figsize=(6.2, 4))
        for col, lab in (("train_mse", "Train"), ("val_mse", "Validation"), ("test_mse", "Test")):
            errplot(ax, s, name, col, lab)
        ax.set_yscale("log"); ax.set_xlabel(xlabel); ax.set_ylabel("MSE (thang log)"); ax.set_title(title); ax.legend()
        if name in ("B", "Bsmall"): ax.set_xscale("log")
        fig.tight_layout(); fig.savefig(f"{FIG}/exp_{name}_mse.png"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.2, 4))
        errplot(ax, s, name, "test_mae", "Test MAE"); errplot(ax, s, name, "test_rmse", "Test RMSE")
        ax.set_yscale("log"); ax.set_xlabel(xlabel); ax.set_ylabel("Sai số (đơn vị của y)"); ax.set_title(title + " – MAE/RMSE"); ax.legend()
        if name in ("B", "Bsmall"): ax.set_xscale("log")
        fig.tight_layout(); fig.savefig(f"{FIG}/exp_{name}_mae_rmse.png"); plt.close(fig)
    if os.path.exists(f"{LOG}/exp_C_summary.csv"):
        s = pd.read_csv(f"{LOG}/exp_C_summary.csv"); h = pd.read_csv(f"{LOG}/exp_C_hist_seed0.csv")
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        for c, g in h.groupby("C"): axs[0].semilogy(g["epoch"], g["val_mse"], label=f"C={c}")
        axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Validation MSE"); axs[0].set_title("Hội tụ theo độ sâu (seed 0)"); axs[0].legend(ncol=2)
        errplot(axs[1], s, "C", "train_time", "Thời gian train (s)", color="tab:purple")
        axs[1].set_xlabel("Số tầng ẩn C"); axs[1].set_ylabel("Giây / 200 epoch"); axs[1].set_title("Thời gian huấn luyện")
        fig.tight_layout(); fig.savefig(f"{FIG}/exp_C_conv_time.png"); plt.close(fig)
    if os.path.exists(f"{LOG}/exp_B_hist_seed0.csv"):
        h = pd.read_csv(f"{LOG}/exp_B_hist_seed0.csv"); fig, ax = plt.subplots(figsize=(6.2, 4))
        for b, g in h.groupby("B"):
            if b in (1, 2, 4, 10, 32): ax.semilogy(g["epoch"], g["val_mse"], label=f"B={b}")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Validation MSE"); ax.set_title("Hội tụ theo độ rộng (seed 0)"); ax.legend()
        fig.tight_layout(); fig.savefig(f"{FIG}/exp_B_conv.png"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("task", choices=["data", "baseline", "A", "B", "Bsmall", "C", "plots", "all"])
    t = ap.parse_args().task
    for d in (RES, FIG, LOG, MOD, DATA): os.makedirs(d, exist_ok=True)
    splits = make_splits()
    if t in ("data", "all"): do_data(splits)
    if t in ("baseline", "all"): do_baseline(splits)
    if t in ("A", "all"): run_A(splits)
    if t in ("B", "all"): run_B(splits)
    if t in ("Bsmall", "all"): run_Bsmall(splits)
    if t in ("C", "all"): run_C(splits)
    make_plots()
    try:                                   # tu dong cap nhat file Excel tong hop
        from make_excel import build_excel
        build_excel()
    except Exception as e:                 # vd: Excel dang mo / chua cai openpyxl
        print(f"[bo qua Excel] {e}")
