"""Vẽ các hình cho Chương 4 (lưu vào experiments/figures/)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from train_mlp import ROOT, EXP_DIR, SEED, get_data, holdout_curves, best_cfg
from preprocess import Preprocessor
from mlp_model import train_model, predict_log, metrics

FIG = f"{EXP_DIR}/figures"
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3, "font.size": 10})


def fig_learning_curves(out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a, name in zip(ax, ["baseline", "tuned"]):
        h = out[name]["hist"]
        a.plot(np.sqrt(h["train_loss"]), label="Train RMSE (log)")
        a.plot(np.sqrt(h["val_loss"]), label="Validation RMSE (log)")
        a.axhline(0.12, color="gray", ls=":", lw=.8)
        a.set_title(f"{'Baseline (Keras mẫu → PyTorch)' if name=='baseline' else 'Cấu hình sau tuning'}\n"
                    f"val RMSE-log cuối = {out[name]['metrics']['rmse_log']:.4f}")
        a.set_xlabel("Epoch"); a.set_ylim(0, 0.45); a.legend()
    ax[0].set_ylabel("RMSE trên log(SalePrice)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_learning_curve.png"); plt.close(fig)


def fig_tuning():
    d = pd.read_csv(f"{EXP_DIR}/tuning_results.csv").sort_values("cv_rmse_log")
    lab = [f"#{i} {h}\ndo={do} wd={wd} {o}" for i, h, do, wd, o in
           zip(d.id, d.hidden, d.dropout, d.weight_decay, d.optimizer)]
    fig, ax = plt.subplots(figsize=(8, 7))
    cols = ["tab:red" if i == 0 else "tab:blue" for i in d.id]
    ax.barh(range(len(d)), d.cv_rmse_log, xerr=d.cv_rmse_log_std, color=cols, alpha=.8, capsize=2)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(lab, fontsize=6); ax.invert_yaxis()
    ax.set_xlim(0.10, 0.18); ax.set_xlabel("CV RMSE (log), 3-fold; đỏ = baseline")
    ax.set_title("Kết quả random search hyperparameter (25 cấu hình)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_tuning.png"); plt.close(fig)


def fig_experiments():
    pf = pd.read_csv(f"{EXP_DIR}/experiment_per_fold.csv")
    g = pf.groupby(["experiment", "config"]).rmse_log.agg(["mean", "std"]).reset_index()
    exps = sorted(g.experiment.unique()); x = np.arange(len(exps)); w = .38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for k, (cfg, c) in enumerate([("baseline", "tab:gray"), ("tuned", "tab:blue")]):
        s = g[g.config == cfg].set_index("experiment").reindex(exps)
        ax.bar(x + (k - .5) * w, s["mean"], w, yerr=s["std"], label=f"MLP {cfg}", color=c, capsize=3)
        for xi, v in zip(x + (k - .5) * w, s["mean"]):
            if not np.isnan(v): ax.text(xi, v + .004, f"{v:.4f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(exps); ax.set_ylim(0.09, 0.15)
    ax.set_ylabel("CV RMSE (log), 5-fold"); ax.set_title("So sánh các experiment (thấp hơn = tốt hơn)")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{FIG}/fig_experiments.png"); plt.close(fig)


def fig_pred_vs_actual():
    """4 biểu đồ chẩn đoán giống 4-in-1 của Minitab trong paper (Fig. 6) cho model tuned trên holdout."""
    train, _, ref = get_data()
    a, b = train_test_split(train, test_size=0.2, random_state=SEED)
    p = Preprocessor().fit(a, ref)
    Xa, Xb, ya, yb = p.transform(a), p.transform(b), p.target(a), p.target(b)
    m, _ = train_model(Xa, ya, Xb, yb, best_cfg(), seed=0)
    pr = predict_log(m, Xb); res = pr - yb
    from scipy import stats
    fig, ax = plt.subplots(2, 2, figsize=(9, 7))
    (osm, osr), (sl, ic, _) = stats.probplot(res, dist="norm")
    ax[0, 0].plot(osm, osr, ".", ms=3); ax[0, 0].plot(osm, sl * np.array(osm) + ic, "r-")
    ax[0, 0].set_title("Normal probability plot (residual)")
    ax[0, 1].scatter(pr, res, s=6); ax[0, 1].axhline(0, color="r"); ax[0, 1].set_title("Residual vs Fitted (log)")
    ax[1, 0].hist(res, bins=30); ax[1, 0].set_title("Histogram residual (log)")
    ax[1, 1].scatter(np.expm1(yb) / 1e3, np.expm1(pr) / 1e3, s=6); lim = [0, 500]
    ax[1, 1].plot(lim, lim, "r-"); ax[1, 1].set_xlabel("Actual ($k)"); ax[1, 1].set_ylabel("Predicted ($k)")
    ax[1, 1].set_title("Predicted vs Actual")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_diagnostics.png"); plt.close(fig)
    return metrics(yb, pr)


if __name__ == "__main__":
    out = holdout_curves()
    fig_learning_curves(out)
    import json
    json.dump({k: v["metrics"] for k, v in out.items()}, open(f"{EXP_DIR}/holdout_metrics.json", "w"), indent=2)
    fig_tuning(); fig_experiments()
    print(fig_pred_vs_actual()); print({k: v["metrics"] for k, v in out.items()})
