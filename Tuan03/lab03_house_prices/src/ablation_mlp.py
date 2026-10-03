"""Bổ sung cho Nhiệm vụ 4 và 5.
(a) Ablation: đổi MỘT hyperparameter so với cấu hình tốt nhất (3-fold CV, EXP-00 features).
(b) Lưu loss history (train/val theo epoch) ra CSV và vẽ learning curve cho từng feature set EXP-00..EXP-04."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from train_mlp import get_data, cv_eval, best_cfg, EXP_DIR, SEED
from preprocess import Preprocessor
from feature_engineering import EXPERIMENTS
from mlp_model import train_model, predict_log, metrics, DEFAULT_CFG

train, _, ref = get_data()
best = best_cfg()

# (b) learning curves theo feature set (holdout 80/20, cfg tuned, seed 0) + lưu CSV
a, b = train_test_split(train, test_size=0.2, random_state=SEED)
plan = [(k, v, False) for k, v in EXPERIMENTS.items()] + [("EXP-04", EXPERIMENTS["EXP-03"], True)]
rows, fig = [], plt.figure(figsize=(9, 4.5))
for name, groups, sel in plan:
    p = Preprocessor(groups, sel).fit(a, ref)
    Xa, Xb, ya, yb = p.transform(a), p.transform(b), p.target(a), p.target(b)
    m, h = train_model(Xa, ya, Xb, yb, best, seed=0)
    for ep, (t, v) in enumerate(zip(h["train_loss"], h["val_loss"]), 1):
        rows.append(dict(experiment=name, epoch=ep, train_loss_mse=t, val_loss_mse=v,
                         train_rmse_log=t ** .5, val_rmse_log=v ** .5))
    plt.plot(np.sqrt(h["val_loss"]), label=f"{name} val"); plt.plot(np.sqrt(h["train_loss"]), "--", lw=.8, color=plt.gca().lines[-1].get_color())
    print(name, metrics(yb, predict_log(m, Xb))["rmse_log"], flush=True)
plt.ylim(0.08, 0.2); plt.xlabel("Epoch"); plt.ylabel("RMSE (log)"); plt.legend(fontsize=7, ncol=2); plt.grid(alpha=.3)
plt.title("Learning curve theo feature set (nét liền = validation, nét đứt = train)")
plt.tight_layout(); plt.savefig(f"{EXP_DIR}/figures/fig_learning_curve_featuresets.png", dpi=130)
pd.DataFrame(rows).to_csv(f"{EXP_DIR}/loss_history_featuresets.csv", index=False)

# (a) ablation
variants = {"cấu hình tốt nhất": {}, "weight_decay=0": dict(weight_decay=0.0), "weight_decay=1e-3": dict(weight_decay=1e-3),
            "epochs=150": dict(epochs=150), "epochs=30": dict(epochs=30), "dropout=0.2": dict(dropout=0.2),
            "lr=3e-3": dict(lr=3e-3), "batch_size=32": dict(batch_size=32), "hidden=(50,25,50)": dict(hidden=(50, 25, 50)),
            "optimizer=adamax": dict(optimizer="adamax")}
out = []
for k, ch in variants.items():
    r = cv_eval(train, ref, {**best, **ch}, n_splits=3)
    out.append(dict(variant=k, cv_rmse_log=r.rmse_log.mean(), std=r.rmse_log.std(), cv_rmse=r.rmse.mean()))
    pd.DataFrame(out).to_csv(f"{EXP_DIR}/ablation_results.csv", index=False); print(out[-1], flush=True)
