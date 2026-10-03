"""Train / tuning / thực nghiệm MLP PyTorch.

Cách chạy (từ thư mục src/):
    python train_mlp.py tune         # random search hyperparameter (EXP-00 features)
    python train_mlp.py experiments  # chạy EXP-00..EXP-05, 5-fold CV
    python train_mlp.py final        # train ensemble cuối, lưu mlp_model.pt + submission_mlp.csv
"""
import json, sys, time, random, itertools, os
import numpy as np, pandas as pd, torch
from sklearn.model_selection import KFold, train_test_split
from preprocess import Preprocessor, load_data, remove_outliers
from feature_engineering import EXPERIMENTS
from mlp_model import train_model, predict_log, metrics, DEFAULT_CFG

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SEED = 42            # random_state chung của nhóm
EXP_DIR = f"{ROOT}/experiments"


def get_data():
    train, test = load_data(f"{ROOT}/data")
    train = remove_outliers(train)
    ref = pd.concat([train.drop(columns="SalePrice"), test], ignore_index=True)
    return train, test, ref


def cv_eval(train, ref, cfg, groups=(), select=False, n_splits=5, seeds=(0,), return_hist=False):
    """K-fold CV. Preprocessor fit riêng trên từng fold train. Báo cáo model ở epoch cuối."""
    kf = KFold(n_splits, shuffle=True, random_state=SEED)
    rows, hists = [], []
    for f, (a, b) in enumerate(kf.split(train)):
        tr, va = train.iloc[a], train.iloc[b]
        p = Preprocessor(groups, select).fit(tr, ref)
        Xa, Xb, ya, yb = p.transform(tr), p.transform(va), p.target(tr), p.target(va)
        preds = []
        for s in seeds:
            m, h = train_model(Xa, ya, Xb, yb, cfg, seed=s)
            preds.append(predict_log(m, Xb)); hists.append(h)
        mt = metrics(yb, np.mean(preds, axis=0))
        mt["fold"], mt["n_features"] = f, Xa.shape[1]
        rows.append(mt)
    df = pd.DataFrame(rows)
    return (df, hists) if return_hist else df


def tune(n_iter=24, n_splits=3):
    train, _, ref = get_data()
    space = dict(hidden=[(50, 25, 50), (64, 32), (128, 64), (128, 128), (256, 128, 64)],
                 dropout=[0.0, 0.1, 0.2, 0.3], weight_decay=[0, 1e-4, 1e-3, 1e-2],
                 lr=[1e-3, 3e-3, 1e-2], batch_size=[16, 32, 64], epochs=[60, 100, 150],
                 optimizer=["adam", "adamw", "adamax"], batchnorm=[False, True])
    rng = random.Random(SEED)
    cfgs = [dict(DEFAULT_CFG, epochs=150)]       # cấu hình baseline (tương đương Keras mẫu) làm mốc
    cfgs += [{k: rng.choice(v) for k, v in space.items()} for _ in range(n_iter)]
    out = f"{EXP_DIR}/tuning_results.csv"
    rows = []
    for i, cfg in enumerate(cfgs):
        t = time.time()
        r = cv_eval(train, ref, cfg, n_splits=n_splits)
        row = dict(id=i, **{k: str(v) for k, v in cfg.items()}, cv_rmse_log=r.rmse_log.mean(),
                   cv_rmse_log_std=r.rmse_log.std(), cv_rmse=r.rmse.mean(), cv_mae=r.mae.mean(),
                   sec=round(time.time() - t, 1))
        rows.append(row)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(i, row["cv_rmse_log"], cfg, flush=True)


def best_cfg():
    """Đọc cấu hình tốt nhất từ tuning_results.csv (CV RMSE-log thấp nhất)."""
    df = pd.read_csv(f"{EXP_DIR}/tuning_results.csv")
    r = df.sort_values("cv_rmse_log").iloc[0]
    cfg = dict(hidden=eval(r["hidden"]), dropout=float(r["dropout"]), batchnorm=r["batchnorm"] in (True, "True"),
               optimizer=r["optimizer"], lr=float(r["lr"]), weight_decay=float(r["weight_decay"]),
               batch_size=int(r["batch_size"]), epochs=int(r["epochs"]), cosine=True)
    return cfg


def experiments(n_splits=5):
    """EXP-00..EXP-05 (thiết kế mục 7). Mỗi experiment chạy với:
       - cấu hình 'baseline' (Keras mẫu chuyển sang PyTorch, không regularization)
       - cấu hình 'tuned' (kết quả tuning), ensemble 3 seed để giảm nhiễu khởi tạo."""
    train, _, ref = get_data()
    tuned = best_cfg()
    base = dict(DEFAULT_CFG, epochs=150)
    rows, per_fold = [], []
    plan = [(k, v, False) for k, v in EXPERIMENTS.items()] + [("EXP-04", EXPERIMENTS["EXP-03"], True)]
    for name, groups, sel in plan:
        for cname, cfg, seeds in [("baseline", base, (0,)), ("tuned", tuned, (0, 1, 2))]:
            t = time.time()
            r = cv_eval(train, ref, cfg, groups, sel, n_splits, seeds)
            rows.append(dict(experiment=name, features="+".join(groups) or "original",
                             selection=sel, config=cname, n_features=int(r.n_features.mean()),
                             **{f"cv_{k}": r[k].mean() for k in ["rmse_log", "rmse", "mae", "bias", "max_dev"]},
                             cv_rmse_log_std=r.rmse_log.std()))
            r["experiment"], r["config"] = name, cname
            per_fold.append(r)
            pd.DataFrame(rows).to_csv(f"{EXP_DIR}/experiment_results.csv", index=False)
            pd.concat(per_fold).to_csv(f"{EXP_DIR}/experiment_per_fold.csv", index=False)
            print(name, cname, round(rows[-1]["cv_rmse_log"], 5), round(time.time() - t), "s", flush=True)
    # EXP-05: feature set tốt nhất (theo cấu hình tuned) + tuned cfg + ensemble 5 seed
    df = pd.DataFrame(rows)
    best = df[df.config == "tuned"].sort_values("cv_rmse_log").iloc[0]
    groups = [g for g in str(best.features).split("+") if g != "original"]
    sel = bool(best.selection)
    r = cv_eval(train, ref, tuned, groups, sel, n_splits, seeds=(0, 1, 2, 3, 4))
    rows.append(dict(experiment="EXP-05", features=("+".join(groups) or "original"), selection=sel,
                     config="tuned (5 seeds)", n_features=int(r.n_features.mean()),
                     **{f"cv_{k}": r[k].mean() for k in ["rmse_log", "rmse", "mae", "bias", "max_dev"]},
                     cv_rmse_log_std=r.rmse_log.std()))
    pd.DataFrame(rows).to_csv(f"{EXP_DIR}/experiment_results.csv", index=False)
    print("EXP-05", rows[-1], flush=True)


def holdout_curves():
    """Learning curve trên split 80/20 (random_state=42) cho baseline và cấu hình tuned."""
    train, _, ref = get_data()
    a, b = train_test_split(train, test_size=0.2, random_state=SEED)
    p = Preprocessor().fit(a, ref)
    Xa, Xb, ya, yb = p.transform(a), p.transform(b), p.target(a), p.target(b)
    out = {}
    for name, cfg in [("baseline", dict(DEFAULT_CFG, epochs=150)), ("tuned", best_cfg())]:
        m, h = train_model(Xa, ya, Xb, yb, cfg, seed=0)
        out[name] = dict(hist=h, metrics=metrics(yb, predict_log(m, Xb)), cfg=cfg)
    return out


def final(n_seeds=10):
    """Train ensemble cuối trên toàn bộ train (đã bỏ outlier), dự đoán test, lưu model + submission."""
    import pickle
    train, test, ref = get_data()
    res = pd.read_csv(f"{EXP_DIR}/experiment_results.csv")
    row = res[res.experiment == "EXP-05"].iloc[0]
    groups = [g for g in str(row.features).split("+") if g != "original"]
    sel, cfg = bool(row.selection), best_cfg()
    p = Preprocessor(groups, sel).fit(train, ref)
    X, y, Xt = p.transform(train), p.target(train), p.transform(test)
    states, preds, tr_rmse = [], [], []
    for s in range(n_seeds):
        m, h = train_model(X, y, None, None, cfg, seed=s)
        states.append({k: v.clone() for k, v in m.state_dict().items()})
        preds.append(predict_log(m, Xt)); tr_rmse.append(h["train_loss"][-1] ** .5)
    pred = np.expm1(np.mean(preds, axis=0))
    # MLP không ngoại suy tốt: chặn dự đoán trong khoảng giá của train (không dùng nhãn test)
    lo, hi = train["SalePrice"].min(), train["SalePrice"].max()
    n_clip = int(((pred < lo) | (pred > hi)).sum())
    pred = np.clip(pred, lo, hi)
    print(f"Đã chặn {n_clip} dự đoán ngoài khoảng [{lo:,.0f}, {hi:,.0f}]")
    sub = pd.DataFrame({"Id": test["Id"], "SalePrice": pred})
    assert len(sub) == len(test) == 1459 and np.isfinite(sub.SalePrice).all() and (sub.SalePrice > 0).all()
    os.makedirs(f"{ROOT}/submissions", exist_ok=True); os.makedirs(f"{ROOT}/models", exist_ok=True)
    sub.to_csv(f"{ROOT}/submissions/submission_mlp.csv", index=False)
    torch.save(dict(state_dicts=states, cfg=cfg, groups=groups, select=sel,
                    input_dim=X.shape[1], feature_names=p.feature_names_), f"{ROOT}/models/mlp_model.pt")
    with open(f"{ROOT}/models/mlp_preprocessor.pkl", "wb") as f:
        pickle.dump(p, f)
    json.dump(dict(groups=groups, select=sel, cfg={k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
                   n_seeds=n_seeds, input_dim=X.shape[1], train_rmse_log_mean=float(np.mean(tr_rmse))),
              open(f"{EXP_DIR}/final_config.json", "w"), indent=2, ensure_ascii=False)
    pd.DataFrame(p.removed_, columns=["feature", "reason"]).to_csv(f"{EXP_DIR}/feature_removed_mlp.csv", index=False)
    print(sub.describe(), "\ntrain rmse_log (full fit):", np.mean(tr_rmse))


if __name__ == "__main__":
    {"tune": tune, "experiments": experiments, "final": final}[sys.argv[1]]()
