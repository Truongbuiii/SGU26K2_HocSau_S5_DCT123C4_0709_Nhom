"""
Vai tro: TV3 - Thuc nghiem va danh gia
File   : aggregate.py - quet results/runs/*, xuat summary.csv, bieu do so sanh, nhat ky xlsx

Cach dung:
    python aggregate.py                 # tao summary.csv + hinh so sanh
    python aggregate.py --check         # kiem tra moi run du file minh chung
    python aggregate.py --merge a.xlsx b.xlsx c.xlsx   # gop nhat ky cua cac thanh vien
"""
import argparse
import glob
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from config import BASELINE, BASELINE_IDS, cfg_from_dict, diff_from_baseline

REQUIRED = ["config.json", "env.json", "train.log", "history.csv", "best.pt",
            "test_metrics.json", "confusion_matrix.png"]
LOG_COLUMNS = ["ExpID", "Ngay gio", "Nguoi chay", "Muc dich", "Thay doi so voi baseline",
               "Seed", "Epoch tot nhat", "Val acc (%)", "Test acc (%)", "Thoi gian chay (phut)",
               "Duong dan run", "Nhan xet"]
NOISE_THRESHOLD = 1.5   # % - chenh lech nho hon muc nay coi la trong nhieu (muc 2.5)


def _pct(x):
    """Dua accuracy ve don vi % (chap nhan ca 0-1 lan 0-100)."""
    return x * 100 if x <= 1.0 else x


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_run(run_dir):
    cfg_d = _read_json(os.path.join(run_dir, "config.json"))
    hist_p = os.path.join(run_dir, "history.csv")
    if cfg_d is None or not os.path.exists(hist_p):
        return None
    cfg = cfg_from_dict(cfg_d)
    h = pd.read_csv(hist_p)
    i = int(h["val_loss"].idxmin())
    tcol = next((c for c in ("elapsed_time", "time", "epoch_time", "time_s") if c in h.columns), None)
    tm = _read_json(os.path.join(run_dir, "test_metrics.json")) or {}
    test_acc = tm.get("test_acc", tm.get("accuracy", np.nan))
    return {
        "exp_id": cfg.exp_id, "name": cfg.exp_name, "seed": cfg.seed, "runner": cfg.runner,
        "change": diff_from_baseline(cfg),
        "n_params": cfg_d.get("num_parameters", cfg_d.get("n_params", np.nan)),
        "epochs_run": len(h),
        "best_epoch": int(h["epoch"].iloc[i]),
        "train_acc": _pct(float(h["train_acc"].iloc[i])),
        "val_acc": _pct(float(h["val_acc"].iloc[i])),
        "val_loss": float(h["val_loss"].iloc[i]),
        "test_acc": _pct(float(test_acc)) if test_acc == test_acc else np.nan,
        "gap_train_val": _pct(float(h["train_acc"].iloc[i])) - _pct(float(h["val_acc"].iloc[i])),
        "minutes": float(h[tcol].sum()) / 60 if tcol else np.nan,
        "run_dir": run_dir,
    }, h


def aggregate(runs_root="results/runs", out_dir="results"):
    rows, hists = [], {}
    for d in sorted(glob.glob(os.path.join(runs_root, "*"))):
        if not os.path.isdir(d) or "_failed_" in d:
            continue
        r = load_run(d)
        if r is None:
            print(f"[!] Bo qua (thieu config.json/history.csv): {d}")
            continue
        rows.append(r[0])
        hists[os.path.basename(d)] = r[1]
    if not rows:
        print("Khong co run nao.")
        return None
    df = pd.DataFrame(rows)

    # so voi baseline cung seed
    base = df[df.exp_id.isin(BASELINE_IDS)]
    for col in ("val_acc", "test_acc"):
        df["d_" + col] = np.nan
    for i, r in df.iterrows():
        b = base[base.seed == r.seed]
        if len(b) and r.exp_id not in BASELINE_IDS:
            df.loc[i, "d_val_acc"] = r.val_acc - float(b.val_acc.iloc[0])
            df.loc[i, "d_test_acc"] = r.test_acc - float(b.test_acc.iloc[0])
    df["trong_nhieu"] = df["d_val_acc"].abs().map(
        lambda v: "" if v != v else ("co" if v < NOISE_THRESHOLD else "khong"))
    df = df.round(3)

    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "summary.csv"), index=False, encoding="utf-8-sig")
    print(df[["exp_id", "name", "seed", "best_epoch", "train_acc", "val_acc", "test_acc",
              "d_val_acc", "trong_nhieu", "minutes"]].to_string(index=False))

    # mean +- std khi co nhieu seed
    g = df.groupby(["exp_id", "name"]).filter(lambda x: len(x) > 1)
    if len(g):
        s = g.groupby(["exp_id", "name"])[["val_acc", "test_acc"]].agg(["mean", "std", "count"]).round(2)
        s.to_csv(os.path.join(out_dir, "summary_seeds.csv"), encoding="utf-8-sig")
        print("\nNhieu seed:\n", s)

    _plot_groups(df, hists, os.path.join(out_dir, "figures"))
    return df


def _plot_groups(df, hists, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)
    df42 = df[df.seed == BASELINE.seed]
    for grp in sorted({e.split("-")[0] for e in df42.exp_id if e not in BASELINE_IDS}):
        sub = df42[(df42.exp_id.str.startswith(grp + "-")) | (df42.exp_id.isin(BASELINE_IDS))]
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        for _, r in sub.iterrows():
            h = hists[os.path.basename(r.run_dir)]
            lab = f"{r.exp_id} {r['name']}"
            ax[0].plot(h["epoch"], h["val_acc"].map(_pct), label=lab, lw=2 if r.exp_id in BASELINE_IDS else 1.2)
            ax[1].plot(h["epoch"], h["val_loss"], label=lab, lw=2 if r.exp_id in BASELINE_IDS else 1.2)
        ax[0].set(title=f"{grp}: Val accuracy (%)", xlabel="epoch")
        ax[1].set(title=f"{grp}: Val loss", xlabel="epoch")
        ax[0].legend(fontsize=8)
        for a in ax:
            a.grid(alpha=.3)
        fig.tight_layout()
        p = os.path.join(fig_dir, f"compare_{grp}_val.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print("Da luu", p)


def check_runs(runs_root="results/runs", log_path="results/experiment_log.xlsx", prefix=None):
    """Kiem tra chéo: du file minh chung + so trong nhat ky khop history.csv / test_metrics.json."""
    ok = True
    log = pd.read_excel(log_path) if os.path.exists(log_path) else None
    for d in sorted(glob.glob(os.path.join(runs_root, "*"))):
        if not os.path.isdir(d) or (prefix and not os.path.basename(d).startswith(prefix)):
            continue
        miss = [f for f in REQUIRED if not os.path.exists(os.path.join(d, f))]
        status = "OK" if not miss else "THIEU " + ", ".join(miss)
        r = load_run(d)
        if r is not None and log is not None:
            row = log[log["Duong dan run"].astype(str) == d]
            if row.empty:
                status += " | CHUA co trong nhat ky"
            else:
                v = float(row["Val acc (%)"].iloc[0])
                if abs(v - r[0]["val_acc"]) > 0.1:
                    status += f" | LECH nhat ky val_acc {v} != {r[0]['val_acc']:.2f}"
        if status != "OK":
            ok = False
        print(f"{os.path.basename(d):45s} {status}")
    print("\nKET QUA:", "TAT CA OK" if ok else "CO MUC CAN SUA")
    return ok


def append_log(run_dir, log_path="results/experiment_log.xlsx", comment=""):
    """Them 1 dong vao nhat ky. 'Nhan xet' dien tay sau khi xem ket qua."""
    r = load_run(run_dir)
    if r is None:
        return
    r = r[0]
    cfg = cfg_from_dict(_read_json(os.path.join(run_dir, "config.json")))
    row = {
        "ExpID": cfg.exp_id, "Ngay gio": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        "Nguoi chay": cfg.runner, "Muc dich": cfg.note, "Thay doi so voi baseline": r["change"],
        "Seed": cfg.seed, "Epoch tot nhat": r["best_epoch"], "Val acc (%)": round(r["val_acc"], 2),
        "Test acc (%)": round(r["test_acc"], 2) if r["test_acc"] == r["test_acc"] else "",
        "Thoi gian chay (phut)": round(r["minutes"], 1) if r["minutes"] == r["minutes"] else "",
        "Duong dan run": run_dir, "Nhan xet": comment,
    }
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    old = pd.read_excel(log_path) if os.path.exists(log_path) else pd.DataFrame(columns=LOG_COLUMNS)
    old = old[old["Duong dan run"].astype(str) != run_dir]      # cap nhat neu da co
    pd.concat([old, pd.DataFrame([row])], ignore_index=True)[LOG_COLUMNS].to_excel(log_path, index=False)


def merge_logs(paths, out="results/experiment_log.xlsx"):
    df = pd.concat([pd.read_excel(p) for p in paths], ignore_index=True)
    df = df.drop_duplicates(subset=["ExpID", "Seed", "Duong dan run"], keep="last")
    df.sort_values(["ExpID", "Seed"]).to_excel(out, index=False)
    print("Da gop", len(df), "dong ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", help="chi kiem tra cac run co ten bat dau bang chuoi nay, vd E2")
    ap.add_argument("--merge", nargs="+")
    a = ap.parse_args()
    if a.merge:
        merge_logs(a.merge)
    elif a.check:
        check_runs(a.runs, prefix=a.only)
    else:
        aggregate(a.runs)
