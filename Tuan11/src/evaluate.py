"""
Author (TV3): Nguyen Thi Ngoc Truc - MSSV: 3123411313 | experiments and evaluation | 2026-11-25
Vai tro: TV3 - Thuc nghiem va danh gia
File   : evaluate.py - danh gia best.pt tren PublicTest (valid) va PrivateTest (test)

    python evaluate.py results/runs/E2-01_sgd_m09_lr1e-2_s42
Test chi danh gia MOT lan: neu test_metrics.json da co thi bo qua (dung --force de ghi de).
Chi so tinh bang numpy/sklearn trong compute_metrics() nen kiem tra duoc ma khong can GPU.
"""
import argparse
import json
import os

import numpy as np

from config import cfg_from_dict


def compute_metrics(y_true, y_pred, class_names):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    n = len(class_names)
    labels = list(range(n))                      # ep dung thu tu CLASS_NAMES
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_pct = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    off = [(class_names[i], class_names[j], float(cm_pct[i, j]))
           for i in labels for j in labels if i != j]
    off.sort(key=lambda t: -t[2])
    return {
        "accuracy": float((np.asarray(y_true) == np.asarray(y_pred)).mean() * 100),
        "macro_f1": float(f.mean() * 100),
        "per_class": {class_names[i]: {"precision": float(p[i] * 100), "recall": float(r[i] * 100),
                                       "f1": float(f[i] * 100), "support": int(s[i])} for i in labels},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_pct": cm_pct.round(1).tolist(),
        "top_confusions": [{"true": a, "pred": b, "pct": round(c, 1)} for a, b, c in off[:6]],
    }


def plot_confusion(cm_pct, class_names, acc, path, title="Test"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm_pct = np.asarray(cm_pct)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm_pct[i, j]:.0f}%", ha="center", va="center",
                    color="white" if cm_pct[i, j] > 50 else "black", fontsize=9)
    ax.set(xlabel="Predicted label", ylabel="True label", title=f"{title} accuracy {acc:.2f}%")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cases(imgs, y_true, y_pred, class_names, path, title, n=32, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if len(imgs) == 0:
        return
    idx = np.random.RandomState(seed).choice(len(imgs), min(n, len(imgs)), replace=False)
    fig, axs = plt.subplots(4, 8, figsize=(14, 7.5))
    for a in axs.ravel():
        a.axis("off")
    for a, i in zip(axs.ravel(), idx):
        im = imgs[i][0]
        a.imshow((im - im.min()) / (im.max() - im.min() + 1e-8), cmap="gray")
        a.set_title(f"{class_names[y_pred[i]]} ({class_names[y_true[i]]})", fontsize=8)
    fig.suptitle(title + "  (Predict / True)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _predict(model, loader, device, keep_images=False):
    import torch
    model.eval()
    ys, ps, imgs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            ps.append(out.argmax(1).cpu().numpy())
            ys.append(np.asarray(y.cpu()).reshape(-1))
            if keep_images:
                imgs.append(x.cpu().numpy())
    return (np.concatenate(ys), np.concatenate(ps),
            np.concatenate(imgs) if keep_images else None)


def evaluate(run_dir, force=False, device=None):
    import torch
    run_dir = str(run_dir)          # train() tra ve Path -> doi sang str
    from dataset import CLASS_NAMES, get_dataloaders     # TV1
    from models import build_model                       # TV2
    out_json = os.path.join(run_dir, "test_metrics.json")
    if os.path.exists(out_json) and not force:
        print(f"[skip] {out_json} da ton tai (test chi danh gia 1 lan). Dung --force de ghi de.")
        return json.load(open(out_json, encoding="utf-8"))

    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg = cfg_from_dict(json.load(f))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _, valid_loader, test_loader, _ = get_dataloaders(
        batch_size=cfg.batch_size, aug_name=cfg.aug_name,
        use_class_weight=cfg.use_class_weight, norm=cfg.norm, seed=cfg.seed)

    model = build_model(cfg).to(device)
    state = torch.load(os.path.join(run_dir, "best.pt"), map_location=device)
    if isinstance(state, dict):                       # phong khi TV2 luu kem optimizer/epoch
        for k in ("model_state_dict", "model_state", "state_dict", "model"):
            if k in state:
                state = state[k]
                break
    model.load_state_dict(state)

    yv, pv, _ = _predict(model, valid_loader, device)
    yt, pt, imgs = _predict(model, test_loader, device, keep_images=True)
    mv = compute_metrics(yv, pv, CLASS_NAMES)
    mt = compute_metrics(yt, pt, CLASS_NAMES)
    res = {"run_dir": run_dir, "class_names": list(CLASS_NAMES),
           "val_acc": mv["accuracy"], "test_acc": mt["accuracy"],
           "val_macro_f1": mv["macro_f1"], "test_macro_f1": mt["macro_f1"],
           "per_class": mt["per_class"], "top_confusions": mt["top_confusions"],
           "confusion_matrix": mt["confusion_matrix"], "confusion_matrix_pct": mt["confusion_matrix_pct"],
           "val_per_class": mv["per_class"]}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=str)

    plot_confusion(mt["confusion_matrix_pct"], CLASS_NAMES, mt["accuracy"],
                   os.path.join(run_dir, "confusion_matrix.png"), "Test (PrivateTest)")
    ok = yt == pt
    plot_cases(imgs[ok], yt[ok], pt[ok], CLASS_NAMES, os.path.join(run_dir, "true_cases.png"), "True predictions")
    plot_cases(imgs[~ok], yt[~ok], pt[~ok], CLASS_NAMES, os.path.join(run_dir, "wrong_cases.png"), "Wrong predictions")

    print(f"VAL acc = {mv['accuracy']:.2f}% | TEST acc = {mt['accuracy']:.2f}% | macro-F1 = {mt['macro_f1']:.2f}%")
    for c, m in mt["per_class"].items():
        print(f"  {c:9s} P={m['precision']:5.1f} R={m['recall']:5.1f} F1={m['f1']:5.1f} n={m['support']}")
    print("Nham lan nhieu nhat:", mt["top_confusions"][:3])
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    evaluate(a.run_dir, force=a.force)
