"""One-time PrivateTest evaluation for a locked E4 transfer-learning run.

Author: Lê Thị Thanh Tuyền – 3123411332 | TV4: E4 final evaluation | 24/11/2026
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from train_pretrained import make_model, prepare
try:
    from dataset import CLASS_NAMES, get_dataloaders
except ImportError:
    from src.dataset import CLASS_NAMES, get_dataloaders


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, type=Path)
    p.add_argument("--csv_path", required=True, type=Path)
    args = p.parse_args()
    cfg = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(cfg["arch"], device)
    model.load_state_dict(torch.load(args.run_dir / "best.pt", map_location=device, weights_only=False))
    model.eval()
    _, _, test_loader, _ = get_dataloaders(batch_size=cfg["batch_size"], aug_name="none", norm="div255", seed=cfg["seed"], csv_path=str(args.csv_path))
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(prepare(x.to(device)))
            y_true.extend(y.numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
    metrics = {"accuracy": accuracy_score(y_true, y_pred), "classification_report": classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0), "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES)))).tolist()}
    (args.run_dir / "private_test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(7), yticks=range(7), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, xlabel="Predicted", ylabel="True", title=f"PrivateTest confusion matrix (acc={metrics['accuracy']:.3f})")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    for (i, j), v in np.ndenumerate(cm): ax.text(j, i, str(v), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(args.run_dir / "private_test_confusion_matrix.png", dpi=180)
    plt.close(fig)
    print(f"PrivateTest accuracy: {metrics['accuracy']:.4f}; saved metrics and confusion matrix under {args.run_dir}")


if __name__ == "__main__": main()
