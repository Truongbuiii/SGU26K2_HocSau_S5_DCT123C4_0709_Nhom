"""PrivateTest evaluator for a locked FERCNN run when TV3 evaluate.py is unavailable.

Author: Lê Thị Thanh Tuyền – 3123411332 | TV4: final model evaluation | 24/11/2026

Use only after configuration and checkpoint selection are finalized on PublicTest.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    from dataset import CLASS_NAMES, get_dataloaders
    from models import build_model
except ImportError:
    from src.dataset import CLASS_NAMES, get_dataloaders
    from src.models import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, type=Path)
    p.add_argument("--csv_path", required=True, type=Path)
    args = p.parse_args()
    cfg = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(args.run_dir / "best.pt", map_location=device, weights_only=False))
    model.eval()
    _, _, loader, _ = get_dataloaders(batch_size=int(cfg.get("batch_size", 64)), aug_name="none", norm=cfg.get("norm", "div255"), seed=int(cfg.get("seed", 42)), csv_path=str(args.csv_path))
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            y_true.extend(y.tolist())
            y_pred.extend(pred.tolist())
    labels = list(range(len(CLASS_NAMES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    result = {"split": "PrivateTest", "accuracy": accuracy_score(y_true, y_pred), "classification_report": classification_report(y_true, y_pred, labels=labels, target_names=CLASS_NAMES, output_dict=True, zero_division=0), "confusion_matrix": cm.tolist()}
    (args.run_dir / "private_test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=labels, yticks=labels, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, xlabel="Predicted", ylabel="True", title=f"PrivateTest confusion matrix ({result['accuracy']:.3f})")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(args.run_dir / "private_test_confusion_matrix.png", dpi=180)
    plt.close(fig)
    print(f"PrivateTest accuracy: {result['accuracy']:.4f}; saved result under {args.run_dir}")


if __name__ == "__main__":
    main()
