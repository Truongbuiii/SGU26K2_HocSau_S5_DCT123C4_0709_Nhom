"""
=============================================================================
BTL02 - CNN nhận diện cảm xúc khuôn mặt (FER2013) - PyTorch
File: src/eda.py
Tác giả (TV1): Bùi Đức Trường - MSSV: 3123411319
Vai trò: Thành viên 1 (Nhóm trưởng) - Dữ liệu và pipeline
Mô tả: Phân tích khám phá dữ liệu, lưu PNG vào results/figures:
       1. eda_class_distribution.png : histogram phân bố lớp của train / valid / test
       2. eda_samples_grid.png       : lưới 4x8 ảnh mẫu có nhãn
       3. eda_augmentation_grid.png  : mỗi cấu hình augmentation, ảnh gốc + 7 biến thể
       Chạy: python src/eda.py [--csv đường_dẫn_fer2013.csv]
=============================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from augment import AUG_NAMES, build_transform  # noqa: E402
from dataset import CLASS_NAMES, NUM_CLASSES, check_split_counts, load_fer2013  # noqa: E402

FIG_DIR = _ROOT_DIR / "results" / "figures"


def plot_class_distribution(data) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=150, sharey=False)
    titles = {"train": "Train (Training)", "valid": "Valid (PublicTest)", "test": "Test (PrivateTest)"}
    for ax, split in zip(axes, ("train", "valid", "test")):
        counts = np.bincount(data[split][1], minlength=NUM_CLASSES)
        bars = ax.bar(CLASS_NAMES, counts, color="#4C72B0")
        ax.set_title(f"{titles[split]} - tổng {counts.sum()} ảnh")
        ax.set_ylabel("Số ảnh")
        ax.tick_params(axis="x", rotation=45)
        for b, c in zip(bars, counts):
            ax.text(b.get_x() + b.get_width() / 2, c, str(c), ha="center", va="bottom", fontsize=8)
    fig.suptitle("Phân bố lớp cảm xúc FER2013", fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "eda_class_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_sample_grid(data, rows: int = 4, cols: int = 8, seed: int = 42) -> Path:
    X, y = data["train"]
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(y), rows * cols, replace=False)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.9), dpi=150)
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(X[i], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{y[i]}: {CLASS_NAMES[y[i]]}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Ảnh mẫu từ tập train (nhãn: số - tên lớp)", fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "eda_samples_grid.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_augmentation_grid(data, n_variants: int = 7, seed: int = 42) -> Path:
    X, y = data["train"]
    img = torch.from_numpy(X[0]).unsqueeze(0)  # (1,48,48) uint8
    names = list(AUG_NAMES)
    fig, axes = plt.subplots(len(names), n_variants + 1, figsize=((n_variants + 1) * 1.5, len(names) * 1.7), dpi=150)
    for r, name in enumerate(names):
        torch.manual_seed(seed)
        tf = build_transform(name, train=True)
        for c in range(n_variants + 1):
            ax = axes[r, c]
            ax.imshow(img[0] if c == 0 else tf(img)[0] * 255, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            if c == 0:
                ax.set_title("gốc" if r == 0 else "", fontsize=8)
                ax.text(-0.15, 0.5, name, transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=8)
    fig.suptitle(f"Augmentation trên 1 ảnh train (nhãn {CLASS_NAMES[y[0]]})", fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "eda_augmentation_grid.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA FER2013 (TV1)")
    parser.add_argument("--csv", type=str, default=None, help="Đường dẫn fer2013.csv")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Đang đọc dữ liệu ...")
    data = load_fer2013(args.csv)
    print("Số lượng ảnh từng tập:")
    check_split_counts(data)
    for fn in (plot_class_distribution, plot_sample_grid, plot_augmentation_grid):
        print("Đã lưu:", fn(data))
