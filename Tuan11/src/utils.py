"""
=============================================================================
BTL02 - CNN nhận diện cảm xúc khuôn mặt (FER2013) - PyTorch
File: src/utils.py
Tác giả (TV2): [HỌ TÊN CỦA BẠN] - MSSV: [MSSV CỦA BẠN]
Vai trò: Thành viên 2 - Mô hình và vòng huấn luyện (Models, Train, Utils)
Ngày tạo: 18/11/2026
Mô tả: Các hàm và lớp tiện ích phục vụ huấn luyện:
       - set_seed: Cố định hạt giống ngẫu nhiên đảm bảo tái lập kết quả.
       - count_parameters: Đếm tổng số tham số huấn luyện được của mô hình.
       - EarlyStopping: Dừng sớm theo val_loss.
       - CSVLogger: Ghi lịch sử huấn luyện ra history.csv sau mỗi epoch.
       - plot_learning_curves: Vẽ biểu đồ Loss và Accuracy từ history.csv.
       - get_git_commit_hash / get_env_info: Lưu thông tin môi trường (env.json).
=============================================================================
"""

from __future__ import annotations

import csv
import datetime
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # Dùng backend không cần GUI để chạy ổn định trên cả Colab/Kaggle/CLI
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """
    Cố định toàn bộ random seed cho Python, NumPy và PyTorch (CPU & GPU)
    để kết quả giữa các lần chạy có thể so sánh công bằng.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Đếm số lượng tham số của mô hình PyTorch.
    - trainable_only=True: Chỉ đếm các tham số có requires_grad=True.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


class EarlyStopping:
    """
    Theo dõi val_loss để dừng sớm khi mô hình không còn cải thiện sau `patience` epoch.
    Hỗ trợ state_dict() và load_state_dict() để nối tiếp (resume) khi Colab ngắt kết nối.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter: int = 0
        self.best_loss: float = float("inf")
        self.best_epoch: int = 0
        self.early_stop: bool = False

    def step(self, val_loss: float, epoch: int) -> bool:
        """
        Cập nhật trạng thái sau mỗi epoch.
        Trả về True nếu epoch hiện tại đạt val_loss tốt nhất từ trước tới nay (`is_best`).
        """
        if np.isnan(val_loss) or np.isinf(val_loss):
            # Nếu loss bị NaN/Inf thì dừng ngay để tránh tốn GPU vô ích
            self.early_stop = True
            return False

        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = float(val_loss)
            self.best_epoch = int(epoch)
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

    def state_dict(self) -> Dict[str, Any]:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "counter": self.counter,
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "early_stop": self.early_stop,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.patience = state.get("patience", self.patience)
        self.min_delta = state.get("min_delta", self.min_delta)
        self.counter = state.get("counter", 0)
        self.best_loss = state.get("best_loss", float("inf"))
        self.best_epoch = state.get("best_epoch", 0)
        self.early_stop = state.get("early_stop", False)


class CSVLogger:
    """
    Ghi log kết quả từng epoch vào file `history.csv` ngay sau khi kết thúc mỗi epoch.
    Các cột chuẩn theo Mục 4:
        epoch, train_loss, train_acc, val_loss, val_acc, lr, elapsed_time
    """

    DEFAULT_FIELDNAMES = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "lr",
        "elapsed_time",
    ]

    def __init__(
        self,
        filepath: Union[str, Path],
        fieldnames: Optional[List[str]] = None,
        append: bool = False,
    ) -> None:
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames or self.DEFAULT_FIELDNAMES

        # Nếu không phải resume (append=False) hoặc file chưa tồn tại thì tạo mới và ghi header
        if not append or not self.filepath.exists() or self.filepath.stat().st_size == 0:
            with open(self.filepath, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row_dict: Dict[str, Any]) -> None:
        """Ghi 1 dòng kết quả của epoch vào file CSV và flush xuống ổ đĩa ngay lập tức."""
        formatted: Dict[str, Any] = {}
        for key in self.fieldnames:
            val = row_dict.get(key, "")
            if isinstance(val, float):
                if key == "lr":
                    formatted[key] = f"{val:.6e}"
                elif key == "elapsed_time":
                    formatted[key] = f"{val:.2f}"
                else:
                    formatted[key] = f"{val:.6f}"
            else:
                formatted[key] = val

        with open(self.filepath, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(formatted)


def plot_learning_curves(
    history_csv_path: Union[str, Path],
    save_path: Optional[Union[str, Path]] = None,
    title_prefix: str = "",
) -> Path:
    """
    Đọc file `history.csv` và vẽ biểu đồ đôi (Model Accuracy & Model Loss theo Epoch)
    giống định dạng báo cáo mẫu (demo_report.pdf trang 8).
    """
    csv_path = Path(history_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file history.csv tại: {csv_path}")

    epochs: List[int] = []
    train_loss: List[float] = []
    train_acc: List[float] = []
    val_loss: List[float] = []
    val_acc: List[float] = []

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            train_acc.append(float(row["train_acc"]))
            val_loss.append(float(row["val_loss"]))
            val_acc.append(float(row["val_acc"]))

    if not epochs:
        raise ValueError(f"File {csv_path} rỗng, chưa có dữ liệu epoch nào.")

    # Tìm epoch có val_loss nhỏ nhất
    best_idx = int(np.argmin(val_loss))
    best_epoch = epochs[best_idx]
    best_val_loss = val_loss[best_idx]
    best_val_acc = val_acc[best_idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
    suptitle = (
        f"{title_prefix} - Best at Epoch {best_epoch} "
        f"(val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})"
        if title_prefix
        else f"Best at Epoch {best_epoch} (val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})"
    )
    fig.suptitle(suptitle, fontsize=11, fontweight="bold")

    # 1. Biểu đồ Accuracy
    ax_acc = axes[0]
    ax_acc.plot(epochs, train_acc, label="train", linewidth=1.8)
    ax_acc.plot(epochs, val_acc, label="valid", linewidth=1.8)
    ax_acc.axvline(best_epoch, color="red", linestyle="--", alpha=0.6, label=f"best ep ({best_epoch})")
    ax_acc.set_title("Model Accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.grid(True, linestyle="--", alpha=0.5)
    ax_acc.legend(loc="lower right")

    # 2. Biểu đồ Loss
    ax_loss = axes[1]
    ax_loss.plot(epochs, train_loss, label="train", linewidth=1.8)
    ax_loss.plot(epochs, val_loss, label="valid", linewidth=1.8)
    ax_loss.axvline(best_epoch, color="red", linestyle="--", alpha=0.6, label=f"best ep ({best_epoch})")
    ax_loss.set_title("Model Loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True, linestyle="--", alpha=0.5)
    ax_loss.legend(loc="upper right")

    plt.tight_layout()

    out_path = Path(save_path) if save_path is not None else csv_path.with_name("learning_curves.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def get_git_commit_hash() -> str:
    """Lấy git commit hash hiện tại (nếu có) để ghi vào config.json làm minh chứng."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out if out else "unknown"
    except Exception:
        return "unknown"


def get_env_info(start_time: Optional[str] = None, end_time: Optional[str] = None) -> Dict[str, Any]:
    """
    Thu thập thông tin môi trường chạy (phiên bản Python, PyTorch, tên GPU, thời gian)
    để lưu thành `env.json` theo đúng yêu cầu Mục 4.
    """
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "CPU (No CUDA GPU)"
    gpu_count = torch.cuda.device_count() if gpu_available else 0

    now_iso = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": gpu_available,
        "cuda_version": torch.version.cuda if gpu_available else None,
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
        "start_time": start_time or now_iso,
        "end_time": end_time,
    }


if __name__ == "__main__":
    # Tự kiểm tra nhanh (Smoke Test) khi chạy trực tiếp: python src/utils.py
    print("=== SMOKE TEST src/utils.py ===")
    set_seed(42)
    dummy_model = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(16 * 48 * 48, 7))
    print(f"1. Seed fixed (42). Dummy model trainable params: {count_parameters(dummy_model):,}")
    print(f"2. Environment info: {get_env_info()}")

    es = EarlyStopping(patience=3)
    sample_losses = [1.50, 1.35, 1.36, 1.37, 1.38]
    for ep, loss_val in enumerate(sample_losses, start=1):
        is_best = es.step(loss_val, ep)
        print(f"   Epoch {ep}: val_loss={loss_val:.2f} | is_best={is_best} | counter={es.counter} | stop={es.early_stop}")
    print("=== SMOKE TEST PASSED ===")
