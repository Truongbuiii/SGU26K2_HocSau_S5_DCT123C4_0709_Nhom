"""
=============================================================================
BTL02 - CNN nhận diện cảm xúc khuôn mặt (FER2013) - PyTorch
File: src/dataset.py
Tác giả (TV1): Bùi Đức Trường - MSSV: 3123411319
Vai trò: Thành viên 1 (Nhóm trưởng) - Dữ liệu và pipeline
Mô tả: Nạp FER2013 từ fer2013.csv (cột emotion, pixels, Usage).
       - CLASS_NAMES: thứ tự nhãn của CSV gốc (0=Angry ... 6=Neutral), cả nhóm import.
       - Usage: Training -> train, PublicTest -> valid, PrivateTest -> test.
       - get_dataloaders(...) là hàm giao diện dùng chung (Mục 3.2 tài liệu phân công).
       Đường dẫn CSV: tham số csv_path, hoặc biến môi trường FER2013_CSV,
       hoặc mặc định <repo>/data/fer2013.csv.
=============================================================================
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from augment import build_transform
except ImportError:  # chạy dạng package: python -m src.dataset
    from src.augment import build_transform

# Thứ tự nhãn theo file CSV gốc và slide đề bài. KHÔNG đổi thứ tự này.
CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
NUM_CLASSES = len(CLASS_NAMES)

# Tên Usage trong CSV -> tên tập
USAGE_TO_SPLIT = {"Training": "train", "PublicTest": "valid", "PrivateTest": "test"}
EXPECTED_COUNTS = {"train": 28709, "valid": 3589, "test": 3589}

_ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CSV = _ROOT_DIR / "data" / "fer2013.csv"


def resolve_csv_path(csv_path: Optional[str] = None) -> Path:
    path = Path(csv_path or os.environ.get("FER2013_CSV") or DEFAULT_CSV)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy fer2013.csv tại: {path}\n"
            "Đặt file vào data/fer2013.csv hoặc đặt biến môi trường FER2013_CSV."
        )
    return path


def load_fer2013(csv_path: Optional[str] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Đọc CSV và tách thành 3 tập.
    Trả về dict {'train'|'valid'|'test': (X uint8 (N,48,48), y int64 (N,))}.
    """
    path = resolve_csv_path(csv_path)
    df = pd.read_csv(path)
    missing = {"emotion", "pixels", "Usage"} - set(df.columns)
    if missing:
        raise ValueError(f"File CSV thiếu cột: {missing}. Cần các cột emotion, pixels, Usage.")

    unknown = set(df["Usage"].unique()) - set(USAGE_TO_SPLIT)
    if unknown:
        raise ValueError(f"Giá trị Usage không nhận ra: {unknown}")

    data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for usage, split in USAGE_TO_SPLIT.items():
        sub = df[df["Usage"] == usage]
        X = np.stack(
            [np.array(p.split(), dtype=np.uint8) for p in sub["pixels"]]
        ).reshape(-1, 48, 48)
        y = sub["emotion"].to_numpy(dtype=np.int64)
        data[split] = (X, y)
    return data


def check_split_counts(data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> bool:
    """In số lượng từng tập và so với số liệu chuẩn 28.709 / 3.589 / 3.589."""
    ok = True
    for split in ("train", "valid", "test"):
        n = len(data[split][1])
        flag = "OK" if n == EXPECTED_COUNTS[split] else "KHÁC SỐ LIỆU CHUẨN"
        if n != EXPECTED_COUNTS[split]:
            ok = False
        print(f"  {split:<6}: {n:>6} ảnh (chuẩn {EXPECTED_COUNTS[split]}) -> {flag}")
    return ok


class FERDataset(Dataset):
    """Dataset FER2013: trả (tensor (1,48,48), nhãn int)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None) -> None:
        self.X = torch.from_numpy(X).unsqueeze(1)  # (N,1,48,48) uint8
        self.y = torch.from_numpy(np.array(y, dtype=np.int64))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        img = self.X[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.y[idx])


def compute_class_weights(y_train: np.ndarray) -> torch.Tensor:
    """Class weight nghịch với tần suất, tính CHỈ trên tập train: w_c = N / (K * n_c)."""
    counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float64)
    weights = counts.sum() / (NUM_CLASSES * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def get_dataloaders(
    batch_size: int = 64,
    aug_name: str = "flip_affine",
    use_class_weight: bool = False,
    norm: str = "div255",
    seed: int = 42,
    num_workers: Optional[int] = None,
    csv_path: Optional[str] = None,
):
    """
    Hàm giao diện dùng chung (Mục 3.2).
    Trả về (train_loader, valid_loader, test_loader, class_weights).
      - aug_name: none | flip | flip_affine | flip_affine_erase | strong (chỉ áp dụng cho train)
      - use_class_weight=True -> class_weights là tensor (7,), False -> None
      - norm: 'div255' (chia 255) hoặc 'meanstd' (mean/std tính trên tập train)
    """
    if norm not in ("div255", "meanstd"):
        raise ValueError("norm phải là 'div255' hoặc 'meanstd'")
    if num_workers is None:
        num_workers = 0 if os.name == "nt" else 2

    data = load_fer2013(csv_path)
    X_tr, y_tr = data["train"]

    mean = std = None
    if norm == "meanstd":
        # Chỉ tính trên tập train (không rò rỉ từ valid/test)
        mean = float(X_tr.mean() / 255.0)
        std = float((X_tr / 255.0).std())

    train_ds = FERDataset(X_tr, y_tr, build_transform(aug_name, train=True, mean=mean, std=std))
    valid_ds = FERDataset(*data["valid"], build_transform("none", train=False, mean=mean, std=std))
    test_ds = FERDataset(*data["test"], build_transform("none", train=False, mean=mean, std=std))

    g = torch.Generator()
    g.manual_seed(seed)
    pin = torch.cuda.is_available()
    common = dict(num_workers=num_workers, pin_memory=pin, worker_init_fn=_seed_worker)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g, **common)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)

    class_weights = compute_class_weights(y_tr) if use_class_weight else None
    return train_loader, valid_loader, test_loader, class_weights


if __name__ == "__main__":
    print("=== KIỂM TRA DỮ LIỆU src/dataset.py ===")
    data = load_fer2013()
    print("Số lượng ảnh từng tập:")
    ok = check_split_counts(data)
    print("Phân bố lớp trong tập train:")
    counts = np.bincount(data["train"][1], minlength=NUM_CLASSES)
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i} {name:<9}: {counts[i]:>5}")
    tr, va, te, cw = get_dataloaders(batch_size=64, use_class_weight=True)
    xb, yb = next(iter(tr))
    print(f"Một batch train: x={tuple(xb.shape)} {xb.dtype}, y={tuple(yb.shape)} {yb.dtype}")
    print(f"Giá trị x trong [{xb.min():.2f}, {xb.max():.2f}]")
    print("Class weights:", [round(w, 3) for w in cw.tolist()])
    print("=== KẾT QUẢ:", "ĐẠT (đủ 28709/3589/3589)" if ok else "CHƯA KHỚP số liệu chuẩn - báo nhóm!", "===")
