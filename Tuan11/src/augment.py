"""
=============================================================================
BTL02 - CNN nhận diện cảm xúc khuôn mặt (FER2013) - PyTorch
File: src/augment.py
Tác giả (TV1): Bùi Đức Trường - MSSV: 3123411319
Vai trò: Thành viên 1 (Nhóm trưởng) - Dữ liệu và pipeline
Mô tả: Các cấu hình augmentation có tên cho E3 (chỉ áp dụng cho tập train).
       - none               : không augmentation
       - flip               : lật ngang
       - flip_affine        : lật ngang + xoay 10° + dịch 0.1 + zoom 0.1 (BASELINE)
       - flip_affine_erase  : flip_affine + RandomErasing
       - strong             : lật ngang + xoay 30° + dịch 0.1 + zoom 0.2 + RandomErasing (E3-04)
       Valid / test chỉ chuyển sang float [0, 1] và chuẩn hóa.
       Ảnh đầu vào là tensor uint8 (1, 48, 48), đầu ra là tensor float32 (1, 48, 48).
=============================================================================
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch
from torchvision.transforms import v2

AUG_NAMES = ("none", "flip", "flip_affine", "flip_affine_erase", "strong")


def _flip() -> List:
    return [v2.RandomHorizontalFlip(p=0.5)]


def _affine(degrees: float, scale: tuple) -> List:
    return [v2.RandomAffine(degrees=degrees, translate=(0.1, 0.1), scale=scale)]


def _erase() -> List:
    # value=0 -> ô bị xóa có màu đen (ảnh đang ở thang [0, 1])
    return [v2.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)]


_AUG_BUILDERS: Dict[str, Callable[[], List]] = {
    "none": lambda: [],
    "flip": lambda: _flip(),
    "flip_affine": lambda: _flip() + _affine(10, (0.9, 1.1)),
    "flip_affine_erase": lambda: _flip() + _affine(10, (0.9, 1.1)) + _erase(),
    "strong": lambda: _flip() + _affine(30, (0.8, 1.2)) + _erase(),
}


def build_transform(
    aug_name: str = "flip_affine",
    train: bool = True,
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> v2.Compose:
    """
    Tạo pipeline biến đổi cho một ảnh.
    - train=True : áp dụng augmentation theo `aug_name`
    - train=False: chỉ chuẩn hóa (valid và test KHÔNG augment)
    - mean/std = None -> chỉ chia 255 (div255); có giá trị -> (x - mean) / std
      (mean/std phải tính trên tập train, xem dataset.py)
    """
    if aug_name not in _AUG_BUILDERS:
        raise ValueError(f"aug_name '{aug_name}' không hợp lệ. Chọn một trong: {AUG_NAMES}")

    ops: List = [v2.ToDtype(torch.float32, scale=True)]  # uint8 [0,255] -> float [0,1]
    if train:
        ops += _AUG_BUILDERS[aug_name]()
    if mean is not None and std is not None:
        ops.append(v2.Normalize(mean=[mean], std=[std]))
    return v2.Compose(ops)


if __name__ == "__main__":
    print("=== SMOKE TEST src/augment.py ===")
    torch.manual_seed(0)
    img = torch.randint(0, 256, (1, 48, 48), dtype=torch.uint8)
    for name in AUG_NAMES:
        out = build_transform(name, train=True)(img)
        print(f"{name:<20} -> shape={tuple(out.shape)} dtype={out.dtype} "
              f"min={out.min():.2f} max={out.max():.2f}")
    out_val = build_transform("strong", train=False)(img)
    assert torch.allclose(out_val, img.float() / 255.0), "Valid/test không được augment!"
    print("Valid/test chỉ chia 255, không augment: OK")
    print("=== SMOKE TEST PASSED ===")
