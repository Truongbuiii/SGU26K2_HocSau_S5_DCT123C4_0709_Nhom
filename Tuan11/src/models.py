"""
=============================================================================
BTL02 - CNN nhận diện cảm xúc khuôn mặt (FER2013) - PyTorch
File: src/models.py
Tác giả (TV2): [HỌ TÊN CỦA BẠN] - MSSV: [MSSV CỦA BẠN]
Vai trò: Thành viên 2 - Mô hình và vòng huấn luyện (Models, Train, Utils)
Ngày tạo: 18/11/2026
Mô tả: Định nghĩa kiến trúc mạng CNN nhận diện cảm xúc khuôn mặt (48x48x1 -> 7 lớp)
       - Hỗ trợ các kiến trúc: Small (2 block), Medium (3 block - Baseline),
         Medium_01 (3 block sâu hơn như demo_report), Large (4 block).
       - Tham số hóa: Hàm kích hoạt (relu, leaky_relu, elu, sigmoid),
         BatchNorm (có/không), mức Dropout (conv_dropout, fc_dropout),
         khởi tạo trọng số (Kaiming / Xavier).
       - Cung cấp 2 hàm giao diện chuẩn cho toàn nhóm (Mục 3.2):
         + build_model(cfg) -> nn.Module
         + get_last_conv(model) -> nn.Conv2d
=============================================================================
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


# Định nghĩa cấu hình số filter của từng block và kích thước lớp FC cho mỗi kiến trúc
# Mỗi phần tử trong `blocks` là danh sách số kênh đầu ra của các lớp Conv2d liên tiếp trong block đó
ARCH_PRESETS: Dict[str, Dict[str, Any]] = {
    # E1-01: Small (2 block MaxPooling: 48 -> 24 -> 12) - Theo demo_report trang 11
    # Block 1: 2x64; Block 2: 2x128 + 2x256 -> Pool -> Flatten (256*12*12) -> FC 1024
    "small": {
        "blocks": [
            [64, 64],
            [128, 128, 256, 256],
        ],
        "fc_dim": 1024,
    },
    # E1-02 / BASELINE: Medium (3 block MaxPooling: 48 -> 24 -> 12 -> 6) - Theo Mục 2.4 & demo_report trang 11-12
    # Block 1: 2x64; Block 2: 3x128; Block 3: 3x256 -> Flatten (256*6*6 = 9216) -> FC 1024
    "medium": {
        "blocks": [
            [64, 64],
            [128, 128, 128],
            [256, 256, 256],
        ],
        "fc_dim": 1024,
    },
    # Tùy chọn thêm: Medium_01 (3x64, 4x128, 4x256) theo demo_report trang 6-7
    "medium_01": {
        "blocks": [
            [64, 64, 64],
            [128, 128, 128, 128],
            [256, 256, 256, 256],
        ],
        "fc_dim": 1024,
    },
    # E1-03: Large (4 block MaxPooling: 48 -> 24 -> 12 -> 6 -> 3) - Theo Mục 6.2 & demo_report trang 13
    # Block 1: 4x64; Block 2: 4x128; Block 3: 4x256; Block 4: 4x512 -> Flatten (512*3*3 = 4608) -> FC 2048
    "large": {
        "blocks": [
            [64, 64, 64, 64],
            [128, 128, 128, 128],
            [256, 256, 256, 256],
            [512, 512, 512, 512],
        ],
        "fc_dim": 2048,
    },
}


def make_activation(name: str) -> nn.Module:
    """Tạo module hàm kích hoạt theo tên cấu hình."""
    act_name = name.lower().strip()
    if act_name == "relu":
        return nn.ReLU(inplace=True)
    if act_name in ("leaky_relu", "leakyrelu"):
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    if act_name == "elu":
        return nn.ELU(alpha=1.0, inplace=True)
    if act_name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(
        f"Hàm kích hoạt '{name}' không được hỗ trợ. Chọn một trong: relu, leaky_relu, elu, sigmoid."
    )


class FERCNN(nn.Module):
    """
    Mạng CNN tham số hóa cho bài toán nhận diện cảm xúc khuôn mặt FER2013 (Input: 1x48x48, Output: 7 logits).
    Lưu ý quan trọng (Mục 2.2 & 6.2):
    - KHÔNG đặt Softmax ở cuối mạng (trả về raw logits để dùng với nn.CrossEntropyLoss).
    - Thứ tự trong mỗi tầng tích chập: Conv2d -> (BatchNorm2d nếu use_bn=True) -> Activation.
    - Cuối mỗi Block: MaxPool2d(2, 2) -> Dropout(conv_dropout).
    - Các lớp Conv được đặt tên rõ ràng: conv1_1, conv1_2, conv2_1...
    """

    def __init__(
        self,
        arch: str = "medium",
        in_channels: int = 1,
        num_classes: int = 7,
        input_size: int = 48,
        activation: str = "relu",
        use_bn: bool = False,
        conv_dropout: float = 0.2,
        fc_dropout: float = 0.5,
        fc_dim: Optional[int] = None,
        custom_blocks: Optional[List[List[int]]] = None,
        init_mode: str = "auto",
    ) -> None:
        super().__init__()
        arch_key = arch.lower().strip()
        if custom_blocks is not None:
            blocks_cfg = custom_blocks
            default_fc = 1024
        else:
            if arch_key not in ARCH_PRESETS:
                raise ValueError(
                    f"Kiến trúc '{arch}' không hợp lệ. Hỗ trợ: {list(ARCH_PRESETS.keys())}"
                )
            blocks_cfg = ARCH_PRESETS[arch_key]["blocks"]
            default_fc = ARCH_PRESETS[arch_key]["fc_dim"]

        self.arch = arch_key
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_size = input_size
        self.activation_name = activation.lower().strip()
        self.use_bn = bool(use_bn)
        self.conv_dropout = float(conv_dropout)
        self.fc_dropout = float(fc_dropout)
        self.fc_dim = int(fc_dim if fc_dim is not None else default_fc)
        self.init_mode = init_mode.lower().strip()

        # Xây dựng các Block tích chập với tên tường minh (conv1_1, conv1_2, ..., pool1, drop1)
        features_dict: "OrderedDict[str, nn.Module]" = OrderedDict()
        curr_channels = in_channels
        spatial_size = input_size

        for b_idx, block_channels in enumerate(blocks_cfg, start=1):
            for l_idx, out_channels in enumerate(block_channels, start=1):
                conv_name = f"conv{b_idx}_{l_idx}"
                # Khi dùng BatchNorm thì tắt bias của Conv2d vì BatchNorm đã có tham số shift (beta)
                features_dict[conv_name] = nn.Conv2d(
                    curr_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not self.use_bn,
                )
                if self.use_bn:
                    features_dict[f"bn{b_idx}_{l_idx}"] = nn.BatchNorm2d(out_channels)
                features_dict[f"act{b_idx}_{l_idx}"] = make_activation(self.activation_name)
                curr_channels = out_channels

            features_dict[f"pool{b_idx}"] = nn.MaxPool2d(kernel_size=2, stride=2)
            spatial_size = spatial_size // 2
            if self.conv_dropout > 0.0:
                features_dict[f"drop{b_idx}"] = nn.Dropout(p=self.conv_dropout)

        if spatial_size < 1:
            raise ValueError(
                f"Ảnh đầu vào {input_size}x{input_size} quá nhỏ cho {len(blocks_cfg)} tầng MaxPool2d(2,2)."
            )

        self.features = nn.Sequential(features_dict)
        self.flatten_dim = curr_channels * spatial_size * spatial_size

        # Xây dựng Classifier (Flatten -> Dense -> Activation -> Dropout -> Dense(7))
        classifier_dict: "OrderedDict[str, nn.Module]" = OrderedDict()
        classifier_dict["flatten"] = nn.Flatten()
        classifier_dict["fc1"] = nn.Linear(self.flatten_dim, self.fc_dim)
        classifier_dict["act_fc1"] = make_activation(self.activation_name)
        if self.fc_dropout > 0.0:
            classifier_dict["drop_fc1"] = nn.Dropout(p=self.fc_dropout)
        classifier_dict["fc_out"] = nn.Linear(self.fc_dim, self.num_classes)
        self.classifier = nn.Sequential(classifier_dict)

        # Khởi tạo trọng số chuẩn
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Khởi tạo trọng số cho Conv2d, Linear và BatchNorm2d:
        - Kaiming Normal (He initialization) cho ReLU, LeakyReLU, ELU (giúp mạng sâu không BN vẫn hội tụ tốt).
        - Xavier Uniform (Glorot) cho Sigmoid.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if self.init_mode == "xavier" or (
                    self.init_mode == "auto" and self.activation_name == "sigmoid"
                ):
                    nn.init.xavier_uniform_(m.weight)
                else:
                    # Kaiming normal (fan_in) cho relu / leaky_relu / elu để giữ phương sai forward ổn định
                    if self.activation_name in ("leaky_relu", "leakyrelu"):
                        nn.init.kaiming_normal_(m.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
                    else:
                        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        logits = self.classifier(x)
        return logits


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Hỗ trợ đọc thuộc tính từ cả dataclass Config (của TV3), SimpleNamespace hoặc dict."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_model(cfg: Any = None) -> nn.Module:
    """
    Hàm giao diện chuẩn (Mục 3.2): Dựng mô hình từ đối tượng cấu hình `cfg`.
    Các trường được đọc từ `cfg` (có giá trị mặc định khớp với BASELINE Mục 2.4):
      - arch / model_name: 'small' | 'medium' | 'medium_01' | 'large' (mặc định: 'medium')
      - activation: 'relu' | 'leaky_relu' | 'elu' | 'sigmoid' (mặc định: 'relu')
      - use_bn / batch_norm: bool (mặc định: False)
      - conv_dropout / dropout: float (mặc định: 0.2)
      - fc_dropout: float (mặc định: 0.5)
      - fc_dim: int | None (mặc định: theo preset, 1024 cho small/medium, 2048 cho large)
      - num_classes: int (mặc định: 7)
      - in_channels: int (mặc định: 1)
    """
    arch = _cfg_get(cfg, "arch", _cfg_get(cfg, "model_name", "medium"))
    activation = _cfg_get(cfg, "activation", "relu")
    use_bn = _cfg_get(cfg, "use_bn", _cfg_get(cfg, "batch_norm", False))
    conv_dropout = _cfg_get(cfg, "conv_dropout", _cfg_get(cfg, "dropout", 0.2))
    fc_dropout = _cfg_get(cfg, "fc_dropout", 0.5)
    fc_dim = _cfg_get(cfg, "fc_dim", None)
    num_classes = _cfg_get(cfg, "num_classes", 7)
    in_channels = _cfg_get(cfg, "in_channels", 1)
    init_mode = _cfg_get(cfg, "init_mode", "auto")

    return FERCNN(
        arch=arch,
        in_channels=in_channels,
        num_classes=num_classes,
        activation=activation,
        use_bn=use_bn,
        conv_dropout=conv_dropout,
        fc_dropout=fc_dropout,
        fc_dim=fc_dim,
        init_mode=init_mode,
    )


def get_last_conv(model: nn.Module) -> nn.Conv2d:
    """
    Hàm giao diện chuẩn (Mục 3.2): Trả về lớp tích chập `nn.Conv2d` cuối cùng của mô hình.
    Hàm này hoạt động được cho cả FERCNN (Small/Medium/Large) lẫn ResNet18/VGG16 của TV4.
    """
    last_conv: Optional[nn.Conv2d] = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("Không tìm thấy lớp nn.Conv2d nào trong mô hình.")
    return last_conv


def get_last_conv_name_and_layer(model: nn.Module) -> Tuple[str, nn.Conv2d]:
    """Trả về cả tên (ví dụ 'features.conv3_3') và đối tượng lớp nn.Conv2d cuối cùng."""
    last_name = ""
    last_conv: Optional[nn.Conv2d] = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_name = name
            last_conv = module
    if last_conv is None:
        raise ValueError("Không tìm thấy lớp nn.Conv2d nào trong mô hình.")
    return last_name, last_conv


if __name__ == "__main__":
    # Tự kiểm tra (Smoke Test) toàn bộ các cấu hình thí nghiệm E1 của Thành viên 2
    print("=== SMOKE TEST src/models.py ===")
    x_dummy = torch.randn(4, 1, 48, 48)

    test_configs = [
        ("E1-01 (Small, 2 block)", {"arch": "small"}),
        ("E1-02 (Medium / BASELINE, 3 block)", {"arch": "medium"}),
        ("E1-03 (Large, 4 block, FC 2048)", {"arch": "large"}),
        ("E1-04 (Medium + BatchNorm)", {"arch": "medium", "use_bn": True}),
        ("E1-05a (Medium + LeakyReLU)", {"arch": "medium", "activation": "leaky_relu"}),
        ("E1-05b (Medium + ELU)", {"arch": "medium", "activation": "elu"}),
        ("E1-05c (Medium + Sigmoid)", {"arch": "medium", "activation": "sigmoid"}),
        ("E1-06a (Medium + Dropout 0.0)", {"arch": "medium", "conv_dropout": 0.0}),
        ("E1-06b (Medium + Dropout 0.5)", {"arch": "medium", "conv_dropout": 0.5}),
    ]

    print(f"{'Cấu hình thí nghiệm':<38} | {'Output shape':<14} | {'Tổng tham số':>14} | {'Lớp Conv cuối'}")
    print("-" * 92)
    for label, cfg_dict in test_configs:
        m = build_model(cfg_dict)
        out = m(x_dummy)
        assert out.shape == (4, 7), f"Sai kích thước đầu ra: {out.shape}"
        num_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        last_name, last_conv = get_last_conv_name_and_layer(m)
        print(
            f"{label:<38} | {str(tuple(out.shape)):<14} | {num_params:>14,} | "
            f"{last_name} ({last_conv.in_channels}->{last_conv.out_channels})"
        )
    print("=== SMOKE TEST PASSED ===")
