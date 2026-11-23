"""
=============================================================================
BTL02 - CNN nhận diện cảm xúc khuôn mặt (FER2013) - PyTorch
File: src/train.py
Tác giả (TV2): Nguyễn Thị Ánh Trâm - MSSV: 3123411306
Vai trò: Thành viên 2 - Mô hình và vòng huấn luyện (Models, Train, Utils)
Ngày tạo: 18/11/2026
Mô tả: Vòng lặp huấn luyện chính `train(cfg) -> run_dir` theo đúng chuẩn Mục 3.2 & Mục 4:
       - Huấn luyện và đánh giá trên tập Validation (PublicTest) sau mỗi epoch.
       - Lưu checkpoint tốt nhất `best.pt` (state_dict) theo `val_loss`.
       - Hỗ trợ EarlyStopping (patience=10), Optimizer (Adam, AdamW, SGD),
         LR Scheduler (ReduceLROnPlateau), Class Weight, Mixed Precision (AMP).
       - Hỗ trợ lưu tạm `last_ckpt.pt` và nối tiếp (resume) khi Colab/Kaggle ngắt phiên.
       - Xuất đầy đủ bộ file minh chứng trong `results/runs/<ExpID>_<ten>_s<seed>/`:
         + config.json
         + env.json
         + train.log
         + history.csv
         + best.pt
         + learning_curves.png
       - Có sẵn chế độ `--overfit_one_batch` để kiểm tra tính đúng đắn của mô hình
         (loss hội tụ về gần 0 trên 1 batch nhỏ).
=============================================================================
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Cho phép import cả khi chạy dạng `python src/train.py` lẫn `python -m src.train`
_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from models import build_model  # noqa: E402
from utils import (  # noqa: E402
    CSVLogger,
    EarlyStopping,
    count_parameters,
    get_env_info,
    get_git_commit_hash,
    local_now,
    plot_learning_curves,
    set_seed,
)


def _cfg_to_dict(cfg: Any) -> Dict[str, Any]:
    """Chuyển cấu hình (dataclass Config của TV3, namespace hoặc dict) về dict chuẩn."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    if dataclasses.is_dataclass(cfg):
        return dataclasses.asdict(cfg)
    if hasattr(cfg, "__dict__"):
        return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    raise TypeError(f"Kiểu cfg không được hỗ trợ: {type(cfg)}")


def _get_default_baseline_cfg() -> Dict[str, Any]:
    """
    Cấu hình BASELINE mặc định theo đúng Mục 2.4 của tài liệu phân công.
    Khi TV3 gọi `train(cfg)`, các trường trong `cfg` của TV3 sẽ ghi đè lên đây.
    """
    return {
        "exp_id": "E1-02",
        "exp_name": "baseline_medium",
        "runner": "TV2",
        "arch": "medium",
        "activation": "relu",
        "use_bn": False,
        "conv_dropout": 0.2,
        "fc_dropout": 0.5,
        "fc_dim": None,
        "num_classes": 7,
        "in_channels": 1,
        "init_mode": "auto",
        # Dữ liệu (TV1)
        "batch_size": 64,
        "aug_name": "flip_affine",
        "use_class_weight": False,
        "norm": "div255",
        # Huấn luyện (TV2 & TV3)
        "optimizer": "adam",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "momentum": 0.9,
        "scheduler": "none",  # 'none' | 'reduce_on_plateau'
        "scheduler_factor": 0.1,
        "scheduler_patience": 3,
        "epochs": 100,
        "patience": 10,
        "seed": 42,
        "use_amp": False,  # Bật hoặc tắt đồng nhất cho toàn bộ thí nghiệm (Mục 2.3)
        # Đường dẫn & chế độ chạy
        "output_root": str(_ROOT_DIR / "results" / "runs"),
        "run_dir": None,
        "resume": True,
        "keep_last_ckpt": False,
    }


class DualLogger:
    """Ghi đồng thời ra màn hình console (stdout) và file `train.log`."""

    def __init__(self, log_path: Path, append: bool = False) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        self._file = open(self.log_path, mode=mode, encoding="utf-8")

    def info(self, msg: str) -> None:
        print(msg, flush=True)
        self._file.write(msg + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()


def build_optimizer(model: nn.Module, cfg_dict: Dict[str, Any]) -> torch.optim.Optimizer:
    """Khởi tạo Optimizer từ cấu hình (hỗ trợ Adam, AdamW, SGD cho cả nhóm E1, E2)."""
    opt_name = str(cfg_dict.get("optimizer", "adam")).lower().strip()
    lr = float(cfg_dict.get("lr", 1e-3))
    weight_decay = float(cfg_dict.get("weight_decay", 0.0))
    momentum = float(cfg_dict.get("momentum", 0.9))

    params = [p for p in model.parameters() if p.requires_grad]
    if opt_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if opt_name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Optimizer '{opt_name}' không hợp lệ. Chọn một trong: adam, adamw, sgd.")


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg_dict: Dict[str, Any]
) -> Optional[torch.optim.lr_scheduler.ReduceLROnPlateau]:
    """Khởi tạo LR Scheduler nếu được bật trong cấu hình (ví dụ E2-05: ReduceLROnPlateau)."""
    sched_name = str(cfg_dict.get("scheduler", "none") or "none").lower().strip()
    if sched_name in ("none", "", "false"):
        return None
    if sched_name in ("reduce_on_plateau", "reducelronplateau", "plateau"):
        factor = float(cfg_dict.get("scheduler_factor", 0.1))
        patience = int(cfg_dict.get("scheduler_patience", 3))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=1e-7,
        )
    raise ValueError(
        f"Scheduler '{sched_name}' không hợp lệ. Hỗ trợ: 'none', 'reduce_on_plateau'."
    )


def _load_dataloaders_from_tv1(
    cfg_dict: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Optional[torch.Tensor]]:
    """
    Gọi hàm `get_dataloaders(batch_size, aug_name, use_class_weight, norm, seed)`
    từ `src/dataset.py` của Thành viên 1 (theo đúng hợp đồng giao diện Mục 3.2).
    """
    try:
        try:
            from src.dataset import get_dataloaders  # type: ignore
        except ImportError:
            from dataset import get_dataloaders  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Không tìm thấy `src/dataset.py` (của Thành viên 1). "
            "Để kiểm tra vòng lặp huấn luyện trước khi TV1 giao dataset.py, "
            "hãy chạy với cờ `--overfit_one_batch` hoặc `--dummy_data`."
        ) from exc

    return get_dataloaders(
        batch_size=int(cfg_dict.get("batch_size", 64)),
        aug_name=str(cfg_dict.get("aug_name", "flip_affine")),
        use_class_weight=bool(cfg_dict.get("use_class_weight", False)),
        norm=str(cfg_dict.get("norm", "div255")),
        seed=int(cfg_dict.get("seed", 42)),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = False,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Tuple[float, float]:
    """
    Chạy 1 epoch huấn luyện.
    Trả về: (train_loss, train_acc) tính trung bình theo tổng số mẫu.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    amp_enabled = bool(use_amp and device.type == "cuda")
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, dtype=torch.long, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if amp_enabled and scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(inputs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += batch_size

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> Tuple[float, float]:
    """
    Đánh giá mô hình trên tập Validation (PublicTest) với `model.eval()` và `torch.no_grad()`.
    Trả về: (val_loss, val_acc).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    amp_enabled = bool(use_amp and device.type == "cuda")
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, dtype=torch.long, non_blocking=True)

        if amp_enabled:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(inputs)
                loss = criterion(logits, targets)
        else:
            logits = model(inputs)
            loss = criterion(logits, targets)

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += batch_size

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def resolve_run_dir(cfg_dict: Dict[str, Any]) -> Path:
    """
    Xác định đường dẫn thư mục lưu kết quả theo đúng quy ước Mục 4:
        results/runs/<ExpID>_<ten>_s<seed>/
    """
    if cfg_dict.get("run_dir"):
        return Path(cfg_dict["run_dir"])

    exp_id = str(cfg_dict.get("exp_id", "EXP"))
    exp_name = str(cfg_dict.get("exp_name", cfg_dict.get("arch", "run"))).replace(" ", "_")
    seed = int(cfg_dict.get("seed", 42))
    folder_name = f"{exp_id}_{exp_name}_s{seed}"
    output_root = Path(cfg_dict.get("output_root", _ROOT_DIR / "results" / "runs"))
    return output_root / folder_name


def train(
    cfg: Any = None,
    dataloaders_override: Optional[
        Tuple[DataLoader, DataLoader, Optional[DataLoader], Optional[torch.Tensor]]
    ] = None,
    model_builder_override: Optional[Callable[[Any], nn.Module]] = None,
) -> Path:
    """
    Hàm giao diện chuẩn của Thành viên 2 (Mục 3.2):
        train(cfg) -> run_dir
    Chạy huấn luyện từ đầu đến cuối (hoặc resume nếu bị ngắt), ghi toàn bộ file minh chứng
    (`config.json`, `env.json`, `train.log`, `history.csv`, `best.pt`, `learning_curves.png`)
    vào `run_dir` và trả về đường dẫn `run_dir`.
    """
    cfg_dict = _get_default_baseline_cfg()
    cfg_dict.update(_cfg_to_dict(cfg))

    seed = int(cfg_dict.get("seed", 42))
    set_seed(seed)

    run_dir = resolve_run_dir(cfg_dict)
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    env_path = run_dir / "env.json"
    log_path = run_dir / "train.log"
    history_path = run_dir / "history.csv"
    best_pt_path = run_dir / "best.pt"
    last_ckpt_path = run_dir / "last_ckpt.pt"

    can_resume = bool(cfg_dict.get("resume", True) and last_ckpt_path.exists())
    logger = DualLogger(log_path, append=can_resume)

    start_iso = local_now().isoformat(timespec="seconds")
    env_info = get_env_info(start_time=start_iso)
    with open(env_path, mode="w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2, ensure_ascii=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # 1. Nạp DataLoaders
        if dataloaders_override is not None:
            train_loader, valid_loader, _, class_weights = dataloaders_override
        else:
            train_loader, valid_loader, _, class_weights = _load_dataloaders_from_tv1(cfg_dict)

        # 2. Khởi tạo Model
        builder = model_builder_override or build_model
        model = builder(cfg_dict).to(device)
        num_params = count_parameters(model, trainable_only=True)

        # Lưu config.json kèm số tham số và git commit hash
        cfg_to_save = dict(cfg_dict)
        cfg_to_save["num_parameters"] = num_params
        cfg_to_save["git_commit"] = get_git_commit_hash()
        cfg_to_save["device"] = str(device)
        cfg_to_save["run_dir"] = str(run_dir)
        with open(config_path, mode="w", encoding="utf-8") as f:
            json.dump(cfg_to_save, f, indent=2, ensure_ascii=False)

        # 3. Khởi tạo Loss, Optimizer, Scheduler, EarlyStopping
        weight_tensor: Optional[torch.Tensor] = None
        if cfg_dict.get("use_class_weight", False) and class_weights is not None:
            weight_tensor = class_weights.to(device, dtype=torch.float32)
        train_criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        val_criterion = nn.CrossEntropyLoss()  # Validation luôn không trọng số để so sánh công bằng giữa các run

        optimizer = build_optimizer(model, cfg_dict)
        scheduler = build_scheduler(optimizer, cfg_dict)
        early_stopping = EarlyStopping(patience=int(cfg_dict.get("patience", 10)))

        use_amp = bool(cfg_dict.get("use_amp", False) and device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        start_epoch = 1
        max_epochs = int(cfg_dict.get("epochs", 100))
        best_val_acc = 0.0

        # 4. Kiểm tra Resume từ `last_ckpt.pt` nếu phiên trước bị ngắt
        if can_resume:
            ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            early_stopping.load_state_dict(ckpt["early_stopping"])
            start_epoch = int(ckpt["epoch"]) + 1
            best_val_acc = float(ckpt.get("best_val_acc", 0.0))
            logger.info(
                f"[RESUME] Nối tiếp huấn luyện từ epoch {start_epoch}/{max_epochs} "
                f"(best_val_loss={early_stopping.best_loss:.4f} tại epoch {early_stopping.best_epoch})"
            )

        csv_logger = CSVLogger(history_path, append=can_resume)

        logger.info("=" * 78)
        logger.info(
            f"Run: {cfg_dict.get('exp_id')} | Name: {cfg_dict.get('exp_name')} | "
            f"Seed: {seed} | Runner: {cfg_dict.get('runner', 'TV2')}"
        )
        logger.info(
            f"Arch: {cfg_dict.get('arch')} | Act: {cfg_dict.get('activation')} | "
            f"BN: {cfg_dict.get('use_bn')} | ConvDrop: {cfg_dict.get('conv_dropout')} | "
            f"Params: {num_params:,}"
        )
        logger.info(
            f"Optimizer: {cfg_dict.get('optimizer')} (lr={cfg_dict.get('lr')}, "
            f"wd={cfg_dict.get('weight_decay')}) | Scheduler: {cfg_dict.get('scheduler')} | "
            f"Batch: {cfg_dict.get('batch_size')} | Device: {device} ({env_info['gpu_name']})"
        )
        logger.info("=" * 78)

        total_train_start = time.perf_counter()

        # 5. Vòng lặp Epoch chính
        for epoch in range(start_epoch, max_epochs + 1):
            ep_start = time.perf_counter()

            train_loss, train_acc = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=train_criterion,
                optimizer=optimizer,
                device=device,
                use_amp=use_amp,
                scaler=scaler,
            )
            val_loss, val_acc = validate_one_epoch(
                model=model,
                loader=valid_loader,
                criterion=val_criterion,
                device=device,
                use_amp=use_amp,
            )

            current_lr = float(optimizer.param_groups[0]["lr"])
            if scheduler is not None:
                scheduler.step(val_loss)

            ep_elapsed = time.perf_counter() - ep_start

            # Ghi vào history.csv ngay sau mỗi epoch
            csv_logger.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": current_lr,
                    "elapsed_time": ep_elapsed,
                }
            )

            # Kiểm tra val_loss tốt nhất & EarlyStopping
            is_best = early_stopping.step(val_loss, epoch)
            best_marker = ""
            if is_best:
                best_val_acc = val_acc
                # Theo Mục 4: best.pt lưu state_dict tốt nhất theo val_loss
                torch.save(model.state_dict(), best_pt_path)
                best_marker = "  [*BEST*]"

            logger.info(
                f"Epoch {epoch:03d}/{max_epochs:03d} | "
                f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
                f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | "
                f"lr: {current_lr:.2e} | time: {ep_elapsed:.2f}s{best_marker}"
            )

            # Lưu checkpoint tạm để hỗ trợ resume nếu Colab bị ngắt kết nối
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "early_stopping": early_stopping.state_dict(),
                    "best_val_acc": best_val_acc,
                },
                last_ckpt_path,
            )

            if early_stopping.early_stop:
                logger.info(
                    f"[EarlyStopping] Dừng sớm tại epoch {epoch} (không cải thiện val_loss sau "
                    f"{early_stopping.patience} epoch). Best epoch: {early_stopping.best_epoch} "
                    f"(best_val_loss={early_stopping.best_loss:.4f}, val_acc={best_val_acc:.4f})."
                )
                break

        # Nếu vì lý do nào đó (ví dụ loss NaN ngay epoch 1) mà best.pt chưa được tạo, vẫn lưu model hiện tại
        if not best_pt_path.exists():
            torch.save(model.state_dict(), best_pt_path)

        total_elapsed = time.perf_counter() - total_train_start
        logger.info("-" * 78)
        logger.info(
            f"[FINISHED] Hoàn tất run tại: {run_dir}\n"
            f"           Best Epoch: {early_stopping.best_epoch} | "
            f"Best Val Loss: {early_stopping.best_loss:.4f} | "
            f"Val Acc tại Best Epoch: {best_val_acc:.4f} | "
            f"Tổng thời gian: {total_elapsed:.2f}s"
        )

        # Vẽ biểu đồ đường cong học tập (learning_curves.png)
        if history_path.exists():
            curve_path = plot_learning_curves(
                history_path,
                save_path=run_dir / "learning_curves.png",
                title_prefix=f"{cfg_dict.get('exp_id')} ({cfg_dict.get('exp_name')})",
            )
            logger.info(f"           Đã lưu biểu đồ đường cong huấn luyện: {curve_path}")

        # Xóa checkpoint tạm `last_ckpt.pt` sau khi hoàn thành để không dư file .pt thừa khi nộp (Mục 9)
        if not cfg_dict.get("keep_last_ckpt", False) and last_ckpt_path.exists():
            last_ckpt_path.unlink()

    finally:
        # Cập nhật end_time vào env.json
        end_iso = local_now().isoformat(timespec="seconds")
        env_info["end_time"] = end_iso
        with open(env_path, mode="w", encoding="utf-8") as f:
            json.dump(env_info, f, indent=2, ensure_ascii=False)
        logger.close()

    return run_dir


def test_overfit_one_batch(
    arch: str = "medium",
    batch_size: int = 16,
    steps: int = 60,
    lr: float = 1e-3,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Kiểm tra tính đúng đắn của mô hình và vòng lặp huấn luyện theo Mục 2.3 & 6.2:
    "Kiểm tra đúng đắn trước khi chạy dài: overfit 1 batch nhỏ, loss phải về gần 0."
    Chạy trực tiếp trên CPU/GPU với 1 batch cố định (tắt dropout).
    """
    print("=" * 72)
    print(f"=== OVERFIT 1 BATCH TEST (arch={arch}, batch_size={batch_size}, steps={steps}) ===")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tạo 1 batch ảnh giả lập chuẩn hóa [0, 1] kích thước (B, 1, 48, 48) và nhãn 0..6
    x_batch = torch.rand(batch_size, 1, 48, 48, device=device)
    y_batch = torch.randint(0, 7, (batch_size,), device=device)

    model = build_model(
        {
            "arch": arch,
            "activation": "relu",
            "use_bn": False,
            "conv_dropout": 0.0,  # Tắt dropout khi test overfit 1 batch
            "fc_dropout": 0.0,
        }
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    initial_loss = 0.0
    final_loss = 0.0
    final_acc = 0.0

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        acc = (preds == y_batch).float().mean().item()
        if step == 1:
            initial_loss = loss.item()
        if step == 1 or step % 10 == 0 or step == steps:
            print(f"  Step {step:03d}/{steps:03d} | loss: {loss.item():.6f} | acc: {acc * 100:6.2f}%")
        final_loss = loss.item()
        final_acc = acc

    print(
        f"-> Kết quả: loss giảm từ {initial_loss:.4f} xuống {final_loss:.6f}, "
        f"accuracy đạt {final_acc * 100:.2f}%"
    )
    assert final_loss < 0.05 and final_acc == 1.0, (
        f"Overfit 1 batch chưa đạt kỳ vọng: final_loss={final_loss}, final_acc={final_acc}"
    )
    print("=== OVERFIT 1 BATCH PASSED (Loss -> ~0, Acc = 100%) ===")
    print("=" * 72)
    return final_loss, final_acc


def run_smoke_test_pipeline(output_dir: Optional[Path] = None) -> Path:
    """
    Chạy kiểm tra toàn bộ pipeline `train(cfg)` trong 3 epoch trên dữ liệu giả lập nhỏ
    để xác nhận đủ 6 file minh chứng được sinh ra đúng chuẩn Mục 4.
    """
    import tempfile

    print("\n=== SMOKE TEST FULL TRAIN PIPELINE (3 EPOCHS) ===")
    set_seed(42)
    x_train = torch.rand(32, 1, 48, 48)
    y_train = torch.randint(0, 7, (32,))
    x_val = torch.rand(16, 1, 48, 48)
    y_val = torch.randint(0, 7, (16,))

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=16, shuffle=True)
    valid_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=16, shuffle=False)

    if output_dir is None:
        smoke_dir = _ROOT_DIR / "results" / "runs" / "_smoke_test_tv2_s42"
        try:
            smoke_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            smoke_dir = Path(tempfile.gettempdir()) / "fer2013_smoke_test_tv2_s42"
    else:
        smoke_dir = Path(output_dir)

    if smoke_dir.exists():
        import shutil
        shutil.rmtree(smoke_dir, ignore_errors=True)

    cfg = {
        "exp_id": "SMOKE",
        "exp_name": "test_tv2",
        "arch": "medium",
        "epochs": 3,
        "patience": 2,
        "batch_size": 16,
        "lr": 1e-3,
        "seed": 42,
        "run_dir": str(smoke_dir),
        "resume": False,
    }

    out_dir = train(cfg, dataloaders_override=(train_loader, valid_loader, None, None))
    expected_files = [
        "config.json",
        "env.json",
        "train.log",
        "history.csv",
        "best.pt",
        "learning_curves.png",
    ]
    for fname in expected_files:
        fpath = out_dir / fname
        assert fpath.exists(), f"Thiếu file minh chứng bắt buộc: {fpath}"
    print(f"=== FULL PIPELINE SMOKE TEST PASSED! Đủ 6 file trong: {out_dir} ===")
    return out_dir


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Huấn luyện CNN nhận diện cảm xúc FER2013 (Thành viên 2)"
    )
    parser.add_argument("--exp_id", type=str, default="E1-02", help="Mã thí nghiệm (VD: E1-01, E1-02)")
    parser.add_argument("--exp_name", type=str, default="baseline_medium", help="Tên ngắn của lần chạy")
    parser.add_argument("--runner", type=str, default="TV2", help="Tên thành viên chạy")
    parser.add_argument(
        "--arch",
        type=str,
        default="medium",
        choices=["small", "medium", "medium_01", "large"],
        help="Kiến trúc mô hình",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "leaky_relu", "elu", "sigmoid"],
        help="Hàm kích hoạt",
    )
    parser.add_argument("--use_bn", action="store_true", help="Bật BatchNorm2d sau mỗi Conv2d")
    parser.add_argument("--conv_dropout", type=float, default=0.2, help="Dropout sau mỗi block Conv")
    parser.add_argument("--fc_dropout", type=float, default=0.5, help="Dropout ở tầng Classifier")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=["none", "reduce_on_plateau"],
        help="Learning rate scheduler",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--aug_name", type=str, default="flip_affine", help="Cấu hình augmentation (TV1)")
    parser.add_argument("--use_class_weight", action="store_true", help="Dùng class weight cho CrossEntropyLoss")
    parser.add_argument("--norm", type=str, default="div255", help="Kiểu chuẩn hóa dữ liệu (TV1)")
    parser.add_argument("--epochs", type=int, default=100, help="Số epoch tối đa")
    parser.add_argument("--patience", type=int, default=10, help="Patience cho EarlyStopping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_amp", action="store_true", help="Bật Mixed Precision (torch.autocast)")
    parser.add_argument("--no_resume", action="store_true", help="Chạy lại từ đầu, không resume")
    parser.add_argument(
        "--overfit_one_batch",
        action="store_true",
        help="Chạy bài test overfit 1 batch nhỏ (loss -> 0) để kiểm tra mô hình",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Chạy thử toàn bộ pipeline train(cfg) 3 epoch trên dữ liệu giả lập",
    )
    return parser


if __name__ == "__main__":
    args = _build_cli_parser().parse_args()

    if args.overfit_one_batch:
        test_overfit_one_batch(arch=args.arch, lr=args.lr, seed=args.seed)
    elif args.smoke_test:
        test_overfit_one_batch(arch=args.arch, lr=args.lr, seed=args.seed)
        run_smoke_test_pipeline()
    else:
        cli_cfg = {
            "exp_id": args.exp_id,
            "exp_name": args.exp_name,
            "runner": args.runner,
            "arch": args.arch,
            "activation": args.activation,
            "use_bn": args.use_bn,
            "conv_dropout": args.conv_dropout,
            "fc_dropout": args.fc_dropout,
            "optimizer": args.optimizer,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
            "batch_size": args.batch_size,
            "aug_name": args.aug_name,
            "use_class_weight": args.use_class_weight,
            "norm": args.norm,
            "epochs": args.epochs,
            "patience": args.patience,
            "seed": args.seed,
            "use_amp": args.use_amp,
            "resume": not args.no_resume,
        }
        train(cli_cfg)
