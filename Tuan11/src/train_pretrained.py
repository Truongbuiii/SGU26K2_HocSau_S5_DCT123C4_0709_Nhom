"""TV4 E4 transfer-learning runs (ResNet18 / VGG16 ImageNet weights).

Author (TV4): [fill in name and student ID] | Role: E4 and final model analysis

Run from Tuan11/: python src/train_pretrained.py --arch resnet18 --seed 42
Kaggle Internet must be enabled for the first run to download torchvision weights.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights, vgg16, VGG16_Weights

try:
    from dataset import get_dataloaders
except ImportError:
    from src.dataset import get_dataloaders


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def make_model(arch, device):
    if arch == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 7)
    else:
        model = vgg16(weights=VGG16_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 7)
    return model.to(device)


def prepare(x):
    # FER input is grayscale [0,1]. Replicate channels, resize, then ImageNet normalize.
    x = F.interpolate(x.repeat(1, 3, 1, 1), size=(96, 96), mode="bilinear", align_corners=False)
    mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (x - mean) / std


def epoch(model, loader, optimizer, device, scaler=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = total = 0
    for x, y in loader:
        x, y = prepare(x.to(device, non_blocking=True)), y.to(device, non_blocking=True)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, enabled=(scaler is not None)):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        if training:
            if scaler: scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else: loss.backward(); optimizer.step()
        loss_sum += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item(); total += len(y)
    return loss_sum / total, correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["resnet18", "vgg16"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--csv_path", type=Path, default=None)
    p.add_argument("--output_root", type=Path, default=Path("results/runs"))
    p.add_argument("--no_amp", action="store_true")
    args = p.parse_args()
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, _, _ = get_dataloaders(batch_size=args.batch_size, aug_name="flip_affine", norm="div255", seed=args.seed, csv_path=str(args.csv_path) if args.csv_path else None)
    model = make_model(args.arch, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if amp else None
    run_dir = args.output_root / f"E4-{args.arch}_{args.arch}_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = vars(args).copy(); cfg.update(device=str(device), pretrained=True, input_size=96, channels=3, norm="ImageNet", amp=amp)
    cfg["csv_path"] = str(args.csv_path) if args.csv_path else None
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
    best, stale, rows = float("inf"), 0, []
    started = time.time()
    for ep in range(1, args.epochs + 1):
        t = time.time()
        tr_loss, tr_acc = epoch(model, train_loader, optimizer, device, scaler)
        va_loss, va_acc = epoch(model, valid_loader, None, device)
        rows.append(dict(epoch=ep, train_loss=tr_loss, train_acc=tr_acc, val_loss=va_loss, val_acc=va_acc, seconds=time.time()-t))
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
        print(f"epoch {ep:02d} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f}", flush=True)
        if va_loss < best:
            best, stale = va_loss, 0
            torch.save(model.state_dict(), run_dir / "best.pt")
            (run_dir / "best_metrics.json").write_text(json.dumps({"epoch": ep, "val_loss": va_loss, "val_acc": va_acc}, indent=2), encoding="utf-8")
        else:
            stale += 1
            if stale >= args.patience: break
    (run_dir / "env.json").write_text(json.dumps({"torch": torch.__version__, "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "seconds": time.time()-started}, indent=2), encoding="utf-8")
    print(f"Saved run to {run_dir}", flush=True)


if __name__ == "__main__": main()
