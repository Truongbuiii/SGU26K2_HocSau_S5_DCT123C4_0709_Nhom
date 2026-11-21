"""TV4 visualizations for FER2013 CNN checkpoints.

Author (TV4): [fill in name and student ID] | Role: visualizations and analysis

Usage from Tuan11/: python src/visualize_filters.py --run_dir results/runs/E3-06_meanstd_s42
Writes first-layer kernels, last-convolution feature maps, and activation-maximization
examples under results/figures. Requires the run's config.json and best.pt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

try:
    from dataset import CLASS_NAMES, get_dataloaders
    from models import build_model, get_last_conv
except ImportError:
    from src.dataset import CLASS_NAMES, get_dataloaders
    from src.models import build_model, get_last_conv


def _model(run_dir: Path, device: torch.device):
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    model = build_model(cfg).to(device)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint)
    model.eval()
    return model, cfg


def plot_first_kernels(model, out: Path, count: int = 32):
    conv = next(m for m in model.modules() if isinstance(m, torch.nn.Conv2d))
    weights = conv.weight.detach().cpu()
    n = min(count, len(weights))
    cols = 8
    fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(12, 1.7 * ((n + cols - 1) // cols)))
    for i, ax in enumerate(np.asarray(axes).reshape(-1)):
        ax.axis("off")
        if i < n:
            k = weights[i].mean(0).numpy()
            ax.imshow(k, cmap="gray")
            ax.set_title(f"kernel {i}", fontsize=8)
    fig.suptitle("First convolution layer kernels")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feature_maps(model, images, labels, out: Path, maps_per_image: int = 16):
    layer = get_last_conv(model)
    captured = []
    hook = layer.register_forward_hook(lambda _m, _i, o: captured.append(o.detach().cpu()))
    with torch.no_grad():
        model(images)
    hook.remove()
    acts = captured[0]
    n, cols = len(images), 4
    fig, axes = plt.subplots(n, cols + 1, figsize=(13, 3.1 * n), squeeze=False)
    for row in range(n):
        axes[row, 0].imshow(images[row, 0].cpu(), cmap="gray")
        axes[row, 0].set_title(CLASS_NAMES[int(labels[row])])
        axes[row, 0].axis("off")
        # Average activation groups keep the 256-channel final layer readable.
        chunks = torch.tensor_split(acts[row], cols)
        for j, ax in enumerate(axes[row, 1:]):
            ax.imshow(chunks[j].mean(0), cmap="magma")
            ax.set_title(f"channels {j * (acts.shape[1] // cols)}–{(j + 1) * (acts.shape[1] // cols) - 1}")
            ax.axis("off")
    fig.suptitle("Last-convolution feature maps (channel-group means)")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def activation_maximization(model, out: Path, channels=(0, 16, 64, 128), steps=160, lr=0.08):
    """Gradient-ascent input patterns for selected last-layer channels."""
    layer = get_last_conv(model)
    captured = []
    handle = layer.register_forward_hook(lambda _m, _i, o: captured.append(o))
    rows = []
    for channel in channels:
        x = torch.rand(1, 1, 48, 48, device=next(model.parameters()).device, requires_grad=True)
        opt = torch.optim.Adam([x], lr=lr)
        for _ in range(steps):
            captured.clear()
            opt.zero_grad(set_to_none=True)
            model(x)
            # Encourage smooth, bounded patterns while maximizing channel response.
            tv = (x[:, :, 1:] - x[:, :, :-1]).abs().mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
            loss = -captured[-1][:, channel].mean() + 0.02 * tv + 0.001 * (x - 0.5).square().mean()
            loss.backward()
            opt.step()
            with torch.no_grad():
                x.clamp_(0, 1)
        rows.append(x.detach().cpu()[0, 0].numpy())
    handle.remove()
    fig, axes = plt.subplots(1, len(rows), figsize=(3 * len(rows), 3))
    for ax, image, channel in zip(np.asarray(axes).reshape(-1), rows, channels):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"filter {channel}")
        ax.axis("off")
    fig.suptitle("Activation maximization: input patterns that excite final-layer filters")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, type=Path)
    p.add_argument("--csv_path", type=Path, default=None)
    p.add_argument("--out_dir", type=Path, default=Path("results/figures"))
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = _model(args.run_dir, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.run_dir.name
    plot_first_kernels(model, args.out_dir / f"{tag}_first_kernels.png")
    _, valid_loader, _, _ = get_dataloaders(batch_size=64, aug_name="none", norm=cfg.get("norm", "div255"), csv_path=str(args.csv_path) if args.csv_path else None)
    selected, labels, seen = [], [], set()
    for x, y in valid_loader:
        for image, label in zip(x, y):
            label = int(label)
            if label not in seen:
                selected.append(image)
                labels.append(label)
                seen.add(label)
            if len(seen) == len(CLASS_NAMES):
                break
        if len(seen) == len(CLASS_NAMES):
            break
    images = torch.stack(selected).to(device)
    plot_feature_maps(model, images, labels, args.out_dir / f"{tag}_last_feature_maps.png")
    activation_maximization(model, args.out_dir / f"{tag}_activation_maximization.png")
    print(f"Saved visualizations to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
