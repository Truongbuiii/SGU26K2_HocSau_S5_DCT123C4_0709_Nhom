"""Optional Grad-CAM for a saved in-repo FERCNN checkpoint.

Author: Lê Thị Thanh Tuyền – 3123411332 | TV4: visualizations and analysis | 24/11/2026
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

try:
    from dataset import CLASS_NAMES, get_dataloaders
    from models import build_model, get_last_conv
except ImportError:
    from src.dataset import CLASS_NAMES, get_dataloaders
    from src.models import build_model, get_last_conv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, type=Path)
    p.add_argument("--csv_path", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("results/figures/gradcam.png"))
    p.add_argument("--count", type=int, default=14)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    model = build_model(cfg).to(device)
    state = torch.load(args.run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state) if isinstance(state, dict) else state)
    model.eval()
    layer = get_last_conv(model)
    features, gradients = [], []
    def save_activation(_module, _inputs, output):
        features.append(output)
        output.register_hook(lambda grad: gradients.append(grad))
    handle = layer.register_forward_hook(save_activation)
    _, loader, _, _ = get_dataloaders(batch_size=args.count, aug_name="none", norm=cfg.get("norm", "div255"), csv_path=str(args.csv_path) if args.csv_path else None)
    x, y = next(iter(loader))
    x = x.to(device)
    display_x = x.detach().clone()
    if cfg.get("norm") == "meanstd":
        normalize = next(t for t in loader.dataset.transform.transforms if hasattr(t, "mean") and hasattr(t, "std"))
        display_x = (display_x * float(normalize.std[0]) + float(normalize.mean[0])).clamp(0, 1)
    logits = model(x)
    targets = logits.argmax(1)
    model.zero_grad(set_to_none=True)
    logits[torch.arange(len(x), device=device), targets].sum().backward()
    handle.remove()
    feat, grad = features[-1].detach(), gradients[-1].detach()
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cams = F.relu((weights * feat).sum(1, keepdim=True))
    cams = F.interpolate(cams, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cams = cams / (cams.amax((1, 2), keepdim=True) + 1e-8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = 7
    fig, axes = plt.subplots((len(x) + cols - 1) // cols, cols, figsize=(15, 2.6 * ((len(x) + cols - 1) // cols)), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < len(x):
            gray = display_x[i, 0].detach().cpu().numpy()
            heat = cams[i].cpu().numpy()
            ax.imshow(gray, cmap="gray")
            ax.imshow(heat, cmap="jet", alpha=0.42, vmin=0, vmax=1)
            ax.set_title(f"true {CLASS_NAMES[int(y[i])]} / pred {CLASS_NAMES[int(targets[i])]}" , fontsize=8)
    fig.suptitle("Grad-CAM on PublicTest")
    fig.tight_layout()
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
