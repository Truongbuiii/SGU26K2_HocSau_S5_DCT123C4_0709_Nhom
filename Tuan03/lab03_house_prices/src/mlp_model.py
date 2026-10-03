"""MLP PyTorch cho bài toán dự đoán giá nhà (Chương 4)."""
import copy
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)


class HousePriceMLP(nn.Module):
    """Bản PyTorch của ANN Keras trong code mẫu: Dense(50)-Dense(25)-Dense(50)-Dense(1).
    Có thêm dropout / batchnorm tuỳ chọn để thử nghiệm."""

    def __init__(self, input_dim, hidden=(50, 25, 50), dropout=0.0, batchnorm=False,
                 y_mean=0.0):
        super().__init__()
        layers, d = [], input_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        self.network = nn.Sequential(*layers)
        # Khởi tạo bias lớp cuối = mean(log SalePrice) để hội tụ nhanh hơn
        nn.init.constant_(self.network[-1].bias, float(y_mean))

    def forward(self, x):
        return self.network(x).squeeze(-1)


DEFAULT_CFG = dict(hidden=(50, 25, 50), dropout=0.0, batchnorm=False, optimizer="adam",
                   lr=1e-3, weight_decay=0.0, batch_size=32, epochs=200, cosine=True)


def _make_opt(cfg, params):
    o = cfg["optimizer"]
    if o == "adam":
        return torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if o == "adamw":
        return torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if o == "adamax":      # optimizer của code Keras mẫu
        return torch.optim.Adamax(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if o == "sgd":
        return torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"])
    raise ValueError(o)


def to_tensor(X, y=None):
    X = torch.as_tensor(np.asarray(X, dtype=np.float32))
    assert torch.isfinite(X).all(), "X còn NaN/inf"
    if y is None:
        return X
    y = torch.as_tensor(np.asarray(y, dtype=np.float32)).reshape(-1)   # shape (N,)
    assert torch.isfinite(y).all(), "y còn NaN/inf"
    return X, y


def train_model(Xtr, ytr, Xva=None, yva=None, cfg=None, seed=0):
    """Training loop: forward -> loss -> backward -> optimizer step -> validation.
    Trả về (model, history). Không chọn epoch theo validation: dùng model ở epoch cuối."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    Xt, yt = to_tensor(Xtr, ytr)
    model = HousePriceMLP(Xt.shape[1], cfg["hidden"], cfg["dropout"], cfg["batchnorm"], yt.mean())
    opt = _make_opt(cfg, model.parameters())
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"]) if cfg["cosine"] else None
    loss_fn = nn.MSELoss()
    has_val = Xva is not None
    if has_val:
        Xv, yv = to_tensor(Xva, yva)
    hist = {"train_loss": [], "val_loss": []}
    n, bs = len(Xt), cfg["batch_size"]
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2 and cfg["batchnorm"]:
                continue
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])     # forward + loss
            loss.backward()                              # backward
            opt.step()                                   # optimizer step
        if sched:
            sched.step()
        model.eval()
        with torch.no_grad():
            hist["train_loss"].append(loss_fn(model(Xt), yt).item())
            if has_val:
                hist["val_loss"].append(loss_fn(model(Xv), yv).item())
    return model, hist


def predict_log(model, X):
    model.eval()
    with torch.no_grad():
        return model(to_tensor(X)).numpy()


def metrics(y_true_log, y_pred_log):
    """RMSE(log) = metric của Kaggle; RMSE/MAE/Bias/MaxDev tính bằng $ (theo paper)."""
    rmse_log = float(np.sqrt(np.mean((y_pred_log - y_true_log) ** 2)))
    yt, yp = np.expm1(y_true_log), np.expm1(y_pred_log)
    e = yp - yt
    return dict(rmse_log=rmse_log, rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))),
                bias=float(np.mean(e)), max_dev=float(np.max(np.abs(e))))
