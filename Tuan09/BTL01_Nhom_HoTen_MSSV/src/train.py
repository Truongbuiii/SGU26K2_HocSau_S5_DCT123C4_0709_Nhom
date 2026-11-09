"""Vong lap train/validation tu xay dung + danh gia."""
import copy, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data import normalize, Y_MAX
from model import MLP

DEFAULT_CFG = dict(hidden=(10,), activation="tanh", optimizer="adam", lr=1e-2,
                   batch_size=32, epochs=200)


def to_tensors(x, y):
    xn, yn = normalize(x, y)
    return (torch.tensor(xn, dtype=torch.float32),
            torch.tensor(yn, dtype=torch.float32).unsqueeze(1))


@torch.no_grad()
def predict(model, x_t):
    model.eval()
    return model(x_t)


def metrics(model, x_t, y_t):
    """MSE/MAE/RMSE/R2 theo DON VI GOC cua y (nhan lai Y_MAX)."""
    p = predict(model, x_t) * Y_MAX
    t = y_t * Y_MAX
    err = p - t
    mse = (err ** 2).mean().item()
    return dict(mse=mse, mae=err.abs().mean().item(), rmse=mse ** 0.5,
                r2=1 - (err ** 2).sum().item() / ((t - t.mean()) ** 2).sum().item())


def train_model(splits, n_train, cfg, seed):
    cfg = {**DEFAULT_CFG, **cfg}
    torch.manual_seed(seed); np.random.seed(seed)
    xtr, ytr = splits["train"][0][:n_train], splits["train"][1][:n_train]   # tap con long nhau
    Xtr, Ytr = to_tensors(xtr, ytr)
    Xva, Yva = to_tensors(*splits["val"])
    Xte, Yte = to_tensors(*splits["test"])

    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=cfg["batch_size"], shuffle=True, generator=g)
    model = MLP(2, 1, cfg["hidden"], cfg["activation"])
    criterion = nn.MSELoss()
    opt_cls = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}[cfg["optimizer"]]
    optimizer = opt_cls(model.parameters(), lr=cfg["lr"])

    hist, best_val, best_state, best_epoch = [], float("inf"), None, -1
    t0 = time.perf_counter()
    for epoch in range(cfg["epochs"]):
        model.train()
        run, n = 0.0, 0
        for xb, yb in loader:
            pred = model(xb)                 # forward
            loss = criterion(pred, yb)       # loss
            optimizer.zero_grad()            # zero_grad
            loss.backward()                  # backward
            optimizer.step()                 # cap nhat tham so
            run += loss.item() * len(xb); n += len(xb)
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xva), Yva).item()
        hist.append((epoch, run / n * Y_MAX ** 2, val_loss * Y_MAX ** 2))   # don vi goc
        if val_loss < best_val:              # checkpoint theo validation loss
            best_val, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
    train_time = time.perf_counter() - t0

    model.load_state_dict(best_state)        # khoi phuc mo hinh tot nhat
    res = dict(best_epoch=best_epoch, train_time=train_time, n_params=sum(p.numel() for p in model.parameters()))
    for name, (X, Y) in dict(train=(Xtr, Ytr), val=(Xva, Yva), test=(Xte, Yte)).items():
        for k, v in metrics(model, X, Y).items():
            res[f"{name}_{k}"] = v
    return model, hist, res
