"""MLP tuy bien: so tang an C, so no-ron moi tang B."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_in=2, n_out=1, hidden=(10,), activation="tanh"):
        super().__init__()
        act = {"tanh": nn.Tanh, "relu": nn.ReLU, "sigmoid": nn.Sigmoid}[activation]
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), act()]
            prev = h
        layers.append(nn.Linear(prev, n_out))   # dau ra tuyen tinh (hoi quy)
        self.net = nn.Sequential(*layers)

    def forward(self, x):                       # y_hat = f_theta(x)
        return self.net(x)
