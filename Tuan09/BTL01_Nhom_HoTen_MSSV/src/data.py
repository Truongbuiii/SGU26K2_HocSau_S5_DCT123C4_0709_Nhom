"""Sinh, chia, luu va khao sat du lieu Gaussian 2 chieu."""
import numpy as np
import pandas as pd

MU = np.array([0.0, 0.0])
COV = np.array([[0.3, -0.5], [-0.5, 2.0]])
RANGE_X1 = (-2.0, 2.0)
RANGE_X2 = (-4.0, 4.0)

# Hang so chuan hoa CO DINH (suy ra tu cong thuc, khong phu thuoc du lieu -> khong ro ri):
X_SCALE = np.array([2.0, 4.0])                                   # dua x ve [-1, 1]
Y_MAX = 1.0 / (2 * np.pi * np.sqrt(np.linalg.det(COV)))          # f(mu) = dinh cua ham

# Seed co dinh cho tung tap -> tap doc lap, tai lap duoc
SEED_TRAIN, SEED_VAL, SEED_TEST = 42, 43, 44
N_TRAIN_POOL, N_VAL, N_TEST = 2000, 500, 100


def gaussian_2d(x, mean=MU, cov=COV):
    """f(x) = 1/(2*pi*sqrt(det C)) * exp(-0.5 (x-mu)^T C^-1 (x-mu)); x co shape (N, 2)."""
    x = np.atleast_2d(x)
    d = x - mean
    inv = np.linalg.inv(cov)
    maha = np.einsum("ni,ij,nj->n", d, inv, d)
    return np.exp(-0.5 * maha) / (2 * np.pi * np.sqrt(np.linalg.det(cov)))


def sample_domain(n, seed):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(*RANGE_X1, size=n)
    x2 = rng.uniform(*RANGE_X2, size=n)
    x = np.stack([x1, x2], axis=1)
    return x, gaussian_2d(x)


def make_splits():
    """Train pool (2000), validation (500), test (100) - 3 seed rieng, doc lap."""
    return {
        "train": sample_domain(N_TRAIN_POOL, SEED_TRAIN),
        "val": sample_domain(N_VAL, SEED_VAL),
        "test": sample_domain(N_TEST, SEED_TEST),
    }


def save_splits(splits, folder="data"):
    for name, (x, y) in splits.items():
        pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "y": y}).to_csv(f"{folder}/{name}.csv", index=False)


def normalize(x, y):
    return x / X_SCALE, y / Y_MAX


if __name__ == "__main__":
    from scipy.stats import multivariate_normal
    x, y = sample_domain(1000, 0)
    ref = multivariate_normal(MU, COV).pdf(x)
    print("max |f_ours - scipy| =", np.abs(y - ref).max())
    print("Y_MAX =", Y_MAX, " max y in pool =", make_splits()["train"][1].max())
