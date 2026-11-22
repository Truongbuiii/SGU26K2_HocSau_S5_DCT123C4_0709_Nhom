"""
Vai tro: TV3 - Thuc nghiem va danh gia
File   : config.py - Config (dataclass), BASELINE, anh xa ExpID -> Config

QUY TAC: moi ExpID chi doi MOT yeu to so voi BASELINE.
Ten truong da KHOP voi src/train.py, models.py, dataset.py cua nhom (da doi chieu):
  aug_name, norm, use_class_weight, batch_size, seed | arch, activation, use_bn, conv_dropout,
  fc_dropout | optimizer, lr, momentum, weight_decay, scheduler, scheduler_factor,
  scheduler_patience, epochs, patience, use_amp, output_root
E4 (pretrained) do TV4 chay bang train_pretrained.py rieng nen khong dinh nghia o day.
"""
from dataclasses import dataclass, asdict, replace, fields
import json
import os


@dataclass
class Config:
    # --- dinh danh ---
    exp_id: str = "E1-02"
    exp_name: str = "baseline_medium"
    seed: int = 42
    # --- mo hinh (TV2) ---
    arch: str = "medium"          # small | medium | large | resnet18 | vgg16
    activation: str = "relu"      # relu | leaky_relu | elu | sigmoid
    use_bn: bool = False
    conv_dropout: float = 0.2
    fc_dropout: float = 0.5
    # --- du lieu (TV1) ---
    aug_name: str = "flip_affine"  # none | flip | flip_affine | flip_affine_erase | strong
    norm: str = "div255"          # div255 | meanstd
    use_class_weight: bool = False
    # --- toi uu (TV3 - E2) ---
    optimizer: str = "adam"       # adam | adamw | sgd
    lr: float = 1e-3
    momentum: float = 0.9         # chi dung cho sgd
    weight_decay: float = 0.0
    batch_size: int = 64
    scheduler: str = "none"       # none | reduce_on_plateau
    scheduler_factor: float = 0.1
    scheduler_patience: int = 3
    epochs: int = 100
    patience: int = 10            # EarlyStopping theo val_loss
    # --- meta (khong phai yeu to thi nghiem) ---
    runner: str = ""              # ten nguoi chay
    output_root: str = "results/runs"
    use_amp: bool = False         # bat/tat dong nhat cho MOI thi nghiem (muc 2.3)
    note: str = ""                # cau hoi thi nghiem can tra loi


# Cac truong KHONG tinh la "thay doi so voi baseline"
META_FIELDS = {"exp_id", "exp_name", "seed", "runner", "output_root", "note"}

BASELINE = Config()   # chot o D3 (muc 2.4b). Sau khi chot KHONG doi nua.


def _e(exp_id, exp_name, note="", **kw):
    return replace(BASELINE, exp_id=exp_id, exp_name=exp_name, note=note, **kw)


_LIST = [
    # ===== E2 (TV3 tu chay) =====
    _e("E2-01", "sgd_m09_lr1e-2", "SGD so voi Adam: toc do hoi tu va muc cuoi?",
       optimizer="sgd", lr=1e-2, momentum=0.9),
    _e("E2-02", "adamw_wd1e-2", "Regularization (weight decay) co giam overfit khong?",
       optimizer="adamw", weight_decay=1e-2),
    _e("E2-03", "adam_lr1e-2", "lr qua lon: dao dong hay phan ky?", lr=1e-2),
    _e("E2-04", "adam_lr1e-4", "lr nho: cham den muc nao, co tot hon khi hoi tu?", lr=1e-4),
    _e("E2-05", "adam_plateau", "ReduceLROnPlateau (0.1, patience 3) co cai thien val?",
       scheduler="reduce_on_plateau", scheduler_factor=0.1, scheduler_patience=3),
    _e("E2-06", "bs32", "Batch nho co tot hon khong, thoi gian tang bao nhieu?", batch_size=32),
    _e("E2-07", "bs128", "Batch lon co nhanh hon, co mat do chinh xac?", batch_size=128),
    # ===== E1 (TV2 chay) - TV3 chi dinh nghia cau hinh =====
    _e("E1-01", "small", "Mo hinh nho dat bao nhieu?", arch="small"),
    _e("E1-03", "large", "Co bao hoa/overfit khi qua sau?", arch="large"),
    _e("E1-04", "medium_bn", "BatchNorm co hoi tu nhanh/on dinh hon?", use_bn=True),
    _e("E1-05a", "leaky_relu", "LeakyReLU", activation="leaky_relu"),
    _e("E1-05b", "elu", "ELU", activation="elu"),
    _e("E1-05c", "sigmoid", "Vi sao sigmoid hoc cham (vanishing gradient)?", activation="sigmoid"),
    _e("E1-06a", "dropout0", "Dropout conv = 0", conv_dropout=0.0),
    _e("E1-06b", "dropout05", "Dropout conv = 0.5", conv_dropout=0.5),
    # ===== E3 (TV1 chay) =====
    _e("E3-01", "no_aug", "Muc overfit khi khong augment?", aug_name="none"),
    _e("E3-02", "flip_only", "Lat ngang dong gop bao nhieu?", aug_name="flip"),
    _e("E3-04", "strong_aug", "Augment qua tay co giam accuracy?", aug_name="strong"),
    _e("E3-05", "class_weight", "Recall Disgust/Fear co cai thien?", use_class_weight=True),
    _e("E3-06", "norm_meanstd", "mean/std co anh huong toc do hoi tu?", norm="meanstd"),
]

EXPERIMENTS = {c.exp_id: c for c in _LIST}
BASELINE_IDS = ("E1-02", "E3-03")   # E3-03 cua TV1 cung la baseline (cung cau hinh)
ALIASES = {"E3-03": "E1-02", "E2-00": "E1-02"}   # dung lai run baseline
EXPERIMENTS["E1-02"] = BASELINE


def get_config(exp_id, seed=None, **overrides):
    """Lay Config theo ExpID. overrides chi dung khi that su can (se bi ghi vao config.json)."""
    exp_id = ALIASES.get(exp_id, exp_id)
    if exp_id not in EXPERIMENTS:
        raise KeyError(f"ExpID khong ton tai: {exp_id}. Hop le: {sorted(EXPERIMENTS)}")
    cfg = EXPERIMENTS[exp_id]
    if seed is not None:
        cfg = replace(cfg, seed=seed)
    return replace(cfg, **overrides) if overrides else cfg


def group_ids(prefix):
    """group_ids('E2') -> ['E2-01', ..., 'E2-07']"""
    return sorted(k for k in EXPERIMENTS if k.startswith(prefix + "-"))


def run_dir_name(cfg):
    return f"{cfg.exp_id}_{cfg.exp_name}_s{cfg.seed}"


def run_dir_path(cfg):
    return os.path.join(cfg.output_root, run_dir_name(cfg))


def cfg_to_dict(cfg):
    return asdict(cfg)


def cfg_from_dict(d):
    names = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in d.items() if k in names})


def diff_from_baseline(cfg):
    """Tra ve chuoi 'lr=0.01, optimizer=sgd' - cac yeu to khac baseline."""
    d = {k: v for k, v in asdict(cfg).items()
         if k not in META_FIELDS and v != getattr(BASELINE, k)}
    return ", ".join(f"{k}={v}" for k, v in d.items()) or "(baseline)"


if __name__ == "__main__":
    for k in group_ids("E2"):
        print(k, "->", run_dir_name(EXPERIMENTS[k]), "|", diff_from_baseline(EXPERIMENTS[k]))
