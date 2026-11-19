# BTL02 – Nhận diện cảm xúc khuôn mặt trên FER2013 (PyTorch)

Bài tập lớn 02 môn **Học sâu** – Huấn luyện mạng CNN bằng PyTorch để phân loại 7 cảm xúc khuôn mặt trên bộ dữ liệu **FER2013** (ảnh xám 48×48).

---

## 1. Bảng phân công thành viên

| Thành viên | Họ tên – MSSV | Vai trò | File code chính | Nhóm thí nghiệm |
|---|---|---|---|---|
| **TV1 (Nhóm trưởng)** | Bùi Đức Trường – 3123411319 | Dữ liệu & pipeline, giữ repo, ghép code | `dataset.py`, `augment.py`, `eda.py`, `README.md` | **E3**: augmentation, class weight, chuẩn hóa |
| **TV2** | *(Điền họ tên – MSSV)* | Mô hình & vòng lặp huấn luyện | `models.py`, `train.py`, `utils.py` | **E1**: kiến trúc CNN, activation, dropout, BatchNorm |
| **TV3** | *(Điền họ tên – MSSV)* | Thực nghiệm hàng loạt, đánh giá & nhật ký | `config.py`, `run_experiments.py`, `evaluate.py`, `aggregate.py` | **E2**: optimizer, lr, scheduler, batch size, weight decay |
| **TV4** | *(Điền họ tên – MSSV)* | Trực quan hóa filter, phân tích, ghép báo cáo/slide | `visualize_filters.py` (+ `gradcam.py`) | **E4**: pretrained fine-tune, đa seed cho mô hình tốt nhất |

---

## 2. Cấu trúc thư mục

```text
Tuan11/
├── README.md
├── requirements.txt
├── docs/                       # Slide đề bài, báo cáo mẫu, tài liệu phân công
├── data/                       # Đặt fer2013.csv tại đây (không commit lên Git)
├── src/
│   ├── dataset.py              # TV1: Nạp CSV, CLASS_NAMES, get_dataloaders(...)
│   ├── augment.py              # TV1: Các cấu hình augmentation (none, flip, flip_affine, ...)
│   ├── eda.py                  # TV1: Vẽ phân bố lớp, lưới ảnh mẫu, lưới augmentation
│   ├── models.py               # TV2: FERCNN (small/medium/large), build_model, get_last_conv
│   ├── train.py                # TV2: Vòng huấn luyện, checkpoint, EarlyStopping, log
│   └── utils.py                # TV2: set_seed, EarlyStopping, CSVLogger, plot_learning_curves
├── results/
│   ├── figures/                # Hình EDA, learning curves, confusion matrix, filter
│   └── runs/                   # Thư mục minh chứng từng run (<ExpID>_<ten>_s<seed>/)
└── evidence/                   # Ảnh chụp màn hình minh chứng chạy theo từng thành viên
```

---

## 3. Quy ước dữ liệu & Baseline

- **Thứ tự nhãn thống nhất (`CLASS_NAMES` trong `src/dataset.py`)**:
  `0: Angry`, `1: Disgust`, `2: Fear`, `3: Happy`, `4: Sad`, `5: Surprise`, `6: Neutral`.
- **Tách tập**: `Training` (28.709) → train; `PublicTest` (3.589) → validation (chọn epoch/siêu tham số); `PrivateTest` (3.589) → test cuối.
- **Baseline mặc định**: `Medium` (2×64, 3×128, 3×256, FC 1024), ReLU, không BatchNorm, conv dropout 0.2, fc dropout 0.5, `flip_affine`, chuẩn hóa chia 255, Adam `lr=1e-3`, `batch_size=64`, tối đa 100 epoch, `EarlyStopping(patience=10)` theo `val_loss`, `seed=42`.

---

## 4. Cài đặt & Chuẩn bị dữ liệu

1. Cài thư viện:
   ```bash
   pip install -r requirements.txt
   ```
2. Tải `fer2013.csv` (301 MB, có đủ 3 cột `emotion, pixels, Usage`) từ Kaggle:
   <https://www.kaggle.com/datasets/deadskull7/fer2013>
   và đặt vào `Tuan11/data/fer2013.csv` (hoặc đặt biến môi trường `FER2013_CSV=/duong_dan/fer2013.csv`).

---

## 5. Hướng dẫn chạy nhanh

Chạy từ thư mục `Tuan11/`:

```bash
# 1. Kiểm tra dữ liệu (phải đạt 28709 / 3589 / 3589)
python src/dataset.py

# 2. Kiểm tra augmentation & xuất hình EDA vào results/figures/
python src/augment.py
python src/eda.py

# 3. Kiểm tra mô hình & overfit 1 batch (trước khi chạy dài)
python src/models.py
python src/train.py --overfit_one_batch

# 4. Huấn luyện 1 cấu hình (ví dụ baseline)
python src/train.py --exp_id E1-02 --exp_name baseline_medium --runner TV2
```

### Giao diện hàm dùng chung (TV3 & TV4 gọi lại)
- `from dataset import CLASS_NAMES, get_dataloaders`
  → `train_loader, valid_loader, test_loader, class_weights = get_dataloaders(batch_size=64, aug_name="flip_affine", use_class_weight=False, norm="div255", seed=42)`
- `from models import build_model, get_last_conv`
- `from train import train` → `run_dir = train(cfg)`
