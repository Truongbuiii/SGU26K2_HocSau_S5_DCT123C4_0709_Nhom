# BTL01 – Xấp xỉ hàm Gaussian 2 chiều bằng MLP (PyTorch)

Môn Học sâu – Bài tập xấp xỉ hàm số dùng mạng MLP.
Hàm mục tiêu: Gaussian 2 chiều với μ = [0, 0], C = [[0.3, −0.5], [−0.5, 2]], miền x1 ∈ [−2, 2], x2 ∈ [−4, 4].

## Cấu trúc thư mục
```
BTL01/
├── requirements.txt
├── README.md
├── src/
│   ├── data.py             # công thức Gaussian, sinh/chia dữ liệu, chuẩn hóa
│   ├── model.py            # MLP (nn.Module, nn.Linear) tuỳ chỉnh số tầng/nơ-ron
│   ├── train.py            # DataLoader, vòng lặp train/val, checkpoint, MSE/MAE/RMSE/R²
│   ├── run_experiments.py  # chạy dữ liệu, baseline, thí nghiệm A/B/C, xuất hình + CSV
│   └── make_tables.py      # bảng JSON cho báo cáo + sơ đồ kiến trúc
├── report/                 # BaoCao_BTL01.docx / .pdf
├── slides/                 # Slide_BTL01.pptx / .pdf
├── data/                   # train.csv (pool 2000), val.csv (500), test.csv (100)
└── results/
    ├── figures/            # hình (PNG)
    ├── logs/               # CSV từng lần chạy (exp_*_runs.csv), tổng hợp (exp_*_summary.csv), history
    └── models/             # baseline_best.pt (trọng số tốt nhất theo validation)
```

## Cài đặt
```
python -m venv .venv && source .venv/bin/activate     # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```
Chỉ cần CPU; chạy GPU không bắt buộc.

## Lệnh chạy (thực hiện trong thư mục BTL01/)
| Mục đích | Lệnh |
|---|---|
| Kiểm tra công thức Gaussian với SciPy | `python src/data.py` |
| Sinh/lưu/vẽ dữ liệu | `python src/run_experiments.py data` |
| Train baseline + hình xấp xỉ | `python src/run_experiments.py baseline` |
| Thí nghiệm A (số mẫu) | `python src/run_experiments.py A` |
| Thí nghiệm B (số nơ-ron) | `python src/run_experiments.py B` |
| Thí nghiệm bổ sung B với A=100 | `python src/run_experiments.py Bsmall` |
| Thí nghiệm C (số tầng ẩn) | `python src/run_experiments.py C` |
| Chạy toàn bộ (≈ 12–15 phút trên 1 nhân CPU) | `python src/run_experiments.py all` |
| Chỉ vẽ lại hình từ CSV có sẵn | `python src/run_experiments.py plots` |

## Thiết kế thực nghiệm (tóm tắt)
* **Dữ liệu**: lấy mẫu đều trong miền, nhãn y = f(x) chính xác (không nhiễu). Ba tập độc lập, seed cố định: train pool 2000 (seed 42), validation 500 (seed 43), test 100 (seed 44). Thí nghiệm A lấy A mẫu *đầu tiên* của train pool (các tập lồng nhau).
* **Chuẩn hóa cố định** (suy ra từ công thức, không dùng thống kê dữ liệu → không rò rỉ): x₁/2, x₂/4, y/f(μ) với f(μ)=1/(2π√det C)≈0.269. Mọi sai số báo cáo theo **đơn vị gốc của y**.
* **Baseline**: A=1000, 1 tầng ẩn 10 nơ-ron, tanh, đầu ra tuyến tính, Adam lr=1e-2, batch 32, 200 epoch, MSELoss, không early-stopping; checkpoint theo validation MSE nhỏ nhất.
* **Test** chỉ dùng để báo cáo cuối; mọi lựa chọn checkpoint dựa trên validation.
* Mỗi cấu hình chạy với 3 seed khởi tạo/xáo trộn (0, 1, 2); bảng báo cáo trung bình ± độ lệch chuẩn.
