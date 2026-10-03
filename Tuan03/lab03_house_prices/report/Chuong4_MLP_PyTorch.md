# Chương 4 – MLP bằng PyTorch
*Người phụ trách: Trúc*

## 4.1. Mục tiêu và thiết lập
Mục tiêu là dự đoán giá bán nhà Ames (De Cock, 2011) bằng mạng nơ-ron nhiều lớp (MLP) viết bằng PyTorch, dựa trên ý tưởng kiến trúc của ANN Keras trong code mẫu, rồi cải tiến bằng thực nghiệm.

- **Dữ liệu:** train 1460 dòng; theo khuyến nghị của paper, loại 4 nhà có `GrLivArea > 4000` còn 1456 dòng. Test 1459 dòng.
- **Target:** `log1p(SalePrice)`. Loss MSE trên thang log, nên RMSE-log chính là metric của Kaggle. Ngoài ra báo cáo RMSE, MAE, Bias, Max Deviation tính bằng đô-la, như paper.
- **Chia dữ liệu:** holdout 80/20 (`random_state=42`) cho learning curve; 5-fold CV (`random_state=42`) cho bảng so sánh. Preprocessor được fit riêng trên từng fold train nên không rò rỉ.

## 4.2. Tiền xử lý (input của MLP)
Bám theo pipeline code mẫu: bỏ `Id, Alley, PoolQC, Fence, MiscFeature`; numeric thiếu → mean, categorical thiếu → mode; one-hot (danh sách category lấy từ train+test). `MSSubClass` được coi là categorical. Bổ sung cho MLP: `log1p` cho 18 biến numeric lệch phải (skew > 0.75) và chuẩn hoá z-score. Input dimension là **288** (EXP-00) và **294** (EXP-03, gồm 6 feature mới). Số này khác 176 của code Keras vì không dùng `drop_first`.

## 4.3. Kiến trúc
Phiên bản chuyển từ Keras: `Linear(d,50)-ReLU-Linear(50,25)-ReLU-Linear(25,50)-ReLU-Linear(50,1)`. Bias lớp cuối khởi tạo bằng mean(log giá) để hội tụ nhanh. Tham số có thể thay đổi: số lớp/nơ-ron, dropout, batch norm.

| | Keras mẫu | PyTorch baseline | Sau tuning |
|---|---|---|---|
| Hidden | 50-25-50 | 50-25-50 | 256-128-64 (mạng 50-25-50 cho kết quả tương đương, xem 4.6) |
| Activation | ReLU | ReLU | ReLU |
| Loss | RMSE | MSE (log) | MSE (log) |
| Optimizer | Adamax | Adam, lr 1e-3 | Adam, lr 1e-3, weight decay 0.01 |
| Batch / Epoch | 10 / 1000 | 32 / 150 | 16 / 60 |
| LR schedule | – | cosine | cosine |

## 4.4. Training
Mỗi batch: forward → loss → backward → optimizer step; cuối mỗi epoch đo train loss và validation loss. **Không chọn epoch tốt nhất theo validation**: luôn dùng model ở epoch cuối, để số liệu không bị lạc quan do chọn theo validation.

## 4.5. Learning curve và overfitting
(Hình `fig_learning_curve.png`.) Baseline giảm train RMSE-log xuống gần 0.005 trong khi validation đứng ở khoảng 0.133, tức là mạng đang ghi nhớ dữ liệu train. Sau tuning, train 0.096 và validation 0.118, khoảng cách nhỏ hơn nhiều. Nguyên nhân được kiểm chứng ở mục 4.6 (ablation).

Giá trị train/validation loss theo từng epoch được lưu trong `experiments/loss_history_featuresets.csv` (cho EXP-00…EXP-04, cấu hình tuned, holdout 80/20) và hình `fig_learning_curve_featuresets.png`. Với cả 5 feature set, validation RMSE-log cuối là 0.1170–0.1180 (EXP-03 thấp nhất 0.1170), đường cong có dạng tương tự nhau và không có dấu hiệu overfit nặng.

## 4.6. Hyperparameter tuning
Random search 25 cấu hình (1 baseline + 24 ngẫu nhiên) trên 3-fold CV. Không gian tìm kiếm: hidden, dropout, weight decay, lr, batch size, epochs, optimizer (adam/adamw/adamax), batch norm.

| Cấu hình | CV RMSE-log |
|---|---|
| Baseline | 0.1320 |
| **Tốt nhất: 256-128-64, wd 0.01, batch 16, 60 epoch** | **0.1127** |
| Hạng 2: 128-64, dropout 0.2, wd 1e-3, adamax | 0.1129 |

Độ lệch chuẩn giữa các fold khoảng 0.009–0.011, nên hạng 1 và 2 không phân biệt được. Chọn cấu hình tốt nhất trong 25 cấu hình có thể làm các số liệu bên dưới hơi lạc quan.

**Ablation – đổi một hyperparameter so với cấu hình tốt nhất** (3-fold CV, EXP-00, 1 seed; `ablation_results.csv`):

| Thay đổi | CV RMSE-log |
|---|---:|
| Cấu hình tốt nhất | 0.1127 |
| hidden = 50-25-50 (mạng nhỏ như Keras mẫu) | **0.1126** |
| batch 32 | 0.1138 |
| optimizer Adamax | 0.1142 |
| epochs 30 | 0.1146 |
| epochs 150 | 0.1159 |
| lr 3e-3 | 0.1168 |
| dropout 0.2 | 0.1169 |
| weight decay 0 | 0.1174 |
| weight decay 1e-3 | 0.1225 |

Kết luận từ ablation: (1) **kích thước mạng không phải nguyên nhân cải thiện**, mạng 50-25-50 đạt gần như y hệt mạng 256-128-64; (2) cải thiện từ 0.132 xuống 0.113 đến từ **tổ hợp** weight decay 0.01, batch nhỏ (16) và huấn luyện ngắn (60 epoch với cosine), vì đổi riêng từng yếu tố trong tổ hợp này đều làm kết quả tệ đi 0.001–0.010; (3) chênh lệch của từng yếu tố riêng lẻ nhỏ so với độ lệch chuẩn giữa các fold (≈0.01), nên chỉ weight decay 1e-3 (+0.0098) là khác biệt tương đối rõ. Vì ablation chỉ chạy 1 seed nên các kết luận mang tính gợi ý. Ở phần còn lại của chương, cấu hình 256-128-64 vẫn được dùng vì đó là cấu hình đã được chọn khi tuning.

## 4.7. Kết quả các feature set (5-fold CV)

| Experiment | Feature | Input dim | Baseline MLP | MLP tuned |
|---|---|---:|---:|---:|
| EXP-00 | Original | 288 | 0.1342 | 0.1124 |
| EXP-01 | + TotalSF | 289 | 0.1314 | 0.1122 |
| EXP-02 | + TotalBathrooms, TotalPorchSF | 291 | 0.1385 | 0.1122 |
| EXP-03 | + HouseAge, RemodAge, GarageAge | 294 | 0.1334 | 0.1119 |
| EXP-04 | EXP-03 + feature selection | 204 | 0.1390 | 0.1156 |
| EXP-05 | EXP-03 + tuned + ensemble 5 seed | 294 | – | **0.1119** |

Chỉ số của EXP-05 (CV): RMSE ≈ **$20,106**, MAE ≈ **$13,174**, Bias ≈ −$1,508, Max Deviation ≈ $131,714, độ lệch chuẩn RMSE-log giữa các fold 0.0077.

**Nhận xét (trung thực về mức ý nghĩa):**
1. Cải thiện lớn nhất đến từ tuning (≈0.134 → 0.112), không đến từ feature mới.
2. Các feature mới (EXP-01→03) chỉ làm RMSE-log thay đổi tối đa 0.0005 với MLP tuned. Mức này nhỏ hơn nhiều so với độ lệch chuẩn giữa các fold (≈0.008), nên **không đủ bằng chứng** nói chúng có giúp hay không. EXP-03 được chọn làm cấu hình cuối vì có số thấp nhất, nhưng sự khác biệt không có ý nghĩa thống kê. Với baseline không regularization, kết quả dao động 0.131–0.139 giữa các feature set, phản ánh nhiễu của MLP hơn là tác dụng của feature.
3. Feature selection (EXP-04) **làm tệ hơn** (0.1156 so với 0.1119). Bước này loại 79/294 cột: 65 dummy hiếm (< 10 nhà) và 14 cột có |corr| < 0.02 với log giá (danh sách trong `feature_removed_EXP04.csv`). Có thể một số dummy hiếm vẫn mang thông tin hữu ích cho mô hình.

## 4.8. Đánh giá trên holdout và chẩn đoán
Holdout 80/20, một seed:

| | RMSE-log | RMSE ($) | MAE ($) | Bias ($) | Max Dev ($) |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.1332 | 22,490 | 15,957 | −3,886 | 112,264 |
| Tuned | 0.1180 | 19,359 | 13,338 | −2,969 | 102,732 |

Bias âm nghĩa là mô hình có xu hướng dự đoán thấp hơn giá thật. Hình `fig_diagnostics.png` gồm bốn biểu đồ tương tự 4-in-1 Minitab của paper (normal probability plot, residual vs fitted, histogram, predicted vs actual).

## 4.9. Model Check
Theo Fig. 2 của paper, mình kiểm tra thủ công quan sát đầu tiên của train (Id = 1, giá thật $208,500):

- Các feature mới tính tay trùng code: TotalSF = 856 + 856 + 854 = 2566, TotalBathrooms = 3.5, TotalPorchSF = 61, HouseAge = RemodAge = GarageAge = 5; chênh lệch sau chuẩn hoá = 0.
- Chuẩn hoá thủ công `GrLivArea`, `LotArea` (log1p), `OverallQual`, `YearBuilt` khớp code; one-hot `Neighborhood_CollgCr = 1`.
- Forward bằng numpy thuần từ trọng số đã lưu so với PyTorch cho cả 10 model: chênh tối đa khoảng 7×10⁻⁷ trên thang log.
- Dự đoán ensemble: **$209,178** (sai số +$678, +0.3%). Lưu ý dòng này thuộc tập train nên sai số nhỏ hơn sai số tổng quát hoá.
- Kết luận: **ĐẠT** (`src/model_check.py`).

## 4.10. Submission và hạn chế
- `submissions/submission_mlp.csv`: 1459 dòng, cột `Id, SalePrice`, không NaN. Model là ensemble 10 seed, train trên toàn bộ train (đã bỏ outlier). Một dự đoán (Id 2550, nhà 5095 sq ft, vượt xa mọi nhà trong train) bị MLP đẩy lên $1.3 triệu nên được chặn về khoảng giá train ($625,000); bước này không dùng nhãn test.
- **Điểm Kaggle: chưa có**, cần nhóm upload file và điền vào bảng so sánh.
- Hạn chế: MLP không ngoại suy tốt; số liệu CV có thể hơi lạc quan do chọn cấu hình tốt nhất trong 25 cấu hình; `feature_engineering.py` là bản tạm cần thay bằng bản chính thức của Tuyền; NA có nghĩa "không có" (ví dụ `FireplaceQu`) hiện được điền bằng mode theo code mẫu, có thể cải thiện bằng cách điền "None".
