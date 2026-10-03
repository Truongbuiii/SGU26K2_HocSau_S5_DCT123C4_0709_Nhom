# LAB 03 - HOUSE PRICES: ADVANCED REGRESSION TECHNIQUES
## ĐH SÀI GÒN (SGU) - KHOA CNTT - BỘ MÔN HỌC SÂU (DEEP LEARNING)
### NHÓM THỰC HIỆN: NHÓM SỐ 3

---

### Danh sách thành viên nhóm

| STT | Họ và tên | Mã sinh viên | Lớp | Vai trò & Phân công chính |
| :---: | :--- | :---: | :---: | :--- |
| 1 | **Bùi Đức Trường** | **3123411319** | DCT123C4 | **Nhóm trưởng** - Nghiên cứu Paper De Cock, EDA, Pipeline chung, Model Check, Tích hợp & Quản lý |
| 2 | **[Ánh Trâm]** | *(Cập nhật)* | *(Cập nhật)* | Mô hình Machine Learning chuẩn Scikit-learn (Ridge/Lasso/Ensemble), Tuning, Đánh giá CV |
| 3 | **[Ngọc Trúc]** | *(Cập nhật)* | *(Cập nhật)* | Mạng Deep Learning PyTorch MLP, Training Loop, Loss Curves, Tuning, Ablation Study |
| 4 | **[Thanh Tuyền]** | *(Cập nhật)* | *(Cập nhật)* | Feature Engineering, Feature Selection, Thiết kế 5 thực nghiệm (EXP-00 -> EXP-04) |

---

### Cấu trúc thư mục dự án thống nhất

```text
lab03_house_prices/
│
├── data/                                      # Dữ liệu thi Kaggle
│   ├── train.csv                              # Tập huấn luyện gốc (1460 dòng, 81 cột)
│   └── test.csv                               # Tập kiểm tra (1459 dòng, 80 cột)
│
├── notebooks/                                 # Toàn bộ 5 Jupyter Notebook theo đúng phân công
│   ├── 01_eda_and_problem_definition.ipynb    # Phần của Trường: EDA & Phân tích bài toán theo paper
│   ├── 02_sklearn_baseline.ipynb              # Phần của Trâm: Xây dựng baseline Scikit-learn
│   ├── 03_sklearn_tuning.ipynb                # Phần của Trâm: Hyperparameter tuning & 5 thực nghiệm
│   ├── 04_pytorch_mlp.ipynb                   # Phần của Trúc: Kiến trúc PyTorch MLP, Loss curve, Ablation
│   └── 05_feature_engineering.ipynb           # Phần của Tuyền: Phân tích tương quan & tạo biến mới
│
├── src/                                       # Mã nguồn module hóa dùng chung
│   ├── preprocess.py                          # Pipeline tiền xử lý & bộ chia 5-Fold CV chuẩn
│   ├── feature_engineering.py                 # Hàm tạo TotalSF, TotalBathrooms, HouseAge, Porch...
│   ├── mlp_model.py                           # Kiến trúc mạng nơ-ron sâu PyTorch MLP
│   ├── train_mlp.py                           # Vòng lặp huấn luyện PyTorch thuần túy
│   ├── preprocess_mlp.py                      # Tiền xử lý cho nhánh PyTorch
│   ├── ablation_mlp.py                        # Script chạy Ablation study cho PyTorch
│   ├── model_check.py                         # Kiểm chứng tính tay độc lập (Hình 2 Paper De Cock)
│   └── make_figures.py                        # Script vẽ biểu đồ chẩn đoán
│
├── figures/                                   # Toàn bộ biểu đồ phân tích & thực nghiệm
│   ├── 01_target_distribution.png             # Phân phối SalePrice trước và sau log1p
│   ├── 02_outliers_grlivarea.png              # Phát hiện ngoại lai theo paper De Cock
│   ├── 03_missing_values.png                  # Tỷ lệ missing values trên tập train
│   ├── 04_top_correlations.png                # Top đặc trưng tương quan mạnh nhất
│   ├── 05_quality_and_neighborhood.png        # Boxplot OverallQual & Neighborhood
│   ├── fig_correlation.png                    # Ma trận tương quan của Tuyền
│   ├── fig_feature_importance.png             # Độ quan trọng biến Random Forest
│   ├── fig_learning_curve.png                 # Đường cong mất mát của Trúc (Overfitting check)
│   ├── fig_diagnostics.png                    # Biểu đồ chẩn đoán phần dư 4-in-1
│   ├── fig_tuning.png                         # Kết quả dò tìm siêu tham số
│   └── fig_experiments.png                    # Biểu đồ đối sánh RMSLE giữa các thực nghiệm
│
├── models/                                    # Trọng số mô hình đã huấn luyện
│   ├── mlp_model.pt                           # Trọng số tối ưu của mạng PyTorch MLP
│   └── mlp_preprocessor.pkl                   # Bộ chuyển đổi dữ liệu đã fit
│
├── results/                                   # Các bảng thống kê kết quả thực nghiệm
│   ├── experiment_results.csv                 # Bảng tổng hợp Mean/Std RMSLE, MAE qua 5 Folds
│   ├── model_check_report.csv                 # Báo cáo kiểm chứng Model Check tính tay vs code
│   ├── selection_report.csv                   # Danh sách 19 đặc trưng chọn lọc của Tuyền
│   ├── tuning_results.csv                     # Kết quả 25 cấu hình tuning của Trúc
│   ├── ablation_results.csv                   # Bảng phân tích bóc tách siêu tham số của Trúc
│   └── loss_history_featuresets.csv           # Lịch sử loss theo từng epoch
│
├── submissions/                               # File kết quả nộp lên Kaggle (1459 dòng)
│   ├── submission_sklearn.csv                 # Dự đoán từ mô hình Scikit-learn tối ưu của Trâm
│   └── submission_mlp.csv                     # Dự đoán từ mô hình PyTorch MLP Ensemble của Trúc
│
├── report/                                    # Báo cáo học phần
│   ├── Chuong3_Scikit_Learn.md                # Bài viết chi tiết Chương 3 (Ánh Trâm)
│   ├── Chuong4_MLP_PyTorch.md                 # Bài viết chi tiết Chương 4 (Ngọc Trúc)
│   └── BaoCao_TongHop_Lab03_HousePrices.md    # BẢN BÁO CÁO TỔNG HỢP HOÀN CHỈNH 6 CHƯƠNG
│
├── requirements.txt                           # Các thư viện phụ thuộc
└── README.md
```

---

### Hướng dẫn chạy kiểm tra lại từ đầu (Reproducibility)

1. **Cài đặt thư viện phụ thuộc:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Chạy kiểm chứng Model Check (theo Paper De Cock):**
   ```bash
   python src/model_check.py
   ```

3. **Chạy các Jupyter Notebook tương ứng:**
   - Notebook 01 (Trường): `jupyter notebook notebooks/01_eda_and_problem_definition.ipynb`
   - Notebook 02 & 03 (Trâm): `jupyter notebook notebooks/02_sklearn_baseline.ipynb` và `03_sklearn_tuning.ipynb`
   - Notebook 04 (Trúc): `jupyter notebook notebooks/04_pytorch_mlp.ipynb`
   - Notebook 05 (Tuyền): `jupyter notebook notebooks/05_feature_engineering.ipynb`
