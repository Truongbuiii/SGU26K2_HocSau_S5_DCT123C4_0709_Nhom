# Chương 3 – Mô hình Machine Learning bằng Scikit-Learn
*Người phụ trách: Bích Trâm*

## 3.1. Mục tiêu và Thiết lập Mô hình
Mục tiêu là xây dựng mô hình hồi quy Machine Learning chuẩn bằng thư viện **Scikit-learn**, làm mốc chuẩn (Benchmark) để đối chứng với mạng Deep Learning PyTorch MLP (Chương 4), đồng thời kiểm tra mức độ cải thiện của mô hình khi bổ sung các đặc trưng phái sinh từ Feature Engineering (Chương 5).

- **Lựa chọn thuật toán:** Mô hình **Ridge Regression (Hồi quy có chính quy hóa L2)**. Trên dữ liệu bất động sản dạng bảng sau khi mã hóa One-Hot (số chiều mở rộng lên tới 250 – 290 chiều), Ridge Regression có ưu thế vượt trội trong việc kiểm soát đa cộng tuyến (multicollinearity), ngăn ngừa quá khớp (overfitting) và duy trì sự ổn định cao.
- **Biến mục tiêu:** Huấn luyện trên $y_{log} = \log(1 + \text{SalePrice})$. Hàm tối ưu MSE trên thang logarit trực tiếp tối thiểu hóa chỉ số **RMSLE** của cuộc thi Kaggle.

---

## 3.2. Tiền xử lý dữ liệu và Chống rò rỉ (Pipeline & No Data Leakage)
Toàn bộ quy trình tiền xử lý được tích hợp chặt chẽ trong `ColumnTransformer` và `Pipeline`:
1. **Biến định lượng (Numerical Features):**
   - Điền giá trị thiếu bằng trung vị (`SimpleImputer(strategy='median')`).
   - Chuẩn hóa thang đo đưa về trung bình 0, phương sai 1 bằng `StandardScaler()`.
2. **Biến định danh (Categorical Features):**
   - Điền giá trị thiếu bằng nhãn `"None"` hoặc yếu vị (`most_frequent`).
   - Mã hóa biến phân loại bằng `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
3. **Nguyên tắc No Data Leakage:** Pipeline chỉ được gọi `fit` trên tập huấn luyện của từng Fold (trong 5-Fold Cross Validation), sau đó mới `transform` sang tập Validation và Test.

---

## 3.3. Tinh chỉnh Siêu tham số (Hyperparameter Tuning)
Nhóm sử dụng kỹ thuật tìm kiếm ngẫu nhiên có kiểm thử chéo **`RandomizedSearchCV`** với 5 Folds:
- Không gian tham số tìm kiếm:
  - `alpha`: [0.01, 0.1, 1.0, 10.0, 100.0]
  - `solver`: ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga']
- **Kết quả tối ưu:** Cấu hình tốt nhất đạt được là:
  $$\text{best\_params\_} = \{\text{'model\_\_alpha'}: 10.0,\ \text{'model\_\_solver'}: \text{'saga'}\}$$

---

## 3.4. Kết quả Thực nghiệm trên các Bộ Đặc trưng (5-Fold Cross Validation)

Đánh giá mô hình tối ưu Ridge ($\alpha=10.0$, solver='saga') trên lần lượt 5 cấu hình đặc trưng của Thanh Tuyền:

| Thực nghiệm | Bộ đặc trưng sử dụng | Số chiều đầu vào | Mean RMSLE (5-Fold CV) | Std RMSLE | Mean MAE ($) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **EXP-00** | Đặc trưng gốc (sau tiền xử lý) | 288 | 0.11450 | 0.00778 | $13,747.46 |
| **EXP-01** | EXP-00 + `TotalSF` | 289 | 0.11450 | 0.00778 | **$13,745.40** |
| **EXP-02** | EXP-01 + `TotalBathrooms` + `TotalPorchSF` | 291 | 0.11450 | 0.00778 | $13,746.29 |
| **EXP-03** | EXP-02 + `HouseAge` + `RemodAge` + `GarageAge` | 294 | 0.11452 | 0.00781 | $13,747.57 |
| **EXP-04** | 19 Đặc trưng chọn lọc từ Random Forest | 204 | 0.13057 | 0.01030 | $16,396.58 |

### Nhận xét & Đánh giá kết quả:
1. **Tính ổn định của Ridge Regression:** Mô hình Ridge đạt độ chính xác rất cao và ổn định ($RMSLE \approx 0.1145$, độ lệch chuẩn nhỏ $0.0078$). 
2. **Tác động của Feature Engineering:** Thêm biến `TotalSF` (EXP-01) giúp sai số tuyệt đối trung bình MAE giảm xuống mức thấp nhất là **$13,745.40**.
3. **Thực nghiệm chọn lọc (EXP-04):** Khi rút gọn chỉ dùng 19 đặc trưng quan trọng nhất, mô hình giảm nhẹ độ chính xác ($RMSLE = 0.13057$) nhưng số lượng biến giảm hơn 70%, giúp mô hình nhẹ hơn đáng kể và dễ diễn giải.

---

## 3.5. Đột phá Hiệu năng: Tác động của việc Lọc Ngoại lai theo Paper De Cock
Một phát hiện thực nghiệm then chốt được kiểm chứng:
- Khi **chưa loại ngoại lai**: Điểm RMSE trên tập huấn luyện gốc bị kéo lên mức **0.1453** do ảnh hưởng của 2 ngôi nhà dị biệt diện tích cực lớn nhưng giá bán một phần (Partial) rất thấp (Id 524 và 1299).
- Khi **áp dụng khuyến nghị của Dean De Cock (2011)** loại bỏ 2 căn nhà này: Điểm RMSE/RMSLE giảm sâu từ **0.1453** xuống **0.1145** (cải thiện hơn **21.2%** độ chính xác)!

---

## 3.6. Xuất File Dự đoán Nộp Kaggle
- Mô hình tối ưu được huấn luyện lại trên toàn bộ tập train đã làm sạch ngoại lai.
- Dự đoán trên 1459 căn nhà của tập test, áp dụng hàm $\text{expm1}$ để khôi phục về đơn vị USD thực tế.
- File xuất: `submissions/submission_sklearn.csv` (1459 dòng, cột `Id` và `SalePrice`, 0 giá trị thiếu).
