import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_default_data_dir():
    """Lấy đường dẫn thư mục data chuẩn của dự án."""
    return Path(__file__).resolve().parent.parent / "data"


def load_raw_data(data_dir=None):
    """
    Đọc dữ liệu train và test nguyên bản từ thư mục data.
    """
    if data_dir is None:
        data_dir = get_default_data_dir()
    else:
        data_dir = Path(data_dir)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {data_dir}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def remove_paper_outliers(df):
    """
    Loại bỏ các ngoại lai theo khuyến nghị của Dean De Cock (2011) đã phân tích trong Notebook 01:
    Các căn nhà có diện tích GrLivArea > 4000 sq ft nhưng giá bán thấp (Partial sales: Id 524 và 1299).
    Chỉ áp dụng trên tập Train, tuyệt đối không xóa dòng ở Test.
    """
    if "SalePrice" in df.columns:
        outlier_mask = (df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)
        cleaned_df = df[~outlier_mask].reset_index(drop=True)
        return cleaned_df
    return df


def build_preprocessor(X):
    """
    Xây dựng pipeline tiền xử lý chống rò rỉ dữ liệu (No Data Leakage):
    - Biến số (Numeric): Điền median + chuẩn hóa StandardScaler
    - Biến phân loại (Categorical): Điền 'None' (với tiện ích không tồn tại) + OneHotEncoder
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def get_kfold_splits(n_samples, n_splits=5, random_state=42):
    """
    Cố định bộ chỉ số 5-Fold Cross Validation để mọi thành viên và mô hình chạy trên cùng một split.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(kf.split(np.zeros(n_samples)))


def calculate_metrics(y_true_log, y_pred_log):
    """
    Tính các độ đo đánh giá:
    - rmsle: RMSE trên log-scale (đúng chuẩn Kaggle)
    - mae: Sai số tuyệt đối trung bình trên giá bán USD thực tế (expm1)
    """
    rmsle = float(np.sqrt(np.mean((y_pred_log - y_true_log) ** 2)))
    
    # Khôi phục về giá bán thực tế để tính MAE
    y_true_orig = np.expm1(y_true_log)
    y_pred_orig = np.expm1(y_pred_log)
    mae = float(np.mean(np.abs(y_pred_orig - y_true_orig)))
    
    return {
        "rmsle": rmsle,
        "mae": mae,
    }
