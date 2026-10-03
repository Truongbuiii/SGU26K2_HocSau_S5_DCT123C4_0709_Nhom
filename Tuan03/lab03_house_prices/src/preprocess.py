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

try:
    from feature_engineering import add_features
except ImportError:
    try:
        from src.feature_engineering import add_features
    except ImportError:
        pass

DROP_COLS = ["Id", "Alley", "PoolQC", "Fence", "MiscFeature"]
FORCE_CATEGORICAL = ["MSSubClass"]          # mã số nhưng thực chất là nominal
OUTLIER_GRLIV = 4000                         # De Cock (2011): bỏ nhà > 4000 sq ft


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


def load_data(path=None):
    """
    Hàm load dữ liệu tương thích cho cả train_mlp.py và các notebook.
    """
    if path is None:
        return load_raw_data()
    p = Path(path)
    train_path = p / "train.csv"
    test_path = p / "test.csv"
    if not train_path.exists() or not test_path.exists():
        return load_raw_data()
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def remove_outliers(train: pd.DataFrame) -> pd.DataFrame:
    """Loại bỏ ngoại lai GrLivArea > 4000 theo khuyến nghị De Cock (2011) cho MLP."""
    return train[train["GrLivArea"] <= OUTLIER_GRLIV].reset_index(drop=True)


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


class Preprocessor:
    """
    Bộ tiền xử lý dùng riêng cho mạng nơ-ron PyTorch MLP (Ngọc Trúc).
    - Chuẩn hóa Z-score sau khi fillna median và log1p các biến bị lệch (skew > 0.75).
    - One-hot encoding với từ điển categories fit từ reference (train + test).
    - Ngăn chặn hoàn toàn data leakage giữa các validation folds.
    """
    def __init__(self, groups=(), select=False, skew_thr=0.75,
                 min_dummy_count=10, min_abs_corr=0.02):
        self.groups, self.select = list(groups), select
        self.skew_thr, self.min_dummy_count, self.min_abs_corr = skew_thr, min_dummy_count, min_abs_corr

    def _prep_raw(self, df):
        df = add_features(df, self.groups)
        df = df.drop(columns=[c for c in DROP_COLS + ["SalePrice"] if c in df.columns])
        for c in FORCE_CATEGORICAL:
            if c in df.columns:
                df[c] = df[c].astype(str)
        return df

    def fit(self, df_train: pd.DataFrame, reference: pd.DataFrame = None):
        """df_train: có SalePrice. reference: train+test (để lấy danh sách category)."""
        y = np.log1p(df_train["SalePrice"].values)
        X = self._prep_raw(df_train)
        ref = self._prep_raw(reference if reference is not None else df_train)
        self.cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        self.num_cols = [c for c in X.columns if c not in self.cat_cols]
        self.num_mean = X[self.num_cols].mean()
        self.cat_mode = X[self.cat_cols].mode().iloc[0]
        self.categories = {c: sorted(ref[c].dropna().unique()) for c in self.cat_cols}
        Xn = X[self.num_cols].fillna(self.num_mean)
        skew = Xn.skew()
        self.log_cols = [c for c in self.num_cols
                         if skew[c] > self.skew_thr and Xn[c].min() >= 0]
        Xn[self.log_cols] = np.log1p(Xn[self.log_cols])
        self.mu, self.sd = Xn.mean(), Xn.std().replace(0, 1.0)
        self.columns_ = None
        self.keep_ = None
        M = self._matrix(X)
        self.columns_ = list(M.columns)
        self.removed_ = []
        if self.select:
            cnt = (M != 0).sum()
            corr = M.apply(lambda s: np.corrcoef(s, y)[0, 1] if s.std() > 0 else 0.0).abs()
            for c in M.columns:
                if c in self.cat_dummies_ and cnt[c] < self.min_dummy_count:
                    self.removed_.append((c, f"dummy hiếm (<{self.min_dummy_count} nhà)"))
                elif corr[c] < self.min_abs_corr:
                    self.removed_.append((c, f"|corr với log(SalePrice)|<{self.min_abs_corr}"))
            drop = {c for c, _ in self.removed_}
            self.keep_ = [c for c in self.columns_ if c not in drop]
        else:
            self.keep_ = self.columns_
        self.feature_names_ = list(self.keep_)
        return self

    def _matrix(self, X):
        Xn = X[self.num_cols].fillna(self.num_mean)
        Xn[self.log_cols] = np.log1p(Xn[self.log_cols].clip(lower=0))
        Xn = (Xn - self.mu) / self.sd
        parts, dummies = [Xn], []
        for c in self.cat_cols:
            s = X[c].fillna(self.cat_mode[c])
            d = pd.DataFrame({f"{c}_{k}": (s == k).astype(float) for k in self.categories[c]},
                             index=X.index)
            parts.append(d)
            dummies += list(d.columns)
        self.cat_dummies_ = set(dummies)
        return pd.concat(parts, axis=1)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        M = self._matrix(self._prep_raw(df))
        return M[self.keep_].values.astype(np.float32)

    @staticmethod
    def target(df):
        return np.log1p(df["SalePrice"].values).astype(np.float32)
