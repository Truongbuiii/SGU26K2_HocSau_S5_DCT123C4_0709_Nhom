"""
preprocess_mlp.py - tien xu ly dau vao cho MLP (Truc).

Bam theo quy trinh chung cua code mau + bang phan cong, voi cac thay doi co chu y:
  * Bo Id va 4 cot thieu >70% (Alley, PoolQC, Fence, MiscFeature)         -> giong code mau
  * Cot so : impute median -> log1p cac cot lech (|skew|>0.75) -> StandardScaler
  * Cot chu: impute hang so "None" -> OneHot (handle_unknown an toan, gop nhom hiem <5)
      - Code mau dien MODE. O day dien "None" vi trong data_description.txt, NA o nhieu
        cot (BsmtQual, GarageType, FireplaceQu...) co nghia la "khong co" chu khong phai thieu.
  * MSSubClass la ma loai nha (20, 30, 60...) -> coi la bien phan loai
  * Moi buoc fit CHI tren tap train cua fold -> khong ro ri thong tin sang validation/test
    (code mau concat train+test truoc khi encode; cach lam o day cho cung cot dau ra nhung sach hon)
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DROP_COLS = ["Id", "Alley", "PoolQC", "Fence", "MiscFeature"]
OUTLIER_GRLIV = 4000  # paper De Cock: nen loai nha > 4000 sq ft (GrLivArea)


def load_raw(data_dir="data"):
    train = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    return train, test


def remove_outliers(train: pd.DataFrame, limit: int = OUTLIER_GRLIV) -> pd.DataFrame:
    """Chi ap dung cho TRAIN (khong bao gio loai dong cua test)."""
    return train[train["GrLivArea"] <= limit].reset_index(drop=True)


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).copy()
    df["MSSubClass"] = df["MSSubClass"].astype(str)
    return df


class SkewLog1p(BaseEstimator, TransformerMixin):
    """log1p cac cot so co |skew| > threshold (chon tren tap fit) va khong am."""

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sk = skew(X, axis=0, nan_policy="omit")
        sk = np.nan_to_num(sk)
        self.cols_ = np.where((np.abs(sk) > self.threshold) & (X.min(axis=0) >= 0))[0]
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        X[:, self.cols_] = np.log1p(np.clip(X[:, self.cols_], 0, None))
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols]
    num = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("skew", SkewLog1p()),
                    ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="None")),
                    ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                         min_frequency=5, sparse_output=False))])
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)],
                             verbose_feature_names_out=False)


def rf_select(X, y, thr=0.95, seed=42):
    """Chon cac cot co tong importance (RandomForest) tich luy >= thr. Tra ve chi so cot giu lai + importance."""
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, max_features=0.33,
                               random_state=seed, n_jobs=-1).fit(X, y)
    imp = rf.feature_importances_
    order = np.argsort(-imp)
    k = int(np.searchsorted(np.cumsum(imp[order]), thr) + 1)
    return np.sort(order[:k]), imp


def fit_encode(df_tr, y_tr, others, select_thr=None, seed=42):
    """
    Fit preprocessor tren df_tr; transform df_tr + cac df trong `others`.
    select_thr: None (giu het) hoac 0.95 (EXP-04, chon feature bang RF importance - CHI tren df_tr).
    """
    pre = build_preprocessor(df_tr)
    Xtr = pre.fit_transform(df_tr)
    names = np.asarray(pre.get_feature_names_out())
    Xo = [pre.transform(o) for o in others]
    imp = None
    keep = np.arange(Xtr.shape[1])
    if select_thr is not None:
        keep, imp = rf_select(Xtr, y_tr, select_thr, seed)
        Xtr = Xtr[:, keep]
        Xo = [x[:, keep] for x in Xo]
    return {"pre": pre, "Xtr": Xtr, "Xo": Xo, "names": names, "keep": keep,
            "kept_names": names[keep], "importance": imp}
