import numpy as np
import pandas as pd


def create_total_sf(df):
    df = df.copy()

    df["TotalSF"] = (
        df["TotalBsmtSF"].fillna(0)
        + df["1stFlrSF"]
        + df["2ndFlrSF"]
    )

    return df


def create_total_bathrooms(df):
    df = df.copy()

    df["TotalBathrooms"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"].fillna(0)
        + 0.5 * df["BsmtHalfBath"].fillna(0)
    )

    return df


def create_total_porch_sf(df):
    df = df.copy()

    df["TotalPorchSF"] = (
        df["OpenPorchSF"]
        + df["3SsnPorch"]
        + df["EnclosedPorch"]
        + df["ScreenPorch"]
        + df["WoodDeckSF"]
    )

    return df


def create_age_features(df):
    df = df.copy()

    df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
    df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)
    
    # Nhà không có garage (GarageYrBlt NaN) -> lấy YearBuilt; clip để tránh năm lỗi (vd. 2207)
    gyr = df["GarageYrBlt"].fillna(df["YearBuilt"])
    df["GarageAge"] = (df["YrSold"] - gyr).clip(lower=0)

    return df


def create_engineered_features(df):
    df = df.copy()

    df = create_total_sf(df)
    df = create_total_bathrooms(df)
    df = create_total_porch_sf(df)
    df = create_age_features(df)

    return df


def add_features(df: pd.DataFrame, groups=()) -> pd.DataFrame:
    """
    Thêm đặc trưng mới trên giá trị thô (chưa impute).
    Tương thích cả nhóm tên ('area', 'bath_porch', 'age')
    lẫn tên cột ('TotalSF', 'TotalBathrooms', 'TotalPorchSF', 'HouseAge', 'RemodAge', 'GarageAge')
    hoặc số level (1, 2, 3).
    """
    df = df.copy()
    if isinstance(groups, (int, float)):
        level = int(groups)
        groups = []
        if level >= 1:
            groups.append("area")
        if level >= 2:
            groups.append("bath_porch")
        if level >= 3:
            groups.append("age")

    g_set = set(groups) if isinstance(groups, (list, tuple, set)) else set([groups])

    if "area" in g_set or "TotalSF" in g_set:
        df["TotalSF"] = (
            df["TotalBsmtSF"].fillna(0)
            + df["1stFlrSF"]
            + df["2ndFlrSF"]
        )

    if (
        "bath_porch" in g_set
        or "TotalBathrooms" in g_set
        or "TotalPorchSF" in g_set
    ):
        df["TotalBathrooms"] = (
            df["FullBath"]
            + 0.5 * df["HalfBath"]
            + df["BsmtFullBath"].fillna(0)
            + 0.5 * df["BsmtHalfBath"].fillna(0)
        )
        df["TotalPorchSF"] = (
            df["OpenPorchSF"]
            + df["3SsnPorch"]
            + df["EnclosedPorch"]
            + df["ScreenPorch"]
            + df["WoodDeckSF"]
        )

    if (
        "age" in g_set
        or "HouseAge" in g_set
        or "RemodAge" in g_set
        or "GarageAge" in g_set
    ):
        df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
        df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)
        gyr = df["GarageYrBlt"].fillna(df["YearBuilt"])
        df["GarageAge"] = (df["YrSold"] - gyr).clip(lower=0)

    return df


EXPERIMENTS = {
    "EXP-00": [],
    "EXP-01": [
        "TotalSF"
    ],
    "EXP-02": [
        "TotalSF",
        "TotalBathrooms",
        "TotalPorchSF"
    ],
    "EXP-03": [
        "TotalSF",
        "TotalBathrooms",
        "TotalPorchSF",
        "HouseAge",
        "RemodAge",
        "GarageAge"
    ],
}


EXP04_FEATURES = [
    "OverallQual",
    "TotalSF",
    "OverallCond",
    "GarageArea",
    "GarageCars",
    "LotArea",
    "GrLivArea",
    "BsmtFinSF1",
    "RemodAge",
    "TotalBathrooms",
    "1stFlrSF",
    "HouseAge",
    "YearBuilt",
    "BsmtUnfSF",
    "YearRemodAdd",
    "LotFrontage",
    "TotalBsmtSF",
    "TotalPorchSF",
    "2ndFlrSF",
]


EXPERIMENTS["EXP-04"] = EXP04_FEATURES