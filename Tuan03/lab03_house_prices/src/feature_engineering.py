import numpy as np
import pandas as pd


def create_total_sf(df):
    df = df.copy()

    df["TotalSF"] = (
        df["TotalBsmtSF"]
        + df["1stFlrSF"]
        + df["2ndFlrSF"]
    )

    return df


def create_total_bathrooms(df):
    df = df.copy()

    df["TotalBathrooms"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"]
        + 0.5 * df["BsmtHalfBath"]
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

    df["HouseAge"] = (
        df["YrSold"] - df["YearBuilt"]
    )

    df["RemodAge"] = (
        df["YrSold"] - df["YearRemodAdd"]
    )

    df["GarageAge"] = np.where(
        df["GarageYrBlt"].notna(),
        df["YrSold"] - df["GarageYrBlt"],
        np.nan
    )

    return df


def create_engineered_features(df):
    df = df.copy()

    df = create_total_sf(df)
    df = create_total_bathrooms(df)
    df = create_total_porch_sf(df)
    df = create_age_features(df)

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