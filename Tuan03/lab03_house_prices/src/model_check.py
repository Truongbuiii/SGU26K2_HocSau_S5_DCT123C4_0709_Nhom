import sys
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

import preprocess
from feature_engineering import create_engineered_features


def perform_model_check():
    """
    Quy trình kiểm tra Model Check theo khuyến nghị trong bài báo của Dean De Cock (2011):
    Kiểm tra độc lập bằng tính toán thủ công trên một dòng quan sát cụ thể để xác minh
    tính chính xác của các biến phái sinh (Feature Engineering).
    """
    print("=" * 65)
    print("THỰC HIỆN KIỂM CHỨNG MODEL CHECK (Dean De Cock, 2011)")
    print("=================================================================")

    train_df, _ = preprocess.load_raw_data()
    row = train_df.iloc[0]

    print(f"\nQuan sát kiểm tra: Id = {row['Id']}")
    print(f"Giá bán thực tế (SalePrice): ${row['SalePrice']:,}")

    # 1. Tính toán thủ công theo định nghĩa toán học
    print("\n--- 1. CÁC GIÁ TRỊ GỐC (RAW VALUES) ---")
    print(f"TotalBsmtSF: {row['TotalBsmtSF']}, 1stFlrSF: {row['1stFlrSF']}, 2ndFlrSF: {row['2ndFlrSF']}")
    print(f"FullBath: {row['FullBath']}, HalfBath: {row['HalfBath']}, BsmtFullBath: {row['BsmtFullBath']}, BsmtHalfBath: {row['BsmtHalfBath']}")
    print(f"WoodDeckSF: {row['WoodDeckSF']}, OpenPorchSF: {row['OpenPorchSF']}, EnclosedPorch: {row['EnclosedPorch']}, 3SsnPorch: {row['3SsnPorch']}, ScreenPorch: {row['ScreenPorch']}")
    print(f"YearBuilt: {row['YearBuilt']}, YearRemodAdd: {row['YearRemodAdd']}, GarageYrBlt: {row['GarageYrBlt']}, YrSold: {row['YrSold']}")

    manual_total_sf = row['TotalBsmtSF'] + row['1stFlrSF'] + row['2ndFlrSF']
    manual_total_bath = row['FullBath'] + 0.5 * row['HalfBath'] + row['BsmtFullBath'] + 0.5 * row['BsmtHalfBath']
    manual_total_porch = row['OpenPorchSF'] + row['3SsnPorch'] + row['EnclosedPorch'] + row['ScreenPorch'] + row['WoodDeckSF']
    manual_house_age = row['YrSold'] - row['YearBuilt']
    manual_remod_age = row['YrSold'] - row['YearRemodAdd']
    manual_garage_age = row['YrSold'] - row['GarageYrBlt']

    # 2. Giá trị tạo từ code pipeline
    featured_df = create_engineered_features(train_df.iloc[[0]])
    code_row = featured_df.iloc[0]

    check_table = [
        {"Feature": "TotalSF", "Manual": manual_total_sf, "Code": code_row["TotalSF"], "Match": manual_total_sf == code_row["TotalSF"]},
        {"Feature": "TotalBathrooms", "Manual": manual_total_bath, "Code": code_row["TotalBathrooms"], "Match": manual_total_bath == code_row["TotalBathrooms"]},
        {"Feature": "TotalPorchSF", "Manual": manual_total_porch, "Code": code_row["TotalPorchSF"], "Match": manual_total_porch == code_row["TotalPorchSF"]},
        {"Feature": "HouseAge", "Manual": manual_house_age, "Code": code_row["HouseAge"], "Match": manual_house_age == code_row["HouseAge"]},
        {"Feature": "RemodAge", "Manual": manual_remod_age, "Code": code_row["RemodAge"], "Match": manual_remod_age == code_row["RemodAge"]},
        {"Feature": "GarageAge", "Manual": manual_garage_age, "Code": code_row["GarageAge"], "Match": manual_garage_age == code_row["GarageAge"]},
    ]

    report_df = pd.DataFrame(check_table)
    print("\n--- 2. KẾT QUẢ ĐỐI SÁNH TÍNH TAY vs CODE PIPELINE ---")
    print(report_df.to_string(index=False))

    all_match = report_df["Match"].all()
    if all_match:
        print("\n KẾT LUẬN: Tất cả các đặc trưng phái sinh khớp 100% với tính toán giải tích!")
    else:
        print("\n CẢNH BÁO: Phát hiện sai lệch trong quá trình sinh biến!")

    # Lưu báo cáo vào kết quả
    check_csv_path = Path(__file__).resolve().parent.parent / "results" / "model_check_report.csv"
    report_df.to_csv(check_csv_path, index=False)
    print(f"Đã lưu kết quả Model Check tại: {check_csv_path}")

    return report_df


def main():
    return perform_model_check()


if __name__ == "__main__":
    main()
