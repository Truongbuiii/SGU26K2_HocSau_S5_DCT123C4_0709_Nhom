"""Gom cac file log trong results/logs thanh 1 file Excel (moi file = 1 sheet).
Chay rieng:  python src/make_excel.py   (tu dong chay sau run_experiments.py)
Can: pip install openpyxl
"""
import json, os
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "results", "logs")
OUT = os.path.join(LOG, "BTL01_logs_tong_hop.xlsx")

DESC = {
    "baseline_history": "Train/Val MSE theo epoch cua baseline (seed 0)",
    "A": "Thi nghiem A: so mau huan luyen",
    "B": "Thi nghiem B: so no-ron tang an (A=1000)",
    "Bsmall": "Thi nghiem bo sung: so no-ron voi A=100",
    "C": "Thi nghiem C: so tang an",
}
ORDER = ["baseline_history"] + [f"exp_{n}_{k}" for n in ("A", "B", "Bsmall", "C")
                                for k in ("summary", "runs", "hist_seed0")]

HFILL = PatternFill("solid", start_color="1F4E79", end_color="1F4E79")
HFONT = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
BORDER = Border(*(Side(style="thin", color="D9D9D9"),) * 4)


def describe(name):
    if name == "baseline_history":
        return DESC[name]
    _, exp, kind = name.split("_", 2)
    what = {"summary": "tong hop mean/std qua 3 seed", "runs": "chi tiet tung lan chay",
            "hist_seed0": "loss theo epoch (seed 0)"}[kind]
    return f"{DESC[exp]} - {what}"


def style_header(ws, ncols):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=1, column=c)
        cell.fill, cell.font = HFILL, HFONT
        cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.freeze_panes = "A2"


def autofit(ws):
    for col in ws.columns:
        width = max(len(str(c.value if c.value is not None else "")) for c in col[:50])
        ws.column_dimensions[get_column_letter(col[0].column)].width = max(width + 3, 12)


def build_excel():
    wb = Workbook()
    idx = wb.active
    idx.title = "Muc_luc"
    rows_index = []

    path = os.path.join(LOG, "baseline_result.json")
    if os.path.exists(path):
        res = json.load(open(path, encoding="utf-8"))
        ws = wb.create_sheet("baseline_result")
        ws.append(["Loai", "Ten", "Gia tri"])
        for k, v in res["config"].items():
            ws.append(["Cau hinh", k, str(v) if isinstance(v, list) else v])
        for k, v in res["result"].items():
            ws.append(["Ket qua", k, v])
        style_header(ws, 3); autofit(ws)
        rows_index.append(("baseline_result", ws.max_row - 1, 3, "Cau hinh va ket qua baseline"))

    for name in ORDER:
        p = os.path.join(LOG, name + ".csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        ws = wb.create_sheet(name)
        ws.append(list(df.columns))
        for row in df.itertuples(index=False):
            ws.append([None if pd.isna(v) else (v.item() if hasattr(v, "item") else v) for v in row])
        for r in ws.iter_rows(min_row=2):
            for cell in r:
                cell.border = BORDER
                if isinstance(cell.value, float):
                    cell.number_format = "0.00E+00" if 0 < abs(cell.value) < 1e-3 else "0.0000"
        style_header(ws, len(df.columns)); autofit(ws)
        rows_index.append((name, len(df), len(df.columns), describe(name)))

    idx.append(["STT", "Ten sheet", "So dong", "So cot", "Mo ta"])
    for i, (n, r, c, d) in enumerate(rows_index, 1):
        idx.append([i, n, r, c, d])
    style_header(idx, 5); autofit(idx)
    wb.save(OUT)
    print(f"Da tao {OUT} ({len(rows_index)} sheet)")


if __name__ == "__main__":
    build_excel()
