import pandas as pd
import openpyxl as op

sheets = pd.read_excel("have_fun.xlsx", sheet_name=None)

# for name, df in sheets.items():
#     print(f"Название листа: {name}")
#     print(df.head(10))

wb = op.load_workbook("have_fun.xlsx", read_only=False)

for ws in wb.worksheets:
    print(ws.title, ws.sheet_state)