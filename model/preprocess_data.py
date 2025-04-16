import pandas as pd


def flatten(self):
    rows = []
    for company_id, company in self.company_dict.items():
        row = {"company_id": company_id, "label": company.label}  # 정답 레이블 포함

        for sheet_name, series in company.data_dict.items():
            for i, val in enumerate(series):
                row[f"{sheet_name}_{i}"] = val

        rows.append(row)
    self.df_model = pd.DataFrame(rows)