import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def standard_scaler(self, sheet_name, df_sheet):
    df_scaled = df_sheet.copy()
    df_train_valid = df_sheet.iloc[:, 9:df_sheet.columns.get_loc(self.split_cutoff_date)].copy()
    all_non_zero_values_list = []
    for column in df_train_valid.columns:
        non_zeros_in_column = df_train_valid.loc[df_train_valid[column] != 0, column].values
        all_non_zero_values_list.extend(non_zeros_in_column)
    if len(all_non_zero_values_list) > 1:
        all_non_zero_values_np = np.array(all_non_zero_values_list).reshape(-1, 1)
        scaler = StandardScaler()
        scaler.fit(all_non_zero_values_np)
        for column in df_scaled.columns[9:]:
            original_column_values = df_scaled[column].values  # 원본 값 사용
            scaled_column_new_values = np.zeros_like(original_column_values, dtype=float)  # 결과 배열 초기화 (0으로)

            non_zero_mask = original_column_values != 0

            if np.any(non_zero_mask):  # 해당 열에 0이 아닌 값이 있다면
                current_col_non_zeros = original_column_values[non_zero_mask].reshape(-1, 1)
                scaled_values_for_col = scaler.transform(current_col_non_zeros)

                scaled_column_new_values[non_zero_mask] = scaled_values_for_col.flatten()

            df_scaled[column] = scaled_column_new_values
        self.scaler_dict[sheet_name] = scaler

    df_scaled_rowwise = df_sheet.copy()
    cols_to_scale = df_sheet.columns[9:]
    df_scaled_rowwise[cols_to_scale] = df_scaled_rowwise[cols_to_scale].astype(float)
    row_scalers_for_sheet = {}  # 현재 시트의 row별 스케일러를 임시 저장
    for index, row in df_sheet.iterrows():
        row_non_zero_values_list = []
        train_data_for_row = df_train_valid.loc[index]
        for value in train_data_for_row:
            if value != 0 and pd.notna(value) and not np.isinf(value):
                row_non_zero_values_list.append(value)

        if len(row_non_zero_values_list) > 1:
            row_non_zero_values_np = np.array(row_non_zero_values_list).reshape(-1, 1)

            scaler_row = StandardScaler()
            scaler_row.fit(row_non_zero_values_np)

            original_row_values = df_scaled_rowwise.loc[index, df_sheet.columns[9:]].values
            scaled_row_new_values = np.zeros_like(original_row_values, dtype=float)

            non_zero_mask = original_row_values != 0

            if np.any(non_zero_mask):
                current_row_non_zeros = original_row_values[non_zero_mask].reshape(-1, 1)
                scaled_values_for_row = scaler_row.transform(current_row_non_zeros)

                scaled_row_new_values[non_zero_mask] = scaled_values_for_row.flatten()

            df_scaled_rowwise.loc[index, df_sheet.columns[9:]] = scaled_row_new_values

            row_scalers_for_sheet[index] = scaler_row

    self.scaler_for_sheet_dict[sheet_name] = row_scalers_for_sheet

    if self.config['scale_by'] == 'feature':
        return df_scaled
    elif self.config['scale_by'] == 'feature_and_company':
        return df_scaled_rowwise
