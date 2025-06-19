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

        # 수집된 모든 0이 아닌 값들에 대해 단일 스케일러 학습
        scaler = StandardScaler()
        scaler.fit(all_non_zero_values_np)

        for column in df_scaled.columns[9:]:
            original_column_values = df_scaled[column].values  # 원본 값 사용
            scaled_column_new_values = np.zeros_like(original_column_values, dtype=float)  # 결과 배열 초기화 (0으로)

            non_zero_mask = original_column_values != 0

            if np.any(non_zero_mask):  # 해당 열에 0이 아닌 값이 있다면
                current_col_non_zeros = original_column_values[non_zero_mask].reshape(-1, 1)
                scaled_values_for_col = scaler.transform(current_col_non_zeros)

                # 스케일링된 값을 원래 0이 아니었던 위치에만 다시 배치
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

        # Global 코드의 `for column in df_train_valid.columns:` 루프와 동일한 역할
        for value in train_data_for_row:
            # [중요] 런타임 오류 방지를 위해 유효한 숫자인지 확인하는 로직은 필수입니다.
            if value != 0 and pd.notna(value) and not np.isinf(value):
                row_non_zero_values_list.append(value)

        # 2. [구조 일치] 수집된 값이 2개 이상일 때만 스케일러를 학습합니다.
        if len(row_non_zero_values_list) > 1:
            row_non_zero_values_np = np.array(row_non_zero_values_list).reshape(-1, 1)

            # [구조 일치] 수집된 값들에 대해 단일 스케일러 학습
            scaler_row = StandardScaler()
            scaler_row.fit(row_non_zero_values_np)

            # 3. [구조 일치] '전체 기간' 데이터에 학습된 스케일러를 적용합니다.
            # Global 코드의 `for column in df_scaled.columns[9:]:` 루프에 해당합니다.

            original_row_values = df_scaled_rowwise.loc[index, df_sheet.columns[9:]].values
            scaled_row_new_values = np.zeros_like(original_row_values, dtype=float)

            # [구조 일치] 0이 아닌 값의 위치를 찾는 마스크 생성
            # [중요] 적용 시점에서도 유효한 숫자인지 확인해야 합니다.
            non_zero_mask = original_row_values != 0

            if np.any(non_zero_mask):
                current_row_non_zeros = original_row_values[non_zero_mask].reshape(-1, 1)
                scaled_values_for_row = scaler_row.transform(current_row_non_zeros)

                scaled_row_new_values[non_zero_mask] = scaled_values_for_row.flatten()

            df_scaled_rowwise.loc[index, df_sheet.columns[9:]] = scaled_row_new_values

            row_scalers_for_sheet[index] = scaler_row

    self.scaler_dict[sheet_name] = row_scalers_for_sheet

    if self.config['scale_by'] == 'feature':
        return df_scaled
    elif self.config['scale_by'] == 'feature_and_company':
        return df_scaled_rowwise
