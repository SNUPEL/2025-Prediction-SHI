import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

def load_and_pivot_data(config):
    print("Step 1: 데이터 로딩 및 기본 전처리를 시작합니다...")
    file_path = config['data_file_path']
    sub_data_file_path = config.get('sub_data_file_path')
    ban_list = config['sheet_ban_list']
    xls = pd.ExcelFile(file_path)
    company_info_df = pd.DataFrame()
    df_sub_data = None

    # 협력사별 데이터 마스킹 관련 로직
    if sub_data_file_path and os.path.exists(sub_data_file_path):
        try:
            df_sub_data_raw = pd.read_excel(sub_data_file_path, sheet_name='추가 정보', skiprows=[0, 2])
            id_col_name_sub = df_sub_data_raw.columns[0]
            masking_col_name_sub = '협력사별 데이터 제거 필요 개월(철수일 기준)'
            if not df_sub_data_raw.empty and masking_col_name_sub in df_sub_data_raw.columns:
                df_sub_data = df_sub_data_raw[[id_col_name_sub, masking_col_name_sub]].copy()
                df_sub_data.columns = ['ID_sub', 'DataMaskingMonths']
                df_sub_data['ID_sub'] = df_sub_data['ID_sub'].astype(str)
                print("INFO: 협력사별 추가 정보 파일을 로드하여 마스킹을 적용합니다.")
            else:
                print("INFO: 협력사별 추가 정보 파일은 존재하나, 마스킹 관련 컬럼이 없거나 비어 마스킹을 건너뜁니다.")
        except Exception as e:
            print(f"경고: 추가 정보 파일 로드 실패. {e}")
    else:
        print("INFO: 협력사별 추가 정보 파일을 찾을 수 없어 마스킹을 건너뜁니다.")

    try:
        raw_info_df = pd.read_excel(xls, sheet_name='기성매출(정상기성)')
        company_info_df = raw_info_df.iloc[:, [0, 4, 5]].copy()
        company_info_df.columns = ['ID', 'OriginalStartDate', 'OriginalEndDate']
        company_info_df['ID'] = company_info_df['ID'].astype(str)
        company_info_df['OriginalStartDate'] = pd.to_datetime(company_info_df['OriginalStartDate'], errors='coerce')
        company_info_df['OriginalEndDate'] = pd.to_datetime(company_info_df['OriginalEndDate'], errors='coerce')

        if df_sub_data is not None:
            company_info_df = pd.merge(company_info_df, df_sub_data, left_on='ID', right_on='ID_sub', how='left').drop(
                columns=['ID_sub'], errors='ignore')
            company_info_df['DataMaskingMonths'] = company_info_df.get('DataMaskingMonths',
                                                                       pd.Series(dtype='float64')).fillna(0).astype(int)

            company_info_df['OriginalEndDate'] = company_info_df.apply(
                lambda row: row['OriginalEndDate'] - pd.DateOffset(months=row['DataMaskingMonths'])
                if pd.notna(row['OriginalEndDate']) and row['DataMaskingMonths'] > 0
                else row['OriginalEndDate'], axis=1
            )
        else:
            company_info_df['DataMaskingMonths'] = 0


    except Exception as e:
        print(f" 에러: 회사 정보 로딩 또는 마스킹 적용 중 오류 발생. {e}")
        return pd.DataFrame(), pd.DataFrame()

    all_sheets_data = {}
    for sheet_name in xls.sheet_names:
        if sheet_name not in ban_list and sheet_name != '기성매출(정상기성)':
            try:
                df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
                if df_sheet.empty or len(df_sheet.columns) < 8: continue

                original_date_cols_names = df_sheet.columns[7:]
                date_columns_converted = []
                for col_name in original_date_cols_names:
                    if isinstance(col_name, str) and '월' in col_name:
                        dt_obj = pd.to_datetime(col_name.replace('월', ''), format='%y.%m', errors='coerce')
                    else:
                        dt_obj = pd.to_datetime(col_name, errors='coerce')
                    date_columns_converted.append(dt_obj.replace(day=1) if pd.notna(dt_obj) else dt_obj)

                df_sheet.columns = list(df_sheet.columns[:7]) + date_columns_converted

                df_sheet = df_sheet.loc[:, ~df_sheet.columns.duplicated()]

                df_sheet.set_index(df_sheet.columns[0], inplace=True)
                all_sheets_data[sheet_name] = df_sheet.iloc[:, 7:].transpose()
            except Exception as e:
                print(f"경고: 시트 '{sheet_name}' 처리 중 오류. {e}")

    if not all_sheets_data:
        print("에러: 피처 데이터를 가진 시트를 찾을 수 없습니다.")
        return pd.DataFrame(), company_info_df

    merged_df = pd.concat(all_sheets_data.values(), axis=1, keys=all_sheets_data.keys())
    merged_df.columns.names = ['Feature', 'ID']
    merged_df.index.name = 'YYYYMM_dt'

    long_df = merged_df.stack(level=['Feature', 'ID']).reset_index(name='Value')
    long_df['Value'] = pd.to_numeric(long_df['Value'], errors='coerce')

    print("\n--- INFO: long_df 중복 제거를 시작합니다. ---")

    long_df_initial_rows = len(long_df)
    long_df.drop_duplicates(subset=['ID', 'YYYYMM_dt', 'Feature'], keep='first', inplace=True)
    long_df_after_dedup = len(long_df)

    if long_df_initial_rows > long_df_after_dedup:
        print(f"long_df에서 (ID, YYYYMM_dt, Feature) 기준 중복된 행 {long_df_initial_rows - long_df_after_dedup}개를 제거했습니다.")
    else:
        print("long_df에 (ID, YYYYMM_dt, Feature) 기준 중복된 행이 없었습니다.")
    print("------------------------------------------")

    pivot_aggfunc = config.get('pivot_table_aggfunc', 'first')
    print(f"INFO: pivot_table aggregation function: '{pivot_aggfunc}'")

    clean_df = long_df.pivot_table(
        index=['ID', 'YYYYMM_dt'],
        columns='Feature',
        values='Value',
        aggfunc=pivot_aggfunc
    ).reset_index().fillna(0)

    clean_df['ID'] = clean_df['ID'].astype(str)
    clean_df['YYYYMM_dt'] = pd.to_datetime(clean_df['YYYYMM_dt'])
    print("데이터 로딩 및 기본 전처리 완료.")
    return clean_df, company_info_df

def create_sequences(base_df, company_info_df, config):
    print("Step 2: 피처 및 라벨 시퀀스 생성을 시작합니다...")
    data_duration = config['data_duration']
    label_duration = config['label_duration']
    config_data_start_date = pd.to_datetime(config['data_start_date'])
    config_label_date = pd.to_datetime(config['label_date'])  # 전역 데이터 처리 종료일
    all_X, all_y, all_info = [], [], []
    feature_cols = sorted([col for col in base_df.columns if col not in ['ID', 'YYYYMM_dt']])
    for company_id, group_df in base_df.groupby('ID'):
        company_info = company_info_df[company_info_df['ID'] == company_id]
        if company_info.empty: continue
        group_df = group_df.sort_values('YYYYMM_dt')

        # actual_end_date는 load_and_pivot_data에서 이미 DataMaskingMonths가 적용된 날짜
        actual_end_date = company_info['OriginalEndDate'].iloc[0]
        original_company_start_date = company_info['OriginalStartDate'].iloc[0]

        effective_start = max(original_company_start_date, config_data_start_date) if pd.notna(
            original_company_start_date) else config_data_start_date
        effective_end = min(actual_end_date, config_label_date) if pd.notna(actual_end_date) else config_label_date
        company_ts = group_df[(group_df['YYYYMM_dt'] >= effective_start) & (group_df['YYYYMM_dt'] <= effective_end)]

        if len(company_ts) < data_duration: continue

        for i in range(len(company_ts) - data_duration + 1):
            feature_window = company_ts.iloc[i: i + data_duration]
            feature_window_end_dt = feature_window['YYYYMM_dt'].iloc[-1]

            label_point_dt = feature_window_end_dt + pd.DateOffset(months=label_duration)

            if label_point_dt > effective_end: break

            label = 0
            if pd.notna(actual_end_date):  # actual_end_date는 마스킹이 적용된 종료일
                # 라벨 결정 로직은 마스킹된 actual_end_date를 사용합니다.
                if (actual_end_date - pd.DateOffset(months=label_duration - 1)) <= label_point_dt <= actual_end_date:
                    label = 1

            unique_sample_id = f"{company_id}_{feature_window_end_dt.strftime('%Y%m')}"
            all_X.append(feature_window[feature_cols].values)
            all_y.append(label)
            all_info.append({'unique_sample_id': unique_sample_id, 'label_point_dt': label_point_dt})

    if not all_X: raise ValueError("유효한 학습 시퀀스를 생성하지 못했습니다.")
    X_all, y_all, info_df = np.array(all_X), np.array(all_y), pd.DataFrame(all_info)
    print(f"피처/라벨 생성 완료. 총 {len(X_all)}개 시퀀스 생성.")
    return X_all, y_all, info_df, feature_cols


def split_data_by_date(X_all, y_all, info_df, config):
    split_cutoff_date = pd.to_datetime(config['split_cutoff_date'])
    train_indices = info_df[info_df['label_point_dt'] < split_cutoff_date].index
    test_indices = info_df[info_df['label_point_dt'] >= split_cutoff_date].index
    X_train, y_train = X_all[train_indices], y_all[train_indices]
    X_test, y_test = X_all[test_indices], y_all[test_indices]
    info_train, info_test = info_df.loc[train_indices], info_df.loc[test_indices]
    print(f"INFO: 데이터를 분할했습니다. (Train: {len(X_train)}, Test: {len(X_test)})")
    return X_train, y_train, X_test, y_test, info_train, info_test


def scale_data(X_train, X_test, config):
    if config['scaler'] != 'standard': return X_train, X_test, None
    print("INFO: StandardScaler로 피처 스케일링을 적용합니다...")
    scaler = StandardScaler()
    original_shape_train, original_shape_test = X_train.shape, X_test.shape
    X_train_reshaped = X_train.reshape(-1, original_shape_train[-1])
    X_test_reshaped = X_test.reshape(-1, original_shape_test[-1])
    scaler.fit(X_train_reshaped)
    X_train_scaled_reshaped, X_test_scaled_reshaped = scaler.transform(X_train_reshaped), scaler.transform(
        X_test_reshaped)
    X_train_scaled, X_test_scaled = X_train_scaled_reshaped.reshape(
        original_shape_train), X_test_scaled_reshaped.reshape(original_shape_test)
    return X_train_scaled, X_test_scaled, scaler


def apply_sampling(X_train, y_train, config):
    if config['oversampling'] == 'SMOTE':
        print(f"INFO: SMOTE 오버샘플링을 적용합니다 (k={config['SMOTE_k_neighbors']})...")
        smote = SMOTE(random_state=config['random_state'], k_neighbors=config['SMOTE_k_neighbors'],
                      sampling_strategy=config['SMOTE_sampling_strategy'])
        X_train, y_train = smote.fit_resample(X_train, y_train)
    if config['undersampling'] == 'ENN':
        print(f"INFO: ENN 언더샘플링을 적용합니다 (n_neighbors={config['ENN_n_neighbors']})...")
        enn = EditedNearestNeighbours(n_neighbors=config['ENN_n_neighbors'])
        X_train, y_train = enn.fit_resample(X_train, y_train)
    return X_train, y_train