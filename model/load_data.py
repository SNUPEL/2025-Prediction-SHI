import pandas as pd
import numpy as np
from normalize_data import *


class Company:
    def __init__(self, company_id, category, subcategory, sub_subcategory, start_date, end_date, comparison_date,
                 date_range, label, condition, age, data_masking_duration, severity_level):
        self.company_id = company_id
        self.category = category
        self.subcategory = subcategory
        self.sub_subcategory = sub_subcategory
        self.start_date = start_date
        self.end_date = end_date
        self.comparison_date = comparison_date  # 이 회사의 date_range의 끝 날짜
        self.date_range = date_range  # 유효 데이터 기간(월별 날짜 리스트)
        self.label = label  # 회사 전체의 최종 레이블 (True: 경영악화, False: 거래중)
        self.condition = condition  # 데이터 가용성/기간 기반 조건 (초기 필터링에 사용)
        self.age = age
        self.data_masking_duration = data_masking_duration
        self.severity_level = severity_level

        self.data_dict = dict()  # 시트별 시계열 데이터를 저장할 딕셔너리


def load_data(self):
    self.df_raw_data_dict = pd.read_excel(self.config['data_file_path'], sheet_name=None)
    self.df_raw_sub_data = pd.read_excel(self.config['sub_data_file_path'], sheet_name='추가 정보', skiprows=[0, 2])
    self.sheet_name_list = list(self.df_raw_data_dict.keys())

    df_temp = self.df_raw_data_dict[self.sheet_name_list[0]]

    info_cols_count = 8
    date_col_start_index = info_cols_count

    if pd.isna(pd.to_datetime(df_temp.columns[-1], errors='coerce')):
        df_temp = df_temp.iloc[:, :-1]
        original_date_cols = df_temp.columns[date_col_start_index:]
        converted_date_cols = pd.to_datetime(original_date_cols, errors='coerce').map(
            lambda dt: dt.replace(day=1) if pd.notna(dt) else dt
        )
        new_cols = list(df_temp.columns[:info_cols_count]) + list(converted_date_cols)
        df_temp.columns = new_cols
    else:
        original_date_cols = df_temp.columns[date_col_start_index:]
        converted_date_cols = pd.to_datetime(original_date_cols, errors='coerce').map(
             lambda dt: dt.replace(day=1) if pd.notna(dt) else dt
         )
        new_cols = list(df_temp.columns[:info_cols_count]) + list(converted_date_cols)
        df_temp.columns = new_cols  # 수정된 컬럼 이름 적용

    # --- 첫 번째 루프: df_temp를 순회하며 company_dict를 회사별 기본 정보로 초기화 ---

    for i, row in df_temp.iterrows():
        # 보조 데이터 파일(self.df_raw_sub_data)에서 해당 행(회사)에 대한 추가 정보(나이, 마스킹 기간, 심각도)를 가져옴
        # 보조 데이터 파일의 행 순서가 메인 데이터 파일의 첫 번째 시트 행 순서와 같다고 가정하고 행 인덱스 'i'를 사용
        age = int(self.df_raw_sub_data.loc[i, '철수 당시/ 현 나이(만)'])
        data_masking_duration = 0 if pd.isna(self.df_raw_sub_data.loc[i, '협력사별 데이터 제거 필요 개월(철수일 기준)']) else int(
            self.df_raw_sub_data.loc[i, '협력사별 데이터 제거 필요 개월(철수일 기준)'])
        severity_level = None if pd.isna(self.df_raw_sub_data.loc[i, '경영악화 경/중 구분']) else self.df_raw_sub_data.loc[
            i, '경영악화 경/중 구분']

        original_company_id = row[df_temp.columns[0]]  # 컬럼 인덱스 0을 사용

        # 회사 데이터의 시작 날짜 (r_start_date)를 결정
        # 회사의 실제 시작 날짜(5번째 컬럼)와 config['data_start_date'] 중 더 나중의 날짜를 선택
        actual_start_date = pd.to_datetime(row[df_temp.columns[4]]).replace(day=1)
        r_start_date = max(actual_start_date, self.config_start_date_dt)

        # 회사의 종료 날짜 (r_end_date)와 레이블 (r_label)을 결정
        actual_end_date_val = row[df_temp.columns[5]]
        r_end_date = None
        r_label = False
        if pd.notna(actual_end_date_val) and actual_end_date_val != '-':
            r_label = True
            actual_end_date_dt = pd.to_datetime(actual_end_date_val).replace(day=1)
            r_end_date = actual_end_date_dt - pd.DateOffset(months=data_masking_duration)

        # comparison_date를 결정. 이 날짜는 이 회사의 데이터 포인트 생성을 고려할 마지막 날짜
        # 조정된 종료 날짜(r_end_date, 회사가 경영악화된 경우)와 self.label_date 중 더 이른 날짜를 사용
        # 회사가 종료되지 않았거나 self.label_date보다 늦게 종료된 경우 self.label_date를 사용
        comparison_date = self.label_date  # 기본값은 config['label_date'] (Data 클래스 초기화 시 설정됨)
        if r_label and r_end_date is not None and r_end_date <= self.label_date:
            comparison_date = r_end_date  # 회사가 경영악화되었고 조정된 종료일이 label_date보다 같거나 이전이면 해당 종료일 사용

        # company.date_range를 생성 r_start_date부터 comparison_date까지의 월별 날짜 리스트
        # 이 범위가 split_data 함수에서 각 회사의 date 순회 범위
        # r_start_date가 comparison_date보다 이후인 경우 빈 리스트
        if comparison_date < r_start_date:
            date_range = []  # 유효한 데이터 포인트 생성 기간이 없으므로 빈 리스트
        else:
            date_range = list(pd.date_range(start=r_start_date, end=comparison_date, freq='MS'))

        r_condition = True if comparison_date >= r_start_date else False  # 모든 로딩된 회사를 일단 포함하는 것으로 간주

        self.company_dict[original_company_id]\
            = Company(original_company_id, row[df_temp.columns[1]],row[df_temp.columns[2]], row[df_temp.columns[3]],
                      r_start_date, r_end_date, comparison_date, date_range, r_label,
                      r_condition, age, data_masking_duration, severity_level)

    self.company_dict = {k: v for k, v in self.company_dict.items() if v.condition}

    for sheet_name in self.sheet_name_list:
        if sheet_name in self.sheet_ban_list:
            continue
        else:
            df_sheet = self.df_raw_data_dict[sheet_name]
            if pd.isna(pd.to_datetime(df_sheet.columns[-1], errors='coerce')):
                df_sheet = df_sheet.iloc[:, :-1]  # 마지막 컬럼이 날짜가 아니면 제외
                original_date_cols = df_sheet.columns[date_col_start_index:]
                converted_date_cols = pd.to_datetime(original_date_cols, errors='coerce').map(
                    lambda dt: dt.replace(day=1) if pd.notna(dt) else dt
                )
                new_cols = list(df_sheet.columns[:info_cols_count]) + list(converted_date_cols)
                df_sheet.columns = new_cols
            else:
                original_date_cols = df_sheet.columns[date_col_start_index:]
                converted_date_cols = pd.to_datetime(original_date_cols, errors='coerce').map(
                     lambda dt: dt.replace(day=1) if pd.notna(dt) else dt
                )
                new_cols = list(df_sheet.columns[:info_cols_count]) + list(converted_date_cols)
                df_sheet.columns = new_cols

            if self.config['scaler'] == 'None':
                pass
            elif self.config['scaler'] == 'standard':
                df_sheet = standard_scaler(self, sheet_name, df_sheet)

            if self.config['use_all_data'] == 'All':
                for i, row in df_sheet.iterrows():
                    original_company_id_from_row = row[df_sheet.columns[0]]
                    if original_company_id_from_row in self.company_dict:
                        company_obj = self.company_dict[original_company_id_from_row]

                        # --- 데이터 로딩 디버깅 출력 추가 (데이터 로딩 과정 확인용) ---
                        print(f"\n--- 로딩 디버그: 회사 ID {original_company_id_from_row}, 시트 {sheet_name} ---")
                        # 이 회사의 company.date_range (split_data에서 데이터 포인트 생성을 고려할 기간)를 확인합니다.
                        print(f"  company.date_range (이 회사의 데이터 예상 기간): {company_obj.date_range}")
                        # 현재 행 Series (row)의 날짜 컬럼 부분만 추출하여 인덱스(날짜)를 확인합니다.
                        row_date_data_series = row[df_sheet.columns[date_col_start_index:]]
                        print(f"  row Series index (현재 Excel 행의 실제 컬럼 날짜): {row_date_data_series.index.tolist()}")

                        # company.date_range에 있는 날짜들이 row Series 인덱스(Excel 컬럼 날짜)에 모두 포함되는지 확인 (잠재적 불일치 진단)
                        missing_dates_in_row = [d for d in company_obj.date_range if d not in row_date_data_series.index]
                        if missing_dates_in_row:
                            print(f"  경고: company.date_range 날짜 중 현재 행의 Excel 컬럼 날짜에 없는 날짜: {missing_dates_in_row}")
                            print("  해당 날짜 데이터는 추출되지 않거나 NaN으로 처리될 수 있습니다.")

                        print(f"  company.date_range에 해당하는 row 데이터 추출 시도...")

                        try:
                            extracted_series = row_date_data_series.reindex(company_obj.date_range)

                            processed_series = extracted_series.fillna(0).astype(float)

                            company_obj.data_dict[sheet_name] = processed_series

                            print(f"  '{sheet_name}' 시트 데이터 company.data_dict에 성공적으로 할당 완료.")
                            if not processed_series.empty:
                                print(f"  할당된 데이터의 실제 시작/끝 날짜: {processed_series.index.min()} ~ {processed_series.index.max()}")
                                print(f"  할당된 데이터 개수: {len(processed_series)}")
                            else:
                                print("  경고: company.data_dict에 할당된 데이터 Series가 비어있습니다.")

                        except Exception as e:
                            print(f"  !!! 데이터 추출/할당 중 오류 발생 (회사 ID {original_company_id_from_row}, 시트 {sheet_name}): {e} !!!")

            elif self.config['use_all_data'] == 'Padding':
                pass
            else:
                print(f"Error: config['use_all_data'] '{self.config['use_all_data']}' was not found.")
                return False

    print('Data has been loaded successfully')
