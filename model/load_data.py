import pandas as pd
import numpy as np
import warnings
from normalize_data import *  # 데이터 정규화 관련 함수(standard_scaler) 임포트


class Company:
    """
    개별 회사(협력사)의 모든 정보를 담는 클래스.
    회사의 메타 정보(ID, 카테고리 등)와 시트별 시계열 데이터를 저장합니다.
    """

    def __init__(self, company_id, category, subcategory, sub_subcategory, start_date, end_date, comparison_date,
                 date_range, label, condition, age, data_masking_duration, severity_level):
        """
        Company 객체 초기화 메서드.
        """
        self.company_id = company_id  # 회사 고유 ID
        self.category = category  # 대분류
        self.subcategory = subcategory  # 중분류
        self.sub_subcategory = sub_subcategory  # 소분류
        self.start_date = start_date  # 유효 데이터 시작일
        self.end_date = end_date  # 경영악화(철수) 발생일. 정상 거래중인 경우 None.
        self.comparison_date = comparison_date  # 이 회사의 데이터 생성 시 고려할 마지막 날짜 (date_range의 끝 날짜)
        self.date_range = date_range  # 이 회사로부터 데이터를 추출할 유효 기간 (월별 날짜 리스트)
        self.label = label  # 이 회사의 최종 레이블 (True: 경영악화, False: 정상)
        self.condition = condition  # 데이터 사용 가능 여부 (유효 기간이 존재하는지)
        self.age = age  # 회사 나이 (만)
        self.data_masking_duration = data_masking_duration  # 경영악화 발생 전 데이터를 제거할 개월 수
        self.severity_level = severity_level  # 경영악화 심각도 (경/중)

        self.data_dict = dict()  # key: 시트명, value: 해당 시트의 시계열 데이터(pd.Series)를 저장할 딕셔너리


def load_data(self):
    """
    메인 데이터와 서브 데이터 엑셀 파일을 로드하여,
    각 회사별로 Company 객체를 생성하고, 각 객체에 시계열 데이터를 채워넣는 함수.
    이 함수는 Data 클래스의 인스턴스 메서드처럼 동작하도록 `self`를 인자로 받습니다.
    """
    # --- 1. 데이터 파일 로드 ---
    # 메인 데이터 파일(모든 시트)과 서브 데이터 파일(추가 정보 시트)을 읽어옴
    self.df_raw_data_dict = pd.read_excel(self.config['data_file_path'], sheet_name=None)
    self.df_raw_sub_data = pd.read_excel(self.config['sub_data_file_path'], sheet_name='추가 정보', skiprows=[0, 2])
    self.sheet_name_list = list(self.df_raw_data_dict.keys())  # 모든 시트 이름 저장

    # 첫 번째 시트를 기준으로 회사 정보를 처리하기 위해 임시 DataFrame 생성
    df_temp = self.df_raw_data_dict[self.sheet_name_list[0]]

    # --- 2. 날짜 컬럼 형식 통일 ---
    # 엑셀 파일의 날짜 컬럼(예: '2023-01-01 00:00:00')을 'YYYY-MM-01' 형식으로 통일
    info_cols_count = 8  # 회사 정보 컬럼 개수
    date_col_start_index = info_cols_count  # 날짜 데이터가 시작되는 컬럼 인덱스

    # 마지막 컬럼이 날짜 형식이 아니면 (예: 주석 컬럼) 제외
    if pd.isna(pd.to_datetime(df_temp.columns[-1], errors='coerce')):
        df_temp = df_temp.iloc[:, :-1]

    original_date_cols = df_temp.columns[date_col_start_index:]  # 원본 날짜 컬럼들
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # 불필요한 경고 메시지 무시
        # 날짜 형식으로 변환하고, 일(day)을 1일로 통일
        converted_date_cols = pd.to_datetime(original_date_cols, errors='coerce').map(
            lambda dt: dt.replace(day=1) if pd.notna(dt) else dt
        )
    # 정보 컬럼 + 변환된 날짜 컬럼으로 새로운 컬럼 리스트 생성
    new_cols = list(df_temp.columns[:info_cols_count]) + list(converted_date_cols)
    df_temp.columns = new_cols  # DataFrame에 새로운 컬럼 적용

    # --- 3. 회사별 Company 객체 생성 및 기본 정보 초기화 ---
    for i, row in df_temp.iterrows():
        # 서브 데이터 파일에서 '나이', '데이터 마스킹 기간', '심각도' 등 추가 정보를 가져옴
        age = int(self.df_raw_sub_data.loc[i, '철수 당시/ 현 나이(만)'])
        data_masking_duration = 0 if pd.isna(self.df_raw_sub_data.loc[i, '협력사별 데이터 제거 필요 개월(철수일 기준)']) else int(
            self.df_raw_sub_data.loc[i, '협력사별 데이터 제거 필요 개월(철수일 기준)'])
        severity_level = None if pd.isna(self.df_raw_sub_data.loc[i, '경영악화 경/중 구분']) else self.df_raw_sub_data.loc[
            i, '경영악화 경/중 구분']

        original_company_id = row[df_temp.columns[0]]  # 회사 ID

        # --- 4. 회사별 유효 데이터 기간(start_date, end_date) 및 레이블(label) 결정 ---
        # 실제 데이터 시작일과 설정된 데이터 시작일 중 더 나중 날짜를 유효 시작일(r_start_date)로 설정
        actual_start_date = pd.to_datetime(row[df_temp.columns[4]]).replace(day=1)
        r_start_date = max(actual_start_date, self.config_start_date_dt)

        # 철수일(경영악화 발생일) 정보를 바탕으로 유효 종료일(r_end_date)과 레이블(r_label) 결정
        actual_end_date_val = row[df_temp.columns[5]]
        r_end_date = None
        r_label = False
        if pd.notna(actual_end_date_val) and actual_end_date_val != '-':  # 철수일이 존재하면
            r_label = True  # 경영악화(label=True)로 판단
            actual_end_date_dt = pd.to_datetime(actual_end_date_val).replace(day=1)
            # 실제 철수일에서 데이터 마스킹 기간을 제외하여 최종 유효 종료일 계산
            r_end_date = actual_end_date_dt - pd.DateOffset(months=data_masking_duration)

        # 데이터 생성의 기준이 되는 마지막 날짜(comparison_date) 결정
        # 정상 기업은 예측 기준일(label_date)이 기준이 되고,
        # 부실 기업은 (조정된) 철수일과 예측 기준일 중 더 빠른 날짜가 기준이 됨
        comparison_date = self.label_date
        if r_label and r_end_date is not None and r_end_date <= self.label_date:
            comparison_date = r_end_date

        # 최종적으로 결정된 시작일과 종료일을 바탕으로, 데이터를 추출할 월별 날짜 리스트(date_range) 생성
        if comparison_date < r_start_date:
            date_range = []  # 유효 기간이 없으면 빈 리스트
        else:
            date_range = list(pd.date_range(start=r_start_date, end=comparison_date, freq='MS'))

        # 유효 기간이 존재하는지 여부(condition)를 판단
        r_condition = True if comparison_date >= r_start_date else False

        # 위에서 계산한 모든 정보를 바탕으로 Company 객체를 생성하여 company_dict에 저장
        self.company_dict[original_company_id] \
            = Company(original_company_id, row[df_temp.columns[1]], row[df_temp.columns[2]], row[df_temp.columns[3]],
                      r_start_date, r_end_date, comparison_date, date_range, r_label,
                      r_condition, age, data_masking_duration, severity_level)

    # 유효 기간이 없는(condition=False) 회사는 분석에서 제외
    self.company_dict = {k: v for k, v in self.company_dict.items() if v.condition}

    # --- 5. 모든 시트를 순회하며 각 회사(Company) 객체에 시계열 데이터 채워넣기 ---
    for sheet_name in self.sheet_name_list:
        if sheet_name in self.sheet_ban_list:  # 제외할 시트면 건너뜀
            continue
        else:
            df_sheet = self.df_raw_data_dict[sheet_name]
            # (위와 동일) 날짜 컬럼 형식 통일 작업 수행
            if pd.isna(pd.to_datetime(df_sheet.columns[-1], errors='coerce')):
                df_sheet = df_sheet.iloc[:, :-1]

            original_date_cols = df_sheet.columns[date_col_start_index:]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                converted_date_cols = pd.to_datetime(original_date_cols, errors='coerce').map(
                    lambda dt: dt.replace(day=1) if pd.notna(dt) else dt
                )
            new_cols = list(df_sheet.columns[:info_cols_count]) + list(converted_date_cols)
            df_sheet.columns = new_cols

            # --- 6. (선택적) 데이터 스케일링 ---
            if self.config['scaler'] == 'None':
                pass
            elif self.config['scaler'] == 'standard':  # StandardScaler 적용
                df_sheet = standard_scaler(self, sheet_name, df_sheet)

            # --- 7. 시트 데이터를 각 Company 객체에 할당 ---
            for i, row in df_sheet.iterrows():
                original_company_id_from_row = row[df_sheet.columns[0]]
                if original_company_id_from_row in self.company_dict:  # 분석 대상인 회사일 경우
                    company_obj = self.company_dict[original_company_id_from_row]

                    # 현재 행의 날짜 데이터(시계열)만 추출
                    row_date_data_series = row[df_sheet.columns[date_col_start_index:]]

                    # 해당 회사의 유효 날짜 범위(date_range)에 해당하는 데이터만 추출
                    # reindex를 사용하여, 엑셀에 데이터가 없는 날짜는 NaN으로 채워짐
                    extracted_series = row_date_data_series.reindex(company_obj.date_range)

                    # 결측치(NaN)는 0으로 채우고, 데이터 타입을 float으로 변환
                    processed_series = extracted_series.fillna(0).astype(float)

                    # 최종 처리된 시계열 데이터를 Company 객체의 data_dict에 저장
                    company_obj.data_dict[sheet_name] = processed_series