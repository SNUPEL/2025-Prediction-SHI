import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE as SMOTESampler
import numpy as np


def split_data(self):
    rows_train = [] # 훈련 데이터의 행들을 저장할 리스트
    rows_test = []  # 테스트 데이터의 행들을 저장할 리스트

    # self.company_dict: load_data에서 채워진 회사 정보 딕셔너리
    for company_id, company in self.company_dict.items():
        for date in company.date_range:

            # 'company_id' 컬럼에 들어갈 식별자: 원본 회사 ID와 데이터 윈도우 시작 날짜 연결
            name = str(company_id) + '_' + str(date.date()) # date.date()로 날짜 부분만 추출

            # 이 데이터 포인트의 레이블(정답) 시점 계산
            # label_date = 데이터 윈도우 시작 ('date') + 데이터 기간 ('data_duration') + 예측 기간 ('label_duration') 간격
            label_date = date + pd.DateOffset(months=self.config['data_duration'] - 1 + self.config['label_duration'])

            # 계산된 label_date가 해당 회사의 데이터 유효 기간 마지막을 벗어나는지 확인
            # load_data에서 company.date_range는 comparison_date(max self.label_date)까지로 제한됩니다.
            # 계산된 label_date가 company.date_range의 마지막 날짜(가장 최근 데이터 시점)보다 미래이면,
            # 해당 시점의 레이블(정답)을 알 수 없으므로 이 회사에 대한 더 이상의 데이터 포인트 생성을 중단합니다.
            if company.date_range and label_date > company.date_range[-1]:
                 break # 이 회사에 대한 date 루프 종료

            # 계산된 label_date를 기준으로 이 데이터 포인트의 레이블(True/False) 결정
            # 이 로직은 기존과 동일하게 유지됩니다.
            if company.label and company.end_date - pd.DateOffset(months=self.config['label_duration'] - 1) <= label_date <= company.end_date:
                 row = {"company_id": name, "label": True} # True: 경영악화
            else:
                 row = {"company_id": name, "label": False} # False: 거래중 또는 다른 상태


            # 이 데이터 포인트에 사용될 특징 데이터의 시간 범위: 'date' (시작) ~ 'date + data_duration - 1 month' (끝)
            window_start_date = date # 특징 데이터 윈도우 시작 날짜
            window_end_date = date + pd.DateOffset(months=self.config['data_duration'] - 1) # 특징 데이터 윈도우 끝 날짜

            if self.config['Flatten']:
                # 각 시트별 데이터 추출 및 특징으로 평탄화
                for sheet_name, full_series in company.data_dict.items():
                    for i in range(self.config['data_duration']):
                        # 윈도우 내 i번째 월에 해당하는 날짜 계산
                        current_date_in_window = window_start_date + pd.DateOffset(months=i)

                        # 해당 날짜의 값을 full_series에서 가져오고, 없으면 np.nan
                        val = full_series.get(current_date_in_window, np.nan)

                        # row 딕셔너리에 특징 컬럼 이름과 값 할당
                        row[f"{sheet_name}_{i}"] = val # np.nan 값 그대로 할당 (나중에 fillna(0))

            else:
                pass

            # 분할 기준: 계산된 label_date와 self.split_cutoff_date 비교
            if label_date <= self.split_cutoff_date:
                # 계산된 label_date가 분할 기준 날짜보다 같거나 이전이면 훈련 세트
                rows_train.append(row)
            elif label_date > self.split_cutoff_date:
                # 계산된 label_date가 분할 기준 날짜보다 이후이면 테스트 세트
                rows_test.append(row)

    # 리스트에 담긴 행들로 DataFrame 생성
    self.df_train = pd.DataFrame(rows_train)
    self.df_test = pd.DataFrame(rows_test)

    # 특징 컬럼들(company_id, label 제외)에 대해 누락된 값(NaN)을 0으로 채우기
    # 플래트닝 로직에서 np.nan으로 할당했으므로 여기서 채워줍니다.
    feature_cols_train = [col for col in self.df_train.columns if col not in ['company_id', 'label']]
    feature_cols_test = [col for col in self.df_test.columns if col not in ['company_id', 'label']]

    # 해당 컬럼들에 대해서만 fillna(0) 적용
    self.df_train[feature_cols_train] = self.df_train[feature_cols_train].fillna(0)
    self.df_test[feature_cols_test] = self.df_test[feature_cols_test].fillna(0)


    # 특징 데이터 (X)와 레이블 (y) 분리
    self.df_x_train = self.df_train.drop(columns=['label'])
    self.df_y_train = self.df_train['label']
    self.df_x_test = self.df_test.drop(columns=['label'])
    self.df_y_test = self.df_test['label']
    # --- DataFrame 생성 및 최종 처리 끝 ---


def apply_SMOTE(self):
    print("\n=== SMOTE 오버샘플링 시작 ===")
    # SMOTE 적용 전 데이터 형태 확인
    company_ids = self.df_x_train['company_id'].values
    X_train = self.df_x_train.drop(columns=["company_id"])
    y_train = self.df_y_train

    # 클래스 불균형 확인
    class_counts = np.bincount(y_train.astype(int))
    print(f"SMOTE 적용 전 클래스 분포: 거래중={class_counts[0]}, 경영악화={class_counts[1] if len(class_counts) > 1 else 0}")
    print(f"총 훈련 데이터: {len(y_train)}개 샘플")

    # SMOTE 적용
    smote = SMOTESampler(random_state=self.config['random_state'])
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # 결과 클래스 분포 확인
    resampled_class_counts = np.bincount(y_resampled.astype(int))
    print(f"SMOTE 적용 후 클래스 분포: 거래중={resampled_class_counts[0]}, 경영악화={resampled_class_counts[1]}")

    # company_id 다시 추가 (원래 있던 id를 복제)
    # 원본 데이터 수
    orig_count = len(y_train)
    # 리샘플링 후 추가된 수
    new_count = len(y_resampled) - orig_count

    # 원본 company_id를 유지하고, 추가 데이터에는 기존 company_id 중 경영악화(label=1) 데이터의 id를 복제
    synthetic_ids = []
    minority_ids = company_ids[y_train == 1]

    if len(minority_ids) > 0:
        # 소수 클래스의 id를 랜덤하게 선택하여 복제
        synthetic_ids = np.random.choice(minority_ids, size=new_count, replace=True)
        print(f"경영악화 기업 ID 개수: {len(minority_ids)}개")
    else:
        # 소수 클래스가 없는 경우, 기존 id에서 랜덤 선택
        synthetic_ids = np.random.choice(company_ids, size=new_count, replace=True)
        print("경영악화 기업이 없어 랜덤하게 ID 생성")

    # id 결합
    all_ids = np.concatenate([company_ids, synthetic_ids])

    # DataFrame으로 변환하고 company_id 추가
    self.df_x_train = pd.DataFrame(X_resampled, columns=X_train.columns)
    self.df_x_train['company_id'] = all_ids
    self.df_y_train = pd.Series(y_resampled, name='label')

    print(f"SMOTE 적용 완료: {orig_count}개 → {len(y_resampled)}개 샘플")
    print(f"합성 데이터 생성: {new_count}개 샘플")
    print("=== SMOTE 오버샘플링 완료 ===\n")


def random_split(self):
    self.df_x = self.df_model.drop(columns=['label'])
    self.df_y = self.df_model['label']

    self.df_x_train, self.df_x_test, self.df_y_train, self.df_y_test = train_test_split(
        self.df_x, self.df_y, test_size=self.config['test_data_ratio'], random_state=self.config['random_state'])


def flatten(self):
    rows = []
    for company_id, company in self.company_dict.items():
        row = {"company_id": company_id, "label": company.label}  # 정답 레이블 포함

        for sheet_name, series in company.data_dict.items():
            for i, val in enumerate(series):
                row[f"{sheet_name}_{i}"] = val

        rows.append(row)
    self.df_model = pd.DataFrame(rows)
