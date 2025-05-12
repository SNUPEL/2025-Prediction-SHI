import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE as SMOTESampler
import numpy as np


def split_data(self):
    rows_train = []
    rows_test = []
    for company_id, company in self.company_dict.items():
        for date in company.date_range:
            name = str(company_id) + '_' + str(date)
            label_date = date + pd.DateOffset(months=self.config['data_duration'] - 1 + self.config['label_duration'])
            if label_date not in company.date_range:
                break
            else:
                if company.label and company.end_date - pd.DateOffset(months=self.config['label_duration'] - 1) <= label_date <= company.end_date:
                    row = {"company_id": name, "label": company.label}
                else:
                    row = {"company_id": name, "label": False}

                if self.config['Flatten']:
                    for sheet_name, series in company.data_dict.items():
                        for i, val in enumerate(series):
                            row[f"{sheet_name}_{i}"] = 0 if pd.isna(val) else val
                            if i == self.config['data_duration']:
                                break
                else:
                    pass

                if self.label_date - pd.DateOffset(months=self.config['label_duration'] - 1) <= label_date <= self.label_date:
                    rows_test.append(row)
                else:
                    if self.config['data_split_type'] == 'no_overlap':
                        if date + pd.DateOffset(months=self.config['data_duration'] * 2 - 1) >= self.label_date:
                            continue
                    elif self.config['data_split_type'] == 'overlap':
                        pass
                    else:
                        print(f"Error: config['data_split_type'] '{self.config['data_split_type']}' was not found.")
                        return False
                    rows_train.append(row)

    self.df_train = pd.DataFrame(rows_train)
    self.df_test = pd.DataFrame(rows_test)

    self.df_x_train = self.df_train.drop(columns=['label'])
    self.df_y_train = self.df_train['label']
    self.df_x_test = self.df_test.drop(columns=['label'])
    self.df_y_test = self.df_test['label']


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
