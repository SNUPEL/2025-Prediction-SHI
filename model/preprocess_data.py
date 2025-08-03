import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import NearMiss


def split_data(self):
    temp_X_train, temp_y_train, temp_name_train = [], [], []
    temp_X_test, temp_y_test, temp_name_test = [], [], []

    # 모든 회사에 공통으로 적용될 feature_names를 미리 정의
    # 실제 데이터에서 추출해야 함. 여기서는 첫 번째 회사를 예시로 사용
    first_company_key = next(iter(self.company_dict))
    feature_names = sorted(list(self.company_dict[first_company_key].data_dict.keys()))
    self.feature_names = feature_names
    num_features = len(feature_names)

    for company_id, company in self.company_dict.items():
        for date in company.date_range:
            name = str(company_id) + '_' + str(date.date())
            if date > self.split_cutoff_date and self.split_cutoff_date + pd.DateOffset(months=self.config['data_duration'] + self.config['label_duration']) > company.date_range[-1]:
                break

            window_start_date = date
            window_end_date = date + pd.DateOffset(months=self.config['data_duration'] - 1)
            label_date = window_end_date + pd.DateOffset(months=self.config['label_duration'])
            if company.date_range and window_end_date >= company.date_range[-1]:
                break

            label = True if company.label and window_end_date < company.end_date <= label_date else False

            # 2. 특징 데이터를 항상 (data_duration, num_features) 매트릭스 형태로 생성
            # instance_features = np.zeros((self.config['data_duration'], num_features))
            # for i in range(self.config['data_duration']):
            #     current_date_in_window = window_start_date + pd.DateOffset(months=i)
            #     for j, feature_name in enumerate(feature_names):
            #         val = company.data_dict[feature_name].get(current_date_in_window, np.nan)
            #         instance_features[i, j] = val
            instance_features = np.zeros((num_features, self.config['data_duration']))
            for i in range(self.config['data_duration']):
                current_date_in_window = window_start_date + pd.DateOffset(months=i)
                for j, feature_name in enumerate(feature_names):
                    val = company.data_dict[feature_name].get(current_date_in_window, np.nan)
                    instance_features[j, i] = val

            # 3. 통합된 오버랩 및 분할 로직
            # 이 부분이 핵심. 모든 데이터는 matrix 형태로 생성된 후, 여기서 train/test로 분리
            if date <= self.split_cutoff_date:
                # 'overlap'이 False이고 윈도우가 분할 기준을 넘어가는 경우, 훈련 세트에 포함시키지 않음
                if not self.config.get('overlap', True) and window_end_date > self.split_cutoff_date:
                    continue
                temp_X_train.append(instance_features)
                temp_y_train.append(label)
                temp_name_train.append(name)
            elif date > self.split_cutoff_date:
                temp_X_test.append(instance_features)
                temp_y_test.append(label)
                temp_name_test.append(name)

    for feature_name in feature_names:
        for i in range(self.config['data_duration']):
            self.flattened_column_names.append(f"{feature_name}_{i}")

    # matrix 형태로 저장
    self.df_x_train_matrix_dict = {temp_name_train[i]: pd.DataFrame(temp_X_train[i], columns=range(temp_X_train[i].shape[1]), index=feature_names) for i in range(len(temp_name_train))}
    self.df_y_train_dict = {temp_name_train[i]: temp_y_train[i] for i in range(len(temp_name_train))}
    self.df_x_test_matrix_dict = {temp_name_test[i]: pd.DataFrame(temp_X_test[i], columns=range(temp_X_test[i].shape[1]), index=feature_names) for i in range(len(temp_name_test))}
    self.df_y_test_dict = {temp_name_test[i]: temp_y_test[i] for i in range(len(temp_name_test))}
    self.df_train_dict = {key: (self.df_x_train_matrix_dict[key], self.df_y_train_dict[key]) for key in temp_name_train}
    self.df_test_dict = {key: (self.df_x_test_matrix_dict[key], self.df_y_test_dict[key]) for key in temp_name_test}

    # 매트릭스 형태의 데이터를 평탄화하여 2D NumPy 배열로 변환
    flattened_X_train_list = []
    flattened_X_test_list = []

    # 훈련 데이터 평탄화
    for matrix_data in temp_X_train:
        # Matrix (data_duration, num_features)를 1D 배열로 평탄화
        flattened_row = np.nan_to_num(matrix_data.flatten(), nan=0.0)
        flattened_X_train_list.append(flattened_row)

    # 테스트 데이터 평탄화
    for matrix_data in temp_X_test:
        flattened_row = np.nan_to_num(matrix_data.flatten(), nan=0.0)
        flattened_X_test_list.append(flattened_row)

    # flatten 형태로 변환
    self.df_x_train_flatten = pd.DataFrame(flattened_X_train_list, columns=self.flattened_column_names, index=temp_name_train)
    self.df_y_train = pd.DataFrame(temp_y_train, columns=['label'], index=temp_name_train)
    self.df_x_test_flatten = pd.DataFrame(flattened_X_test_list, columns=self.flattened_column_names, index=temp_name_test)
    self.df_y_test = pd.DataFrame(temp_y_test, columns=['label'], index=temp_name_test)
    self.df_train = pd.concat([self.df_x_train_flatten, self.df_y_train], axis=1)
    self.df_test = pd.concat([self.df_x_test_flatten, self.df_y_test], axis=1)

    self.name_train = temp_name_train
    self.name_test = temp_name_test


def apply_undersampling(self):
    x_train = self.df_x_train_flatten.copy()
    y_train = self.df_y_train.copy()

    class_counts_before = y_train['label'].value_counts()

    # 클래스 불균형 확인
    print(f"  총 훈련 데이터: {y_train.shape[0]}개")
    print(f"  undersampling 적용 전 클래스 분포: 거래중={class_counts_before.get(False, 0)}, 경영악화={class_counts_before.get(True, 0)}")

    if self.config['undersampling'] == 'ENN':
        sampler = EditedNearestNeighbours(**self.config['ENN_parameter'], kind_sel='all', n_jobs=-1)
    elif self.config['undersampling'] == 'tomek_link':
        sampler = TomekLinks(sampling_strategy='auto', n_jobs=-1)
    elif self.config['undersampling'] == 'nearmiss':
        sampler = NearMiss(version=1, n_jobs=-1)
    else:
        print(f"  오류: 지원되지 않는 undersampling 방법 '{self.config['undersampling']}'입니다.")
        return

    X_sampled, y_sampled = sampler.fit_resample(x_train, y_train['label'])

    self.df_x_train_flatten_after_sampling = x_train.iloc[sampler.sample_indices_]
    self.df_y_train_after_sampling = y_train.iloc[sampler.sample_indices_]
    self.df_train_after_sampling = pd.concat([self.df_x_train_flatten_after_sampling, self.df_y_train_after_sampling], axis=1)
    class_counts_after = self.df_y_train_after_sampling['label'].value_counts()
    print(f"  undersampling 적용 후 클래스 분포: 거래중={class_counts_after.get(False, 0)}, 경영악화={class_counts_after.get(True, 0)}")
    print(f"  undersampling 적용 완료: 원본 {y_train.shape[0]}개 → 최종 {self.df_y_train_after_sampling.shape[0]}개")
    print(f"  제거된 데이터 수: {y_train.shape[0] - self.df_y_train_after_sampling.shape[0]}개")


def apply_oversampling(self):
    if self.config['undersampling']:
        x_train = self.df_x_train_flatten_after_sampling.copy()
        y_train = self.df_y_train_after_sampling.copy()
    else:
        x_train = self.df_x_train_flatten.copy()
        y_train = self.df_y_train.copy()

    class_counts_before = y_train['label'].value_counts()

    # 클래스 불균형 확인
    print(f"  총 훈련 데이터: {y_train.shape[0]}개 샘플")
    print(
        f"  oversampling 적용 전 클래스 분포: 거래중={class_counts_before.get(False, 0)}, 경영악화={class_counts_before.get(True, 0)}")

    if self.config['oversampling'] == 'BorderlineSMOTE':
        sampler = BorderlineSMOTE(sampling_strategy='auto', k_neighbors=self.config.get('SMOTE_k_neighbors', 5),
                                  kind='borderline-1', random_state=self.config['random_state'])
    elif self.config['oversampling'] == 'SMOTE':  # 일반 SMOTE
        sampler = SMOTE(**self.config['SMOTE_parameter'],
                        random_state=self.config['random_state'])
    else:
        print(f"  오류: 지원되지 않는 oversampling 방법 '{self.config['oversampling']}'입니다.")
        return

    X_sampled, y_sampled = sampler.fit_resample(x_train, y_train)

    new_count = len(X_sampled) - len(x_train)
    if new_count > 0:
        synthetic_names = [f"synthetic_{i + 1}" for i in range(new_count)]
        new_index = list(x_train.index) + synthetic_names
        X_sampled.index = new_index
        y_sampled.index = new_index
        self.df_x_train_flatten_after_sampling = X_sampled
        self.df_y_train_after_sampling = y_sampled
    else:
        self.df_x_train_flatten_after_sampling = x_train
        self.df_y_train_after_sampling = y_train
    self.df_train_after_sampling = pd.concat(
        [self.df_x_train_flatten_after_sampling, self.df_y_train_after_sampling], axis=1)

    class_counts_after = self.df_y_train_after_sampling['label'].value_counts()
    print(f"  oversampling 적용 후 클래스 분포: 거래중={class_counts_after.get(False, 0)}, 경영악화={class_counts_after.get(True, 0)}")
    print(f"  oversampling 적용 완료: 원본 {len(x_train)}개 → 최종 {len(X_sampled)}개")
    print(f"  생성 된 데이터 수: {new_count}개")


def make_matrix_data(self):
    df_x_train_after_sampling = self.df_x_train_flatten_after_sampling
    matrix_dict = dict()
    time_duration = self.config['data_duration']
    for sample_name, row_data in df_x_train_after_sampling.iterrows():
        matrix_rows = []
        for feature_name in self.feature_names:
            feature_over_time = []
            for i in range(time_duration):
                col_name = f"{feature_name}_{i}"
                feature_over_time.append(row_data[col_name])
            matrix_rows.append(feature_over_time)

        matrix = pd.DataFrame(matrix_rows, index=self.feature_names)
        matrix_dict[sample_name] = matrix

    self.df_x_train_matrix_dict_after_sampling = matrix_dict
    self.df_y_train_dict_after_sampling = {index: row['label'] for index, row in self.df_y_train_after_sampling.iterrows()}
    self.df_train_dict_after_sampling = {key: (self.df_x_train_matrix_dict_after_sampling[key], self.df_y_train_dict_after_sampling[key]) for key in matrix_dict}
