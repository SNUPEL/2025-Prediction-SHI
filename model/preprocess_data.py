import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import NearMiss
from dtaidistance import dtw


def split_data(self):
    """
    각 회사 별 데이터를 기반으로 train valid test 분리, sampling을 진행
    """
    temp_X_train, temp_y_train, temp_name_train = [], [], []
    temp_X_test, temp_y_test, temp_name_test = [], [], []

    # 모든 회사에 공통으로 적용될 feature_names를 미리 정의
    first_company_key = next(iter(self.company_dict))
    feature_names = sorted(list(self.company_dict[first_company_key].data_dict.keys()))
    self.feature_names = feature_names
    num_features = len(feature_names)

    for company_id, company in self.company_dict.items():
        # 회사별 date range에 속하는 모든 회사에 대해 반복
        for date in company.date_range[:-self.config['data_duration']]:
            name = str(company_id) + '_' + str(date.date())
            window_start_date = date
            window_end_date = date + pd.DateOffset(months=self.config['data_duration'] - 1)
            label_date = window_end_date + pd.DateOffset(months=self.config['label_duration'])

            # window 뒤의 기간을 기반으로 label 설정
            label = True if company.label and window_end_date < company.end_date <= label_date else False

            # 경영악화이며, 경/중이 경이면 제외
            if self.config['severity_label_type'] and label and company.severity_level == 'B':
                continue

            # 2차원 형태로 데이터 정의
            instance_features = np.zeros((num_features, self.config['data_duration']))
            for i in range(self.config['data_duration']):
                current_date_in_window = window_start_date + pd.DateOffset(months=i)
                for j, feature_name in enumerate(feature_names):
                    val = company.data_dict[feature_name].get(current_date_in_window, np.nan)
                    instance_features[j, i] = val

            # 기준에 따라 train, test로 분류
            if (company.end_date is not None and company.end_date < self.label_date) or (date <= self.split_cutoff_date):
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

    # 일정 비율을 validation으로 분류
    train_indicies = range(len(temp_y_train))
    train_idx, valid_idx = train_test_split(
        train_indicies,
        test_size=self.config['validation_ratio'],
        random_state=self.config['random_state'],
        stratify=temp_y_train
    )

    temp_X_valid = [temp_X_train[i] for i in valid_idx]
    temp_y_valid = [temp_y_train[i] for i in valid_idx]
    temp_name_valid = [temp_name_train[i] for i in valid_idx]
    temp_X_train = [temp_X_train[i] for i in train_idx]
    temp_y_train = [temp_y_train[i] for i in train_idx]
    temp_name_train = [temp_name_train[i] for i in train_idx]

    # matrix 형태로 저장
    self.df_x_train_matrix_dict = {temp_name_train[i]: pd.DataFrame(temp_X_train[i], columns=range(temp_X_train[i].shape[1]), index=feature_names) for i in range(len(temp_name_train))}
    self.df_y_train_dict = {temp_name_train[i]: temp_y_train[i] for i in range(len(temp_name_train))}
    self.df_x_valid_matrix_dict = {temp_name_valid[i]: pd.DataFrame(temp_X_valid[i], columns=range(temp_X_valid[i].shape[1]), index=feature_names) for i in range(len(temp_name_valid))}
    self.df_y_valid_dict = {temp_name_valid[i]: temp_y_valid[i] for i in range(len(temp_name_valid))}
    self.df_x_test_matrix_dict = {temp_name_test[i]: pd.DataFrame(temp_X_test[i], columns=range(temp_X_test[i].shape[1]), index=feature_names) for i in range(len(temp_name_test))}
    self.df_y_test_dict = {temp_name_test[i]: temp_y_test[i] for i in range(len(temp_name_test))}
    self.df_train_dict = {key: (self.df_x_train_matrix_dict[key], self.df_y_train_dict[key]) for key in temp_name_train}
    self.df_valid_dict = {key: (self.df_x_valid_matrix_dict[key], self.df_y_valid_dict[key]) for key in temp_name_valid}
    self.df_test_dict = {key: (self.df_x_test_matrix_dict[key], self.df_y_test_dict[key]) for key in temp_name_test}

    # 매트릭스 형태의 데이터를 평탄화하여 2D NumPy 배열로 변환
    flattened_X_train_list = []
    flattened_X_valid_list = []
    flattened_X_test_list = []

    # 훈련 데이터 평탄화
    for matrix_data in temp_X_train:
        # Matrix (data_duration, num_features)를 1D 배열로 평탄화
        flattened_row = np.nan_to_num(matrix_data.flatten(), nan=0.0)
        flattened_X_train_list.append(flattened_row)

    # 검증 데이터 평탄화
    for matrix_data in temp_X_valid:
        flattened_row = np.nan_to_num(matrix_data.flatten(), nan=0.0)
        flattened_X_valid_list.append(flattened_row)

    # 테스트 데이터 평탄화
    for matrix_data in temp_X_test:
        flattened_row = np.nan_to_num(matrix_data.flatten(), nan=0.0)
        flattened_X_test_list.append(flattened_row)

    # flatten 형태로 변환
    self.df_x_train_flatten = pd.DataFrame(flattened_X_train_list, columns=self.flattened_column_names, index=temp_name_train)
    self.df_y_train = pd.DataFrame(temp_y_train, columns=['label'], index=temp_name_train)
    self.df_x_valid_flatten = pd.DataFrame(flattened_X_valid_list, columns=self.flattened_column_names, index=temp_name_valid)
    self.df_y_valid = pd.DataFrame(temp_y_valid, columns=['label'], index=temp_name_valid)
    self.df_x_test_flatten = pd.DataFrame(flattened_X_test_list, columns=self.flattened_column_names, index=temp_name_test)
    self.df_y_test = pd.DataFrame(temp_y_test, columns=['label'], index=temp_name_test)
    self.df_train = pd.concat([self.df_x_train_flatten, self.df_y_train], axis=1)
    self.df_valid = pd.concat([self.df_x_valid_flatten, self.df_y_valid], axis=1)
    self.df_test = pd.concat([self.df_x_test_flatten, self.df_y_test], axis=1)

    self.name_train = temp_name_train
    self.name_valid = temp_name_valid
    self.name_test = temp_name_test


def apply_undersampling(self):
    if self.config['sampling_order'][0] == 'undersampling':
        x_train = self.df_x_train_flatten.copy()
        y_train = self.df_y_train.copy()
    else:
        x_train = self.df_x_train_flatten_after_sampling.copy()
        y_train = self.df_y_train_after_sampling.copy()

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

    # sampler를 통해 유효한 샘플만 저장
    self.df_x_train_flatten_after_sampling = x_train.iloc[sampler.sample_indices_]
    self.df_y_train_after_sampling = y_train.iloc[sampler.sample_indices_]
    self.df_train_after_sampling = pd.concat([self.df_x_train_flatten_after_sampling, self.df_y_train_after_sampling], axis=1)
    class_counts_after = self.df_y_train_after_sampling['label'].value_counts()
    print(f"  undersampling 적용 후 클래스 분포: 거래중={class_counts_after.get(False, 0)}, 경영악화={class_counts_after.get(True, 0)}")
    print(f"  undersampling 적용 완료: 원본 {y_train.shape[0]}개 → 최종 {self.df_y_train_after_sampling.shape[0]}개")
    print(f"  제거된 데이터 수: {y_train.shape[0] - self.df_y_train_after_sampling.shape[0]}개")


def apply_oversampling(self):
    # --- 1. 표준 SMOTE / BorderlineSMOTE 로직 ---
    if self.config['oversampling'] in ['BorderlineSMOTE', 'SMOTE']:
        print(f"\n==== '{self.config['oversampling']}' 오버샘플링 시작 (평탄화 데이터 기반) ====")
        if self.config['sampling_order'][0] == 'undersampling':
            x_train = self.df_x_train_flatten_after_sampling.copy()
            y_train = self.df_y_train_after_sampling.copy()
        else:
            x_train = self.df_x_train_flatten.copy()
            y_train = self.df_y_train.copy()

        class_counts_before = y_train['label'].value_counts()
        print(f"  오버샘플링 적용 전: 거래중={class_counts_before.get(False, 0)}, 경영악화={class_counts_before.get(True, 0)}")

        if self.config['oversampling'] == 'SMOTE':
            sampler = SMOTE(**self.config.get('SMOTE_parameter', {}), random_state=self.config['random_state'])
        else:  # BorderlineSMOTE
            sampler = BorderlineSMOTE(**self.config.get('BorderlineSMOTE_parameter', {}),
                                      random_state=self.config['random_state'])

        X_sampled, y_sampled = sampler.fit_resample(x_train, y_train['label'])

        # DataFrame으로 변환
        X_sampled = pd.DataFrame(X_sampled, columns=x_train.columns)
        y_sampled = pd.DataFrame(y_sampled, columns=['label'])

        # 인덱스 재설정
        new_count = len(X_sampled) - len(x_train)
        if new_count > 0:
            synthetic_names = [f"synthetic_{i + 1}" for i in range(new_count)]
            new_index = list(x_train.index) + synthetic_names
            X_sampled.index = new_index
            y_sampled.index = new_index

        # 샘플링 후의 샘플로 변경
        self.df_x_train_flatten_after_sampling = X_sampled
        self.df_y_train_after_sampling = y_sampled
        self.df_train_after_sampling = pd.concat(
            [self.df_x_train_flatten_after_sampling, self.df_y_train_after_sampling], axis=1)

        make_matrix_data(self)  # 매트릭스 데이터 재생성

    # --- 2. TSSMOTE / Borderline-TSSMOTE 로직 ---
    elif 'TSSMOTE' in self.config['oversampling']:
        is_borderline = 'Borderline' in self.config['oversampling']
        method_name = "Borderline-TSSMOTE" if is_borderline else "TSSMOTE"
        print(f"\n==== '{method_name}' 오버샘플링 시작 (시계열 데이터 기반) ====")

        if self.config['sampling_order'][0] == 'undersampling':
            x_dict, y_dict = self.df_x_train_matrix_dict_after_sampling, self.df_y_train_dict_after_sampling
        else:
            x_dict, y_dict = self.df_x_train_matrix_dict.copy(), self.df_y_train_dict.copy()

        class_counts_before = pd.Series(y_dict).value_counts()
        print(f"  오버샘플링 적용 전: 거래중={class_counts_before.get(False, 0)}, 경영악화={class_counts_before.get(True, 0)}")

        tssmote_params = self.config.get('TSSMOTE_parameter', {})
        k_neighbors = tssmote_params.get('k_neighbors', 5)

        all_keys = list(x_dict.keys())
        minority_keys = {key for key, label in y_dict.items() if label}
        majority_keys = {key for key, label in y_dict.items() if not label}

        if not minority_keys or len(all_keys) <= k_neighbors:
            print("경고: 샘플 수가 부족하여 TSSMOTE 계열 오버샘플링을 적용할 수 없습니다.")
            return

        base_minority_keys = list(minority_keys)

        # Borderline-TSSMOTE일 경우에만 경계선 샘플 식별
        if is_borderline:
            all_x_list = [x_dict[key].values for key in all_keys]
            print(f"    {method_name}: 전체 샘플 간 DTW 거리 계산 중 (시간 소요)...")
            full_dist_matrix = dtw.distance_matrix(all_x_list, use_c=True, parallel=True)

            borderline_keys = []
            print(f"    {method_name}: {k_neighbors}개의 이웃을 확인하여 경계선 샘플 식별 중...")
            for i, key in enumerate(all_keys):
                if key in minority_keys:
                    neighbor_indices = np.argsort(full_dist_matrix[i])[1:k_neighbors + 1]
                    majority_neighbor_count = sum(1 for idx in neighbor_indices if all_keys[idx] in majority_keys)
                    if majority_neighbor_count > 0:
                        borderline_keys.append(key)
            print(f"    {method_name}: 총 {len(minority_keys)}개 소수 샘플 중 {len(borderline_keys)}개의 경계선 샘플 발견.")

            if len(borderline_keys) > k_neighbors:
                base_minority_keys = borderline_keys
            else:
                print("    경고: 경계선 샘플이 부족하여 표준 TSSMOTE 방식으로 대체합니다.")

        # 샘플링 수행
        base_minority_x_list = [x_dict[key].values for key in base_minority_keys]
        minority_dist_matrix = dtw.distance_matrix(base_minority_x_list, use_c=True, parallel=True)

        num_synthetic = int(len(majority_keys) * tssmote_params.get('sampling_strategy', 1.0)) - len(minority_keys)
        new_samples = []

        if num_synthetic > 0:
            print(f"    {method_name}: {num_synthetic}개의 새로운 시계열 데이터 생성 중...")
            for _ in range(num_synthetic):
                sample_idx = np.random.randint(0, len(base_minority_keys))
                distances = minority_dist_matrix[sample_idx]
                neighbor_indices = np.argsort(distances)[1:k_neighbors + 1]
                chosen_neighbor_idx = np.random.choice(neighbor_indices)
                base_sample = base_minority_x_list[sample_idx]
                neighbor_sample = base_minority_x_list[chosen_neighbor_idx]
                synthetic_sample = base_sample + np.random.random() * (neighbor_sample - base_sample)
                new_samples.append(synthetic_sample)

        x_resampled, y_resampled = x_dict.copy(), y_dict.copy()
        feat_names = list(x_dict[list(minority_keys)[0]].index)
        for i, data in enumerate(new_samples):
            key = f"synthetic_{i}"
            x_resampled[key] = pd.DataFrame(data, index=feat_names)
            y_resampled[key] = True

        self.df_x_train_matrix_dict_after_sampling, self.df_y_train_dict_after_sampling = x_resampled, y_resampled
        self.df_train_dict_after_sampling = {k: (x_resampled[k], y_resampled[k]) for k in x_resampled}

        flat_data, labels, new_idx = [], [], []
        for name, matrix in x_resampled.items():
            flat_data.append(matrix.values.flatten())
            labels.append(y_resampled[name])
            new_idx.append(name)
        self.df_x_train_flatten_after_sampling = pd.DataFrame(flat_data, index=new_idx,
                                                              columns=self.flattened_column_names)
        self.df_y_train_after_sampling = pd.DataFrame(labels, index=new_idx, columns=['label'])
        self.df_train_after_sampling = pd.concat(
            [self.df_x_train_flatten_after_sampling, self.df_y_train_after_sampling], axis=1)

    # --- 최종 결과 출력 ---
    class_counts_after = self.df_y_train_after_sampling['label'].value_counts()
    print(f"  오버샘플링 적용 후: 거래중={class_counts_after.get(False, 0)}, 경영악화={class_counts_after.get(True, 0)}")


def make_matrix_data(self):
    """
    flatten data를 matrix로 변형하는 함수
    """
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
