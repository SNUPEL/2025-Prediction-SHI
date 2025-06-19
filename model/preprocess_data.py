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

    # 모든 회사에 공통으로 적용될 feature_names를 미리 정의 (한 번만)
    # 실제 데이터에서 추출해야 함. 여기서는 첫 번째 회사를 예시로 사용
    first_company_key = next(iter(self.company_dict))
    feature_names = sorted(list(self.company_dict[first_company_key].data_dict.keys()))
    num_features = len(feature_names)

    for company_id, company in self.company_dict.items():
        for date in company.date_range:
            name = str(company_id) + '_' + str(date.date())

            label_date = date + pd.DateOffset(months=self.config['data_duration'] - 1 + self.config['label_duration'])

            if company.date_range and label_date > company.date_range[-1]:
                break

            # 레이블 결정 로직은 동일
            label = True if company.label and company.end_date - pd.DateOffset(
                months=self.config['label_duration'] - 1) <= label_date <= company.end_date else False

            window_start_date = date
            window_end_date = date + pd.DateOffset(months=self.config['data_duration'] - 1)

            # 2. 특징 데이터를 항상 (data_duration, num_features) 매트릭스 형태로 생성
            instance_features = np.zeros((self.config['data_duration'], num_features))
            for i in range(self.config['data_duration']):
                current_date_in_window = window_start_date + pd.DateOffset(months=i)
                for j, feature_name in enumerate(feature_names):
                    val = company.data_dict[feature_name].get(current_date_in_window, np.nan)
                    instance_features[i, j] = val

            # 3. 통합된 오버랩 및 분할 로직 (temp_X_raw에 추가하기 전에 적용)
            # 이 부분이 핵심. 모든 데이터는 matrix 형태로 생성된 후, 여기서 train/test로 분리됨.
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

    if self.config['data_shape'] == 'flatten':
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

        # NumPy 배열로 최종 할당
        self.X_train = np.array(flattened_X_train_list)
        self.y_train = np.array(temp_y_train)
        self.name_train = temp_name_train  # 이름 정보는 그대로 리스트로 유지

        self.X_test = np.array(flattened_X_test_list)
        self.y_test = np.array(temp_y_test)
        self.name_test = temp_name_test  # 이름 정보는 그대로 리스트로 유지

        print("\n전통적인 머신러닝용 Flattened NumPy 배열이 생성되었습니다.")
        print(f"훈련 데이터 X 형태: {self.X_train.shape}")
        print(f"훈련 데이터 y 형태: {self.y_train.shape}")
        print(f"테스트 데이터 X 형태: {self.X_test.shape}")
        print(f"테스트 데이터 y 형태: {self.y_test.shape}")

    elif self.config['data_shape'] == 'matrix':
        # Matrix 형태 데이터를 NumPy 배열로 변환하고 NaN 처리
        self.X_train = np.nan_to_num(np.array(temp_X_train), nan=0.0)
        self.y_train = np.array(temp_y_train)
        self.name_train = temp_name_train

        self.X_test = np.nan_to_num(np.array(temp_X_test), nan=0.0)
        self.y_test = np.array(temp_y_test)
        self.name_test = temp_name_test

        print("\n딥러닝용 Matrix(Tensor) 생성이 완료되었습니다.")
        print(f"훈련 데이터 X 형태: {self.X_train.shape}")
        print(f"훈련 데이터 y 형태: {self.y_train.shape}")
        print(f"테스트 데이터 X 형태: {self.X_test.shape}")
        print(f"테스트 데이터 y 형태: {self.y_test.shape}")

    print("=== 데이터 분할 작업 완료 ===")


def apply_undersampling(self):  # 이 함수는 DataLoader 클래스(Data 클래스)의 메서드
    print("\n=== undersampling 시작 ===")

    # self.X_train은 NumPy 배열, self.y_train은 NumPy 배열, self.name_train은 리스트
    X_input = self.X_train  # 샘플링 함수에 전달할 X
    y_input = self.y_train  # 샘플링 함수에 전달할 y
    name_input = self.name_train  # 샘플링 후 company_id를 업데이트할 리스트

    # Imblearn 샘플러는 2D 입력 (n_samples, n_features)를 기대합니다.
    # self.X_train.ndim을 직접 사용하여 차원 확인
    if self.X_train.ndim > 2:
        num_samples_original = self.X_train.shape[0]
        # 3D -> 2D 평탄화 (n_samples, data_duration * num_features)
        X_input_reshaped_for_sampler = self.X_train.reshape(num_samples_original, -1)
        print(f"  X_train이 {self.X_train.ndim}D이므로 샘플링을 위해 2D로 평탄화했습니다: {X_input_reshaped_for_sampler.shape}")
    else:
        # 2D (flatten)인 경우 그대로 사용
        X_input_reshaped_for_sampler = self.X_train
        print(f"  X_train이 2D이므로 샘플링을 위해 그대로 사용합니다: {X_input_reshaped_for_sampler.shape}")

    # 클래스 불균형 확인
    class_counts_before = np.bincount(y_input.astype(int))
    print(
        f"  undersampling 적용 전 클래스 분포: 거래중={class_counts_before[0]}, 경영악화={class_counts_before[1] if len(class_counts_before) > 1 else 0}")
    orig_total_samples = len(y_input)
    print(f"  총 훈련 데이터: {orig_total_samples}개 샘플")

    if self.config['undersampling'] == 'ENN':
        sampler = EditedNearestNeighbours(sampling_strategy='auto', n_neighbors=self.config['ENN_n_neighbors'],
                                          kind_sel='all', n_jobs=-1)
    elif self.config['undersampling'] == 'tomek_link':
        sampler = TomekLinks(sampling_strategy='auto', n_jobs=-1)
    elif self.config['undersampling'] == 'nearmiss':
        sampler = NearMiss(version=1, n_jobs=-1)
    else:
        print(f"  오류: 지원되지 않는 undersampling 방법 '{self.config['undersampling']}'입니다.")
        return  # 함수 종료

    X_resampled_2d, y_resampled = sampler.fit_resample(X_input_reshaped_for_sampler, y_input)

    # 결과 클래스 분포 확인
    resampled_class_counts = np.bincount(y_resampled.astype(int))
    class0_count_after = resampled_class_counts[0] if len(resampled_class_counts) > 0 else 0
    class1_count_after = resampled_class_counts[1] if len(resampled_class_counts) > 1 else 0
    print(f"  undersampling 적용 후 클래스 분포: 거래중={class0_count_after}, 경영악화={class1_count_after}")

    # self.X_train 업데이트 (원래의 차원으로 복원)
    # self.X_train.ndim을 직접 사용하여 차원 확인
    if self.X_train.ndim > 2:
        # 2D -> 3D 복원 (n_samples, data_duration, num_features)
        original_matrix_shape = self.X_train.shape[1:]  # 원본 shape의 나머지 차원 (tuple)
        self.X_train = X_resampled_2d.reshape(-1, *original_matrix_shape)
    else:
        self.X_train = X_resampled_2d  # 2D인 경우 그대로 할당

    self.y_train = y_resampled  # y_train 업데이트

    # name_train 업데이트: 유지된 샘플에 해당하는 name만 필터링
    if hasattr(sampler, 'sample_indices_') and sampler.sample_indices_ is not None:
        self.name_train = [name_input[idx] for idx in sampler.sample_indices_]
        print(f"  유지된 샘플에 대해 'company_id' (name_train)를 필터링했습니다. (총 {len(self.name_train)}개)")
    else:
        # 샘플러가 sample_indices_를 제공하지 않거나 (예: NearMiss)
        # 또는 일부 샘플러에서 sample_indices_가 None인 경우
        if len(y_resampled) != len(name_input):
            print(f"  경고: 샘플러가 인덱스를 명시적으로 제공하지 않아 name_train 필터링이 정확하지 않을 수 있습니다. y_resampled 길이에 맞춰 조정합니다.")
            self.name_train = name_input[:len(y_resampled)]
        else:
            self.name_train = name_input

    new_total_samples = len(y_resampled)
    removed_samples_count = orig_total_samples - new_total_samples

    print(f"undersampling 적용 완료: 원본 {orig_total_samples}개 → 최종 {new_total_samples}개 샘플")
    print(f"제거된 샘플 수: {removed_samples_count}개 샘플")
    print("=== undersampling 완료 ===\n")


def apply_SMOTE(self):  # 이 함수는 DataLoader 클래스(Data 클래스)의 메서드
    print("\n=== SMOTE 오버샘플링 시작 ===")

    X_input = self.X_train
    y_input = self.y_train
    name_input = self.name_train  # company_id 리스트

    # Imblearn 샘플러는 2D 입력 (n_samples, n_features)를 기대합니다.
    if self.X_train.ndim > 2:  # self.X_train.ndim을 직접 사용
        num_samples_original = self.X_train.shape[0]
        X_input_reshaped_for_sampler = self.X_train.reshape(num_samples_original, -1)
        print(f"  X_train이 {self.X_train.ndim}D이므로 샘플링을 위해 2D로 평탄화했습니다: {X_input_reshaped_for_sampler.shape}")
    else:
        X_input_reshaped_for_sampler = self.X_train
        print(f"  X_train이 2D이므로 샘플링을 위해 그대로 사용합니다: {X_input_reshaped_for_sampler.shape}")

    # 클래스 불균형 확인
    class_counts = np.bincount(y_input.astype(int))
    print(f"  SMOTE 적용 전 클래스 분포: 거래중={class_counts[0]}, 경영악화={class_counts[1] if len(class_counts) > 1 else 0}")
    orig_total_samples = len(y_input)
    print(f"  총 훈련 데이터: {orig_total_samples}개 샘플")

    # SMOTE 적용
    if self.config['oversampling'] == 'BorderlineSMOTE':
        smote_sampler = BorderlineSMOTE(sampling_strategy='auto', k_neighbors=self.config.get('SMOTE_k_neighbors', 5),
                                        kind='borderline-1', random_state=self.config['random_state'], n_jobs=-1)
    elif self.config['oversampling'] == 'SMOTE':  # 일반 SMOTE
        smote_sampler = SMOTE(sampling_strategy=self.config['SMOTE_sampling_strategy'], k_neighbors=self.config['SMOTE_k_neighbors'],
                              random_state=self.config['random_state'], n_jobs=-1)
    else:
        print(f"  오류: 지원되지 않는 oversampling 방법 '{self.config['oversampling']}'입니다.")
        return  # 함수 종료

    X_resampled_2d, y_resampled = smote_sampler.fit_resample(X_input_reshaped_for_sampler, y_input)

    # 결과 클래스 분포 확인
    resampled_class_counts = np.bincount(y_resampled.astype(int))
    class0_count_after = resampled_class_counts[0] if len(resampled_class_counts) > 0 else 0
    class1_count_after = resampled_class_counts[1] if len(resampled_class_counts) > 1 else 0
    print(f"  SMOTE 적용 후 클래스 분포: 거래중={class0_count_after}, 경영악화={class1_count_after}")

    # self.X_train 업데이트 (원래의 차원으로 복원)
    if self.X_train.ndim > 2:  # self.X_train.ndim을 직접 사용
        original_matrix_shape = self.X_train.shape[1:]
        self.X_train = X_resampled_2d.reshape(-1, *original_matrix_shape)
    else:
        self.X_train = X_resampled_2d  # 2D인 경우 그대로 할당

    self.y_train = y_resampled  # y_train 업데이트

    # name_train 업데이트: 합성된 샘플에 대한 company_id 생성
    new_count = len(y_resampled) - orig_total_samples  # 추가된 샘플 수

    # 원본 name_input에 추가될 합성 ID를 생성
    synthetic_names = []
    minority_names = [name_input[i] for i, label in enumerate(y_input) if label == 1]

    if len(minority_names) > 0:
        synthetic_names_array = np.random.choice(np.array(minority_names), size=new_count, replace=True)
        synthetic_names = synthetic_names_array.tolist()
        print(f"  경영악화 기업 ID 개수 (원본 소수 클래스): {len(minority_names)}개")
    else:
        print("  경고: 소수 클래스(경영악화)가 없어 SMOTE가 작동하지 않습니다. 합성 ID를 생성하지 않습니다.")
        synthetic_names = []  # 합성 ID를 만들지 않음

    # id 결합
    self.name_train = name_input + synthetic_names

    print(f"SMOTE 적용 완료: 원본 {orig_total_samples}개 → 최종 {len(y_resampled)}개 샘플")
    print(f"합성 데이터 생성: {new_count}개 샘플")
    print("=== SMOTE 오버샘플링 완료 ===\n")


def random_split(self):
    self.df_x = self.df_model.drop(columns=['label'])
    self.df_y = self.df_model['label']

    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
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
