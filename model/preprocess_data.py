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
        self.name_train = temp_name_train

        self.X_test = np.array(flattened_X_test_list)
        self.y_test = np.array(temp_y_test)
        self.name_test = temp_name_test

        print("머신러닝용 Flattened NumPy 배열 생성 완료.")
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

        print("\n딥러닝용 Matrix(Tensor) 생성 완료.")
        print(f"훈련 데이터 X 형태: {self.X_train.shape}")
        print(f"훈련 데이터 y 형태: {self.y_train.shape}")
        print(f"테스트 데이터 X 형태: {self.X_test.shape}")
        print(f"테스트 데이터 y 형태: {self.y_test.shape}")

    elif self.config['data_shape'] == 'multichannel':
        # 각 시트별로 독립적인 채널 생성
        multichannel_X_train = []
        multichannel_X_test = []
        
        # 훈련 데이터 변환
        for matrix_data in temp_X_train:
            # (data_duration, num_features) → (num_channels, data_duration, 1)
            multichannel_sample = np.zeros((num_features, self.config['data_duration'], 1))
            for channel_idx in range(num_features):  # 각 시트 = 각 채널
                for time_idx in range(self.config['data_duration']):
                    multichannel_sample[channel_idx, time_idx, 0] = matrix_data[time_idx, channel_idx]
            multichannel_X_train.append(multichannel_sample)
        
        # 테스트 데이터 변환
        for matrix_data in temp_X_test:
            multichannel_sample = np.zeros((num_features, self.config['data_duration'], 1))
            for channel_idx in range(num_features):  # 각 시트 = 각 채널
                for time_idx in range(self.config['data_duration']):
                    multichannel_sample[channel_idx, time_idx, 0] = matrix_data[time_idx, channel_idx]
            multichannel_X_test.append(multichannel_sample)

        # NumPy 배열로 변환
        self.X_train = np.nan_to_num(np.array(multichannel_X_train), nan=0.0)
        self.y_train = np.array(temp_y_train)
        self.name_train = temp_name_train

        self.X_test = np.nan_to_num(np.array(multichannel_X_test), nan=0.0)
        self.y_test = np.array(temp_y_test)
        self.name_test = temp_name_test

        print("MultiChannel용 텐서 생성 완료.")
        print(f"훈련 데이터 X 형태: {self.X_train.shape} (samples, channels, time_points, features)")
        print(f"테스트 데이터 X 형태: {self.X_test.shape}")
        print(f"채널 수: {num_features}개 (시트: {feature_names})")


def apply_undersampling(self):
    y_input = self.y_train
    name_input = self.name_train

    # 3D -> 2D 평탄화 (n_samples, data_duration * num_features)
    if self.X_train.ndim > 2:
        num_samples_original = self.X_train.shape[0]
        X_input_reshaped_for_sampler = self.X_train.reshape(num_samples_original, -1)
        print(f"  샘플링을 위한 X_train 변환 결과: {X_input_reshaped_for_sampler.shape}")
    else:
        # 2D (flatten)인 경우 그대로 사용
        X_input_reshaped_for_sampler = self.X_train

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
        return

    X_resampled_2d, y_resampled = sampler.fit_resample(X_input_reshaped_for_sampler, y_input)

    # 결과 클래스 분포 확인
    resampled_class_counts = np.bincount(y_resampled.astype(int))
    class0_count_after = resampled_class_counts[0] if len(resampled_class_counts) > 0 else 0
    class1_count_after = resampled_class_counts[1] if len(resampled_class_counts) > 1 else 0
    print(f"  undersampling 적용 후 클래스 분포: 거래중={class0_count_after}, 경영악화={class1_count_after}")

    # 2D -> 3D 복원 (n_samples, data_duration, num_features)
    if self.X_train.ndim > 2:
        original_matrix_shape = self.X_train.shape[1:]
        self.X_train = X_resampled_2d.reshape(-1, *original_matrix_shape)
    else:
        self.X_train = X_resampled_2d

    self.y_train = y_resampled  # y_train 업데이트

    if hasattr(sampler, 'sample_indices_') and sampler.sample_indices_ is not None:
        self.name_train = [name_input[idx] for idx in sampler.sample_indices_]
    else:
        if len(y_resampled) != len(name_input):
            self.name_train = name_input[:len(y_resampled)]
        else:
            self.name_train = name_input

    new_total_samples = len(y_resampled)
    removed_samples_count = orig_total_samples - new_total_samples

    print(f"undersampling 적용 완료: 원본 {orig_total_samples}개 → 최종 {new_total_samples}개 샘플")
    print(f"제거된 샘플 수: {removed_samples_count}개 샘플")

def apply_SMOTE(self):
    y_input = self.y_train
    name_input = self.name_train

    if self.X_train.ndim > 2:
        num_samples_original = self.X_train.shape[0]
        X_input_reshaped_for_sampler = self.X_train.reshape(num_samples_original, -1)
        print(f"  샘플링을 위한 X_train 변환 결과: {X_input_reshaped_for_sampler.shape}")
    else:
        X_input_reshaped_for_sampler = self.X_train

    # 클래스 불균형 확인
    class_counts = np.bincount(y_input.astype(int))
    print(f"  SMOTE 적용 전 클래스 분포: 거래중={class_counts[0]}, 경영악화={class_counts[1] if len(class_counts) > 1 else 0}")
    orig_total_samples = len(y_input)
    print(f"  총 훈련 데이터: {orig_total_samples}개 샘플")

    if self.config['oversampling'] == 'BorderlineSMOTE':
        smote_sampler = BorderlineSMOTE(sampling_strategy='auto', k_neighbors=self.config.get('SMOTE_k_neighbors', 5),
                                        kind='borderline-1', random_state=self.config['random_state'])
    elif self.config['oversampling'] == 'SMOTE':  # 일반 SMOTE
        smote_sampler = SMOTE(sampling_strategy=self.config['SMOTE_sampling_strategy'], k_neighbors=self.config['SMOTE_k_neighbors'],
                              random_state=self.config['random_state'])
    elif self.config['oversampling'] == 'TSSMOTE':  # TSSMOTE 추가
        return apply_TSSMOTE(self)
    else:
        print(f"  오류: 지원되지 않는 oversampling 방법 '{self.config['oversampling']}'입니다.")
        return

    X_resampled_2d, y_resampled = smote_sampler.fit_resample(X_input_reshaped_for_sampler, y_input)

    # 결과 클래스 분포 확인
    resampled_class_counts = np.bincount(y_resampled.astype(int))
    class0_count_after = resampled_class_counts[0] if len(resampled_class_counts) > 0 else 0
    class1_count_after = resampled_class_counts[1] if len(resampled_class_counts) > 1 else 0
    print(f"  SMOTE 적용 후 클래스 분포: 거래중={class0_count_after}, 경영악화={class1_count_after}")

    if self.X_train.ndim > 2:
        original_matrix_shape = self.X_train.shape[1:]
        self.X_train = X_resampled_2d.reshape(-1, *original_matrix_shape)
    else:
        self.X_train = X_resampled_2d

    self.y_train = y_resampled

    # name_train 업데이트: 합성된 샘플에 대한 company_id 생성
    new_count = len(y_resampled) - orig_total_samples

    # 원본 name_input에 추가될 합성 ID를 생성
    synthetic_names = [f"synthetic_{i + 1}" for i in range(new_count)]

    # id 결합
    self.name_train = name_input + synthetic_names

    print(f"SMOTE 적용 완료: 원본 {orig_total_samples}개 → 최종 {len(y_resampled)}개 샘플")
    print(f"합성 데이터 생성: {new_count}개 샘플")

def apply_TSSMOTE(self):
    """
    TimeSeries SMOTE 적용 (MultiChannel 데이터용)
    """
    print("  TSSMOTE 적용 시작...")
    
    y_input = self.y_train
    name_input = self.name_train
    X_input = self.X_train
    
    # 클래스 분포 확인
    class_counts = np.bincount(y_input.astype(int))
    print(f"  TSSMOTE 적용 전 클래스 분포: 거래중={class_counts[0]}, 경영악화={class_counts[1] if len(class_counts) > 1 else 0}")
    orig_total_samples = len(y_input)
    print(f"  총 훈련 데이터: {orig_total_samples}개 샘플")
    
    # 소수 클래스가 충분히 있는지 확인
    if len(class_counts) <= 1 or class_counts[1] < 2:
        print("  소수 클래스 샘플이 부족하여 TSSMOTE를 적용할 수 없습니다.")
        return
    
    # 소수/다수 클래스 분리
    minority_indices = np.where(y_input == 1)[0]
    majority_indices = np.where(y_input == 0)[0]
    
    # 필요한 샘플 수 계산
    n_samples_needed = max(0, len(majority_indices) - len(minority_indices))
    
    if n_samples_needed == 0:
        print("  클래스가 이미 균형 상태입니다. TSSMOTE를 적용하지 않습니다.")
        return
    
    # 소수 클래스 데이터 추출
    X_minority = X_input[minority_indices]
    n_minority = len(minority_indices)
    k = min(self.config.get('SMOTE_k_neighbors', 5), n_minority - 1)
    
    # 간단한 유클리드 거리 기반 시계열 유사도 계산
    print(f"  시계열 유사도 계산 중... (k={k})")
    
    synthetic_X = []
    synthetic_names = []
    
    # 각 소수 클래스 샘플에 대해 합성 샘플 생성
    samples_per_minority = int(np.ceil(n_samples_needed / n_minority))
    
    for i in range(n_minority):
        # 현재 샘플과 다른 모든 소수 클래스 샘플들 간의 거리 계산
        distances = []
        for j in range(n_minority):
            if i != j:
                # 평탄화하여 유클리드 거리 계산
                dist = np.linalg.norm(X_minority[i].flatten() - X_minority[j].flatten())
                distances.append((dist, j))
        
        # k개 최근접 이웃 선택
        distances.sort(key=lambda x: x[0])
        neighbors = [idx for _, idx in distances[:k]]
        
        # 필요한 만큼 합성 샘플 생성
        for _ in range(samples_per_minority):
            if len(synthetic_X) >= n_samples_needed:
                break
                
            # 이웃 중 하나 선택
            nn_idx = np.random.choice(neighbors)
            
            # 보간 비율 (0.2-0.8 사이)
            alpha = 0.2 + 0.6 * np.random.random()
            
            # 새 샘플 생성 (선형 보간)
            new_sample = X_minority[i] * alpha + X_minority[nn_idx] * (1 - alpha)
            synthetic_X.append(new_sample)
            synthetic_names.append(f"synthetic_tssmote_{len(synthetic_X)}")
    
    # 결과 합치기
    if synthetic_X:
        X_resampled = np.vstack([X_input, np.array(synthetic_X)])
        y_resampled = np.concatenate([y_input, np.ones(len(synthetic_X))])
        name_resampled = name_input + synthetic_names
        
        self.X_train = X_resampled
        self.y_train = y_resampled
        self.name_train = name_resampled
        
        # 결과 로깅
        new_class_counts = np.bincount(y_resampled.astype(int))
        print(f"  TSSMOTE 적용 후 클래스 분포: 거래중={new_class_counts[0]}, 경영악화={new_class_counts[1]}")
        print(f"  생성된 합성 샘플: {len(synthetic_X)}개")
    else:
        print("  유효한 합성 샘플을 생성할 수 없습니다.")


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
