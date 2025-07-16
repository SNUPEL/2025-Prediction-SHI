import numpy as np
import pandas as pd
import os
import load_data as dp
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight

class Data:
    def __init__(self, config):
        self.config = config
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
        self.name_train, self.name_test = None, None
        self.y_pred, self.y_pred_proba = None, None
        self.feature_names = None
        self.scaler = None
        self.company_dict = {}

    def prepare_data(self):
        base_df, company_info_df = dp.load_and_pivot_data(self.config)
        X_all, y_all, info_all, self.feature_names = dp.create_sequences(base_df, company_info_df, self.config)

        # 이 결과가 스케일링 전의 원본 분리 데이터입니다.
        X_train_before_scale, y_train_before_scale, X_test_before_scale, y_test_before_scale, info_train, info_test = \
            dp.split_data_by_date(X_all, y_all, info_all, self.config)

        # self.name_train과 self.name_test는 info_train/info_test에서 가져옵니다.
        self.name_train = info_train['unique_sample_id'].values
        self.name_test = info_test['unique_sample_id'].values

        output_folder = self.config['result_folder_path']
        print("\nINFO: 스케일링 전 원본 시퀀스 데이터를 평탄화하여 CSV 파일로 저장합니다...")

        # 1. 모든 훈련 샘플을 평탄화하여 하나의 큰 CSV 파일로 저장
        n_samples_train, n_timesteps_train, n_features_train = X_train_before_scale.shape
        flattened_column_names_train = []
        for t in range(n_timesteps_train):
            for f_name in self.feature_names:
                flattened_column_names_train.append(f"{f_name}_월{t+1}") # 월을 1부터 시작

        X_train_raw_flattened = X_train_before_scale.reshape(n_samples_train, -1)

        df_all_train_samples_flattened = pd.DataFrame(X_train_raw_flattened, columns=flattened_column_names_train)
        df_all_train_samples_flattened.insert(0, 'sample_id', self.name_train) # ID 컬럼 추가
        df_all_train_samples_flattened.insert(1, 'label', y_train_before_scale) # Label 컬럼 추가

        all_train_flattened_csv_path = os.path.join(output_folder, "all_train_raw_flattened_sequences.csv")
        df_all_train_samples_flattened.to_csv(all_train_flattened_csv_path, index=False, encoding='utf-8-sig')
        print(f"모든 훈련 RAW 샘플의 평탄화된 데이터가 '{all_train_flattened_csv_path}' 에 저장되었습니다.")

        # 2. 모든 테스트 샘플을 평탄화하여 하나의 큰 CSV 파일로 저장
        n_samples_test, n_timesteps_test, n_features_test = X_test_before_scale.shape
        flattened_column_names_test = []
        for t in range(n_timesteps_test):
            for f_name in self.feature_names:
                flattened_column_names_test.append(f"{f_name}_월{t+1}")

        X_test_raw_flattened = X_test_before_scale.reshape(n_samples_test, -1)

        df_all_test_samples_flattened = pd.DataFrame(X_test_raw_flattened, columns=flattened_column_names_test)
        df_all_test_samples_flattened.insert(0, 'sample_id', self.name_test)
        df_all_test_samples_flattened.insert(1, 'label', y_test_before_scale)

        all_test_flattened_csv_path = os.path.join(output_folder, "all_test_raw_flattened_sequences.csv")
        df_all_test_samples_flattened.to_csv(all_test_flattened_csv_path, index=False, encoding='utf-8-sig')
        print(f"모든 테스트 RAW 샘플의 평탄화된 데이터가 '{all_test_flattened_csv_path}' 에 저장되었습니다.")


        # --- 스케일링 적용 (X_train_before_scale, X_test_before_scale을 인자로 전달) ---
        X_train_after_scale, X_test_after_scale, self.scaler = \
            dp.scale_data(X_train_before_scale, X_test_before_scale, self.config)

        # --- 데이터 형태 변환 및 샘플링 적용 ---
        if self.config['data_shape'] == 'flatten':
            n_samples, n_timesteps, n_features = X_train_after_scale.shape
            X_train_flatten = X_train_after_scale.reshape(n_samples, -1)
            X_test_flatten = X_test_after_scale.reshape(X_test_after_scale.shape[0], -1)

            self.X_train, self.y_train = dp.apply_sampling(X_train_flatten, y_train_before_scale, self.config)
            self.X_test = X_test_flatten
            self.y_test = y_test_before_scale

        else: # matrix (ConvLSTM, Transformer 등 3D 모델용)
            if self.config.get('use_smote_for_keras', False):
                print("INFO: Keras 모델에 SMOTE 오버샘플링을 적용합니다...")
                n_samples, n_timesteps, n_features = X_train_after_scale.shape
                X_reshaped = X_train_after_scale.reshape(n_samples, -1)
                smote = SMOTE(random_state=self.config['random_state'])
                X_res, y_res = smote.fit_resample(X_reshaped, y_train_before_scale)
                self.X_train, self.y_train = X_res.reshape(-1, n_timesteps, n_features), y_res
                print(f"SMOTE 적용 완료. Train 데이터 수: {n_samples} -> {len(y_res)}")
            else:
                self.X_train, self.y_train = X_train_after_scale, y_train_before_scale

            self.X_test = X_test_after_scale
            self.y_test = y_test_before_scale

        # --- 클래스 가중치 자동 계산 및 적용 코드 ---

        unique_classes = np.unique(self.y_train)
        if len(unique_classes) > 1: # 두 개 이상의 클래스가 있을 때만 계산
            computed_class_weights = class_weight.compute_class_weight(
                class_weight='balanced', # 'balanced' 옵션으로 불균형 자동 보정
                classes=unique_classes,
                y=self.y_train # 최종 훈련 라벨 사용
            )
            self.config['class_weight'] = {cls: weight for cls, weight in zip(unique_classes, computed_class_weights)}
            print(f"계산된 클래스 가중치: {self.config['class_weight']}")
        else:
            print("경고: 훈련 데이터에 단일 클래스만 존재하여 클래스 가중치를 계산할 수 없습니다. 가중치를 1로 설정합니다.")
            self.config['class_weight'] = {unique_classes[0]: 1.0}
        print("모든 데이터 준비 완료.")