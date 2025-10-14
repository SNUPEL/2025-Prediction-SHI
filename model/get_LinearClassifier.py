from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
import pandas as pd
import joblib
import numpy as np


def get_RidgeClassifier(self):
    self.model = RidgeClassifier(
        **self.config['model_parameter'],       # 모델 파라미터: alpha, solver, tol
        class_weight=self.config['class_weight'],       # 클래스별 가중치: {0(거래중): a, 1(경영악화): b} 형식으로 사용(a, b는 자연수)
        random_state=self.config['random_state']    # 랜덤시드 값
    )

    self.model.fit(self.data.df_x_train_flatten, self.data.df_y_train['label'])     # 모델 학습

    # feature별 가중치(계수) 데이터프레임 생성
    df_features = pd.DataFrame({'Feature_name': self.model.feature_names_in_, 'Feature_coef': self.model.coef_})
    file_path = self.config["result_folder_path"] + '/Ridge_features_coef.xlsx'     # 결과 폴더에 feature별 계수 파일 저장
    df_features.to_excel(file_path, index=False)

    max_feature_idx = np.where(self.model.coef_ == np.amax(self.model.coef_))[0][0]     # 가장 계수가 큰 feature
    max_feature = self.model.feature_names_in_[max_feature_idx]

    min_feature_idx = np.where(self.model.coef_ == np.amin(self.model.coef_))[0][0]     # 가장 계수가 작은 feature
    min_feature = self.model.feature_names_in_[min_feature_idx]

    print()
    print('=== Feature별 계수 분석 결과 ===')
    print('가중치가 가장 큰 feature:', max_feature,' |  값:', np.amax(self.model.coef_))
    print('가중치가 가장 작은 feature:', min_feature, ' |  값:', np.amin(self.model.coef_))
    print()

    self.data.df_y_pred = pd.DataFrame(         # 예측 결과 데이터프레임 생성
        self.model.predict(self.data.df_x_test_flatten),
        index=self.data.name_test,
        columns=['label']
    )

    if self.config['save_model']:       # 학습한 모델 저장
        joblib.dump(self.model, self.config['result_folder_path'] + '/RidgeClassifier model.joblib')


def get_SGDClassifier(self):
    self.model = SGDClassifier(
        **self.config['model_parameter'],       # 모델 파라미터: loss, penalty, alpha, max_iter, learning_rate
        class_weight=self.config['class_weight'],       # 클래스별 가중치: {0(거래중): a, 1(경영악화): b} 형식으로 사용(a, b는 자연수)
        random_state=self.config['random_state']        # 랜덤시드 값
    )

    self.model.fit(self.data.df_x_train_flatten, self.data.df_y_train['label'])     # 모델 학습

    self.data.df_y_pred = pd.DataFrame(     # 예측 결과 데이터프레임 생성
        self.model.predict(self.data.df_x_test_flatten),
        index=self.data.name_test,
        columns=['label']
    )

    if self.config['save_model']:       # 학습한 모델 저장
        joblib.dump(self.model, self.config['result_folder_path'] + '/SGDClassifier model.joblib')