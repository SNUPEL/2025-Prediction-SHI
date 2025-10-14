from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
import pandas as pd
import joblib
import numpy as np


def get_AdaBoostClassifier(self):
    if 'estimator_parameter' in self.config:
        self.model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(**self.config['estimator_parameter'],      # estimator 파라미터: max_depth, min_samples_split
                                             random_state=self.config['random_state']),     # 랜덤시드 값
            **self.config['model_parameter'], random_state=self.config['random_state'])     # 모델 파라미터: n_estimators, learning_rate
    else:       # estimator 파라미터를 설정하지 않은 경우 사용
        self.model = AdaBoostClassifier(**self.config['model_parameter'], random_state=self.config['random_state'])

    self.model.fit(self.data.df_x_train_flatten, self.data.df_y_train['label'],         # 모델 학습
                   sample_weight=compute_sample_weight(class_weight=self.config['class_weight'],        # sample_weight 추정
                                                       y=self.data.df_y_train['label']))

    # feature importance 데이터프레임 생성
    df_features = pd.DataFrame({'Feature_name': self.model.feature_names_in_, 'Feature_importance': self.model.feature_importances_})
    file_path = self.config["result_folder_path"] + '/AdaBoost_features_importance.xlsx'        # 결과 폴더에 feature importance 파일 저장
    df_features.to_excel(file_path, index=False)

    max_feature_idx = np.where(self.model.feature_importances_ == np.amax(self.model.feature_importances_))[0][0]       # 가장 중요도가 큰 feature
    max_feature = self.model.feature_names_in_[max_feature_idx]

    min_feature_idx = np.where(self.model.feature_importances_ == np.amin(self.model.feature_importances_))[0][0]       # 가장 중요도가 작은 feature
    min_feature = self.model.feature_names_in_[min_feature_idx]

    print()
    print('=== Feature별 중요도 분석 결과 ===')
    print('중요도가 가장 큰 feature:', max_feature, ' 값:', np.amax(self.model.feature_importances_))
    print('중요도가 가장 작은 feature:', min_feature, ' 값:', np.amin(self.model.feature_importances_))
    print()

    self.data.df_y_pred = pd.DataFrame(     # 예측 결과 데이터프레임 생성
        self.model.predict(self.data.df_x_test_flatten),
        index=self.data.name_test,
        columns=['label']
    )

    if self.config['save_model']:       # 학습한 모델 저장
        joblib.dump(self.model, self.config['result_folder_path'] + '/AdaBoostClassifier model.joblib')
