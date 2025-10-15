from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.utils.class_weight import compute_sample_weight
import pandas as pd
import joblib


def get_SVC(self):
    self.model = SVC(
        **self.config['model_parameter'],       # 모델 파라미터: probability, C, kernel, gamma
        class_weight=self.config['class_weight'],       # 클래스별 가중치
        random_state=self.config['random_state']        # 랜덤시드 값
    )

    self.model.fit(self.data.df_x_train_flatten, self.data.df_y_train['label'])     # 모델 학습

    self.data.df_y_pred = pd.DataFrame(     # 예측 결과 데이터프레임 생성
        self.model.predict(self.data.df_x_test_flatten),
        index=self.data.name_test,
        columns=['label']
    )

    if self.config['save_model']:       # 학습한 모델 저장
        joblib.dump(self.model, self.config['result_folder_path'] + '/SVC model.joblib')


def get_nuSVC(self):
    self.model = NuSVC(
        **self.config['model_parameter']        # 모델 파라미터: probability, nu, kernel, gamma
    )

    self.model.fit(self.data.df_x_train_flatten, self.data.df_y_train['label'],     # 모델 학습
                   sample_weight=compute_sample_weight(class_weight=self.config['class_weight'],        # sample_weight 추정
                                                       y=self.data.df_y_train['label']))

    self.data.df_y_pred = pd.DataFrame(     # 예측 결과 데이터프레임 생성
        self.model.predict(self.data.df_x_test_flatten),
        index=self.data.name_test,
        columns=['label']
    )
