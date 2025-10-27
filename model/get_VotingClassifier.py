from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import RidgeClassifier
import pandas as pd


def get_VotingClassifier(self, voting):
    # SVC 추정기
    svc = SVC(
        **self.config['SVC_parameter'],  # probability, C, kernel, gamma
        class_weight=self.config['class_weight'],  # 클래스별 가중치
        random_state=self.config['random_state']  # 랜덤시드 값
    )

    # nuSVC 추정기
    nuSVC = NuSVC(**self.config['nuSVC_parameter'],  # probability, nu, kernel, gamma
                  class_weight=self.config['class_weight'],  # 클래스별 가중치
                  random_state=self.config['random_state'])  # 랜덤시드 값

    # AdaBoostClassifier 추정기
    adaboost = AdaBoostClassifier(**self.config['AdaBoostClassifier_parameter'],  # n_estimators, learning_rate

                                  random_state=self.config['random_state'])  # 랜덤시드 값

    # RidgeClassifier 추정기
    ridge = RidgeClassifier(**self.config['RidgeClassifier_parameter'],  # alpha, solver, tol
                            class_weight=self.config['class_weight'],  # 클래스별 가중치
                            random_state=self.config['random_state']  # 랜덤시드 값
                            )

    svc = svc.fit(self.data.df_x_train_flatten, self.data.df_y_train['label'])  # SVC 학습
    nuSVC = nuSVC.fit(self.data.df_x_train_flatten, self.data.df_y_train['label'])  # NuSVC 학습
    adaboost = adaboost.fit(self.data.df_x_train_flatten, self.data.df_y_train['label'])  # AdaBoost 학습
    ridge = ridge.fit(self.data.df_x_train_flatten, self.data.df_y_train['label'])  # RidgeClassifier 학습

    self.model = VotingClassifier(
        estimators=[  # 추정기 설정
            ('svc', svc),
            ('nusvc', nuSVC),
            ('adaboost', adaboost),
            ('ridge', ridge)
        ],
        weights=self.config['model_parameter']['weights'],  # 모델별 가중치
        voting=voting).fit(self.data.df_x_train_flatten, self.data.df_y_train['label'])

    self.data.df_y_pred = pd.DataFrame(  # 예측 결과 데이터프레임 생성
        self.model.predict(self.data.df_x_test_flatten),
        index=self.data.name_test,
        columns=['label']
    )
