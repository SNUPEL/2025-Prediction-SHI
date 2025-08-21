from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import RidgeClassifier
import pandas as pd


def get_VotingClassifier(self, voting):
    svc = SVC(
        **self.config['SVC_parameter'],
        class_weight=self.config['class_weight'],
        random_state=self.config['random_state']
    )

    nuSVC = NuSVC(**self.config['nuSVC_parameter'],
        class_weight=self.config['class_weight'],
        random_state=self.config['random_state'])

    adaboost = AdaBoostClassifier(**self.config['AdaBoostClassifier_parameter'],

        random_state=self.config['random_state'])

    ridge = RidgeClassifier(**self.config['RidgeClassifier_parameter'],
                  class_weight=self.config['class_weight'],
                  random_state=self.config['random_state']
                  )

    svc = svc.fit(self.data.df_x_train_flatten, self.data.df_y_train['label'])
    nuSVC = nuSVC.fit(self.data.df_x_train_flatten, self.data.df_y_train['label'])
    adaboost = adaboost.fit(self.data.df_x_train_flatten, self.data.df_y_train['label'])
    ridge = ridge.fit(self.data.df_x_train_flatten, self.data.df_y_train['label'])

    self.model = VotingClassifier(
        estimators=[
        ('svc', svc),
        ('nusvc', nuSVC),
        ('adaboost', adaboost),
        ('ridge', ridge)
                                ],
        weights=[1, 1, 1, 2],
        voting=voting).fit(self.data.df_x_train_flatten, self.data.df_y_train['label'])

    self.data.df_y_pred = pd.DataFrame(
        self.model.predict(self.data.df_x_test_flatten),
        index=self.data.name_test,
        columns=['label']
    )