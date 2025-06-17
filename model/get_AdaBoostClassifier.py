from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
import pandas as pd
import joblib


def get_AdaBoostClassifier(self):
    if 'AdaBoost_estimator_parameter' in self.config:
        self.model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(**self.config['AdaBoost_estimator_parameter'],
                                             random_state=self.config['random_state']),
            **self.config['AdaBoost_parameter'], random_state=self.config['random_state'])
    else:
        self.model = AdaBoostClassifier(**self.config['AdaBoost_parameter'], random_state=self.config['random_state'])
    self.model.fit(self.data.df_x_train.drop(columns=["company_id"]), self.data.df_y_train,
                   sample_weight=compute_sample_weight(class_weight=self.config['class_weight'], y=self.data.df_y_train))
    self.data.df_y_pred = pd.Series(
        self.model.predict(self.data.df_x_test.drop(columns=["company_id"])),
        index=self.data.df_y_test.index,
        name='label'
    )
    self.models_hyperparameters = self.model.get_params()
    if self.config['save_model']:
        joblib.dump(self.model, self.config['result_folder_path'] + '/AdaBoostClassifier model.joblib')
