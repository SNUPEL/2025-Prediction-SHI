from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
import pandas as pd
import joblib


def get_AdaBoostClassifier(self):
    if 'estimator_parameter' in self.config:
        self.model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(**self.config['estimator_parameter'],
                                             random_state=self.config['random_state']),
            **self.config['model_parameter'], random_state=self.config['random_state'])
    else:
        self.model = AdaBoostClassifier(**self.config['model_parameter'], random_state=self.config['random_state'])

    self.model.fit(self.data.df_x_train_flatten, self.data.df_y_train['label'],
                   sample_weight=compute_sample_weight(class_weight=self.config['class_weight'],
                                                       y=self.data.df_y_train['label']))

    self.data.df_y_pred = pd.DataFrame(
        self.model.predict(self.data.df_x_test_flatten),
        index=self.data.name_test,
        columns=['label']
    )

    if self.config['save_model']:
        joblib.dump(self.model, self.config['result_folder_path'] + '/AdaBoostClassifier model.joblib')
