from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
import pandas as pd
import joblib


def get_RidgeClassifier(self):
    self.model = RidgeClassifier(
        **self.config['model_parameter'],
        class_weight=self.config['class_weight'],
        random_state=self.config['random_state']
    )

    self.model.fit(self.data.df_x_train_flatten, self.data.df_y_train['label'])

    self.data.df_y_pred = pd.DataFrame(
        self.model.predict(self.data.df_x_test_flatten),
        index=self.data.name_test,
        columns=['label']
    )

    if self.config['save_model']:
        joblib.dump(self.model, self.config['result_folder_path'] + '/RidgeClassifier model.joblib')


def get_SGDClassifier(self):
    self.model = SGDClassifier(
        **self.config['model_parameter'],
        class_weight=self.config['class_weight'],
        random_state=self.config['random_state']
    )

    self.model.fit(self.data.df_x_train_flatten, self.data.df_y_train['label'])

    self.data.df_y_pred = pd.DataFrame(
        self.model.predict(self.data.df_x_test_flatten),
        index=self.data.name_test,
        columns=['label']
    )

    if self.config['save_model']:
        joblib.dump(self.model, self.config['result_folder_path'] + '/SGDClassifier model.joblib')