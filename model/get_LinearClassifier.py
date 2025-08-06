from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
import pandas as pd
import joblib

def get_RidgeClassifier(self):
    self.model = RidgeClassifier(
        random_state=self.config['random_state'],
        class_weight=self.config['class_weight']
    )

    self.model.fit(self.data.X_train, self.data.y_train)
    self.data.y_pred = pd.Series(
        self.model.predict(self.data.X_test),
        index=self.data.name_test,
        name='label'
    )

    self.models_hyperparameters = self.model.get_params()

    # save model
    if self.config['save_model']:
        joblib.dump(self.model, self.config['result_folder_path'] + '/RidgeClassifier model.joblib')

def get_SGDClassifier(self):
    self.model = SGDClassifier(
        random_state=self.config['random_state']
    )

    self.model.fit(self.data.X_train, self.data.y_train)
    self.data.y_pred = pd.Series(
        self.model.predict(self.data.X_test),
        index=self.data.name_test,
        name='label'
    )

    self.models_hyperparameters = self.model.get_params()

    # save model
    if self.config['save_model']:
        joblib.dump(self.model, self.config['result_folder_path'] + '/SGDClassifier model.joblib')