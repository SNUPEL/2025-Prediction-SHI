from sklearn.svm import SVC
import pandas as pd
import joblib

def get_SVC(self):
    self.model = SVC(
        C=10,
        kernel='rbf',
        random_state=self.config['random_state']
    )

    self.model.fit(self.data.X_train.drop(columns=["company_id"]), self.data.y_train)
    self.data.y_pred = pd.Series(
        self.model.predict(self.data.X_test.drop(columns=["company_id"])),
        index=self.data.y_test.index,
        name='label'
    )

    self.models_hyperparameters = self.model.get_params()

    # save model
    if self.config['save_model']:
        joblib.dump(self.model, self.config['result_folder_path'] + '/SVC model.joblib')