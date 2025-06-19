from xgboost import XGBClassifier
import pandas as pd
import joblib


def get_XGBClassifier(self):
    self.model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.001,
        max_depth=50,
        gamma=0.1,
        min_child_weight=1,
        random_state=self.config['random_state'],
        n_jobs=-1
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
        joblib.dump(self.model, self.config['result_folder_path'] + '/XGBClassifier model.joblib')