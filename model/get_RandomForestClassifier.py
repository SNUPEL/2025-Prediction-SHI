from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd


def get_RandomForestClassifier(self):
    self.model = RandomForestClassifier(n_estimators=1000,
                                        max_depth=25,
                                        min_samples_split=2,
                                        min_samples_leaf=20,
                                        max_features='sqrt',
                                        random_state=self.config['random_state'],
                                        n_jobs=-1)

    self.model.fit(self.data.X_train, self.data.y_train)
    # self.data.df_y_pred = self.model.predict(self.data.df_x_test.drop(columns=["company_id"]))
    self.data.y_pred = pd.Series(
        self.model.predict(self.data.X_test),
        index=self.data.name_test,
        name='label'
    )

    self.models_hyperparameters = self.model.get_params()

    # save model
    if self.config['save_model']:
        joblib.dump(self.model, self.config['result_folder_path'] + '/RandomForestClassifier model.joblib')
