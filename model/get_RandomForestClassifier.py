from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd


def get_RandomForestClassifier(self):
    self.model = RandomForestClassifier(n_estimators=100,
                                        max_depth=20,
                                        min_samples_split=3,
                                        min_samples_leaf=1,
                                        max_features='log2',
                                        random_state=self.config['random_state'])

    self.model.fit(self.data.df_x_train.drop(columns=["company_id"]), self.data.df_y_train)
    # self.data.df_y_pred = self.model.predict(self.data.df_x_test.drop(columns=["company_id"]))
    self.data.df_y_pred = pd.Series(
        self.model.predict(self.data.df_x_test.drop(columns=["company_id"])),
        index=self.data.df_y_test.index,
        name='label'
    )

    self.models_hyperparameters = self.model.get_params()

    # save model
    if self.config['save_model']:
        joblib.dump(self.model, self.config['result_folder_path'] + '/RandomForestClassifier model.joblib')
