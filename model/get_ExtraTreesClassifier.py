from sklearn.ensemble import ExtraTreesClassifier
import joblib
import pandas as pd


def get_ExtraTreesClassifier(self):
    self.model = ExtraTreesClassifier(
        n_estimators=1000,
        max_depth=50,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=self.config['random_state']
        # n_jobs=-1
    )

    self.model.fit(self.data.df_x_train.drop(columns=["company_id"]), self.data.df_y_train)
    self.data.df_y_pred = pd.Series(
        self.model.predict(self.data.df_x_test.drop(columns=["company_id"])),
        index=self.data.df_y_test.index,
        name='label'
    )

    self.models_hyperparameters = self.model.get_params()

    # save model
    if self.config['save_model']:
        joblib.dump(self.model, self.config['result_folder_path'] + '/ExtraTreesClassifier model.joblib')