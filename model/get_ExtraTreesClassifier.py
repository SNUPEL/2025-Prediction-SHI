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

    self.model.fit(self.data.X_train, self.data.y_train)
    self.data.y_pred = pd.Series(
        self.model.predict(self.data.X_test),
        index=self.data.name_test,
        name='label'
    )

    self.models_hyperparameters = self.model.get_params()

    # save model
    if self.config['save_model']:
        joblib.dump(self.model, self.config['result_folder_path'] + '/ExtraTreesClassifier model.joblib')