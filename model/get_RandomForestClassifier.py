from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd


def get_RandomForestClassifier(self):
    self.model = RandomForestClassifier(
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
        joblib.dump(self.model, self.config['result_folder_path'] + '/RandomForestClassifier model.joblib')
