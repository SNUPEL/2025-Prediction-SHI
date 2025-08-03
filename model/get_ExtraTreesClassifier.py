from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import pandas as pd


def get_ExtraTreesClassifier(self):
    self.model = ExtraTreesClassifier(
        **self.config['model_parameter'],
        random_state=self.config['random_state']
    )

    self.model.fit(self.data.df_x_train_flatten, self.data.df_y_train['label'],
                   sample_weight=compute_sample_weight(class_weight=self.config.get('class_weight'),
                                                       y=self.data.df_y_train['label']))
    self.data.df_y_pred = pd.DataFrame(
        self.model.predict(self.data.df_x_test_flatten),
        index=self.data.name_test,
        columns=['label']
    )

    self.models_hyperparameters = self.model.get_params()

    if self.config['save_model']:
        joblib.dump(self.model, self.config['result_folder_path'] + '/ExtraTreesClassifier model.joblib')