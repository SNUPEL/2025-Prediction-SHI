from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.utils.class_weight import compute_sample_weight
import pandas as pd
import joblib


def get_SVC(self):
    self.model = SVC(
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
        joblib.dump(self.model, self.config['result_folder_path'] + '/SVC model.joblib')


def get_nuSVC(self):
    self.model = NuSVC(
        **self.config['model_parameter']
    )

    self.model.fit(self.data.df_x_train_flatten, self.data.df_y_train['label'],
                   sample_weight=compute_sample_weight(class_weight=self.config['class_weight'],
                                                       y=self.data.df_y_train['label']))

    self.data.df_y_pred = pd.DataFrame(
        self.model.predict(self.data.df_x_test_flatten),
        index=self.data.name_test,
        columns=['label']
    )
