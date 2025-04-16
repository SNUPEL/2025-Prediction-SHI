from sklearn.model_selection import train_test_split
import pandas as pd


def random_split(self):
    self.df_x = self.df_model.drop(columns=['label'])
    self.df_y = self.df_model['label']

    self.df_x_train, self.df_x_test, self.df_y_train, self.df_y_test = train_test_split(
        self.df_x, self.df_y, test_size=self.config['test_data_ratio'], random_state=self.config['random_state'])

def SMOTE(self):
    pass
