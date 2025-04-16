import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def evaluate_classifier(self):
    self.result['accuracy'] = accuracy_score(self.data.df_y_test, self.data.df_y_pred)
    self.result['precision'] = precision_score(self.data.df_y_test, self.data.df_y_pred)
    self.result['recall'] = recall_score(self.data.df_y_test, self.data.df_y_pred)
    self.result['f1_score'] = f1_score(self.data.df_y_test, self.data.df_y_pred)
    print(f"Accuracy: {self.result['accuracy']}, Precision: {self.result['precision']}")


def evaluate_regressor(self):
    self.result['MSE'] = mean_squared_error(self.data.df_y_test, self.data.df_y_pred)
    self.result['RMSE'] = np.sqrt(self.result['MSE'])
    self.result['MAE'] = mean_absolute_error(self.data.df_y_test, self.data.df_y_pred)
    self.result['MAPE'] = mean_absolute_percentage_error(self.data.df_y_test, self.data.df_y_pred)
