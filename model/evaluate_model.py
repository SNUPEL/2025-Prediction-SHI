import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def evaluate_classifier(self):
    y_test_arr = self.data.df_y_test['label'].values
    y_pred_arr = self.data.df_y_pred['label'].values

    # 혼동 행렬에서 TP, FN, FP, TN 추출
    cm = confusion_matrix(y_test_arr, y_pred_arr, labels=[1, 0])
    self.result['confusion_matrix'] = cm.tolist()
    self.result['TP'] = cm[0][0]
    self.result['FN'] = cm[0][1]
    self.result['FP'] = cm[1][0]
    self.result['TN'] = cm[1][1]

    # 전체 예측 건수 집계
    self.result['total_count'] = len(y_test_arr)
    self.result['correct_count'] = int(np.sum(y_test_arr == y_pred_arr))
    self.result['wrong_count'] = int(np.sum(y_test_arr != y_pred_arr))

    # 클래스별 분포 및 정답 수
    self.result['class_1_total'] = int(np.sum(y_test_arr == 1))
    self.result['class_0_total'] = int(np.sum(y_test_arr == 0))
    self.result['class_1_correct'] = int(self.result['TP'])
    self.result['class_0_correct'] = int(self.result['TN'])

    # 주요 분류 지표 계산
    self.result['accuracy'] = accuracy_score(y_test_arr, y_pred_arr)
    self.result['precision'] = precision_score(y_test_arr, y_pred_arr, zero_division=0.0)
    self.result['recall'] = recall_score(y_test_arr, y_pred_arr, zero_division=0.0)
    self.result['f1_score'] = f1_score(y_test_arr, y_pred_arr, zero_division=0.0)
    self.result['specificity'] = self.result['TN'] / self.result['class_0_total'] if self.result['class_0_total'] > 0 else 0.0


def evaluate_regressor(self):
    # 회귀 문제용 핵심 지표 계산
    self.result['MSE'] = mean_squared_error(self.data.y_test, self.data.y_pred)
    self.result['RMSE'] = np.sqrt(self.result['MSE'])
    self.result['MAE'] = mean_absolute_error(self.data.y_test, self.data.y_pred)
    self.result['MAPE'] = mean_absolute_percentage_error(self.data.y_test, self.data.y_pred)
