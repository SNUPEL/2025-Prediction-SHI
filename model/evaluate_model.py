import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os
import sys


def evaluate_classifier(self):
    # 한글 폰트 설정
    self._set_korean_font()

    y_test_arr = self.data.y_test
    y_pred_arr = self.data.y_pred

    # 기본 평가 지표 계산
    self.result['accuracy'] = accuracy_score(y_test_arr, y_pred_arr)
    self.result['precision'] = precision_score(y_test_arr, y_pred_arr, zero_division=0.0, pos_label=1)
    self.result['recall'] = recall_score(y_test_arr, y_pred_arr, zero_division=0.0, pos_label=1)
    self.result['f1_score'] = f1_score(y_test_arr, y_pred_arr, zero_division=0.0, pos_label=1)

    # 혼동 행렬 계산
    cm = confusion_matrix(y_test_arr, y_pred_arr, labels=[1, 0]).tolist()
    self.result['confusion_matrix'] = cm

    # 예측 결과 개수 계산
    self.result['correct_count'] = np.sum(y_test_arr == y_pred_arr)
    self.result['wrong_count'] = np.sum(y_test_arr != y_pred_arr)
    self.result['total_count'] = len(y_test_arr)

    # 클래스별 통계 계산
    self.result['class_0_total'] = np.sum(y_test_arr == 0)  # 거래중(0) 총 개수
    self.result['class_1_total'] = np.sum(y_test_arr == 1)  # 경영악화(1) 총 개수

    self.result['class_0_correct'] = np.sum((y_test_arr == 0) & (y_pred_arr == 0))  # 거래중 맞춘 개수
    self.result['class_1_correct'] = np.sum((y_test_arr == 1) & (y_pred_arr == 1))  # 경영악화 맞춘 개수

    # 클래스별 정확도
    self.result['class_0_accuracy'] = self.result['class_0_correct'] / self.result['class_0_total'] if self.result['class_0_total'] > 0 else 0.0
    self.result['class_1_accuracy'] = self.result['class_1_correct'] / self.result['class_1_total'] if self.result['class_1_total'] > 0 else 0.0

    # 상세 결과 출력
    if self.config['detailed_results']:
        print("\n===== 모델 성능 평가 결과 =====")
        print(f"정확도(Accuracy): {self.result['accuracy']:.4f}")
        print(f"정밀도(Precision): {self.result['precision']:.4f}")
        print(f"재현율(Recall): {self.result['recall']:.4f}")
        print(f"F1 점수: {self.result['f1_score']:.4f}")

        print("\n----- 예측 결과 요약 -----")
        print(f"전체 테스트 샘플: {self.result['total_count']}개")
        print(f"맞은 예측: {self.result['correct_count']}개 ({self.result['correct_count']/self.result['total_count']:.2%})")
        print(f"틀린 예측: {self.result['wrong_count']}개 ({self.result['wrong_count']/self.result['total_count']:.2%})")

        print("\n----- 클래스별 성능 -----")
        print(f"경영악화(1) 클래스: {self.result['class_1_total']}개 중 {self.result['class_1_correct']}개 맞춤 ({self.result['class_1_accuracy']:.2%})")
        print(f"거래중(0) 클래스: {self.result['class_0_total']}개 중 {self.result['class_0_correct']}개 맞춤 ({self.result['class_0_accuracy']:.2%})")

        print("\n----- 혼동 행렬 -----")
        print(f"        | 예측: 경영악화(1) | 예측: 거래중(0)")
        print(f"실제: 경영악화(1) | {cm[0][0]}            | {cm[0][1]}")  # cm[0][0] = TP, cm[0][1] = FN
        print(f"실제: 거래중(0)  | {cm[1][0]}            | {cm[1][1]}")  # cm[1][0] = FP, cm[1][1] = TN
        print("==========================\n")
    else:
        print(f"Accuracy: {self.result['accuracy']}, Precision: {self.result['precision']}")

    # 혼동 행렬 시각화 및 저장
    if self.config['save_confusion_matrix']:
        plt.figure(figsize=(8, 6))
        classes_plot_labels = ['경영악화(1)', '거래중(0)']

        ax = sns.heatmap(np.array(cm), annot=True, fmt='d', cmap='Blues',
                         xticklabels=classes_plot_labels, yticklabels=classes_plot_labels,
                         annot_kws={"size": 14, "weight": "bold"})

        ax.set_xlabel('예측', fontsize=12, fontweight='bold')
        ax.set_ylabel('실제', fontsize=12, fontweight='bold')
        ax.set_title(f'{self.config["model_type"]} 혼동 행렬', fontsize=14, fontweight='bold')

        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)

        plt.tight_layout()

        cm_path = os.path.join(self.config['result_folder_path'], 'confusion_matrix_positive_first.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()

    # 예측 결과 상세 저장
    if self.config['save_predictions']:
        predictions = pd.DataFrame({
            'company_id': self.data.name_test,
            'actual': y_test_arr,
            'predicted': y_pred_arr,
            'correct': (y_test_arr == y_pred_arr)
        })

        predictions['actual_label'] = predictions['actual'].map({0: '거래중', 1: '경영악화'})
        predictions['predicted_label'] = predictions['predicted'].map({0: '거래중', 1: '경영악화'})

        pred_path = os.path.join(self.config['result_folder_path'], 'predictions_detail.csv')
        predictions.to_csv(pred_path, index=False, encoding='utf-8-sig')

    # 성능 지표 저장
    if self.config['save_metrics']:
        metrics_to_save = {k: v for k, v in self.result.items() if not isinstance(v, list)}
        metrics_df = pd.DataFrame([metrics_to_save])
        metrics_path = os.path.join(self.config['result_folder_path'], 'detailed_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')


def evaluate_regressor(self):
    self.result['MSE'] = mean_squared_error(self.data.y_test, self.data.y_pred)
    self.result['RMSE'] = np.sqrt(self.result['MSE'])
    self.result['MAE'] = mean_absolute_error(self.data.y_test, self.data.y_pred)
    self.result['MAPE'] = mean_absolute_percentage_error(self.data.y_test, self.data.y_pred)
