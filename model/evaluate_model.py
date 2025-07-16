import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import tensorflow as tf  # Keras 모델(Transformer) DeepExplainer를 위해 필요
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    precision_recall_curve


def evaluate_classifier(self):
    """
    분류 모델의 성능 지표 (정확도, 정밀도, 재현율, F1-Score 등)를 계산합니다.
    """
    y_test_arr, y_pred_arr = self.data.y_test, self.data.y_pred
    self.result['accuracy'] = accuracy_score(y_test_arr, y_pred_arr)
    self.result['precision'] = precision_score(y_test_arr, y_pred_arr, zero_division=0, pos_label=1)
    self.result['recall'] = recall_score(y_test_arr, y_pred_arr, zero_division=0, pos_label=1)
    self.result['f1_score'] = f1_score(y_test_arr, y_pred_arr, zero_division=0, pos_label=1)
    self.result['confusion_matrix'] = confusion_matrix(y_test_arr, y_pred_arr, labels=[1, 0]).tolist()
    self.result['TP'] = int(np.sum((y_test_arr == 1) & (y_pred_arr == 1)))  # True Positives
    self.result['FN'] = int(np.sum((y_test_arr == 1) & (y_pred_arr == 0)))  # False Negatives
    self.result['FP'] = int(np.sum((y_test_arr == 0) & (y_pred_arr == 1)))  # False Positives
    self.result['TN'] = int(np.sum((y_test_arr == 0) & (y_pred_arr == 0)))  # True Negatives
    self.result['total_count'] = len(y_test_arr)
    self.result['correct_count'] = int(np.sum(y_test_arr == y_pred_arr))
    self.result['wrong_count'] = int(np.sum(y_test_arr != y_pred_arr))
    self.result['class_1_total'] = int(np.sum(y_test_arr == 1))
    self.result['class_0_total'] = int(np.sum(y_test_arr == 0))
    self.result['class_1_correct'] = int(np.sum((y_test_arr == 1) & (y_pred_arr == 1)))
    self.result['class_0_correct'] = int(np.sum((y_test_arr == 0) & (y_pred_arr == 0)))
    self.result['class_1_accuracy'] = self.result['class_1_correct'] / self.result['class_1_total'] if self.result[
                                                                                                           'class_1_total'] > 0 else 0.0
    self.result['class_0_accuracy'] = self.result['class_0_correct'] / self.result['class_0_total'] if self.result[
                                                                                                           'class_0_total'] > 0 else 0.0


def evaluate_keras_model(self):
    """
    Keras 모델의 성능을 평가하고 최적의 임계값을 찾습니다.
    """
    y_true, y_pred_proba = self.data.y_test, self.data.y_pred_proba

    # Precision-Recall Curve를 사용하여 임계값 탐색
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

    min_precision = self.config['min_precision_for_threshold']
    candidate_indices = np.where(precision[:-1] >= min_precision)[0]

    best_threshold = 0.5  # 기본 임계값 설정

    if len(candidate_indices) > 0:
        # 최소 정밀도 조건을 만족하는 임계값 중 재현율이 가장 높은 임계값 선택
        best_recall_idx = candidate_indices[np.argmax(recall[candidate_indices])]
        best_threshold = thresholds[best_recall_idx]
        print(f"성공: 정밀도 {min_precision:.0%} 이상에서 재현율을 극대화하는 최적 임계값({best_threshold:.2f})을 찾았습니다.")
    else:
        # 정밀도 조건을 만족하는 임계값이 없으면 F1-Score를 최대화하는 임계값 선택
        f1_scores = 2 * (precision * recall) / (precision + recall)
        # NaN 값(분모 0)을 0으로 처리
        f1_scores = np.nan_to_num(f1_scores)
        best_f1_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_f1_idx]
        print(f" 경고: 정밀도 조건을 만족하는 임계값이 없어 F1-Score를 최대화하는 임계값({best_threshold:.2f})으로 전환합니다.")

    # 최적 임계값을 결과에 저장하고, 이진 예측값(y_pred) 업데이트
    self.result['best_threshold'] = best_threshold
    self.data.y_pred = (y_pred_proba >= best_threshold).astype(int)

    # 최종적으로 이진 예측값으로 분류기 성능 지표 계산
    evaluate_classifier(self)


def run_shap_analysis(self):
    """
    SHAP 분석을 실행합니다. KernelExplainer를 사용하여 전체 요약 플롯과
    개별 샘플에 대한 상세 waterfall 플롯을 생성합니다.
    """
    print("\nINFO: SHAP 고급 분석을 시작합니다 (KernelExplainer 사용. 매우 오래 걸릴 수 있음)...")
    try:
        path = self.config['result_folder_path']

        # 1. 배경 데이터 요약 (Explainer 생성을 위해 한 번만 실행)
        X_train_2d = self.data.X_train.reshape(self.data.X_train.shape[0], -1)
        background_data_summary = shap.kmeans(X_train_2d, 50).data  # 배경 데이터는 X_train에서 가져옴

        # 2. KernelExplainer에 맞는 예측 함수(f) 정의
        def f(x):
            # x의 형태는 (샘플 수, 평탄화된 피처 수) -> (샘플 수, 시퀀스 길이, 피처 수)로 변환
            original_shape = (-1, self.data.X_train.shape[1], self.data.X_train.shape[2])
            x_3d = x.reshape(original_shape)
            return self.model.predict(x_3d)

        # 3. KernelExplainer 생성
        explainer = shap.KernelExplainer(f, background_data_summary)
        print("SHAP Explainer 생성이 완료되었습니다.")

        # A. 전체 경향성 분석 (Summary Plot)

        print("\n[A] 전체 경향성 분석(Summary Plot)을 시작합니다...")
        summary_samples = shap.sample(self.data.X_test, 50)  # 50개 샘플로 경향성 분석
        summary_samples_2d = summary_samples.reshape(summary_samples.shape[0], -1)

        summary_shap_values = explainer.shap_values(summary_samples_2d)

        # 이진 분류 모델 (sigmoid 출력)의 경우 shap_values는 보통 리스트 [클래스0_shap_values, 클래스1_shap_values]
        # 양성 클래스(1)에 대한 SHAP 값을 선택
        shap_values_for_summary = summary_shap_values[1] if isinstance(summary_shap_values,
                                                                       list) else summary_shap_values

        # 3D SHAP 값을 시간 축으로 평균
        shap_values_3d = shap_values_for_summary.reshape(summary_samples.shape)  # (50, seq_len, features)
        mean_shap_values = shap_values_3d.mean(axis=1)  # (50, features)

        # mean_test_sample_for_plot도 시간 축으로 평균
        mean_test_sample_for_plot = summary_samples.mean(axis=1)  # (50, features)

        # summary_plot은 DataFrame을 직접 받지 않으므로 NumPy 배열로 전달
        # Pandas DataFrame으로 변환하여 컬럼명을 연결
        summary_feature_names = self.data.feature_names  # 기본 feature names (15개)

        plt.figure(figsize=(10, 7))
        # summary_plot에 feature_names를 직접 전달
        shap.summary_plot(mean_shap_values, mean_test_sample_for_plot, feature_names=summary_feature_names, show=False,
                          max_display=20)
        plt.title('SHAP Feature Importance Summary (KernelExplainer, Time-averaged)')
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'shap_summary_plot_kernel.png'))
        plt.close()
        print("전체 요약 플롯 저장이 완료되었습니다.")

        # B. 개별 샘플 심층 분석 (Waterfall Plot) <- 코드 수정 필요 ..

        # 분석하고 싶은 샘플의 인덱스를 지정 (예: 테스트 데이터의 0, 1, 2번째 샘플)
        indices_to_analyze = [0, 1, 2]  # 원하는 인덱스로 변경 가능

        # SHAP expected_value (base_value)는 explainer에서 가져옵니다.
        # KernelExplainer의 expected_value도 리스트일 수 있으므로 긍정 클래스 값을 선택
        expected_value_for_waterfall = explainer.expected_value[1] if isinstance(explainer.expected_value,
                                                                                 list) else explainer.expected_value

        for index in indices_to_analyze:
            print(f"\n[B-{index}] {index}번째 테스트 샘플에 대한 개별 분석(Waterfall Plot)을 시작합니다...")

            # 단일 샘플 추출 (3D 형태)
            sample_to_analyze = self.data.X_test[index:index + 1]  # shape (1, seq_len, features)

            # KernelExplainer에 맞게 2D로 평탄화
            sample_to_analyze_2d = sample_to_analyze.reshape(1, -1)  # shape (1, seq_len * features)

            # 단일 샘플에 대한 SHAP 값 계산
            individual_shap_values = explainer.shap_values(sample_to_analyze_2d)

            # SHAP 값 처리 (긍정 클래스 선택)
            shap_values_for_plot_single = individual_shap_values[1] if isinstance(individual_shap_values,
                                                                                  list) else individual_shap_values

            # Waterfall plot에 사용할 최종 SHAP 값과 피처 값 (모두 1D 배열)
            # shap_values_for_plot_single은 (1, 평탄화된_피처_수) 형태일 수 있으므로 [0]으로 첫 번째 샘플 선택
            final_shap_values_single = shap_values_for_plot_single[0]  # (평탄화된_피처_수,)
            final_feature_values_single = sample_to_analyze_2d[0]  # (평탄화된_피처_수,)

            # 🔑 핵심 수정: Waterfall Plot을 위한 피처 이름 리스트 재구성 (Flattened)
            # 전체 시퀀스(data_duration)의 각 월별 피처 이름을 생성
            flattened_feature_names = [
                f"{name}_월{t + 1}"
                for t in range(self.data.X_test.shape[1])  # 시퀀스 길이
                for name in self.data.feature_names  # 원본 피처 이름
            ]

            # shap.Explanation 객체 생성
            explanation = shap.Explanation(
                values=final_shap_values_single,  # 1D SHAP 값
                base_values=expected_value_for_waterfall,  # explainer의 기대값
                data=final_feature_values_single,  # 1D 피처 값
                feature_names=flattened_feature_names
            )

            plt.figure(figsize=(12, 8))


            shap.waterfall_plot(explanation, max_display=20, show=False)

            # 샘플 ID 명시 (self.data.name_test에서 가져옴)
            sample_id_text = self.data.name_test[index] if len(self.data.name_test) > index else f"Sample #{index}"
            plt.title(f'SHAP Waterfall Plot for Test Sample ID: {sample_id_text}')
            plt.tight_layout()

            file_path = os.path.join(path, f'shap_waterfall_plot_sample_{index}.png')
            plt.savefig(file_path)
            plt.close()
            print(f"개별 분석 플롯이 '{file_path}'에 저장되었습니다.")

    except Exception as e:
        print(f" 경고: SHAP 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        self.result['shap_analysis_error'] = str(e)