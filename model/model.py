import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from get_RandomForestClassifier import get_RandomForestClassifier
from get_AdaBoostClassifier import get_AdaBoostClassifier
from get_ExtraTreesClassifier import get_ExtraTreesClassifier
from get_LinearClassifier import get_RidgeClassifier, get_SGDClassifier
from get_XGBClassifier import get_XGBClassifier
from get_SVC import get_SVC
from get_ConvLSTM import get_ConvLSTM
from get_Transformer import get_Transformer
from evaluate_model import evaluate_classifier, evaluate_keras_model ,run_shap_analysis

class Model:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.model = None
        self.history = None
        self.models_hyperparameters = None
        self.result = dict()
        self._set_korean_font()

    def _set_korean_font(self):
        if sys.platform.startswith('win'):
            font_name = 'Malgun Gothic'
        elif sys.platform.startswith('darwin'):
            font_name = 'AppleGothic'
        else:
            font_name = 'NanumGothic'
        if font_name in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams['font.family'] = font_name
        plt.rcParams['axes.unicode_minus'] = False

    def make_model(self):
        model_type = self.config['model_type']
        print(f"\n=== '{model_type}' 모델 생성 및 학습 시작 ===")

        model_function_map = {
            'RandomForestClassifier': get_RandomForestClassifier,
            'AdaBoostClassifier': get_AdaBoostClassifier,
            'ExtraTreesClassifier': get_ExtraTreesClassifier,
            'RidgeClassifier': get_RidgeClassifier,
            'SGDClassifier': get_SGDClassifier,
            'XGBClassifier': get_XGBClassifier,
            'SVC': get_SVC,
            'ConvLSTM': get_ConvLSTM,
            'Transformer': get_Transformer,
        }

        if model_type in model_function_map:
            model_function_map[model_type](self)
        else:
            raise ValueError(f"지원하지 않는 모델 타입입니다: '{model_type}'")

    def evaluate_model(self):
        model_type = self.config['model_type']
        print(f"\n=== '{model_type}' 모델 성능 평가 시작 ===")
        if model_type in ['ConvLSTM', 'Transformer']:
            evaluate_keras_model(self)
        else:
            evaluate_classifier(self)

        # 3D 모델(Keras)이고, config [run_shap_analysis] = True 일 경우에만 실행
        # "ConvLSTM" 같은 경우에는 모델에 맞춰서 다시 함수 생성해야함
        if self.config['run_shap_analysis'] and model_type in ['Transformer']:
            run_shap_analysis(self)
        elif self.config['run_shap_analysis']:
            print("\n경고: SHAP 분석은 'Transformer'에서만 지원되므로, 현재 모델에서는 분석을 건너뜁니다.")

    def save_result(self):
        print(f"\n=== 결과 저장 시작 ===")
        path = self.config['result_folder_path']

        # 혼동 행렬 이미지 저장
        if self.config['save_confusion_matrix']:
            plt.figure(figsize=(8, 6))
            sns.heatmap(self.result['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                        xticklabels=['경영악화(1)', '거래중(0)'], yticklabels=['경영악화(1)', '거래중(0)'])
            plt.title(f'{self.config["model_type"]} 혼동 행렬');
            plt.xlabel('예측');
            plt.ylabel('실제')
            plt.savefig(os.path.join(path, 'confusion_matrix.png'));
            plt.close()

        # 성능 지표 엑셀 저장
        if self.config['save_metrics']:
            pd.DataFrame.from_dict(self.result, orient='index', columns=['value']).to_excel(
                os.path.join(path, 'metrics.xlsx'))

        # 상세 예측 결과 엑셀 저장
        if self.config['save_predictions']:
            df_pred = pd.DataFrame({
                'id': self.data.name_test,
                'true_label': self.data.y_test,
                'predicted_label': self.data.y_pred
            })
            if self.data.y_pred_proba is not None:
                df_pred['predicted_probability'] = self.data.y_pred_proba
            df_pred.to_excel(os.path.join(path, 'predictions.xlsx'), index=False)

        print(" 모든 결과 저장 완료.")