from get_RandomForestClassifier import *
from evaluate_model import *
from get_AdaBoostClassifier import *
from get_ExtraTreesClassifier import *
from get_LinearClassifier import *
from get_XGBClassifier import *
from get_SVC import *
import os
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sys


class Model:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.model = None
        self.models_hyperparameters = None
        self.result = dict()
        
        # 한글 폰트 설정
        self._set_korean_font()

    def _set_korean_font(self):
        # 운영체제별 폰트 설정
        if sys.platform.startswith('win'):
            # 윈도우용 폰트
            fonts = ['Malgun Gothic', '맑은 고딕', 'NanumGothic', '나눔고딕']
            for font in fonts:
                if font in [f.name for f in fm.fontManager.ttflist]:
                    plt.rcParams['font.family'] = font
                    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
                    return True
        elif sys.platform.startswith('darwin'):
            # macOS용 폰트
            for font in ['AppleGothic', 'Apple Gothic', 'Nanum Gothic']:
                if font in [f.name for f in fm.fontManager.ttflist]:
                    plt.rcParams['font.family'] = font
                    plt.rcParams['axes.unicode_minus'] = False
                    return True
        elif sys.platform.startswith('linux'):
            # 리눅스용 폰트
            for font in ['NanumGothic', 'NanumBarunGothic']:
                if font in [f.name for f in fm.fontManager.ttflist]:
                    plt.rcParams['font.family'] = font
                    plt.rcParams['axes.unicode_minus'] = False
                    return True
        
        # 폰트를 찾지 못한 경우
        print("Warning: 한글 폰트를 찾을 수 없습니다. 그래프와 결과에 한글이 깨질 수 있습니다.")
        return False

    def make_model(self):
        if self.config['model_type'] == 'RandomForestClassifier':
            get_RandomForestClassifier(self)
        elif self.config['model_type'] == 'AdaBoostClassifier':
            get_AdaBoostClassifier(self)
        elif self.config['model_type'] == 'ExtraTreesClassifier':
            get_ExtraTreesClassifier(self)
        elif self.config['model_type'] == 'RidgeClassifier':
            get_RidgeClassifier(self)
        elif self.config['model_type'] == 'SGDClassifier':
            get_SGDClassifier(self)
        elif self.config['model_type'] == 'XGBClassifier':
            get_XGBClassifier(self)
        elif self.config['model_type'] == 'SVC':
            get_SVC(self)
        else:
            print(f"Error: config['model_type'] '{self.config['model_type']}' was not found.")
        print(f"Model {self.config['model_type']} has been trained successfully")

    def evaluate_model(self):
        evaluate_classifier(self)
        print('Model has been evaluated successfully')

    def save_result(self):
        # 예측값과 실제값 변환 (Series → 문자열)
        test_labels = pd.Series(
            np.where(self.data.df_y_test == 0, 'Normal', 'Caution'),
            index=self.data.df_y_test.index,
            name='true_label')
        pred_labels = pd.Series(
            np.where(self.data.df_y_pred == 0, 'Normal', 'Caution'),
            index=self.data.df_y_pred.index,
            name='predicted_label')
            
        # 예측 정확성 추가
        correct_prediction = pd.Series(
            self.data.df_y_test == self.data.df_y_pred,
            index=self.data.df_y_test.index,
            name='is_correct')
            
        # 두 Series를 하나의 DataFrame으로 합치기
        df_predict_result = pd.concat([self.data.df_x_test, test_labels, pred_labels, correct_prediction], axis=1)
        test_company_list = list(self.data.df_x_test['company_id'])
        rows = []
        for company_id in test_company_list:
            company = self.data.company_dict[company_id]
            row = {"company_id": company_id,
                   "start_date": company.date_range[0],
                   "end_date": company.date_range[-1]}
            rows.append(row)
        df_test_company_results = pd.DataFrame(rows)
        df_final_result = pd.merge(df_predict_result, df_test_company_results, on='company_id')
        df_final_result = df_final_result.sort_values(by='company_id').reset_index(drop=True)

        # 기본 결과 저장
        df_final_result.to_excel(self.config['result_folder_path'] + '/predict result.xlsx', index=True)

        # 모델 결과 저장
        model_result_df = pd.json_normalize(self.result, sep='_').transpose()
        model_result_df.to_excel(self.config['result_folder_path'] + '/model result.xlsx', index=True)

        # 결과 요약 저장
        summary_path = os.path.join(self.config['result_folder_path'], 'result_summary.txt')
        with open(summary_path, 'w', encoding='utf-8-sig') as f:
            f.write(f"===== {self.config['model_type']} 모델 예측 결과 요약 =====\n\n")
            
            # 성능 지표
            f.write("1. 성능 지표\n")
            f.write(f"   - 정확도(Accuracy): {self.result['accuracy']:.4f}\n")
            f.write(f"   - 정밀도(Precision): {self.result['precision']:.4f}\n")
            f.write(f"   - 재현율(Recall): {self.result['recall']:.4f}\n")
            f.write(f"   - F1 점수: {self.result['f1_score']:.4f}\n\n")
            
            # 예측 결과 요약
            f.write("2. 예측 결과 요약\n")
            f.write(f"   - 전체 테스트 샘플: {self.result['total_count']}개\n")
            f.write(f"   - 맞은 예측: {self.result['correct_count']}개 ({self.result['correct_count']/self.result['total_count']:.2%})\n")
            f.write(f"   - 틀린 예측: {self.result['wrong_count']}개 ({self.result['wrong_count']/self.result['total_count']:.2%})\n\n")
            
            # 클래스별 성능
            f.write("3. 클래스별 성능\n")
            f.write(f"   - 거래중(Normal) 클래스: {self.result['class_0_total']}개 중 {self.result['class_0_correct']}개 맞춤 ({self.result['class_0_accuracy']:.2%})\n")
            f.write(f"   - 경영악화(Caution) 클래스: {self.result['class_1_total']}개 중 {self.result['class_1_correct']}개 맞춤 ({self.result['class_1_accuracy']:.2%})\n\n")
            
            # 혼동 행렬
            cm = self.result['confusion_matrix']
            f.write("4. 혼동 행렬\n")
            f.write(f"            | 예측: Normal | 예측: Caution\n")
            f.write(f"   실제: Normal  | {cm[0][0]}          | {cm[0][1]}\n")
            f.write(f"   실제: Caution | {cm[1][0]}          | {cm[1][1]}\n\n")
            
            f.write(f"===== 저장 시간: {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        
        print(f"\n결과 요약 저장 완료: {summary_path}")

        # 모델 하이퍼파라미터 저장
        try:
            models_hyperparameters_df = pd.json_normalize(self.models_hyperparameters, sep='_').transpose()
            models_hyperparameters_df.to_excel(self.config['result_folder_path'] + '/models hyperparameter.xlsx', index=True)
        except Exception as e:
            print(f"Error: 모델 하이퍼파라미터 저장 중 오류 발생: {str(e)}")

