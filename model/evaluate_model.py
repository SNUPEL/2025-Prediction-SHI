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

# 한글 폰트 설정
def set_korean_font():
    # 윈도우의 경우 기본 한글 폰트
    fonts = ['Malgun Gothic', '맑은 고딕', 'NanumGothic', '나눔고딕']
    
    # 시스템 확인
    if sys.platform.startswith('win'):
        # 윈도우용 폰트
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
    print("Warning: 한글 폰트를 찾을 수 없습니다. 그래프에 한글이 깨질 수 있습니다.")
    return False

# 한글 폰트 설정 적용
set_korean_font()

def evaluate_classifier(self):
    # 기본 평가 지표 계산
    self.result['accuracy'] = accuracy_score(self.data.df_y_test, self.data.df_y_pred)
    self.result['precision'] = precision_score(self.data.df_y_test, self.data.df_y_pred)
    self.result['recall'] = recall_score(self.data.df_y_test, self.data.df_y_pred)
    self.result['f1_score'] = f1_score(self.data.df_y_test, self.data.df_y_pred)
    
    # 혼동 행렬 계산
    self.result['confusion_matrix'] = confusion_matrix(self.data.df_y_test, self.data.df_y_pred).tolist()
    cm = self.result['confusion_matrix']
    
    # 분류 결과 상세 분석
    y_test = self.data.df_y_test.values
    y_pred = self.data.df_y_pred.values
    
    # 맞은 개수와 틀린 개수 계산
    self.result['correct_count'] = np.sum(y_test == y_pred)
    self.result['wrong_count'] = np.sum(y_test != y_pred)
    self.result['total_count'] = len(y_test)
    
    # 클래스별 분석
    self.result['class_0_total'] = np.sum(y_test == 0)  # 거래중(0) 총 개수
    self.result['class_1_total'] = np.sum(y_test == 1)  # 경영악화(1) 총 개수
    
    self.result['class_0_correct'] = np.sum((y_test == 0) & (y_pred == 0))  # 거래중 맞춘 개수
    self.result['class_1_correct'] = np.sum((y_test == 1) & (y_pred == 1))  # 경영악화 맞춘 개수
    
    # 클래스별 정확도
    if self.result['class_0_total'] > 0:
        self.result['class_0_accuracy'] = self.result['class_0_correct'] / self.result['class_0_total']
    else:
        self.result['class_0_accuracy'] = 0
        
    if self.result['class_1_total'] > 0:
        self.result['class_1_accuracy'] = self.result['class_1_correct'] / self.result['class_1_total']
    else:
        self.result['class_1_accuracy'] = 0
    
    # 상세 결과 출력 (config 설정에 따라)
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
        print(f"거래중(0) 클래스: {self.result['class_0_total']}개 중 {self.result['class_0_correct']}개 맞춤 ({self.result['class_0_accuracy']:.2%})")
        print(f"경영악화(1) 클래스: {self.result['class_1_total']}개 중 {self.result['class_1_correct']}개 맞춤 ({self.result['class_1_accuracy']:.2%})")
        
        print("\n----- 혼동 행렬 -----")
        print(f"        | 예측: 거래중(0) | 예측: 경영악화(1)")
        print(f"실제: 거래중(0)  | {cm[0][0]}            | {cm[0][1]}")
        print(f"실제: 경영악화(1) | {cm[1][0]}            | {cm[1][1]}")
        print("==========================\n")
    else:
        print(f"Accuracy: {self.result['accuracy']}, Precision: {self.result['precision']}")
    
    # 혼동 행렬 시각화 및 저장
    if self.config['save_confusion_matrix']:
        # 한글 폰트 적용 확인
        has_korean_font = set_korean_font()
        
        plt.figure(figsize=(8, 6))
        classes = ['거래중(0)', '경영악화(1)']
        cm_display = confusion_matrix(self.data.df_y_test, self.data.df_y_pred)
        
        # 시각화 개선
        ax = sns.heatmap(cm_display, annot=True, fmt='d', cmap='Blues', 
                     xticklabels=classes, yticklabels=classes, 
                     annot_kws={"size": 14, "weight": "bold"})
        
        # 텍스트 크기 및 스타일 설정
        ax.set_xlabel('예측', fontsize=12, fontweight='bold')
        ax.set_ylabel('실제', fontsize=12, fontweight='bold')
        ax.set_title(f'{self.config["model_type"]} 혼동 행렬', fontsize=14, fontweight='bold')
        
        # 테두리 추가
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)
            
        plt.tight_layout()
        
        # 저장 - DPI 높여서 해상도 개선
        cm_path = os.path.join(self.config['result_folder_path'], 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # 예측 결과 상세 저장
    if self.config['save_predictions']:
        # 테스트 데이터와 예측 결과 병합
        predictions = pd.DataFrame({
            'company_id': self.data.df_x_test['company_id'],
            'actual': self.data.df_y_test,
            'predicted': self.data.df_y_pred,
            'correct': self.data.df_y_test == self.data.df_y_pred
        })
        
        # 레이블을 읽기 쉬운 형태로 변환
        predictions['actual_label'] = predictions['actual'].map({0: '거래중', 1: '경영악화'})
        predictions['predicted_label'] = predictions['predicted'].map({0: '거래중', 1: '경영악화'})
        
        # CSV 파일로 저장
        pred_path = os.path.join(self.config['result_folder_path'], 'predictions_detail.csv')
        predictions.to_csv(pred_path, index=False, encoding='utf-8-sig')
    
    # 성능 지표 저장
    if self.config['save_metrics']:
        metrics_to_save = {k: v for k, v in self.result.items() if not isinstance(v, list)}  # 리스트 형태의 결과 제외
        metrics_df = pd.DataFrame([metrics_to_save])
        metrics_path = os.path.join(self.config['result_folder_path'], 'detailed_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')


def evaluate_regressor(self):
    self.result['MSE'] = mean_squared_error(self.data.df_y_test, self.data.df_y_pred)
    self.result['RMSE'] = np.sqrt(self.result['MSE'])
    self.result['MAE'] = mean_absolute_error(self.data.df_y_test, self.data.df_y_pred)
    self.result['MAPE'] = mean_absolute_percentage_error(self.data.df_y_test, self.data.df_y_pred)
