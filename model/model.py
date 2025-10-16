# RandomForest 모델을 생성하고 학습시키는 헬퍼 함수 모음
from get_RandomForestClassifier import *
# 기본 AdaBoost 모델 빌더 함수
from get_AdaBoostClassifier import *
# 최적화된 AdaBoost 모델 구성 함수
from get_AdaBoostClassifier_optimized import *
# ExtraTrees 분류 모델 학습 함수
from get_ExtraTreesClassifier import *
# 선형 분류기(SGD 등) 학습 함수
from get_LinearClassifier import *
# Ridge 분류기 최적화 함수
from get_RidgeClassifier_optimized import *
# XGBoost 분류기 학습 함수
from get_XGBClassifier import *
# 커널 SVC 모델 학습 함수
from get_SVC import *
# 하이퍼파라미터 최적화된 SVC 학습 함수
from get_SVC_optimized import *
# 앙상블 투표 분류기 학습 함수
from get_VotingClassifier import *
# ConvLSTM 기반 시계열 모델 학습 함수
from get_ConvLSTM import *
# 다중 채널 CNN-LSTM 모델 학습 함수
from get_MultiChannelCNNLSTM import *
# Transformer 기반 모델 학습 함수
from get_Transformer import *
# ResNet 기반 분류 모델 학습 함수
from get_ResNet import *
# MLP-Mixer 구조 학습 함수
from get_MLP_Mixer import *
# AutoEncoder 모델 학습 함수
from get_AutoEncoder import *
# 공통 평가 함수
from evaluate_model import *
# SHAP 해석기 생성 함수
from get_SHAP_explainer import *
# 파일 경로 조작 용도
import os
# 시각화 생성 라이브러리
import matplotlib.pyplot as plt
# 시스템 폰트 탐색 라이브러리
import matplotlib.font_manager as fm
# 통계형 시각화 스타일
import seaborn as sns
# 플랫폼 판별을 위해 필요
import sys


class Model:
    def __init__(self, config, data):
        # 전체 파이프라인 설정 값을 그대로 보관
        self.config = config
        # 데이터 래퍼 객체를 그대로 보관
        self.data = data
        # 학습된 모델 인스턴스를 담을 자리
        self.model = None
        # 딥러닝 학습 시 손실 곡선을 담을 자리
        self.history = None
        # 평가 결과와 계산된 지표를 보관할 딕셔너리
        self.result = dict()
        
        # 한글 그래프 출력을 위해 폰트를 미리 세팅
        self._set_korean_font()

    def _set_korean_font(self):
        # 실행 플랫폼에 따라 폰트 후보를 다르게 조회
        if sys.platform.startswith('win'):
            # Windows 환경에서 우선 적용할 한글 폰트 목록
            fonts = ['Malgun Gothic', '맑은 고딕', 'NanumGothic', '나눔고딕']
            for font in fonts:
                # 폰트 매니저에 해당 폰트가 등록되어 있는지 확인
                if font in [f.name for f in fm.fontManager.ttflist]:
                    # 그래프 전역 폰트를 설정하고
                    plt.rcParams['font.family'] = font
                    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
                    return True
        elif sys.platform.startswith('darwin'):
            # macOS 환경에서 사용할 수 있는 한글 폰트 목록
            for font in ['AppleGothic', 'Apple Gothic', 'Nanum Gothic']:
                if font in [f.name for f in fm.fontManager.ttflist]:
                    # macOS에서 사용할 폰트를 설정
                    plt.rcParams['font.family'] = font
                    plt.rcParams['axes.unicode_minus'] = False
                    return True
        elif sys.platform.startswith('linux'):
            # Linux 배포판에서 자주 사용하는 나눔 폰트 목록
            for font in ['NanumGothic', 'NanumBarunGothic']:
                if font in [f.name for f in fm.fontManager.ttflist]:
                    # 그래프 폰트를 지정하고 마이너스 표시 깨짐 방지
                    plt.rcParams['font.family'] = font
                    plt.rcParams['axes.unicode_minus'] = False
                    return True
        
        # 위의 모든 후보를 찾지 못하면 경고 메시지를 출력
        print("Warning: 한글 폰트를 찾을 수 없습니다. 그래프와 결과에 한글이 깨질 수 있습니다.")
        return False

    def make_model(self):
        # 구성 파일에 지정된 모델 타입에 따라 분기한다
        if self.config['model_type'] == 'RandomForestClassifier':
            # 랜덤 포레스트 분류기를 학습
            get_RandomForestClassifier(self)
        elif self.config['model_type'] == 'AdaBoostClassifier':
            # 기본 AdaBoost 분류기를 학습
            get_AdaBoostClassifier(self)
            # get_AdaBoostClassifier_optimized(self)
        elif self.config['model_type'] == 'ExtraTreesClassifier':
            # ExtraTrees 분류기를 학습
            get_ExtraTreesClassifier(self)
        elif self.config['model_type'] == 'RidgeClassifier':
            # Ridge 분류기를 학습
            get_RidgeClassifier(self)
            # get_RidgeClassifier_optimized(self)
        elif self.config['model_type'] == 'SGDClassifier':
            # SGD 기반 선형 분류기를 학습
            get_SGDClassifier(self)
        elif self.config['model_type'] == 'XGBClassifier':
            # XGBoost 분류기를 학습
            get_XGBClassifier(self)
        elif self.config['model_type'] == 'SVC':
            # 커널 SVC를 학습
            get_SVC(self)
            # get_SVC_optimized(self)
        elif self.config['model_type'] == 'nuSVC':
            # nu-SVC 변형 모델을 학습
            get_nuSVC(self)
            # get_nuSVC_optimized(self)
        elif self.config['model_type'] == 'VotingClassifier':
            # 하위 모델 조합을 투표 방식으로 학습
            get_VotingClassifier(self, self.config['model_parameter']['voting'])
        elif self.config['model_type'] == 'ConvLSTM':
            # ConvLSTM 시계열 모델 학습
            get_ConvLSTM(self)
        elif self.config['model_type'] == 'MultiChannelCNNLSTM':
            # 다중 채널 CNN-LSTM 모델 학습
            get_MultiChannelCNNLSTM(self)
        elif self.config['model_type'] == 'Transformer':
            # Transformer 기반 모델 학습
            get_Transformer(self)
        elif self.config['model_type'] == 'ResNet':
            # ResNet 기반 모델 학습
            get_ResNet(self)
        elif self.config['model_type'] == 'MLP_Mixer':
            # MLP-Mixer 모델 학습
            get_MLP_Mixer(self)
        elif self.config['model_type'] == 'AutoEncoder':
            # AutoEncoder 기반 모델 학습
            get_AutoEncoder(self)
        else:
            # 정의되지 않은 모델 타입은 경고 메시지를 출력
            print(f"Error: config['model_type'] '{self.config['model_type']}' was not found.")

        if self.config['save_train_data']:
            # 학습 입력 데이터를 엑셀로 남길지 여부 확인
            print(f"==== 학습 데이터 저장 시작 ====")
            # 샘플링을 수행했다면 조정된 데이터를, 아니면 원본 데이터를 사용
            if self.config['undersampling'] or self.config['oversampling']:
                matrix_dict_to_save = self.data.df_x_train_matrix_dict_after_sampling
                labels_dict_to_save = self.data.df_y_train_dict_after_sampling
            else:
                matrix_dict_to_save = self.data.df_x_train_matrix_dict
                labels_dict_to_save = self.data.df_y_train_dict

            # 결과 폴더 아래 학습 데이터 저장 경로를 생성
            matrix_train_folder_path = os.path.join(self.config['result_folder_path'], 'train_data_matrix')
            os.makedirs(matrix_train_folder_path, exist_ok=True)

            # 라벨 메타 정보를 누적할 리스트 초기화
            labels_data = []
            # 각 샘플별로 데이터를 엑셀 파일로 기록
            for sample_name, df_sample in matrix_dict_to_save.items():
                # 현재 샘플 라벨을 조회
                sample_label = labels_dict_to_save[sample_name]

                # 라벨 정보를 파일명에 담아 구분 가능하게 만든다
                file_name_excel = f"{sample_name}_label_{int(sample_label)}.xlsx"

                # 특성 행렬을 엑셀 파일로 저장한다
                df_sample.to_excel(os.path.join(matrix_train_folder_path, file_name_excel), index=True)

                # 이후 CSV 요약 파일 작성을 위해 메타 정보를 저장
                labels_data.append({
                    'filename': file_name_excel,
                    'company_id_date': sample_name,
                    'label': int(sample_label)
                })

            # 라벨 정보를 담은 CSV 파일 저장
            labels_csv_path = os.path.join(matrix_train_folder_path, 'train_data_labels.csv')
            # 누적한 라벨 정보를 DataFrame으로 변환
            labels_df = pd.DataFrame(labels_data)
            # CSV 파일로 라벨 메타데이터를 저장
            labels_df.to_csv(labels_csv_path, index=False, encoding='utf-8-sig')
            print(f"    총 {len(matrix_dict_to_save)}개 데이터 및 라벨 저장")
            print(f"==== 학습 데이터 저장 완료 ====")

        if self.config['save_validation_data']:
            # 검증 입력 데이터를 별도로 보관하도록 요청된 경우
            print(f"==== 검증 데이터 저장 시작 ====")
            matrix_dict_to_save = self.data.df_x_valid_matrix_dict
            labels_dict_to_save = self.data.df_y_valid_dict

            # 검증 데이터 저장 경로 준비
            matrix_valid_folder_path = os.path.join(self.config['result_folder_path'], 'valid_data_matrix')
            os.makedirs(matrix_valid_folder_path, exist_ok=True)

            # 검증 라벨 메타 정보를 쌓을 리스트 초기화
            labels_data = []
            # 검증 데이터 셋 전체를 순회
            for sample_name, df_sample in matrix_dict_to_save.items():
                # 해당 샘플의 라벨을 가져온다
                sample_label = labels_dict_to_save[sample_name]

                # 검증 데이터도 라벨이 포함된 파일명으로 저장
                file_name_excel = f"{sample_name}_label_{int(sample_label)}.xlsx"

                # 특성 행렬을 엑셀 파일로 내보낸다
                df_sample.to_excel(os.path.join(matrix_valid_folder_path, file_name_excel), index=True)

                # CSV 라벨 요약 생성을 위한 정보 누적
                labels_data.append({
                    'filename': file_name_excel,
                    'company_id_date': sample_name,
                    'label': int(sample_label)
                })

            # 라벨 정보를 담은 CSV 파일 저장
            labels_csv_path = os.path.join(matrix_valid_folder_path, 'valid_data_labels.csv')
            # 검증 라벨 메타데이터를 DataFrame으로 정리
            labels_df = pd.DataFrame(labels_data)
            # CSV 파일로 내보내 추후 확인 가능하게 한다
            labels_df.to_csv(labels_csv_path, index=False, encoding='utf-8-sig')
            print(f"    총 {len(matrix_dict_to_save)}개 데이터 및 라벨 저장")
            print(f"==== 검증 데이터 저장 완료 ====")

    def evaluate_model(self):
        # 분류 성능 지표와 혼동 행렬을 계산
        evaluate_classifier(self)
        if self.config['shap_analysis']:
            # 사용자가 설정한 경우 SHAP 분석을 통해 피처 중요도를 도출
            get_SHAP_explainer(self)

    def save_result(self):
        # 저장 작업 직전에 다시 한번 폰트를 초기화
        self._set_korean_font()

        # 콘솔에 주요 성능 지표를 출력
        print("\n===== 모델 성능 평가 결과 요약 =====")
        print(f"  - 정확도(Accuracy): {self.result['accuracy']:.4f}")
        print(f"  - 정밀도(Precision): {self.result['precision']:.4f}")
        print(f"  - 재현율(Recall): {self.result['recall']:.4f}")
        print(f"  - F1 점수: {self.result['f1_score']:.4f}")
        print("\n--- 혼동 행렬 ---")
        # 혼동 행렬을 텍스트 표 형태로 정리
        print(f"        | 예측: 경영악화(1) | 예측: 거래중(0)")
        print(f"실제: 경영악화(1) | {self.result['TP']:<15} | {self.result['FN']}")
        print(f"실제: 거래중(0)  | {self.result['FP']:<15} | {self.result['TN']}")
        print("==================================\n")

        # 예측 결과 DataFrame의 열 이름을 이해하기 쉽게 변경
        df_pred = self.data.df_y_pred.rename(columns={'label': 'predicted_label'})
        # 실제 라벨도 동일한 규칙으로 정리
        df_true = self.data.df_y_test.rename(columns={'label': 'true_label'})
        # 예측 확률 DataFrame이 예측 라벨과 동일한 길이를 갖는지 확인
        if self.data.df_y_pred.shape[0] == self.data.df_y_pred_proba.shape[0]:
            # 예측 확률이 존재하는 경우 가독성을 위해 열 이름을 변경
            df_pred_proba = self.data.df_y_pred_proba.rename(columns={'label': 'probability'})

            df_final_result = pd.concat([df_true, df_pred, df_pred_proba], axis=1)
        else:
            df_final_result = pd.concat([df_true, df_pred], axis=1)
        # 실제 라벨과 예측 라벨이 일치하는지 여부를 부울 값으로 기록
        df_final_result['is_correct'] = (df_final_result['true_label'] == df_final_result['predicted_label'])
        # 고유 식별자로 사용할 인덱스명을 명시
        df_final_result.index.name = 'company_id_date'

        # 개별 샘플별 예측 결과를 엑셀로 저장
        df_final_result.to_excel(os.path.join(self.config['result_folder_path'], 'predict_result.xlsx'), index=True)

        # 텍스트 요약 파일 경로를 구성
        summary_path = os.path.join(self.config['result_folder_path'], 'result_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"===== {self.config['model_type']} 모델 예측 결과 요약 =====\n\n")
            f.write("1. 예측 결과 요약\n")
            f.write(f"   - 전체 테스트 샘플: {self.result['total_count']}개\n")
            f.write(f"   - 맞은 예측: {self.result['correct_count']}개 ({self.result['correct_count'] / self.result['total_count']:.2%})\n")
            f.write(f"   - 틀린 예측: {self.result['wrong_count']}개 ({self.result['wrong_count'] / self.result['total_count']:.2%})\n\n")
            f.write("2. 클래스별 성능\n")
            f.write(f"   - 경영악화(1) 클래스: {self.result['class_1_total']}개 중 {self.result['class_1_correct']}개 맞춤 ({self.result['recall']:.2%})\n")
            f.write(f"   - 거래중(0) 클래스: {self.result['class_0_total']}개 중 {self.result['class_0_correct']}개 맞춤 ({self.result['specificity']:.2%})\n\n")
            f.write("3. 성능 지표\n")
            f.write(f"   - 정확도(Accuracy): {self.result['accuracy']:.4f}\n")
            f.write(f"   - 정밀도(Precision): {self.result['precision']:.4f}\n")
            f.write(f"   - 재현율(Recall): {self.result['recall']:.4f}\n")
            f.write(f"   - F1 점수: {self.result['f1_score']:.4f}\n\n")
            f.write("4. 혼동 행렬\n")
            f.write(f"            | 예측: 경영악화(1) | 예측: 거래중(0)\n")
            f.write(f"   실제: 경영악화(1) | {self.result['TP']:<15} | {self.result['FN']}\n")
            f.write(f"   실제: 거래중(0)  | {self.result['FP']:<15} | {self.result['TN']}\n")

        # result 딕셔너리를 평탄화하여 개별 지표를 확인하기 쉽게 만든다
        result_df = pd.json_normalize(self.result, sep='_').transpose()
        # 세부 지표를 엑셀로 내보낸다
        result_df.to_excel(self.config['result_folder_path'] + '/specific result.xlsx', index=True)

        if self.config['save_loss_history'] and self.history is not None:
            # 학습 과정의 손실 변화를 그림으로 남긴다
            plt.figure(figsize=(10, 5))
            # 훈련 손실 변화를 선 그래프로 표시
            plt.plot(self.history['train_loss'], label='Train Loss')
            # 검증 손실 변화를 같은 축 위에 표시
            plt.plot(self.history['val_loss'], label='Validation Loss')
            plt.title('Learning Curves')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            # 학습 곡선을 이미지 파일로 저장
            plt.savefig(os.path.join(self.config['result_folder_path'], 'loss_curves.png'))
            # 메모리 확보를 위해 그림을 닫는다
            plt.close()

        if self.config['save_confusion_matrix']:
            # 혼동 행렬을 히트맵 이미지로 시각화
            plt.figure(figsize=(8, 6))
            ax = sns.heatmap(np.array(self.result['confusion_matrix']), annot=True, fmt='d', cmap='Blues',
                             xticklabels=['경영악화(1)', '거래중(0)'], yticklabels=['경영악화(1)', '거래중(0)'])
            ax.set_xlabel('예측된 라벨', fontsize=12)
            ax.set_ylabel('실제 라벨', fontsize=12)
            ax.set_title(f'{self.config["model_type"]} Confusion Matrix', fontsize=14)
            # 히트맵을 이미지로 저장
            plt.savefig(os.path.join(self.config['result_folder_path'], 'confusion_matrix.png'), dpi=300)
            # Figure 인스턴스를 정리
            plt.close()

        if self.config['save_test_data']:
            print("\n==== 테스트 데이터 저장 시작 ====")
            # 저장에 필요한 데이터 로드
            matrix_dict_to_save = self.data.df_x_test_matrix_dict
            true_labels = self.data.df_y_test['label']
            pred_labels = self.data.df_y_pred['label']

            # 테스트 데이터 저장 위치를 생성
            test_data_folder_path = os.path.join(self.config['result_folder_path'], 'test_data_matrix')
            os.makedirs(test_data_folder_path, exist_ok=True)

            for sample_name, df_sample in matrix_dict_to_save.items():
                # 각 샘플의 실제 라벨과 예측 라벨 가져오기
                true_label = true_labels.get(sample_name)
                pred_label = pred_labels.get(sample_name)

                # 파일명에 실제 라벨과 예측 라벨 모두 포함
                file_name_excel = f"{sample_name}_true_{int(true_label)}_pred_{int(pred_label)}.xlsx"
                file_path_excel = os.path.join(test_data_folder_path, file_name_excel)

                # 특성 이름(인덱스)을 포함하여 엑셀 파일로 저장
                df_sample.to_excel(file_path_excel, index=True)

            print(f"    총 {len(matrix_dict_to_save)}개 테스트 데이터 저장 완료")
            print(f"==== 테스트 데이터 저장 완료 ====")

        # 모든 결과 저장 프로세스가 완결되었음을 알린다
        print("\n===== 모든 평가 결과 저장 완료 =====")
