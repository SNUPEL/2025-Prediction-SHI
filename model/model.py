from get_RandomForestClassifier import *
from evaluate_model import *
from get_AdaBoostClassifier import *
from get_AdaBoostClassifier_optimized import *
from get_ExtraTreesClassifier import *
from get_LinearClassifier import *
from get_XGBClassifier import *
from get_SVC import *
from get_ConvLSTM import *
from get_MultiChannelCNNLSTM import *
from get_AutoEncoder import *
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import sys


class Model:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.model = None
        self.history = None
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
        # --- 모델 선택 및 학습 ---
        if self.config['model_type'] == 'RandomForestClassifier':
            get_RandomForestClassifier(self)
        elif self.config['model_type'] == 'AdaBoostClassifier':
            get_AdaBoostClassifier(self)
            # get_AdaBoostClassifier_optimized(self)
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
        elif self.config['model_type'] == 'nuSVC':
            get_nuSVC(self)
        elif self.config['model_type'] == 'ConvLSTM':
            get_ConvLSTM(self)
        elif self.config['model_type'] == 'MultiChannelCNNLSTM':
            get_MultiChannelCNNLSTM(self)
        elif self.config['model_type'] == 'AutoEncoder':
            get_AutoEncoder(self)
        else:
            print(f"Error: config['model_type'] '{self.config['model_type']}' was not found.")

        if self.config['save_train_data']:
            # --- 학습 입력 데이터 엑셀 파일 저장 ---
            print(f"==== 학습 데이터 저장 시작 ====")
            if self.config['undersampling'] or self.config['oversampling']:
                matrix_dict_to_save = self.data.df_x_train_matrix_dict_after_sampling
                labels_dict_to_save = self.data.df_y_train_dict_after_sampling
            else:
                matrix_dict_to_save = self.data.df_x_train_matrix_dict
                labels_dict_to_save = self.data.df_y_train_dict

            matrix_train_folder_path = os.path.join(self.config['result_folder_path'], 'train_data_matrix')
            os.makedirs(matrix_train_folder_path, exist_ok=True)

            labels_data = []
            for sample_name, df_sample in matrix_dict_to_save.items():
                sample_label = labels_dict_to_save[sample_name]

                file_name_excel = f"{sample_name}_label_{int(sample_label)}.xlsx"

                df_sample.to_excel(os.path.join(matrix_train_folder_path, file_name_excel), index=True)

                labels_data.append({
                    'filename': file_name_excel,
                    'company_id_date': sample_name,
                    'label': int(sample_label)
                })

            # 라벨 정보를 담은 CSV 파일 저장
            labels_csv_path = os.path.join(matrix_train_folder_path, 'train_data_labels.csv')
            labels_df = pd.DataFrame(labels_data)
            labels_df.to_csv(labels_csv_path, index=False, encoding='utf-8-sig')
            print(f"    총 {len(matrix_dict_to_save)}개 데이터 및 라벨 저장")
            print(f"==== 학습 데이터 저장 완료 ====")

    def evaluate_model(self):
        evaluate_classifier(self)

    def save_result(self):
        # 한글 폰트 설정
        self._set_korean_font()

        print("\n===== 모델 성능 평가 결과 요약 =====")
        print(f"  - 정확도(Accuracy): {self.result['accuracy']:.4f}")
        print(f"  - 정밀도(Precision): {self.result['precision']:.4f}")
        print(f"  - 재현율(Recall): {self.result['recall']:.4f}")
        print(f"  - F1 점수: {self.result['f1_score']:.4f}")
        print("\n--- 혼동 행렬 ---")
        print(f"        | 예측: 경영악화(1) | 예측: 거래중(0)")
        print(f"실제: 경영악화(1) | {self.result['TP']:<15} | {self.result['FN']}")
        print(f"실제: 거래중(0)  | {self.result['FP']:<15} | {self.result['TN']}")
        print("==================================\n")

        df_pred = self.data.df_y_pred.rename(columns={'label': 'predicted_label'})
        df_true = self.data.df_y_test.rename(columns={'label': 'true_label'})

        df_final_result = pd.concat([df_true, df_pred], axis=1)
        df_final_result['is_correct'] = (df_final_result['true_label'] == df_final_result['predicted_label'])
        df_final_result.index.name = 'company_id_date'

        df_final_result.to_excel(os.path.join(self.config['result_folder_path'], 'predict_result.xlsx'), index=True)

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

        result_df = pd.json_normalize(self.result, sep='_').transpose()
        result_df.to_excel(self.config['result_folder_path'] + '/specific result.xlsx', index=True)

        if self.config['save_loss_history'] and self.history is not None:
            plt.figure(figsize=(10, 5))
            plt.plot(self.history['train_loss'], label='Train Loss')
            plt.plot(self.history['val_loss'], label='Validation Loss')
            plt.title('Learning Curves')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.config['result_folder_path'], 'loss_curves.png'))
            plt.close()

        if self.config['save_confusion_matrix']:
            plt.figure(figsize=(8, 6))
            ax = sns.heatmap(np.array(self.result['confusion_matrix']), annot=True, fmt='d', cmap='Blues',
                             xticklabels=['경영악화(1)', '거래중(0)'], yticklabels=['경영악화(1)', '거래중(0)'])
            ax.set_xlabel('예측된 라벨', fontsize=12)
            ax.set_ylabel('실제 라벨', fontsize=12)
            ax.set_title(f'{self.config["model_type"]} Confusion Matrix', fontsize=14)
            plt.savefig(os.path.join(self.config['result_folder_path'], 'confusion_matrix.png'), dpi=300)
            plt.close()

        if self.config['save_test_data']:
            print("\n==== 테스트 데이터 저장 시작 ====")
            # 저장에 필요한 데이터 로드
            matrix_dict_to_save = self.data.df_x_test_matrix_dict
            true_labels = self.data.df_y_test['label']
            pred_labels = self.data.df_y_pred['label']

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

        print("\n===== 모든 평가 결과 저장 완료 =====")
