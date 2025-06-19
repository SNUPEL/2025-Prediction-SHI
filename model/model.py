from get_RandomForestClassifier import *
from evaluate_model import *
from get_AdaBoostClassifier import *
from get_AdaBoostClassifier_optimized import *
from get_ExtraTreesClassifier import *
from get_LinearClassifier import *
from get_XGBClassifier import *
from get_SVC import *
from get_ConvLSTM import *
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
        elif self.config['model_type'] == 'ConvLSTM':
            get_ConvLSTM(self)
        else:
            print(f"Error: config['model_type'] '{self.config['model_type']}' was not found.")

        # --- 학습 입력 데이터 엑셀 파일 저장 코드 (모델 학습 후) ---
        if self.model is not None:
            try:
                if self.config['data_shape'] == 'flatten':
                    # Flatten 모드: 폴더 없이 기존처럼 단일 엑셀 파일로 저장
                    train_data_excel_path = os.path.join(self.config['result_folder_path'], 'training_data.xlsx')

                    # self.data.X_train은 (샘플수, 총_피처수) 2D NumPy 배열
                    first_company_key = next(iter(self.data.company_dict))
                    feature_names_base = sorted(list(self.data.company_dict[first_company_key].data_dict.keys()))
                    data_duration = self.config['data_duration']

                    flattened_column_names = []
                    for d_idx in range(data_duration):
                        for f_name in feature_names_base:
                            flattened_column_names.append(f"{f_name}_{d_idx}")

                    df_X_train_to_save = pd.DataFrame(self.data.X_train, columns=flattened_column_names)
                    df_X_train_to_save['company_id'] = self.data.name_train
                    df_X_train_to_save['label'] = self.data.y_train  # 라벨도 함께 저장

                    df_X_train_to_save.to_excel(train_data_excel_path, index=False)
                    print(f"\n=== 학습 입력 데이터 (Flattened) 저장 완료: {train_data_excel_path} ===")

                elif self.config['data_shape'] == 'matrix':
                    # Matrix 모드: 'training_data_matrix' 폴더 생성 후 개별 .xlsx 파일 및 라벨 CSV 저장
                    matrix_train_folder_path = os.path.join(self.config['result_folder_path'], 'training_data_matrix')
                    os.makedirs(matrix_train_folder_path, exist_ok=True)  # 폴더 없으면 생성

                    # 라벨 정보를 저장하기 위한 CSV 파일 준비
                    labels_csv_path = os.path.join(matrix_train_folder_path, 'matrix_training_labels.csv')
                    labels_data = []

                    # self.data.X_train은 (샘플수, data_duration, num_features) 3D NumPy 배열이어야 함
                    for i in range(len(self.data.X_train)):
                        sample_matrix = self.data.X_train[i]  # (data_duration, num_features) 2D 배열
                        sample_label = self.data.y_train[i]
                        sample_name = self.data.name_train[i]  # company_id_date 형식

                        # 각 2D 행렬(NumPy 배열)을 Pandas DataFrame으로 변환 후 엑셀로 저장
                        # 엑셀 파일 내의 컬럼 이름을 정의 (필요에 따라)
                        first_company_key = next(iter(self.data.company_dict))
                        feature_names_base = sorted(list(self.data.company_dict[first_company_key].data_dict.keys()))

                        df_sample = pd.DataFrame(sample_matrix, columns=feature_names_base)

                        # 파일명에 company_id_date와 라벨 포함
                        file_name_excel = f"{sample_name}_label_{sample_label}.xlsx"
                        file_path_excel = os.path.join(matrix_train_folder_path, file_name_excel)

                        df_sample.to_excel(file_path_excel, index=False)  # 인덱스 없이 저장

                        # 라벨 CSV 파일에 기록할 데이터 수집
                        labels_data.append({
                            'filename': file_name_excel,
                            'company_id_date': sample_name,
                            'label': int(sample_label)  # bool이면 int로 변환
                        })

                    # 라벨 CSV 파일 저장
                    labels_df = pd.DataFrame(labels_data)
                    labels_df.to_csv(labels_csv_path, index=False, encoding='utf-8-sig')

                    print(f"\n=== 학습 입력 데이터 (Matrix, 개별 .xlsx 파일) 저장 완료: {matrix_train_folder_path} 내 ===")
                    print(f"    총 {len(self.data.X_train)}개 샘플 (.xlsx) 및 라벨 ({labels_csv_path}) 저장됨.")

                else:
                    print(f"경고: 알 수 없는 data_shape '{self.config['data_shape']}' 입니다. 학습 입력 데이터를 저장하지 않습니다.")

            except Exception as e:
                print(f"\n!!! 학습 입력 데이터 저장 중 오류 발생: {e} !!!")

            print(f"Model {self.config['model_type']} has been trained successfully")
        else:
            print("모델 학습이 제대로 완료되지 않아 입력 데이터를 저장하지 않습니다.")

    def evaluate_model(self):
        evaluate_classifier(self)
        print('Model has been evaluated successfully')

    def save_result(self):
        # 모든 Series에 사용할 공통 인덱스 (company_id_날짜)
        common_index = self.data.name_test

        # 실제/예측 레이블 및 정확성 Series 생성
        test_labels_series = pd.Series(
            np.where(self.data.y_test == 0, 'Normal', 'Caution'),
            index=common_index,
            name='true_label')
        pred_labels_series = pd.Series(
            np.where(self.data.y_pred == 0, 'Normal', 'Caution'),
            index=common_index,
            name='predicted_label')
        correct_prediction_series = pd.Series(
            self.data.y_test == self.data.y_pred,
            index=common_index,
            name='is_correct')

        # 예측 결과 및 레이블 정보를 담는 DataFrame 생성
        df_predict_result_core = pd.concat([
            test_labels_series,
            pred_labels_series,
            correct_prediction_series
        ], axis=1)
        df_predict_result_core.index.name = 'company_id'
        df_predict_result_core.reset_index(inplace=True)

        # 데이터 형태에 따라 특징 데이터 추가
        if self.config['data_shape'] == 'flatten':
            # Flatten 모드: 2D NumPy 특징 배열을 DataFrame 컬럼으로 추가
            first_company_key = next(iter(self.data.company_dict))
            feature_names_base = sorted(list(self.data.company_dict[first_company_key].data_dict.keys()))
            data_duration = self.config['data_duration']

            flattened_column_names = []
            for d_idx in range(data_duration):
                for f_name in feature_names_base:
                    flattened_column_names.append(f"{f_name}_{d_idx}")

            df_features_test = pd.DataFrame(self.data.X_test, columns=flattened_column_names, index=common_index)
            df_features_test.reset_index(inplace=True)
            df_features_test.rename(columns={'index': 'company_id'}, inplace=True)

            df_predict_result = pd.merge(df_predict_result_core, df_features_test, on='company_id', how='left')

        elif self.config['data_shape'] == 'matrix':
            # Matrix 모드: 특징 데이터는 최종 결과 DataFrame에 직접 포함하지 않음
            df_predict_result = df_predict_result_core
            pass

        # 회사 정보(전체 데이터 기간) 추가
        rows_company_info = []
        for full_id_name in common_index:
            original_company_id = int(full_id_name.split('_', 1)[0])
            company = self.data.company_dict[original_company_id]
            row = {"company_id": full_id_name,
                   "original_company_id": original_company_id,
                   "company_actual_start_date": company.date_range[0],
                   "company_actual_end_date": company.date_range[-1]}
            rows_company_info.append(row)

        df_test_company_results = pd.DataFrame(rows_company_info)

        # 최종 결과 DataFrame 병합 및 정렬
        df_final_result = pd.merge(df_predict_result, df_test_company_results, on='company_id', how='left')
        df_final_result = df_final_result.sort_values(by='company_id').reset_index(drop=True)

        # 예측 결과 엑셀 파일 저장
        df_final_result.to_excel(self.config['result_folder_path'] + '/predict_result.xlsx', index=False)

        # 모델 성능 지표 엑셀 파일 저장
        model_result_df = pd.json_normalize(self.result, sep='_').transpose()
        model_result_df.to_excel(self.config['result_folder_path'] + '/model_result.xlsx', index=True)

        # 결과 요약 텍스트 파일 저장
        summary_path = os.path.join(self.config['result_folder_path'], 'result_summary.txt')
        with open(summary_path, 'w', encoding='utf-8-sig') as f:
            f.write(f"===== {self.config['model_type']} 모델 예측 결과 요약 =====\n\n")

            # 성능 지표 섹션
            f.write("1. 성능 지표\n")
            f.write(f"   - 정확도(Accuracy): {self.result['accuracy']:.4f}\n")
            f.write(f"   - 정밀도(Precision): {self.result['precision']:.4f}\n")
            f.write(f"   - 재현율(Recall): {self.result['recall']:.4f}\n")
            f.write(f"   - F1 점수: {self.result['f1_score']:.4f}\n\n")

            # 예측 결과 요약 섹션
            f.write("2. 예측 결과 요약\n")
            f.write(f"   - 전체 테스트 샘플: {self.result['total_count']}개\n")
            f.write(
                f"   - 맞은 예측: {self.result['correct_count']}개 ({self.result['correct_count'] / self.result['total_count']:.2%})\n")
            f.write(
                f"   - 틀린 예측: {self.result['wrong_count']}개 ({self.result['wrong_count'] / self.result['total_count']:.2%})\n\n")

            # 클래스별 성능 섹션
            f.write("3. 클래스별 성능\n")
            f.write(
                f"   - 경영악화(Caution) 클래스: {self.result['class_1_total']}개 중 {self.result['class_1_correct']}개 맞춤 ({self.result['class_1_accuracy']:.2%})\n")
            f.write(
                f"   - 거래중(Normal) 클래스: {self.result['class_0_total']}개 중 {self.result['class_0_correct']}개 맞춤 ({self.result['class_0_accuracy']:.2%})\n\n")

            # 혼동 행렬 섹션
            cm = self.result['confusion_matrix']
            f.write("4. 혼동 행렬\n")
            f.write(f"            | 예측: Caution | 예측: Normal\n")
            f.write(f"   실제: Caution | {cm[0][0]}          | {cm[0][1]}\n")
            f.write(f"   실제: Normal  | {cm[1][0]}          | {cm[1][1]}\n\n")

            f.write(f"===== 저장 시간: {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")

        print(f"\n결과 요약 저장 완료: {summary_path}")

        # 모델 하이퍼파라미터 엑셀 파일 저장
        try:
            models_hyperparameters_df = pd.json_normalize(self.models_hyperparameters, sep='_').transpose()
            models_hyperparameters_df.to_excel(self.config['result_folder_path'] + '/models_hyperparameter.xlsx',
                                               index=True)
        except Exception as e:
            print(f"Error: 모델 하이퍼파라미터 저장 중 오류 발생: {str(e)}")
