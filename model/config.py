# config.py 파일

import os
import time
import pandas as pd


def create_config():
    config = dict()

    # 데이터 파일 및 보조 파일 경로 설정
    config['data_file_path'] = '../data/사내협력사 현황(철수사&거래 협력사)_ Data_추가(250417)_예진.xlsx'
    config['sub_data_file_path'] = '../data/사내협력사 현황(철수사&거래 협력사)_출근인력(추가).xlsx'
    # 데이터 로딩 방식 설정 (All: 당사 투입일 ~ 당사 철수일, Padding: 모든 기간)
    config['use_all_data'] = 'All'
    # 데이터 사용의 가장 이른 시작 시점 설정
    config['data_start_date'] = '2016-02-01'
    # config['data_start_date'] = '2020-01-01'
    # config['data_end_date'] = '2024-10-01' # 현재 코드 로직에서 사용되지 않는 설정 (원본 코드 유지)

    # 데이터 포인트의 계산된 label_date가 이 날짜보다 같거나 이전이면 훈련 세트, 이후이면 테스트 세트
    # 일반적으로 config['label_date']보다 이전 날짜로 설정(label_duration 및 데이터 마스킹 기간까지 고려하여 설정)
    # config['split_cutoff_date'] = '2023-07-01'  # 굉장히 넉넉히 줄 것
    config['split_cutoff_date'] = '2023-03-01'  # 굉장히 넉넉히 줄 것
    config['overlap'] = False

    # 각 데이터 포인트의 과거 데이터 기간(길이) 설정
    # config['data_duration'] = 6
    config['data_duration'] = 12
    # 테스트 예측 기간의 끝점을 정의하거나 load_data에서 회사 데이터 기간 제한에 사용되는 날짜
    # config['label_date'] = '2024-08-01'
    config['label_date'] = '2024-08-01'
    # 데이터 기간 끝 ~ 레이블 시점 간격 설정 (테스트 예측 기간 계산에도 사용)
    config['label_duration'] = 1

    # 데이터 마스킹 기간 설정 (load_data에서 회사의 종료일 조정에 사용) 사용안함
    # config['data_masking_duration'] = 0
    # 데이터 로딩 시 제외할 시트 이름 목록
    # config['sheet_ban_list'] = ['경영 영향1', '경영 영향2', '종합평가', '입사자', '입사율', '퇴사율', '본공률(시급월급)',
    #                             '4대보험 가입자', '본공률(4대보험)', '안전사고 건수(중간, 낮음)', '안전사고 건수(높음)',
    #                             '공수능률',  '시급,월급제 인원', '투입인원', '환산능률']
    config['sheet_ban_list'] = ['경영 영향1', '경영 영향2', '종합평가', '본공률(4대보험)', '본공률(시급월급)'
                                '4대보험 가입자', '안전사고 건수(중간, 낮음)', '안전사고 건수(높음)']

    # 정규화: None, standard
    config['scaler'] = 'standard'
    # 정규화 기준: feature, feature_and_company
    config['scale_by'] = 'feature'

    # 데이터 분석 및 가시화 그래프를 저장, matrix시 불가능
    config['save_graph'] = True

    config['test_data_ratio'] = 0.3  # 사용안함

    # 랜덤 시드 설정
    config['random_state'] = 42  # 랜덤 시드

    # undersampling: None, tomek_link, ENN, nearmiss
    config['undersampling'] = 'ENN'
    config['ENN_n_neighbors'] = 5

    # SMOTE 오버샘플링 사용 여부 설정
    # SMOTE 사용시 경영 악화 데이터 증강으로 예측도 높아짐
    # oversampling: SMOTE, BorderlineSMOTE
    config['oversampling'] = 'SMOTE'
    config['SMOTE_k_neighbors'] = 5
    config['SMOTE_sampling_strategy'] = 1

    # ML model: 'RandomForestClassifier', 'AdaBoostClassifier', 'ExtraTreesClassifier',
    # 'RidgeClassifier', 'SGDClassifier', 'XGBClassifier', 'SVC',
    # DL model: 'ConvLSTM'
    config['model_type'] = 'ConvLSTM'  # 사용할 모델 타입

    # flatten, matrix
    config['data_shape'] = 'matrix'
    config['sub_window_size'] = 3

    # model parameter:
    config['AdaBoost_parameter'] = {'n_estimators': 10, "learning_rate": 0.01}
    # config['AdaBoost_estimator_parameter'] = {'max_depth': 4, 'min_samples_split': 2}
    config['class_weight'] = {0: 1, 1: 10000}
    # 학습된 모델 저장 여부 설정
    config['save_model'] = False

    config['detailed_results'] = True  # 상세 결과 출력 여부
    config['save_predictions'] = True  # 예측 결과 저장 여부
    config['save_confusion_matrix'] = True  # 혼동 행렬 저장 여부
    config['save_metrics'] = True  # 성능 지표 저장 여부

    # 결과 저장 폴더 설정 (타임스탬프 기반)
    config['ymd'] = time.strftime('%Y%m%d')
    config['hour'] = str(time.localtime().tm_hour)
    config['minute'] = str(time.localtime().tm_min)
    config['second'] = str(time.localtime().tm_sec)
    config["result_folder_path"] = '../results/{0}_{1}h_{2}m_{3}s'.format(
        config['ymd'], config['hour'], config['minute'], config['second'])

    # 결과 폴더 생성 (폴더가 없으면 생성)
    if not os.path.exists(config["result_folder_path"]):
        os.mkdir(config["result_folder_path"])

    # 현재 설정 정보를 엑셀 파일로 저장
    config_df = pd.json_normalize(config, sep='_').transpose()
    config_df.to_excel(config['result_folder_path'] + '/configuration.xlsx', index=True)

    return config
