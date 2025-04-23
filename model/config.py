import os
import time
import pandas as pd


def create_config():
    config = dict()

    config['data_file_path'] = '../data/사내협력사 현황(철수사&거래 협력사)_ Data_송부_수정.xlsx'
    # All: 당사 투입일 ~ 당사 철수일, Duration: 당사철수일까지의 특정 개월 수, Padding: 모든 기간
    config['use_all_data'] = 'Duration'
    config['data_start_date'] = '2016-02-01'
    config['data_end_date'] = '2024-10-01'
    config['data_duration'] = 12
    config['label_date'] = '2024-10-01'  # 해당 월 기준으로 계약 종결 여부 판단, 직전월까지의 데이터 사용
    config['data_masking_duration'] = 0
    config['sheet_ban_list'] =  ['임금체불', '4대보험 체납', '종합평가', '입사자', '퇴사율', '퇴사자', '본공률(시급월급)', '4대보험 가입자', '안전사고 건수(높음)']

    config['preprocessing'] = 'Flatten'

    config['data_split_type'] = 'Random'
    config['test_data_ratio'] = 0.3
    config['random_state'] = 42
    config['SMOTE'] = False

    # model: 'RandomForestClassifier'
    # config['model_type'] = 'RandomForestClassifier'
    # config['save_model'] = True

    # model: 'AdaBoostClassifier'
    # config['model_type'] = 'AdaBoostClassifier'
    # config['save_model'] = True

    # model: 'ExtraTreesClassifier'
    # config['model_type'] = 'ExtraTreesClassifier'
    # config['save_model'] = True

    # model: 'RidgeClassifier'
    # config['model_type'] = 'RidgeClassifier'
    # config['save_model'] = True

    # model: 'SGDClassifier'
    # config['model_type'] = 'SGDClassifier'
    # config['save_model'] = True

    # model: 'XGBClassifier'
    # config['model_type'] = 'XGBClassifier'
    # config['save_model'] = True

    # model: 'SVC'
    config['model_type'] = 'SVC'
    config['save_model'] = True

    # 평가 결과 관련 설정
    config['detailed_results'] = True  # 상세 결과 출력 여부
    config['save_predictions'] = True  # 예측 결과 저장 여부
    config['save_confusion_matrix'] = True  # 혼동 행렬 저장 여부
    config['save_metrics'] = True  # 성능 지표 저장 여부

    # 결과 저장
    config['ymd'] = time.strftime('%Y%m%d')
    config['hour'] = str(time.localtime().tm_hour)
    config['minute'] = str(time.localtime().tm_min)
    config['second'] = str(time.localtime().tm_sec)
    # 결과 파일 경로
    config["result_folder_path"] = '../results/{0}_{1}h_{2}m_{3}s'.format(
        config['ymd'], config['hour'], config['minute'], config['second'])

    if not os.path.exists(config["result_folder_path"]):
        os.mkdir(config["result_folder_path"])

    config_df = pd.json_normalize(config, sep='_').transpose()
    config_df.to_excel(config['result_folder_path'] + '/configuration.xlsx', index=True)

    return config
