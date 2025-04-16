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
    config['data_masking_duration'] = 2
    config['sheet_ban_list'] = ['4대보험 체납', '종합평가', '임금체불']

    config['SMOTE'] = True

    # model: 'SVC', 'RandomForestClassifier'
    config['model_type'] = None
    config['save_model'] = True

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