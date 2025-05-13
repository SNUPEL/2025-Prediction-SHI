# config.py 파일

import os
import time
import pandas as pd


def create_config():
    config = dict()

    # 데이터 파일 및 보조 파일 경로 설정
    config['data_file_path'] = '../data/사내협력사 현황(철수사&거래 협력사)_ Data_추가(250417).xlsx'
    config['sub_data_file_path'] = '../data/사내협력사 현황(철수사&거래 협력사)_출근인력(추가).xlsx'
    # 데이터 로딩 방식 설정 (All: 당사 투입일 ~ 당사 철수일, Padding: 모든 기간)
    config['use_all_data'] = 'All'
    # 데이터 사용의 가장 이른 시작 시점 설정
    config['data_start_date'] = '2016-02-01'
    # config['data_end_date'] = '2024-10-01' # 현재 코드 로직에서 사용되지 않는 설정 (원본 코드 유지)

    # 데이터 포인트의 계산된 label_date가 이 날짜보다 같거나 이전이면 훈련 세트, 이후이면 테스트 세트
    # 일반적으로 config['label_date']보다 이전 날짜로 설정(label_duration 및 데이터 마스킹 기간까지 고려하여 설정)
    config['split_cutoff_date'] = '2024-03-01'

    # 각 데이터 포인트의 과거 데이터 기간(길이) 설정
    config['data_duration'] = 6
    # 테스트 예측 기간의 끝점을 정의하거나 load_data에서 회사 데이터 기간 제한에 사용되는 날짜
    config['label_date'] = '2024-08-01'
    # 데이터 기간 끝 ~ 레이블 시점 간격 설정 (테스트 예측 기간 계산에도 사용)
    config['label_duration'] = 3

    # 데이터 마스킹 기간 설정 (load_data에서 회사의 종료일 조정에 사용) 사용안함
    # config['data_masking_duration'] = 0
    # 데이터 로딩 시 제외할 시트 이름 목록
    config['sheet_ban_list'] = ['경영 영향1', '경영 영향2', '종합평가', '입사자', '퇴사율', '퇴사자', '본공률(시급월급)', '4대보험 가입자', '안전사고 건수(높음)']
    config['Flatten'] = True



    config['data_split_type'] = 'overlap' # 사용안함
    config['test_data_ratio'] = 0.3 # 사용안함

    # 랜덤 시드 설정
    config['random_state'] = 42 # 랜덤 시드

    # SMOTE 오버샘플링 사용 여부 설정
    # SMOTE 사용시 경영 악화 데이터 증강으로 예측도 높아짐
    config['SMOTE'] = True

    # model: 'RandomForestClassifier', 'AdaBoostClassifier', 'ExtraTreesClassifier', 'RidgeClassifier', 'SGDClassifier', 'XGBClassifier', 'SVC'
    config['model_type'] = 'AdaBoostClassifier' # 사용할 모델 타입
    # 학습된 모델 저장 여부 설정
    config['save_model'] = True

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