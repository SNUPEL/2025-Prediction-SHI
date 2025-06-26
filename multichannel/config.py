"""
설정값 및 상수 관리 모듈
"""
import os
import pandas as pd
from datetime import datetime

# 저장 경로 설정
OUTPUT_DIR = "./Last"

# 현재 시간
NOW = datetime.now().strftime("%Y%m%d_%H%M%S")

# 로그 파일 경로
LOG_FILE = f"{OUTPUT_DIR}/anl_2_log_{NOW}.txt"

# 기준 날짜 (train/test 데이터 분리 기준)
CUTOFF_DATE = pd.Timestamp('2024-01-01')

# 테스트 데이터 종료 날짜 (None이면 모든 데이터 사용)
TEST_END_DATE = None

# # 기준 날짜 (train/test 데이터 분리 기준)
# CUTOFF_DATE = pd.Timestamp('2023-01-01')

# # 테스트 데이터 종료 날짜 (None이면 모든 데이터 사용)
# TEST_END_DATE = pd.Timestamp('2024-01-01')

# 마스킹 설정 (패딩된 값을 모델에서 무시)
USE_MASKING = True  # True: 마스킹 사용, False: 기존 방식 사용

# 오버샘플링 방법 설정
# "SMOTE": 일반 SMOTE 사용
# "TSSMOTE": 시계열 특화 SMOTE 사용 (DTW 거리 기반)
# "NONE": 오버샘플링 사용 안함 (클래스 가중치만 사용)
OVERSAMPLING_METHOD = "TSSMOTE"

# 언더샘플링 설정 (오버샘플링 전에 적용)
USE_OSS = True  # True: Tomek Links 언더샘플링 사용, False: 언더샘플링 사용 안함

# 데이터 분할 설정
REGENERATE_SPLIT_DATA = True  # True: 데이터 새로 분할, False: 기존 CSV 사용
SPLIT_OUTPUT_DIR = "./split_data"  # 분할 데이터 저장 경로

# 데이터 디렉토리 경로
DATA_DIRECTORIES = [
    "./split_data",
    "./split_data/test",
    "./split_data/train"
]

# 회사 그룹 CSV 파일 경로
COMPANY_GROUPS_PATH = "./split_data/company_groups.csv"
COMPANY_GROUPS_ALT_PATH = "./split_data/company_groups.csv"

# 모델 기본 하이퍼파라미터
DEFAULT_MODEL_PARAMS = {
    'hidden_size': 64,
    'dropout': 0.5,
    'weight_decay': 0.0001
}

# 학습 파라미터
TRAIN_PARAMS = {
    'batch_size': 32,
    'epochs': 1000,
    'patience': 100,
    'learning_rate': 0.001
}

# 기본 파일 경로
DEFAULT_DATA_FILE = "./Data.xlsx"
DEFAULT_ADD_DATA_FILE = "./add_data.xlsx"

# 기본 제외 시트
DEFAULT_EXCLUDE_SHEETS = [
    '본공률(4대보험)',
    '종합평가',
    '입사자',
    '퇴사자',
    '안전사고 건수(중간, 낮음)', 
    '안전사고 건수(높음)',
    '4대보험 가입자',
    '경영영향2',
    '경영영향1'
]

# 최소 시간점 개수
MIN_TIME_POINTS = 6 
# 테스트 데이터 최소 시간점 개수
MIN_TIME_POINTS_TEST = 1

# 확장된 테스트 데이터 사용 여부 (기존 거래중 회사의 cutoff_date 이후 데이터도 test에 포함)
# True: 확장된 테스트 데이터 사용 - cutoff_date 이전부터 거래중인 회사의 cutoff_date 이후 데이터도 test에 추가
#       이를 통해 데이터 활용률이 크게 개선되고 더 현실적인 예측 시나리오 평가 가능
# False: 기존 방식 - 4분할 데이터만 사용 (일부 데이터 버림 현상 발생)
USE_EXTENDED_TEST_DATA = True

# 최적 임계값 자동 탐색 설정
AUTO_THRESHOLD = False  # True: 검증 데이터에서 최적 임계값 자동 찾기, False: 고정 임계값 사용
FIXED_THRESHOLD = 0.5  # AUTO_THRESHOLD가 False일 때 사용할 고정 임계값