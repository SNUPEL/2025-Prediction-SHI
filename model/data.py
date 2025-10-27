import numpy as np
import pandas as pd
from load_data import *  # 데이터 로딩 함수 (load_data) 임포트
from preprocess_data import *  # 데이터 전처리 관련 함수들 (split_data, apply_oversampling 등) 임포트
from save_graph import *  # 그래프 저장 관련 함수 (save_PCA_LDA) 임포트


class Data:
    """
    데이터 로딩, 전처리, 분할, 샘플링 등 데이터 관련 모든 작업을 총괄하는 클래스.
    모델 학습에 필요한 모든 데이터를 속성으로 저장하고 관리합니다.
    """

    def __init__(self, config):
        """
        Data 클래스 초기화 메서드.
        config 파일로부터 각종 설정 값을 읽어와 클래스 속성들을 초기화합니다.
        데이터를 담을 다양한 변수들(DataFrame, dict, list 등)을 미리 선언합니다.
        """
        # --- 기본 설정 및 경로 정보 ---
        self.config = config  # 전체 프로젝트의 설정을 담고 있는 딕셔너리
        self.label_date = pd.to_datetime(self.config['label_date']).replace(day=1)  # 예측 기준이 되는 날짜 (YYYY-MM-01 형식)
        self.config_start_date_dt = pd.to_datetime(self.config['data_start_date']).replace(day=1)  # 데이터 사용 시작 날짜
        self.sheet_ban_list = self.config['sheet_ban_list']  # 분석에서 제외할 엑셀 시트 이름 리스트

        # --- 원본 데이터 및 회사 정보 ---
        self.df_raw_data_dict = dict()  # 원본 엑셀 파일의 각 시트를 담을 딕셔너리 (key: 시트명, value: DataFrame)
        self.sheet_name_list = list()  # 엑셀 파일의 모든 시트 이름 리스트
        self.company_dict = dict()  # 각 회사(Company) 객체를 담을 딕셔너리 (key: 회사 ID, value: Company 객체)

        # --- 데이터 분할 기준 날짜 ---
        # 예측 시점(label_date)으로부터 (분석 기간 + 예측 기간) 만큼 이전 날짜를 계산하여 train/test 분할 기준으로 삼음
        self.split_cutoff_date = self.label_date - pd.DateOffset(
            months=self.config['data_duration'] + self.config['label_duration'])

        # --- 모델 학습/검증/테스트를 위한 데이터셋 ---
        self.feature_names = list()  # 모델에 사용될 피처(feature) 이름 리스트

        # 3D 형태의 시계열 데이터 (샘플, 피처, 시간) - 주로 딥러닝 모델용
        self.df_x_train_matrix_dict = dict()  # Train X (3D)
        self.df_y_train_dict = dict()  # Train Y
        self.df_x_valid_matrix_dict = dict()  # Validation X (3D)
        self.df_y_valid_dict = dict()  # Validation Y
        self.df_x_test_matrix_dict = dict()  # Test X (3D)
        self.df_y_test_dict = dict()  # Test Y

        # 각 데이터셋(train/valid/test)의 원본 정보를 담는 딕셔너리
        self.df_train_dict = dict()
        self.df_valid_dict = dict()
        self.df_test_dict = dict()
        self.df_y_pred_dict = dict()  # 예측 결과를 담을 딕셔너리

        # 2D 형태의 데이터 (샘플, 피처) - 주로 머신러닝 모델용 (시계열 데이터를 flatten)
        self.df_x_train_flatten = pd.DataFrame()  # Train X (2D)
        self.df_y_train = pd.DataFrame()  # Train Y
        self.df_x_valid_flatten = pd.DataFrame()  # Validation X (2D)
        self.df_y_valid = pd.DataFrame()  # Validation Y
        self.df_x_test_flatten = pd.DataFrame()  # Test X (2D)
        self.df_y_test = pd.DataFrame()  # Test Y

        # flatten된 데이터셋의 원본 정보를 담는 DataFrame
        self.df_train = pd.DataFrame()
        self.df_valid = pd.DataFrame()
        self.df_test = pd.DataFrame()
        self.flattened_column_names = list()  # flatten된 컬럼 이름 리스트

        # 각 데이터셋에 포함된 회사(샘플) 이름 리스트
        self.name_train = list()
        self.name_valid = list()
        self.name_test = list()

        # 최종 예측 결과 저장용 DataFrame
        self.df_y_pred = pd.DataFrame()  # 예측된 클래스(0 또는 1)
        self.df_y_pred_proba = pd.DataFrame()  # 예측 확률

        # --- 데이터 샘플링 후 데이터를 저장할 변수들 ---
        # Oversampling 또는 Undersampling 적용 후의 데이터셋
        self.df_x_train_flatten_after_sampling = pd.DataFrame()
        self.df_y_train_after_sampling = pd.DataFrame()
        self.df_train_after_sampling = pd.DataFrame()

        self.df_x_train_matrix_dict_after_sampling = dict()
        self.df_y_train_dict_after_sampling = dict()
        self.df_train_dict_after_sampling = dict()

        # --- 데이터 스케일링(정규화) 관련 ---
        self.scaler_dict = dict()  # 피처별로 학습된 스케일러(Scaler) 객체를 저장
        self.scaler_for_sheet_dict = dict()  # 시트별로 학습된 스케일러 객체를 저장

    def load_data(self):
        """
        load_data.py 파일의 load_data 함수를 호출하여 원본 데이터를 로딩하고,
        Company 객체를 생성하여 각 회사별 메타 정보 및 시계열 데이터를 정리합니다.
        """
        load_data(self)

    def preprocess_data(self):
        """
        데이터 전처리 파이프라인을 실행하는 메서드.
        설정(config)에 따라 데이터 분할, 시각화, 샘플링 등을 순차적으로 수행합니다.
        """
        print("==== 데이터 분할 시작 ====")
        split_data(self)  # preprocess_data.py의 함수. 데이터를 train/valid/test로 분할.
        print("==== 데이터 분할 완료 ====\n")

        # 설정에서 save_graph가 True일 경우, PCA/LDA 분석 그래프를 저장
        if self.config['save_graph']:
            print("\n==== 데이터 분할 후 PCA, LDA 분석 시작 ====")
            save_PCA_LDA(self, graph_name='after_split')
            print("==== 데이터 분할 후 PCA, LDA 분석 완료 ====\n")

        # 설정된 샘플링 순서('oversampling', 'undersampling')에 따라 샘플링을 적용
        for sampling_order in self.config['sampling_order']:
            if sampling_order == 'oversampling' and self.config['oversampling']:
                print("\n==== 데이터 oversampling 적용 시작 ====")
                apply_oversampling(self)  # 소수 클래스 데이터 증강
                print("==== 데이터 oversampling 적용 완료 =====\n")
            elif sampling_order == 'undersampling' and self.config['undersampling']:
                print("\n==== 데이터 undersampling 적용 시작 ====")
                apply_undersampling(self)  # 다수 클래스 데이터 감소
                print("==== 데이터 undersampling 적용 완료 =====\n")
                # Undersampling 후 matrix 형태 데이터로 변경
                make_matrix_data(self)

        # 샘플링 적용 후, 그래프 저장이 활성화되어 있으면 다시 PCA/LDA 분석을 수행
        if self.config['save_graph'] and (self.config['undersampling'] or self.config['oversampling']):
            print("\n==== 데이터 증강 후 PCA, LDA 분석 시작 ====")
            save_PCA_LDA(self, graph_name='after_under_oversampling')
            print("==== 데이터 증강 후 PCA, LDA 분석 완료 ====\n")