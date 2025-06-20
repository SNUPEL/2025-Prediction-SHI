import numpy as np
import pandas as pd
from load_data import *
from preprocess_data import *
from save_graph import *


class Data:
    def __init__(self, config):
        self.config = config
        self.label_date = pd.to_datetime(self.config['label_date']).replace(day=1)
        self.config_start_date_dt = pd.to_datetime(self.config['data_start_date']).replace(day=1)
        self.sheet_ban_list = self.config['sheet_ban_list']
        self.df_raw_data_dict = dict()
        self.sheet_name_list = list()
        self.company_dict = dict()
        self.split_cutoff_date = pd.to_datetime(self.config['split_cutoff_date']).replace(day=1)
        self.df_model = pd.DataFrame()
        self.df_x = pd.DataFrame()
        self.df_y = pd.DataFrame()
        self.df_train = pd.DataFrame()
        self.df_test = pd.DataFrame()
        self.X_train = list()
        self.y_train = list()
        self.X_test = list()
        self.y_test = list()
        self.y_pred = list()

        self.scaler_dict = dict()

    def load_data(self):
        load_data(self)

    def preprocess_data(self):
        print("==== 데이터 분할 시작 ====")
        split_data(self)
        print("==== 데이터 분할 완료 =====\n")

        print("\n==== 데이터 분할 후 PCA, LDA 분석 시작 ===")
        if self.config['save_graph']:
            save_PCA_LDA(self, graph_name='after_split')
        print("==== 데이터 분할 후 PCA, LDA 분석 완료 =====\n")

        print("\n==== 데이터 undersampling 적용 시작 ====")
        if self.config['undersampling']:
            apply_undersampling(self)
        print("==== 데이터 undersampling 적용 완료 =====\n")

        print("\n==== 데이터 oversampling 적용 시작 ====")
        if self.config['oversampling']:
            apply_SMOTE(self)
        print("==== 데이터 oversampling 적용 완료 =====\n")

        print("\n==== 데이터 증강 후 PCA, LDA 분석 시작 ====")
        if self.config['save_graph']:
            save_PCA_LDA(self, graph_name='after_under_oversampling')
        print("==== 데이터 증강 후 PCA, LDA 분석 완료 =====\n")

