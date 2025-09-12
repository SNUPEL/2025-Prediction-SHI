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
        # self.split_cutoff_date = pd.to_datetime(self.config['split_cutoff_date']).replace(day=1)
        self.split_cutoff_date = self.label_date - pd.DateOffset(months=self.config['data_duration'] + self.config['label_duration'])
        # self.df_model = pd.DataFrame()
        self.feature_names = list()

        self.df_x_train_matrix_dict = dict()
        self.df_y_train_dict = dict()
        self.df_x_valid_matrix_dict = dict()
        self.df_y_valid_dict = dict()
        self.df_x_test_matrix_dict = dict()
        self.df_y_test_dict = dict()
        self.df_train_dict = dict()
        self.df_valid_dict = dict()
        self.df_test_dict = dict()
        self.df_y_pred_dict = dict()

        self.df_x_train_flatten = pd.DataFrame()
        self.df_y_train = pd.DataFrame()
        self.df_x_valid_flatten = pd.DataFrame()
        self.df_y_valid = pd.DataFrame()
        self.df_x_test_flatten = pd.DataFrame()
        self.df_y_test = pd.DataFrame()
        self.df_train = pd.DataFrame()
        self.df_valid = pd.DataFrame()
        self.df_test = pd.DataFrame()
        self.flattened_column_names = list()
        self.name_train = list()
        self.name_valid = list()
        self.name_test = list()
        self.df_y_pred = pd.DataFrame()
        self.df_y_pred_proba = pd.DataFrame()

        self.df_x_train_flatten_after_sampling = pd.DataFrame()
        self.df_y_train_after_sampling = pd.DataFrame()
        self.df_train_after_sampling = pd.DataFrame()

        self.df_x_train_matrix_dict_after_sampling = dict()
        self.df_y_train_dict_after_sampling = dict()
        self.df_train_dict_after_sampling = dict()

        self.scaler_dict = dict()
        self.scaler_for_sheet_dict = dict()

    def load_data(self):
        load_data(self)

    def preprocess_data(self):
        print("==== 데이터 분할 시작 ====")
        split_data(self)
        print("==== 데이터 분할 완료 ====\n")

        if self.config['save_graph']:
            print("\n==== 데이터 분할 후 PCA, LDA 분석 시작 ====")
            save_PCA_LDA(self, graph_name='after_split')
            print("==== 데이터 분할 후 PCA, LDA 분석 완료 ====\n")

        for sampling_order in self.config['sampling_order']:
            if sampling_order == 'oversampling' and self.config['oversampling']:
                print("\n==== 데이터 oversampling 적용 시작 ====")
                apply_oversampling(self)
                print("==== 데이터 oversampling 적용 완료 =====\n")
            elif sampling_order == 'undersampling' and self.config['undersampling']:
                print("\n==== 데이터 undersampling 적용 시작 ====")
                apply_undersampling(self)
                print("==== 데이터 undersampling 적용 완료 =====\n")
                make_matrix_data(self)

        # if self.config['oversampling']:
        #     print("\n==== 데이터 oversampling 적용 시작 ====")
        #     apply_oversampling(self)
        #     print("==== 데이터 oversampling 적용 완료 ====\n")

        if self.config['save_graph'] and (self.config['undersampling'] or self.config['oversampling']):
            print("\n==== 데이터 증강 후 PCA, LDA 분석 시작 ====")
            save_PCA_LDA(self, graph_name='after_under_oversampling')
            print("==== 데이터 증강 후 PCA, LDA 분석 완료 ====\n")

