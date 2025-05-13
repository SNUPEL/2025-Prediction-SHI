import numpy as np
import pandas as pd
from load_data import *
from preprocess_data import *

class Data:
    def __init__(self, config):
        self.config = config
        self.label_date = pd.to_datetime(self.config['label_date']).replace(day=1)
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
        self.df_x_train = pd.DataFrame()
        self.df_y_train = pd.DataFrame()
        self.df_x_test = pd.DataFrame()
        self.df_y_test = pd.DataFrame()
        self.df_y_pred = pd.DataFrame()

    def load_data(self):
        load_data(self)
        print('Data has been loaded successfully')

    def preprocess_data(self):
        split_data(self)
        # 분할 방식에 대한 메시지 출력 수정
        print('Data has been split successfully using standard chronological split')

        # flatten 함수 호출
        if self.config['Flatten']:
            flatten(self)
        else:
            pass

        # SMOTE 오버샘플링 적용
        if self.config['SMOTE']:
            apply_SMOTE(self)
        else:
            pass