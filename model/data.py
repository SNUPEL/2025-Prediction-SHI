import numpy as np
import pandas as pd
from preprocess_data import *
from split_data import *


class Company:
    def __init__(self, company_id, category, subcategory, sub_subcategory, start_date, end_date, date_range, label, condition):
        self.company_id = company_id
        self.category = category
        self.subcategory = subcategory
        self.sub_subcategory = sub_subcategory
        self.start_date = start_date
        self.end_date = end_date
        self.date_range = date_range
        self.label = label
        self.condition = condition

        self.data_dict = dict()


class Data:
    def __init__(self, config):
        self.config = config
        self.sheet_ban_list = self.config['sheet_ban_list']
        self.df_raw_data_dict = dict()
        self.sheet_name_list = list()
        self.company_dict = dict()

        self.df_model = pd.DataFrame()
        self.df_x = pd.DataFrame()
        self.df_y = pd.DataFrame()
        self.df_x_train = pd.DataFrame()
        self.df_y_train = pd.DataFrame()
        self.df_x_test = pd.DataFrame()
        self.df_y_test = pd.DataFrame()
        self.df_y_pred = pd.DataFrame()

    def load_data(self):
        self.df_raw_data_dict = pd.read_excel(self.config['data_file_path'], sheet_name=None)
        self.sheet_name_list = list(self.df_raw_data_dict.keys())

        df_temp = self.df_raw_data_dict[self.sheet_name_list[0]]
        column = list(df_temp.columns)[:8]
        if pd.isna(pd.to_datetime(df_temp.columns[-1], errors='coerce')):
            df_temp = df_temp.iloc[:, :-1]
        new_cols = (list(df_temp.columns[:9]) + list(
            pd.to_datetime(df_temp.columns[9:]).map(lambda dt: dt.replace(day=1))))
        date_cols = new_cols[9:]

        data_duration = self.config['data_duration']
        label_date = self.config['label_date']
        data_masking_duration = self.config['data_masking_duration']

        for i, row in df_temp.iterrows():
            r_start_date = pd.to_datetime(row[column[4]]).replace(day=1)
            r_end_date = pd.to_datetime(label_date) - pd.DateOffset(months=1) if row[column[
                5]] == '-' else pd.to_datetime(row[column[5]]).replace(day=1)
            r_label = True if pd.to_datetime(label_date) >= r_end_date + pd.DateOffset(
                months=1 - data_masking_duration) else False
            r_label = False if row[column[5]] == '-' else r_label

            # 후 : 종결된 회사, 활동 중 회사까지 마스킹 처리 되도록 수정 , 전 : 종결된 회사, 활동 중 회사 구분 되어 마스킹 됨 (성유)
            date_range = list(
                pd.date_range(start=r_end_date - pd.DateOffset(months=data_duration + data_masking_duration - 1),
                              end=r_end_date - pd.DateOffset(months=data_masking_duration), freq='MS'))
            # if r_label:
            #     date_range = list(
            #         pd.date_range(start=r_end_date - pd.DateOffset(months=data_duration + data_masking_duration - 1),
            #                       end=r_end_date - pd.DateOffset(months=data_masking_duration), freq='MS'))
            # else:
            #     date_range = list(pd.date_range(start=pd.to_datetime(label_date) - pd.DateOffset(months=data_duration),
            #                                     end=pd.to_datetime(label_date) - pd.DateOffset(months=1), freq='MS'))
            r_condition = all(element in iter(date_cols) for element in date_range) and r_start_date <= date_range[
                0] and r_end_date >= date_range[-1]
            self.company_dict[row[column[0]]] = Company(row[column[0]], row[column[1]], row[column[2]], row[column[3]],
                                                        r_start_date, r_end_date, date_range, r_label, r_condition)

        self.company_dict = {k: v for k, v in self.company_dict.items() if v.condition}

        for sheet_name in self.sheet_name_list:
            if sheet_name in self.sheet_ban_list:
                continue
            else:
                df_sheet = self.df_raw_data_dict[sheet_name]

                if pd.isna(pd.to_datetime(df_sheet.columns[-1], errors='coerce')):
                    df_sheet = df_sheet.iloc[:, :-1]
                new_cols = (list(df_sheet.columns[:9]) + list(
                    pd.to_datetime(df_temp.columns[9:]).map(lambda dt: dt.replace(day=1))))
                df_sheet.columns = new_cols
                self.df_raw_data_dict[sheet_name] = df_sheet

                for i, row in df_sheet.iterrows():
                    if self.config['use_all_data'] == 'All':
                        pass
                    elif self.config['use_all_data'] == 'Duration':
                        if i + 1 in self.company_dict.keys():
                            self.company_dict[i + 1].data_dict[sheet_name] = (row[self.company_dict[i + 1].date_range]
                                                                              .fillna(0).astype(float))
                    elif self.config['use_all_data'] == 'Padding':
                        # 현진이가 한 방식인데 향후 수정 필요
                        pass
                    else:
                        print(f"Error: config['use_all_data'] '{self.config['use_all_data']}' was not found.")
                        return False
        print('Data has been loaded successfully')

    def preprocess_data(self):
        if self.config['preprocessing'] == 'Flatten':
            flatten(self)
        else:
            print(f"Error: config['preprocessing'] '{self.config['preprocessing']}' was not found.")
        print('Data has been preprocessed successfully')

    def split_data(self):
        if self.config['data_split_type'] == 'Random':
            random_split(self)
        else:
            print(f"Error: config['data_split_type'] '{self.config['data_split_type']}' was not found.")
        print('Data has been split successfully')

        if self.config['SMOTE']:
            SMOTE(self)
        else:
            pass



