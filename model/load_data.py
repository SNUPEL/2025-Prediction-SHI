import pandas as pd


class Company:
    def __init__(self, company_id, category, subcategory, sub_subcategory, start_date, end_date, comparison_date,
                 date_range, label, condition, age, data_masking_duration, severity_level):
        self.company_id = company_id
        self.category = category
        self.subcategory = subcategory
        self.sub_subcategory = sub_subcategory
        self.start_date = start_date
        self.end_date = end_date
        self.comparison_date = comparison_date
        self.date_range = date_range
        self.label = label
        self.condition = condition
        self.age = age
        self.data_masking_duration = data_masking_duration
        self.severity_level = severity_level

        self.data_dict = dict()


def load_data(self):
    self.df_raw_data_dict = pd.read_excel(self.config['data_file_path'], sheet_name=None)
    self.df_raw_sub_data = pd.read_excel(self.config['sub_data_file_path'], sheet_name='추가 정보', skiprows=[0, 2])
    self.sheet_name_list = list(self.df_raw_data_dict.keys())

    df_temp = self.df_raw_data_dict[self.sheet_name_list[0]]
    column = list(df_temp.columns)[:8]
    if pd.isna(pd.to_datetime(df_temp.columns[-1], errors='coerce')):
        df_temp = df_temp.iloc[:, :-1]
    new_cols = (list(df_temp.columns[:9]) + list(
        pd.to_datetime(df_temp.columns[9:]).map(lambda dt: dt.replace(day=1))))
    date_cols = new_cols[9:]


    for i, row in df_temp.iterrows():
        age = int(self.df_raw_sub_data.loc[i, '철수 당시/ 현 나이(만)'])
        data_masking_duration = 0 if pd.isna(self.df_raw_sub_data.loc[i, '협력사별 데이터 제거 필요 개월(철수일 기준)']) else int(
            self.df_raw_sub_data.loc[i, '협력사별 데이터 제거 필요 개월(철수일 기준)'])
        severity_level = None if pd.isna(self.df_raw_sub_data.loc[i, '경영악화 경/중 구분']) else self.df_raw_sub_data.loc[
            i, '경영악화 경/중 구분']

        r_start_date = pd.to_datetime(row[column[4]]).replace(day=1) if pd.to_datetime(row[column[4]]).replace(
            day=1) >= pd.to_datetime(self.config['data_start_date']).replace(day=1) else pd.to_datetime(
            self.config['data_start_date']).replace(day=1)
        r_end_date = None if row[column[5]] == '-' else pd.to_datetime(row[column[5]]).replace(day=1) - pd.DateOffset(months=data_masking_duration)
        r_label = False if row[column[5]] == '-' else True
        comparison_date = r_end_date if r_label and r_end_date <= self.label_date else self.label_date
        date_range = list(pd.date_range(start=r_start_date, end=comparison_date, freq='MS'))
        # r_condition = True if comparison_date >= r_start_date + pd.DateOffset(
        #     months=self.config['data_duration'] + self.config['label_duration'] + data_masking_duration) else False
        r_condition = True

        self.company_dict[row[column[0]]] = Company(row[column[0]], row[column[1]], row[column[2]], row[column[3]],
                                                    r_start_date, r_end_date, comparison_date, date_range, r_label,
                                                    r_condition, age, data_masking_duration, severity_level)

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

            if self.config['use_all_data'] == 'All':
                for i, row in df_sheet.iterrows():
                    if i + 1 in self.company_dict.keys():
                        self.company_dict[i + 1].data_dict[sheet_name] = (row[self.company_dict[i + 1].date_range]
                                                                          .fillna(0).astype(float))
            elif self.config['use_all_data'] == 'Padding':
                # 현진이가 한 방식인데 향후 수정 필요
                pass
            else:
                print(f"Error: config['use_all_data'] '{self.config['use_all_data']}' was not found.")
                return False
