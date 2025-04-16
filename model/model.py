from get_RandomForestClassifier import *
from evaluate_model import *


class Model:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.model = None
        self.models_hyperparameters = None
        self.result = dict()

    def make_model(self):
        if self.config['model_type'] == 'RandomForestClassifier':
            get_RandomForestClassifier(self)
        else:
            print(f"Error: config['model_type'] '{self.config['model_type']}' was not found.")
        print(f"Model {self.config['model_type']} has been trained successfully")

    def evaluate_model(self):
        evaluate_classifier(self)
        print('Model has been evaluated successfully')

    def save_result(self):
        # 예측값과 실제값 변환 (Series → 문자열)
        test_labels = pd.Series(
            np.where(self.data.df_y_test == 0, 'Normal', 'Caution'),
            index=self.data.df_y_test.index,
            name='true_label')
        pred_labels = pd.Series(
            np.where(self.data.df_y_pred == 0, 'Normal', 'Caution'),
            index=self.data.df_y_pred.index,
            name='predicted_label')
        # 두 Series를 하나의 DataFrame으로 합치기
        df_predict_result = pd.concat([self.data.df_x_test, test_labels, pred_labels], axis=1)
        test_company_list = list(self.data.df_x_test['company_id'])
        rows = []
        for company_id in test_company_list:
            company = self.data.company_dict[company_id]
            row = {"company_id": company_id,
                   "start_date": company.date_range[0],
                   "end_date": company.date_range[-1]}
            rows.append(row)
        df_test_company_results = pd.DataFrame(rows)
        df_final_result = pd.merge(df_predict_result, df_test_company_results, on='company_id')
        df_final_result = df_final_result.sort_values(by='company_id').reset_index(drop=True)

        df_final_result.to_excel(self.config['result_folder_path'] + '/predict result.xlsx', index=True)

        model_result_df = pd.json_normalize(self.result, sep='_').transpose()
        model_result_df.to_excel(self.config['result_folder_path'] + '/model result.xlsx', index=True)

        try:
            models_hyperparameters_df = pd.json_normalize(self.models_hyperparameters, sep='_').transpose()
            models_hyperparameters_df.to_excel(self.config['result_folder_path'] + '/models hyperparameter.xlsx', index=True)
        except NotImplementedError:
            print("Error: json_normalize could not process self.models_hyperparameters. Skipping the save operation.")

