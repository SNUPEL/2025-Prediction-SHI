# import matplotlib
# matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import tensorflow as tf
import pandas as pd
import numpy as np
import torch
from get_ConvLSTM import create_convlstm_5d_sequences_internal


def save_shap_stats(self, shap_values_df):
    """
    shap 값들의 통계값을 저장하는 함수
    """
    shap_values_stats_df = shap_values_df.describe()
    shap_values_stats_df.loc['평균절대값(중요도)'] = shap_values_df.abs().mean()
    sorted_columns = shap_values_stats_df.loc['평균절대값(중요도)'].sort_values(ascending=False).index
    stats_df_sorted = shap_values_stats_df[sorted_columns]
    stats_df_sorted.to_excel(os.path.join(self.config['result_folder_path'], 'shap_stats.xlsx'))


def plot_shap(self, shap_values_df, X_test):
    """
    shap value의 가시화 함수
    """
    grouper = [col.split('_')[0] for col in shap_values_df.columns]
    new_columns = {}
    for name, group_df in shap_values_df.groupby(grouper, axis=1):
        concatenated_values = pd.concat(
            [group_df[col] for col in group_df.columns],
            ignore_index=True
        )
        new_columns[name] = concatenated_values
    grouped_shap_values_df = pd.DataFrame(new_columns)

    new_data_columns = {}
    for name, group_df in X_test.groupby(grouper, axis=1):  # X_test_df에 적용
        concatenated_values = pd.concat(
            [group_df[col] for col in group_df.columns],
            ignore_index=True
        )
        new_data_columns[name] = concatenated_values
    grouped_X_test_df = pd.DataFrame(new_data_columns)

    feature_importance = grouped_shap_values_df.abs().mean().sort_values(ascending=False)

    sorted_feature_names = feature_importance.index.tolist()

    shap_values_sorted = grouped_shap_values_df[sorted_feature_names]
    X_test_sorted = grouped_X_test_df[sorted_feature_names]

    plt.figure()
    shap.summary_plot(
        shap_values_sorted.values,
        X_test_sorted.values,
        feature_names=shap_values_sorted.columns,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(self.config['result_folder_path'], 'shap_summary_plot.png'))
    plt.show()
    plt.close()
    save_shap_stats(self, shap_values_df)


def get_SHAP_explainer(self):
    TREE_MODELS = [
        'RandomForestClassifier', 'AdaBoostClassifier',
        'ExtraTreesClassifier', 'XGBClassifier'
    ]
    LINEAR_MODELS = ['RidgeClassifier', 'SGDClassifier']
    DEEP_MODELS = [
        'ConvLSTM', 'MultiChannelCNNLSTM', 'Transformer',
        'ResNet', 'MLP_Mixer', 'AutoEncoder'
    ]
    model = self.model
    y_pred = self.data.df_y_pred['label']
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    # 모델 별 데이터 전처리 방식 적용
    if self.config['data_shape'] == 'flatten':
        X_train = self.data.df_x_train_flatten
        y_train = self.data.df_y_train['label']
        X_test = self.data.df_x_test_flatten
        y_test = self.data.df_y_test['label']
    elif self.config['model_type'] == 'ConvLSTM':
        if self.config['undersampling'] or self.config['oversampling']:
            X_train_3d = np.array(list(self.data.df_x_train_matrix_dict_after_sampling.values()))
            y_train = np.array(list(self.data.df_y_train_dict_after_sampling.values()))
        else:
            X_train_3d = np.array(list(self.data.df_x_train_matrix_dict.values()))
            y_train = np.array(list(self.data.df_y_train_dict.values()))
        X_test_3d = np.array(list(self.data.df_x_test_matrix_dict.values()))
        y_test = np.array(list(self.data.df_y_test_dict.values()))
        X_train = create_convlstm_5d_sequences_internal(X_train_3d, self.config['sub_window_size'])
        X_test = create_convlstm_5d_sequences_internal(X_test_3d, self.config['sub_window_size'])
    elif self.config['model_type'] == 'MultiChannelCNNLSTM':
        if self.config['undersampling'] or self.config['oversampling']:
            x_dict = self.data.df_x_train_matrix_dict_after_sampling
            y_dict = self.data.df_y_train_dict_after_sampling
        else:
            x_dict = self.data.df_x_train_matrix_dict
            y_dict = self.data.df_y_train_dict
        X_train_3d = np.array([df.values for df in x_dict.values()])
        y_train = np.array(list(y_dict.values()))
        X_test_3d = np.array([df.values for df in self.data.df_x_test_matrix_dict.values()])
        y_test = np.array(list(self.data.df_y_test_dict.values()))
        X_train = np.expand_dims(X_train_3d, axis=-1)
        X_test = np.expand_dims(X_test_3d, axis=-1)
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train).unsqueeze(1)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test).unsqueeze(1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        X_train = X_train.to(device)
        X_test = X_test.to(device)
        self.model.eval()
    elif self.config['model_type'] == 'Transformer':
        if self.config.get('undersampling') or self.config.get('oversampling'):
            X_train = np.array(list(self.data.df_x_train_matrix_dict_after_sampling.values()))
            y_train = np.array(list(self.data.df_y_train_dict_after_sampling.values()))
        else:
            X_train = np.array(list(self.data.df_x_train_matrix_dict.values()))
            y_train = np.array(list(self.data.df_y_train_dict.values()))
        X_test = np.array(list(self.data.df_x_test_matrix_dict.values()))
        y_test = np.array(list(self.data.df_y_test_dict.values()))
        X_train = X_train.transpose(0, 2, 1).astype('float32')
        y_train = y_train.astype('float32')
        X_test = X_test.transpose(0, 2, 1).astype('float32')
        y_test = y_test.astype('float32')
    elif self.config['model_type'] == 'ResNet':
        if self.config['undersampling'] or self.config['oversampling']:
            X_train = np.array(list(self.data.df_x_train_matrix_dict_after_sampling.values()))
            y_train = np.array(list(self.data.df_y_train_dict_after_sampling.values()))
        else:
            X_train = np.array(list(self.data.df_x_train_matrix_dict.values()))
            y_train = np.array(list(self.data.df_y_train_dict.values()))
        X_test = np.array(list(self.data.df_x_test_matrix_dict.values()))
        y_test = np.array(list(self.data.df_y_test_dict.values()))

        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
    elif self.config['model_type'] == 'MLP_Mixer':
        if self.config.get('undersampling') or self.config.get('oversampling'):
            X_train = np.array(list(self.data.df_x_train_matrix_dict_after_sampling.values()))
            y_train = np.array(list(self.data.df_y_train_dict_after_sampling.values()))
        else:
            X_train = np.array(list(self.data.df_x_train_matrix_dict.values()))
            y_train = np.array(list(self.data.df_y_train_dict.values()))
        X_test = np.array(list(self.data.df_x_test_matrix_dict.values()))
        y_test = np.array(list(self.data.df_y_test_dict.values()))
    else:
        return 0

    # 모델 별 예측 함수 적용
    if self.config['model_type'] in TREE_MODELS:
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_test)
        shap_values_df = pd.DataFrame(shap_values[:, :, 1], columns=self.data.flattened_column_names)
        plot_shap(self, shap_values_df, X_test)

    elif self.config['model_type'] in LINEAR_MODELS:
        explainer = shap.LinearExplainer(self.model, X_train)
        shap_values = explainer.shap_values(X_test)
        shap_values_df = pd.DataFrame(shap_values, columns=self.data.flattened_column_names)
        plot_shap(self, shap_values_df, X_test)

    elif self.config['model_type'] in DEEP_MODELS:
        if self.config['model_type'] == 'ConvLSTM':
            explainer = shap.GradientExplainer(self.model, shap.sample(X_train, 20))
            shap_values = explainer.shap_values(X_test)
            shap_values = shap_values.squeeze().transpose(0, 2, 1, 3).reshape(X_test.shape[0], X_test.shape[2], -1)
            X_test = X_test.squeeze().transpose(0, 2, 1, 3).reshape(X_test.shape[0], X_test.shape[2], -1)
            shap_values_dict = {name: [] for name in self.data.feature_names}
            for shap_value in shap_values:
                for i, name in enumerate(self.data.feature_names):
                    shap_values_dict[name].extend(shap_value[i])
            shap_values_df = pd.DataFrame(shap_values_dict)

            X_test_dict = {name: [] for name in self.data.feature_names}
            for x_val in X_test:
                # GPU 텐서를 CPU로 이동
                if hasattr(x_val, 'cpu'):
                    x_val = x_val.cpu()
                if hasattr(x_val, 'numpy'):
                    x_val = x_val.numpy()
                
                for i, name in enumerate(self.data.feature_names):
                    X_test_dict[name].extend(x_val[i])
            X_test_df = pd.DataFrame(X_test_dict)

            feature_importance = shap_values_df.abs().mean().sort_values(ascending=False)
            sorted_feature_names = feature_importance.index.tolist()
            shap_values_sorted = shap_values_df[sorted_feature_names]
            X_test_sorted = X_test_df[sorted_feature_names]

            plt.figure()
            shap.summary_plot(
                shap_values_sorted.values,
                X_test_sorted.values,
                feature_names=shap_values_sorted.columns,
                show=False
            )
            plt.tight_layout()
            plt.savefig(os.path.join(self.config['result_folder_path'], 'shap_summary_plot.png'))
            plt.show()
            plt.close()
            save_shap_stats(self, shap_values_df)

        elif self.config['model_type'] == 'MultiChannelCNNLSTM':
            explainer = shap.GradientExplainer(self.model, X_train[:20])
            self.model.train()
            shap_values = explainer.shap_values(X_test)
            self.model.eval()
            shap_values = shap_values.squeeze()
            X_test = X_test.squeeze()
            shap_values_dict = {name: [] for name in self.data.feature_names}
            for shap_value in shap_values:
                for i, name in enumerate(self.data.feature_names):
                    shap_values_dict[name].extend(shap_value[i])
            shap_values_df = pd.DataFrame(shap_values_dict)

            X_test_dict = {name: [] for name in self.data.feature_names}
            for x_val in X_test:
                # GPU 텐서를 CPU로 이동
                if hasattr(x_val, 'cpu'):
                    x_val = x_val.cpu()
                if hasattr(x_val, 'numpy'):
                    x_val = x_val.numpy()
                
                for i, name in enumerate(self.data.feature_names):
                    X_test_dict[name].extend(x_val[i])
            X_test_df = pd.DataFrame(X_test_dict)

            feature_importance = shap_values_df.abs().mean().sort_values(ascending=False)
            sorted_feature_names = feature_importance.index.tolist()
            shap_values_sorted = shap_values_df[sorted_feature_names]
            X_test_sorted = X_test_df[sorted_feature_names]

            plt.figure()
            shap.summary_plot(
                shap_values_sorted.values,
                X_test_sorted.values,
                feature_names=shap_values_sorted.columns,
                show=False
            )
            plt.tight_layout()
            plt.savefig(os.path.join(self.config['result_folder_path'], 'shap_summary_plot.png'))
            plt.show()
            plt.close()
            save_shap_stats(self, shap_values_df)

        elif self.config['model_type'] == 'Transformer':
            explainer = shap.GradientExplainer(self.model, shap.sample(X_train, 20))
            shap_values = explainer.shap_values(X_test)
            shap_values = shap_values.squeeze().transpose(0, 2, 1).astype('float32')
            X_test = X_test.transpose(0, 2, 1).astype('float32')
            shap_values_dict = {name: [] for name in self.data.feature_names}
            for shap_value in shap_values:
                for i, name in enumerate(self.data.feature_names):
                    shap_values_dict[name].extend(shap_value[i])
            shap_values_df = pd.DataFrame(shap_values_dict, columns=self.data.feature_names)
            X_test_dict = {name: [] for name in self.data.feature_names}
            for shap_value in X_test:
                for i, name in enumerate(self.data.feature_names):
                    X_test_dict[name].extend(shap_value[i])
            X_test_df = pd.DataFrame(X_test_dict, columns=self.data.feature_names)
            feature_importance = shap_values_df.abs().mean().sort_values(ascending=False)

            sorted_feature_names = feature_importance.index.tolist()

            shap_values_sorted = shap_values_df[sorted_feature_names]
            X_test_sorted = X_test_df[sorted_feature_names]

            plt.figure()
            shap.summary_plot(
                shap_values_sorted.values,
                X_test_sorted.values,
                feature_names=shap_values_sorted.columns,
                show=False
            )
            plt.tight_layout()
            plt.savefig(os.path.join(self.config['result_folder_path'], 'shap_summary_plot.png'))
            plt.show()
            plt.close()
            save_shap_stats(self, shap_values_df)

        elif self.config['model_type'] == 'ResNet':
            explainer = shap.GradientExplainer(self.model, shap.sample(X_train, 20))
            shap_values = explainer.shap_values(X_test)
            shap_values_dict = {name: [] for name in self.data.feature_names}
            for shap_value in shap_values:
                for i, name in enumerate(self.data.feature_names):
                    shap_values_dict[name].extend(shap_value[i])
            shap_values_df = pd.DataFrame(shap_values_dict, columns=self.data.feature_names)
            X_test_dict = {name: [] for name in self.data.feature_names}
            for shap_value in X_test:
                for i, name in enumerate(self.data.feature_names):
                    X_test_dict[name].extend(shap_value[i])
            X_test_df = pd.DataFrame(X_test_dict, columns=self.data.feature_names)
            feature_importance = shap_values_df.abs().mean().sort_values(ascending=False)

            sorted_feature_names = feature_importance.index.tolist()

            shap_values_sorted = shap_values_df[sorted_feature_names]
            X_test_sorted = X_test_df[sorted_feature_names]

            plt.figure()
            shap.summary_plot(
                shap_values_sorted.values,
                X_test_sorted.values,
                feature_names=shap_values_sorted.columns,
                show=False
            )
            plt.tight_layout()
            plt.savefig(os.path.join(self.config['result_folder_path'], 'shap_summary_plot.png'))
            plt.show()
            plt.close()
            save_shap_stats(self, shap_values_df)

        elif self.config['model_type'] == 'MLP_Mixer':
            pass
        else:
            return 0

    else:
        explainer = shap.KernelExplainer(self.model.predict_proba, shap.sample(X_train, 50))
        shap_values = explainer.shap_values(X_test)
        shap_values_df = pd.DataFrame(shap_values[:, :, 1], columns=self.data.flattened_column_names)
        plot_shap(self, shap_values_df, X_test)
