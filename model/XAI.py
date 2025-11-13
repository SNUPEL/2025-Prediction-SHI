import dice_ml
import pandas as pd
import os
import numpy as np
import torch
from get_ConvLSTM import create_convlstm_5d_sequences_internal
from raiutils.exceptions import UserConfigValidationException
import warnings
from pandas.errors import PerformanceWarning
warnings.filterwarnings('ignore', category=PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class ModelWrapper:
    def __init__(self, prediction_function):
        self.prediction_function = prediction_function

    def predict_proba(self, x):
        return self.prediction_function(x)


def prediction_wrapper(model, reshaper, backend='sklearn'):
    def predict_proba(x):
        x_np = x.values if isinstance(x, pd.DataFrame) else x

        x_reshaped = reshaper(x_np)

        if backend == 'pytorch':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x_reshaped).to(device)
                outputs = model(x_tensor)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                probs = torch.sigmoid(outputs).cpu().numpy()
                return np.hstack((1 - probs, probs))
        elif backend in ['tensorflow', 'keras']:
            probs = model.predict(x_reshaped, verbose=0)
            return np.hstack((1 - probs, probs))
        elif backend == 'sklearn':
            if model == 'VotingClassifier':
                pass
            else:
                if hasattr(model, 'predict_proba'):
                    return model.predict_proba(x_reshaped)
                elif hasattr(model, 'decision_function'):
                    scores = model.decision_function(x_reshaped)
                    probs = 1 / (1 + np.exp(-scores))
                    return np.vstack([1 - probs, probs]).T
                else:
                    raise AttributeError("모델에 .predict_proba() 또는 .decision_function() 메소드가 없습니다.")
        else:
            return model.predict_proba(x_reshaped)

    return predict_proba


def get_dice(config, data, model):
    """
    DiCE 모델 적용
    """
    dice_params = config['dice_parameter']

    num_features = len(data.feature_names)
    timesteps = config['data_duration']

    # 모델 별 입력 데이터 변형 함수 적용
    base_reshaper = lambda x: x.reshape(-1, num_features, timesteps)

    if config['data_shape'] == 'flatten':
        reshaper = lambda x: x
    elif config['model_type'] == 'ConvLSTM':
        sub_window_size = config['sub_window_size']
        reshaper = lambda x: create_convlstm_5d_sequences_internal(base_reshaper(x), sub_window_size)
    elif config['model_type'] in ['MultiChannelCNNLSTM', 'ResNet']:
        reshaper = lambda x: np.expand_dims(base_reshaper(x), axis=-1)
    elif config['model_type'] == 'Transformer':
        reshaper = lambda x: base_reshaper(x).transpose(0, 2, 1)
    elif config['model_type'] == 'MLP_Mixer':
        reshaper = base_reshaper
    else:
        print(f"DiCE를 지원하지 않는 모델 타입: {config['model_type']}")
        return 0

    # 훈련 데이터 복사
    train_df_for_dice = data.df_x_train_flatten.copy()
    train_df_for_dice['label'] = data.df_y_train['label'].values
    feature_names = data.df_x_train_flatten.columns.tolist()

    # DiCE 모델 구성
    dice_data = dice_ml.Data(dataframe=train_df_for_dice, continuous_features=feature_names, outcome_name='label')
    wrapped_predict_function = prediction_wrapper(model.model, reshaper, backend=config['back_end'])
    wrapped_model_object = ModelWrapper(wrapped_predict_function)
    dice_model = dice_ml.Model(model=wrapped_model_object, backend='sklearn')

    explainable_model = dice_ml.Dice(dice_data, dice_model, method=dice_params['method'])

    # 반사실적 설명 생성할 대상 지정
    X_test_flat = data.df_x_test_flatten
    predicted_positives = (data.df_y_pred['label'] == 1).values
    misclassified = (data.df_y_test['label'] != data.df_y_pred['label']).values
    query_mode = dice_params['query_instance_mode']
    if query_mode == 'predicted_positives':
        query_instances = X_test_flat[predicted_positives]
        print(f"설명 대상: '경영악화'로 예측된 {len(query_instances)}개 샘플")
    elif query_mode == 'misclassified':
        query_instances = X_test_flat[misclassified]
        print(f"설명 대상: 잘못 예측된 {len(query_instances)}개 샘플")
    else:
        print(f"알 수 없는 query_instance_mode: '{query_mode}'")
        return
    # 만약 경영 악화 예측 샘플이 없는 경우 적용
    if query_instances.empty:
        print("설명할 대상 샘플이 없습니다.")
        return

    last_timestep_index = int(config['data_duration'] - 1)

    # n개월 및 feature에 대해 변경 가능한 특성 지정
    features_to_vary = [
        col for col in feature_names
        # 조건 1과 2를 한 줄에 결합
        if any(str(config['data_duration'] - i) in col for i in range(1, dice_params['month_length'] + 1)) and \
           not any(word in col for word in dice_params['features_to_ban'])
    ]

    if not features_to_vary:
        print(f"경고: 마지막 타임스텝({last_timestep_index}) 피처를 찾지 못했습니다. 전체 피처를 대상으로 합니다.")
        features_to_vary = feature_names

    percentage_change = dice_params['percentage_change']

    # 반 사실적 설명 생성 및 저장
    cf_examples_list = []
    for i in range(len(query_instances)):
        single_query_instance = query_instances.iloc[[i]]
        instance_name = single_query_instance.index[0]

        # feature 별 실제 조정 가능 범위 생성
        dynamic_permitted_range = {}
        for feature_name in features_to_vary:
            base_feature_name = feature_name.rsplit('_', 1)[0]

            scaler = None
            if config['scaler']:
                if config['scale_by'] == 'feature':
                    scaler = data.scaler_dict[base_feature_name]
                elif config['scale_by'] == 'feature_and_company':
                    company_id = instance_name.split('_')[0]
                    scaler = data.scaler_for_sheet_dict[base_feature_name][company_id]
            # scaling 고려
            if scaler:
                scaled_value = single_query_instance[feature_name].iloc[0]
                original_value = scaler.inverse_transform(np.array([[scaled_value]]))[0, 0]
            else:
                original_value = single_query_instance[feature_name].iloc[0]

            if original_value > 0:
                min_val_orig = original_value * (1 - percentage_change)
                max_val_orig = original_value * (1 + percentage_change)
            elif original_value < 0:
                min_val_orig = original_value * (1 + percentage_change)
                max_val_orig = original_value * (1 - percentage_change)
            else:
                # 원본 값이 0인 경우, 역정규화 후 작은 절대값 범위를 줌
                min_val_orig, max_val_orig = -0.05, 0.05
            if scaler:
                min_max_orig_array = np.array([[min_val_orig], [max_val_orig]])
                scaled_range = scaler.transform(min_max_orig_array).flatten()
                dynamic_permitted_range[feature_name] = sorted(list(scaled_range))
            else:
                dynamic_permitted_range[feature_name] = [min_val_orig, max_val_orig]

        # 설명 가능한 대안 생성 시도
        try:
            dice_explain_single = explainable_model.generate_counterfactuals(
                single_query_instance,
                total_CFs=dice_params['total_CFs'],
                desired_class=dice_params['desired_class'],
                permitted_range=dynamic_permitted_range,
                features_to_vary=list(dynamic_permitted_range.keys())
            )
            if dice_explain_single.cf_examples_list:
                cf_examples_list.append(dice_explain_single.cf_examples_list[0])
                print(f" - 샘플 '{instance_name}'에 대한 대안을 생성했습니다.")
            else:
                print(f" - 샘플 '{instance_name}'에 대한 대안을 찾지 못했습니다. (결과 비어있음)")

        except UserConfigValidationException:
            print(f" - 샘플 '{instance_name}'에 대한 대안을 찾지 못했습니다. (생성 실패)")
            continue

    # 설명 가능한 대안을 샘플 별로 최종 저장
    cf_folder_path = os.path.join(config['result_folder_path'], 'counterfactuals')
    os.makedirs(cf_folder_path, exist_ok=True)

    for i, cf_example in enumerate(cf_examples_list):
        instance_name = query_instances.index[i]
        if cf_example.final_cfs_df is not None and not cf_example.final_cfs_df.empty:
            original_df = query_instances.iloc[[i]]
            total_result = pd.concat([original_df, cf_example.final_cfs_df], ignore_index=True)
            total_result.index = ['원본 데이터'] + [f'대안 {j + 1}' for j in range(len(cf_example.final_cfs_df))]

            columns_to_save = features_to_vary.copy()
            outcome_name = explainable_model.data_interface.outcome_name
            if outcome_name not in columns_to_save:
                columns_to_save.append(outcome_name)
            columns_to_save = [col for col in columns_to_save if col in total_result.columns]

            result_to_save = total_result[columns_to_save].copy()

            for feature_col in result_to_save.columns:
                if feature_col in feature_names:
                    base_feature_name = feature_col.rsplit('_', 1)[0]

                    scaler = None
                    if config['scaler']:
                        if config['scale_by'] == 'feature':
                            scaler = data.scaler_dict[base_feature_name]
                        elif config['scale_by'] == 'feature_and_company':
                            company_id = instance_name.split('_')[0]
                            scaler = data.scaler_for_sheet_dict[base_feature_name][company_id]

                    if scaler:
                        scaled_values = total_result[feature_col].values.reshape(-1, 1)
                        original_values = scaler.inverse_transform(scaled_values)
                        result_to_save[feature_col] = original_values.flatten()

            result_to_save.to_csv(os.path.join(cf_folder_path, f'cf_{instance_name}.csv'), index=True,
                                  encoding='utf-8-sig')
        else:
            print(f" - 샘플 '{instance_name}'에 대한 대안을 찾지 못했습니다.")
