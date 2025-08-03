import dice_ml
import pandas as pd
import os


def get_dice(config, data, model):
    dice_params = config['dice_parameter']

    if config['undersampling'] or config['oversampling']:
        train_df = data.df_train_after_sampling
    else:
        train_df = data.df_train

    feature_names = [col for col in train_df.columns if col != 'label']

    dice_data = dice_ml.Data(dataframe=train_df, continuous_features=feature_names, outcome_name='label')
    dice_model = dice_ml.Model(model=model.model, backend=config['back_end'])
    explainable_model = dice_ml.Dice(dice_data, dice_model, method=dice_params['method'])

    predicted_positives = data.df_y_pred['label'] == 1
    misclassified = data.df_y_test['label'] != data.df_y_pred['label']

    query_mode = dice_params['query_instance_mode']
    if query_mode == 'predicted_positives':
        query_instances = data.df_x_test_flatten[predicted_positives]
        print(f"설명 대상: '경영악화'로 예측된 {len(query_instances)}개 샘플")
    elif query_mode == 'misclassified':
        query_instances = data.df_x_test_flatten[misclassified]
        print(f"설명 대상: 잘못 예측된 {len(query_instances)}개 샘플")
    else:
        print(f"알 수 없는 query_instance_mode: '{query_mode}'")
        return

    if query_instances.empty:
        print("설명할 대상 샘플이 없습니다.")
        return

    features_to_vary = []
    for substring in dice_params['features_to_vary_substrings']:
        features_to_vary.extend([f for f in feature_names if substring in f])

    permitted_range = {feature: dice_params['permitted_range'] for feature in feature_names}

    dice_explain = explainable_model.generate_counterfactuals(
        query_instances,
        total_CFs=dice_params['total_CFs'],
        desired_class=dice_params['desired_class'],
        permitted_range=permitted_range,
        features_to_vary=features_to_vary if features_to_vary else 'all'
    )

    cf_folder_path = os.path.join(config['result_folder_path'], 'counterfactuals')
    os.makedirs(cf_folder_path, exist_ok=True)

    for i, cf_example in enumerate(dice_explain.cf_examples_list):
        instance_name = query_instances.index[i]
        if cf_example.final_cfs_df is not None and not cf_example.final_cfs_df.empty:
            original_df = query_instances.iloc[[i]]
            total_result = pd.concat([original_df, cf_example.final_cfs_df], ignore_index=True)
            total_result.index = ['원본 데이터'] + [f'대안 {j + 1}' for j in range(len(cf_example.final_cfs_df))]

            # 개별 CSV 파일로 저장
            total_result.to_csv(os.path.join(cf_folder_path, f'cf_{instance_name}.csv'), index=True,
                                encoding='utf-8-sig')
        else:
            print(f" - 샘플 '{instance_name}'에 대한 대안을 찾지 못했습니다.")


def grad_cam(data, model):
    pass
