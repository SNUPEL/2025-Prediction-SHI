import os
import time
import pandas as pd

def create_config():
    """프로젝트의 모든 설정을 담는 딕셔너리를 생성합니다."""
    config = dict()

    # [1] 경로 및 기본 설정

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
    except NameError:
        project_root = os.getcwd()

    config['project_root'] = project_root
    config['data_file_path'] = os.path.join(project_root, 'data', '사내협력사 현황(철수사&거래 협력사)_ Data_추가(250417).xlsx')
    config['sub_data_file_path'] = os.path.join(project_root, 'data', '사내협력사 현황(철수사&거래 협력사)_출근인력(추가).xlsx')
    config['data_start_date'] = '2016-02-01'
    config['label_date'] = '2024-10-01'
    config['split_cutoff_date'] = '2023-05-01'
    config['data_duration'] = 12
    config['label_duration'] = 3
    config['sheet_ban_list'] = ['4대보험 가입자', '공수능률', '본공률(4대보험)', '본공률(시급월급)']
    config['random_state'] = 970517

    # [2] 모델 선택
    # 사용 가능 모델: 'RandomForestClassifier', 'AdaBoostClassifier', 'ExtraTreesClassifier',
    #              'RidgeClassifier', 'SGDClassifier', 'XGBClassifier', 'SVC',
    #              'ConvLSTM', 'Transformer'
    config['model_type'] = 'Transformer'

    # [3] 데이터 형태 및 전처리 설정
    if config['model_type'] in ['ConvLSTM', 'Transformer']:
        config['data_shape'] = 'matrix'
        config['oversampling'] = None
        config['undersampling'] = None
        config['use_smote_for_keras'] = False # 3D 데이터 형태일때 SMOTE 적용
        config['scaler'] = 'standard'
    else:
        config['data_shape'] = 'flatten'
        config['oversampling'] = 'SMOTE'
        config['undersampling'] = 'ENN'
        config['scaler'] = 'standard'

    config['ENN_n_neighbors'] = 5
    config['SMOTE_k_neighbors'] = 5
    config['SMOTE_sampling_strategy'] = 1

    # [4] 모델별 하이퍼파라미터

    # [4-1] Transformer 전용
    config['model_embed_dim'] = 128  # 입력 피처의 임베딩 차원 (각 월별 데이터가 변환되는 크기)
    config['transformer_ff_dim'] = 256  # Transformer 블록 내 Feed-Forward 네트워크의 내부 확장 차원 (embed_dim의 2~4배 권장)
    config['transformer_num_blocks'] = 4  # Transformer Encoder 블록의 수 (모델의 깊이)
    config['latent_dim'] = 64  # 최종 예측 직전, 시퀀스 요약 후 Dense 레이어의 차원 embed_dim의 같게 하거나 , 1/2 or 1/4 (과적합시)
    config['mha_num_heads'] = 8  # Multi-Head Attention 헤드의 수 (다양한 관점 학습)
    config['dropout_rate'] = 0.5  # 드롭아웃 비율 (과적합 방지)

    # [4-4] ConvLSTM 전용 파라미터
    config['sub_window_size'] = 4  # 추천: data_duration의 약수 중 하나

    # [4-3] 딥러닝(Keras) 공통
    config['epochs'] = 200
    config['batch_size'] = 32
    config['learning_rate'] = 1e-4
    config['weight_decay'] = 1e-4
    #config['class_weight'] = {0: 1, 1: 10} <- data.py에서 자동으로 class weight 계산 되도록 코드 수정함 **주석 해제시 수동 조정 가능!**

    # [4-4] AdaBoostClassifier 전용 파라미터
    config['AdaBoost_parameter'] = {
        'n_estimators': 100,  # 부스팅 단계 수 (기본 50, 더 늘려볼 수 있음)
        'learning_rate': 1.0,  # 각 약분류기의 기여도 (기본 1.0, 0.01~1.0 사이 조정)
        # 'algorithm': 'SAMME.R', # SAMME.R (실수 예측) 또는 SAMME (이산 예측), 기본 SAMME.R
    }
    config['AdaBoost_estimator_parameter'] = {
        'max_depth': 1,  # DecisionTreeClassifier의 깊이 (기본 1, '약한 학습기')
        # 'min_samples_leaf': 1,
    }


    # [5] 평가 및 결과 저장 설정
    config['save_model'] = True
    config['save_graph'] = True
    config['save_predictions'] = True
    config['save_confusion_matrix'] = True
    config['save_metrics'] = True
    config['run_shap_analysis'] = True # "Transformer" 구조에만 가능함....
    config['min_precision_for_threshold'] = 0.15


    # 결과 폴더 생성
    ymd_hms = time.strftime('%Y%m%d_%H%M%S')
    config["result_folder_path"] = os.path.join(project_root, 'results', f'{ymd_hms}_Advanced_{config["model_type"]}')
    if not os.path.exists(config["result_folder_path"]):
        os.makedirs(config["result_folder_path"])

    # 설정 파일 저장
    config_to_save = {k: str(v) if not isinstance(v, (str, int, float, bool, list, dict)) else v for k, v in config.items()}
    pd.DataFrame.from_dict(config_to_save, orient='index', columns=['value']).to_excel(
        os.path.join(config['result_folder_path'], 'configuration.xlsx')
    )
    return config