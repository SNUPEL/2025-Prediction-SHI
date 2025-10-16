# config.py 파일

import os
import time
import pandas as pd
from tensorflow.keras.metrics import Recall


def create_config():
    config = dict()

    # 랜덤 시드 설정
    config['random_state'] = 42  # 랜덤 시드

    # 데이터 파일 및 보조 파일 경로 설정
    config['data_file_path'] = '../data/사내협력사 현황(철수사&거래 협력사)_ Data_최종수정본_3.xlsx'
    config['sub_data_file_path'] = '../data/사내협력사 현황(철수사&거래 협력사)_출근인력(추가).xlsx'

    # 데이터 사용의 가장 이른 시작 시점 설정
    config['data_start_date'] = '2016-02-01'
    # 입력 데이터 기간(길이) 설정
    config['data_duration'] = 12
    # 예측하는 시점을 정의(해당 일자까지 데이터가 존재한다고 가정)
    config['label_date'] = '2024-06-01'
    # 라벨 판단 기준에 필요한 길이
    config['label_duration'] = 3
    # True면 경/중 경을 포함하지 않음 (label이 True인 기간만 제외, False인 기간은 사용)
    config['severity_label_type'] = False
    # train, test 겹치는 기간 사용 여부
    config['overlap'] = True
    # validation set 비율 0~1
    config['validation_ratio'] = 0.1

    # 데이터 로딩 시 제외할 시트 이름 목록
    # config['sheet_ban_list'] = ['경영 영향1', '경영 영향2', '종합평가', '입사자', '입사율', '퇴사율', '본공률(시급월급)',
    #                             '4대보험 가입자', '본공률(4대보험)', '안전사고 건수(중간, 낮음)', '안전사고 건수(높음)',
    #                             '공수능률',  '시급,월급제 인원', '투입인원', '환산능률']
    config['sheet_ban_list'] = ['경영 영향1', '경영 영향2', '종합평가', '본공률(4대보험)', '본공률(시급월급)',
                                '4대보험 가입자', '안전사고 건수(중간, 낮음)', '안전사고 건수(높음)', '종합평가_점수_과거']

    # 정규화: None, standard
    config['scaler'] = 'standard'
    # 정규화 기준: feature, feature_and_company
    config['scale_by'] = 'feature'

    # 데이터 분석 및 가시화 그래프를 저장
    config['save_graph'] = False

    # undersampling: None, tomek_link, ENN, nearmiss
    # oversampling: None, SMOTE, BorderlineSMOTE
    config['sampling_order'] = ['undersampling', 'oversampling']

    # ML model: 'RandomForestClassifier', 'AdaBoostClassifier', 'ExtraTreesClassifier',
    # 'RidgeClassifier', 'SGDClassifier', 'XGBClassifier', 'SVC', 'nuSVC', 'VotingClassifier'
    # DL model: 'ConvLSTM', 'MultiChannelCNNLSTM', 'Transformer', 'ResNet', 'MLP_Mixer', 'AutoEncoder'
    config['model_type'] = 'RidgeClassifier'  # 사용할 모델 타입
    config['shap_analysis'] = True
    config['DiCE'] = False  # True, False

    model_config = {
        'RandomForestClassifier': {
            'back_end': 'sklearn',
            'data_shape': 'flatten',
            'undersampling': 'ENN',
            'ENN_parameter': {'sampling_strategy': 'auto', 'n_neighbors': 200},
            'oversampling': 'SMOTE',
            'SMOTE_parameter': {'sampling_strategy': 0.5, 'k_neighbors': 30},
            'TSSMOTE_parameter': {'sampling_strategy': 0.5, 'k_neighbors': 30},
            'model_parameter': {'n_estimators': 1000, 'max_depth': 25, 'min_samples_split': 2,      # n_estimators: 트리의 개수, max_depth: 트리의 최대 한도 깊이, min_samples_split: 자식 노드를 갖기 위한 최소한의 데이터 개수
                                'min_samples_leaf': 1, 'max_features': 'sqrt'},     # min_samples_leaf: 리프 노드의 최소 데이터 개수, max_features: 자식 노드를 최적으로 분할할 수 있는 특성의 개수 ('sqrt', 'log2', None, 실수값 지정)
            'class_weight': {0: 1, 1: 1000}     # 클래스별 가중치 ('balanced', {0(거래중): a, 1(경영악화): b}(a, b는 자연수))
        },

        'AdaBoostClassifier': {
            'back_end': 'sklearn',
            'data_shape': 'flatten',
            'undersampling': 'ENN',
            'ENN_parameter': {'sampling_strategy': 'auto', 'n_neighbors': 200},
            'oversampling': 'SMOTE',
            'SMOTE_parameter': {'sampling_strategy': 0.5, 'k_neighbors': 30},
            'TSSMOTE_parameter': {'sampling_strategy': 0.5, 'k_neighbors': 30},
            'model_parameter': {'n_estimators': 1000, "learning_rate": 0.001},      # n_estimators: estimator의 개수, learning_rate: 학습률
            'estimator_parameter': {'max_depth': 2, 'min_samples_split': 2},    # estimator로 사용하는 DecisionTreeClassifier의 파라미터      # max_depth: 트리의 최대 한도 깊이, min_samples_split: 자식 노드를 갖기 위한 최소한의 데이터 개수
            'class_weight': {0: 1, 1: 1000}     # 클래스별 가중치 ('balanced', {0(거래중): a, 1(경영악화): b}(a, b는 자연수))
        },

        'ExtraTreesClassifier': {
            'back_end': 'sklearn',
            'data_shape': 'flatten',
            'undersampling': 'ENN',
            'ENN_parameter': {'sampling_strategy': 'auto', 'n_neighbors': 200},
            'oversampling': 'SMOTE',
            'SMOTE_parameter': {'sampling_strategy': 0.5, 'k_neighbors': 30},
            'TSSMOTE_parameter': {'sampling_strategy': 0.5, 'k_neighbors': 30},
            'model_parameter': {'n_estimators': 1000, 'max_depth': 50, 'min_samples_split': 2,      # n_estimators: 트리의 개수, max_depth: 트리의 최대 한도 깊이, min_samples_split: 자식 노드를 갖기 위한 최소한의 데이터 개수
                                'min_samples_leaf': 1, 'max_features': 'sqrt'},     # min_samples_leaf: 리프 노드의 최소 데이터 개수, max_features: 자식 노드를 최적으로 분할할 수 있는 특성의 개수 ('sqrt', 'log2', None, 정수/실수값 지정)
            'class_weight': {0: 1, 1: 1000}     # 클래스별 가중치 ('balanced', {0(거래중): a, 1(경영악화): b}(a, b는 자연수))
        },

        'RidgeClassifier': {
            'back_end': 'sklearn',
            'data_shape': 'flatten',
            'undersampling': None,
            'ENN_parameter': {'sampling_strategy': 'auto', 'n_neighbors': 200},
            'oversampling': None,
            'SMOTE_parameter': {'sampling_strategy': 0.5, 'k_neighbors': 30},
            'TSSMOTE_parameter': {'sampling_strategy': 0.5, 'k_neighbors': 201},
            # get_RidgeClassifier_optimized를 이용해서 최적화한 파라미터(label_date: 2023-03-01 기준)
            'model_parameter': {
                'alpha': 0.9887,          # 규제 강도
                'solver': 'auto',      # 계산 알고리즘 ('auto', 'svd', 'cholesky', 'sparse_cg', 'lsqr', 'sag', 'saga', 'lbfgs')
                'tol': 1e-4            # 중단 기준 정밀도 ('svd', 'cholesky', 'sparse_cg', 'lsqr', 'sag', 'saga', 'lbfgs', 실수값)
            },
            'class_weight': {0: 6, 1: 770}      # 클래스별 가중치 ('balanced', {0(거래중): a, 1(경영악화): b}(a, b는 자연수))

        },

        'SGDClassifier': {
            'back_end': 'sklearn',
            'data_shape': 'flatten',
            'undersampling': 'ENN',
            'ENN_parameter': {'sampling_strategy': 'auto', 'n_neighbors': 200},
            'oversampling': 'SMOTE',
            'SMOTE_parameter': {'sampling_strategy': 0.5, 'k_neighbors': 30},
            'TSSMOTE_parameter': {'sampling_strategy': 0.5, 'k_neighbors': 30},
            'model_parameter': {
                'loss': 'hinge',       # 손실 함수 ('hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive')
                'penalty': 'l2',       # 규제 종류 ('l2', 'l1', 'elasticnet', None)
                'alpha': 0.0001,       # 규제 강도
                'max_iter': 1000,      # 최대 반복 횟수(에포크)
                'learning_rate': 'optimal'  # 학습률 스케줄 ('constant', 'optimal', 'invscaling', 'adaptive')
            },
            'class_weight': {0: 1, 1: 1000}  # 클래스별 가중치 ('balanced', {0(거래중): a, 1(경영악화): b}(a, b는 자연수))
        },

        'XGBClassifier': {
            'back_end': 'xgboost',
            'data_shape': 'flatten',
            'undersampling': 'ENN',
            'ENN_parameter': {'sampling_strategy': 'auto', 'n_neighbors': 200},
            'oversampling': 'SMOTE',
            'SMOTE_parameter': {'sampling_strategy': 0.5, 'k_neighbors': 30},
            'TSSMOTE_parameter': {'sampling_strategy': 0.5, 'k_neighbors': 30},
            'model_parameter': {
                'n_estimators': 1000,       # 트리의 개수
                'learning_rate': 0.01,      # 학습률
                'max_depth': 6,             # 트리의 최대 한도 깊이
                'gamma': 0.1,               # 리프 노드 분할을 위한 최소 손실 감소
                'min_child_weight': 1,      # 자식 노드에 필요한 최소 가중치 합
            },

        },

        'SVC': {
            'back_end': 'sklearn',
            'data_shape': 'flatten',
            'undersampling': 'ENN',
            'ENN_parameter': {'sampling_strategy': 'auto', 'n_neighbors': 200},
            'oversampling': 'SMOTE',
            'SMOTE_parameter': {'sampling_strategy': 0.5, 'k_neighbors': 30},
            'TSSMOTE_parameter': {'sampling_strategy': 0.5, 'k_neighbors': 30},
            'model_parameter': {
                'probability': True,   # 확률 추정 사용 여부, XAI 사용시 True
                'C': 0.188609965,      # 규제 파라미터. 작을수록 규제가 강함
                'kernel': 'rbf',       # 커널 종류 ('linear', 'rbf', 'poly', 'sigmoid')
                'gamma': 'scale'       # 커널 계수 ('scale', 'auto' 또는 실수값)
            },
            'class_weight': {0: 1, 1: 374}      # 클래스별 가중치 ('balanced', {0(거래중): a, 1(경영악화): b}(a, b는 자연수))
        },

        'nuSVC': {
            'back_end': 'sklearn',
            'data_shape': 'flatten',
            'undersampling': 'ENN',
            'ENN_parameter': {'sampling_strategy': 'auto', 'n_neighbors': 200},
            'oversampling': 'TSSMOTE',
            'SMOTE_parameter': {'sampling_strategy': 0.5, 'k_neighbors': 30},
            'TSSMOTE_parameter': {'sampling_strategy': 0.5, 'k_neighbors': 30},
            'model_parameter': {
                'probability': True,   # 확률 추정 사용 여부, XAI 사용시 True
                'nu': 0.01,            # 오류와 서포트 벡터의 비율을 조정하는 파라미터
                'kernel': 'rbf',       # 커널 종류 ('linear', 'rbf', 'poly', 'sigmoid')
                'gamma': 'scale'       # 커널 계수 ('scale', 'auto' 또는 실수값)
            },
            'class_weight': {0: 1, 1: 1000}         # 클래스별 가중치 ('balanced', {0(거래중): a, 1(경영악화): b}(a, b는 자연수))
        },

        'VotingClassifier': {
            'back_end': 'sklearn',
            'data_shape': 'flatten',
            'undersampling': 'ENN',
            'ENN_parameter': {'sampling_strategy': 'auto', 'n_neighbors': 200},
            'oversampling': 'TSSMOTE',
            'SMOTE_parameter': {'sampling_strategy': 0.5, 'k_neighbors': 30},
            'TSSMOTE_parameter': {'sampling_strategy': 0.5, 'k_neighbors': 30},
            'SVC_parameter': {
                'probability': True,  # 확률 추정 사용 여부, XAI 사용시 True
                'C': 0.1,  # 규제 파라미터. 작을수록 규제가 강함
                'kernel': 'rbf',  # 커널 종류 ('linear', 'rbf', 'poly', 'sigmoid')
                'gamma': 'scale'  # 커널 계수 ('scale', 'auto' 또는 실수값)
            },
            'nuSVC_parameter': {
                'probability': True,  # 확률 추정 사용 여부, XAI 사용시 True
                'nu': 0.01,
                'kernel': 'rbf',  # 커널 종류 ('linear', 'rbf', 'poly', 'sigmoid')
                'gamma': 'scale'  # 커널 계수 ('scale', 'auto' 또는 실수값)
            },
            'AdaBoostClassifier_parameter': {'n_estimators': 1000, "learning_rate": 0.001},     # n_estimators: estimator의 개수, learning_rate: 학습률
            'RidgeClassifier_parameter': {
                'alpha': 0.978892658777735,  # 규제 강도
                'solver': 'auto',  # 계산 알고리즘
                'tol': 1e-4  # 중단 기준 정밀도
            },
            'model_parameter': {
                'voting': 'hard',   # voting 수준: soft(소프트 보팅)/hard(하드 보팅)
                'weights': [2, 3, 3, 5],   # estimator 가중치, SVC-nuSVC-AdaBoost-Ridge 순서
                'n_jobs': None,       # 사용할 cpu 코어 개수
                'flatten_transform': True       # voting이 soft일 때만 사용, transform output에 영향을 미치는 요소
            },
            'class_weight': {0: 1, 1: 1000}
        },

        'AutoEncoder': {
            'back_end': 'tensorflow',
            'data_shape': 'flatten',
            'undersampling': None,
            'oversampling': None,
            'model_parameter': {
                'encoder_layers': [
                    {'units': 1024, 'activation': 'relu'},
                    {'units': 256, 'activation': 'relu'},
                    {'units': 64, 'activation': 'relu'}
                ],
                'latent_space': {'units': 16, 'activation': 'relu'},
                'decoder_layers': [
                    {'units': 64, 'activation': 'relu'},
                    {'units': 256, 'activation': 'relu'},
                    {'units': 1024, 'activation': 'relu'}
                ],
                'output_activation': 'linear'
            },
            'compile_parameter': {
                'learning_rate': 0.001,
                'loss': 'mean_squared_error'
            },
            'fit_parameter': {
                'epochs': 100,
                'batch_size': 64,
                'shuffle': True
            },
            'threshold_percentile': 97.5
        },

        'ConvLSTM': {
            'back_end': 'tensorflow',
            'data_shape': 'matrix',
            'undersampling': 'ENN',
            'ENN_parameter': {'sampling_strategy': 'auto', 'n_neighbors': 100},
            'oversampling': None,
            'SMOTE_parameter': {'sampling_strategy': 0.5, 'k_neighbors': 30},
            'TSSMOTE_parameter': {'sampling_strategy': 0.5, 'k_neighbors': 10},
            'sub_window_size': 4,
            'model_parameter': {
                'dropout_rate': 0.3,
                'convlstm_layers': [
                    {'filters': 32, 'kernel_width': 2, 'padding': "same", 'return_sequences': True, 'activation': 'relu'},
                    {'filters': 16, 'kernel_width': 2, 'padding': "same", 'return_sequences': True, 'activation': 'relu'},
                    {'filters': 8, 'kernel_width': 2, 'padding': "same", 'return_sequences': True, 'activation': 'relu'}
                ],
                'bilstm_layers': [
                    {'units': 32, 'return_sequences': True, 'activation': 'relu', 'l2_reg': 0.001},
                    {'units': 16, 'return_sequences': True, 'activation': 'relu', 'l2_reg': 0.001},
                    {'units': 8, 'return_sequences': False, 'activation': 'relu', 'l2_reg': 0.001}
                ],
                'output_layer': {'units': 1, 'activation': 'sigmoid'}
            },
            'compile_parameter': {
                'learning_rate': 0.0001,
                'clipnorm': 1.0,
                'loss': 'binary_crossentropy',
                'metrics': ['accuracy', Recall()]
            },
            'fit_parameter': {
                'epochs': 100,
                'batch_size': 64,
                'class_weight': {0: 1, 1: 20},
                'shuffle': True
            },
            'threshold': 0.5
        },

        'MultiChannelCNNLSTM': {
            'back_end': 'pytorch',
            'data_shape': 'multichannel',
            'undersampling': 'ENN',
            # 다중 채널 모델은 과도한 이상치 제거를 위해 ENN(Edited Nearest Neighbours)을 기본으로 사용
            # 'undersampling': None,
            # 'ENN_parameter': {'sampling_strategy': 'auto', 'n_neighbors': 20},
            # ENN 이웃 수를 30으로 확대해 필요 이상의 샘플 제거를 방지
            'ENN_parameter': {'sampling_strategy': 'auto', 'n_neighbors': 30},
            # 'ENN_parameter': {'sampling_strategy': 'auto', 'n_neighbors': 15},
            # 'oversampling': 'SMOTE',
            # 시계열 채널 간 상관이 높으므로 오버샘플링은 기본적으로 비활성화
            'oversampling': None,
            # 'oversampling': 'Borderline-TSSMOTE',  # Borderline-TSSMOTE, TSSMOTE, SMOTE
            # 필요 시 사용할 수 있는 oversampling 파라미터 세트
            'SMOTE_parameter': {'sampling_strategy': 0.1, 'k_neighbors': 5},
            'TSSMOTE_parameter': {'sampling_strategy': 1.0, 'k_neighbors': 15},
            'model_parameter': {
                # 기본 LSTM hidden size (층 정보가 없을 때 사용)
                'hidden_size': 512,
                # 시계열 정보를 차례로 처리할 다층 LSTM 구성
                'lstm_layers': [
                    {'hidden_size': 128, 'dropout': 0.1},
                    {'hidden_size': 512, 'dropout': 0.1},
                    {'hidden_size': 64, 'dropout': 0.1}
                ],
                'cnn_filters': 512,  # 하위 호환성용 (cnn_layers가 없을 때 사용)
                'cnn_layers': [
                    {'filters': 128, 'kernel_size': 3, 'dropout': 0.1},
                    {'filters': 512, 'kernel_size': 3, 'dropout': 0.1},
                    {'filters': 64, 'kernel_size': 3, 'dropout': 0.1}
                ],
                'kernel_size': 2,
                'dropout': 0.5,
                'use_channel_attention': True,
                'use_temporal_attention': True,
                # Bi-LSTM 및 Transformer 설정 추가
                # 양방향 LSTM으로 전후 맥락을 함께 학습
                'use_bidirectional': True,  # Bi-LSTM 사용 여부
                # 시계열 추출 후 특징 보강을 위해 Transformer 활성화
                'use_transformer': True,  # Transformer 사용 여부
                # Transformer를 적용할 위치를 LSTM 뒤로 배치
                'transformer_position': 'after_lstm',  # 'after_cnn' or 'after_lstm'
                'positional_encoding': False,  # 채널 순서 무관하므로 False
                'transformer_config': {
                    'num_heads': 4,  # Multi-head attention 헤드 수
                    'num_encoder_layers': 2,  # Transformer encoder 층 수
                    # d_model은 자동 계산됨: after_cnn이면 CNN 마지막 필터 수, after_lstm이면 LSTM 마지막 hidden_size * bidirectional_factor
                    'dim_feedforward': 128,  # FFN 차원
                    'dropout': 0.1  # Transformer 내부 dropout
                },
                # Residual Connection 설정
                # 첫 LSTM 층 출력을 잔차 연결로 보강할지 여부
                'use_residual_connection': True,  # Residual Connection 사용 여부
                'residual_weight': 0.1,  # Residual Connection 가중치
                # Cross-Channel Transformer 설정
                # 채널 간 상호작용을 학습하기 위한 Transformer 사용 여부
                'use_cross_channel_transformer': True,  # 채널 간 Transformer 사용 여부
                'cross_channel_transformer_config': {
                    'num_heads': 4,  # 채널 수와 호환되도록 자동 조정됨
                    'num_encoder_layers': 2,  # Cross-channel encoder 층 수
                    'dim_feedforward': 128,  # FFN 차원
                    'dropout': 0.1  # Dropout
                }
            },
            'optimizer_parameter': {
                # Adam 학습률과 L2 정규화 계수
                'lr': 0.0001,
                'weight_decay': 1e-4
            },
            'scheduler_parameter': {
                'mode': 'min',
                'factor': 0.5,
                'patience': 10
            },
            'fit_parameter': {
                'epochs': 30,
                'batch_size': 64,
                'early_stopping_patience': 15
            },
            'threshold': 0.5
        },

        'Transformer': {
            'back_end': 'tensorflow',
            'data_shape': 'matrix',
            'undersampling': 'ENN',
            'ENN_parameter': {'sampling_strategy': 'auto', 'n_neighbors': 7},
            'oversampling': None,  # Borderline-TSSMOTE, TSSMOTE, SMOTE
            'SMOTE_parameter': {'sampling_strategy': 1.0, 'k_neighbors': 5},
            'TSSMOTE_parameter': {'sampling_strategy': 1.0, 'k_neighbors': 15},
            'model_parameter': {
                'embed_dim': 128,         # 각 타임스텝의 피처를 임베딩할 차원
                'num_blocks': 4,         # 쌓을 Transformer Encoder Block의 수
                'num_heads': 8,          # Multi-Head Attention의 헤드 수
                'dropout_rate': 0.1,
                'output_layer': {'units': 1, 'activation': 'sigmoid'}
            },
            'compile_parameter': {
                'learning_rate': 0.0001,
                'weight_decay': 0.01,
                'loss': 'binary_crossentropy',
                'metrics': ['accuracy', Recall(name='recall')]  # Recall 메트릭 추가
            },
            'fit_parameter': {
                'epochs': 50,
                'batch_size': 64,
                'class_weight': 'auto',
                'shuffle': True
            },
            'threshold': 0.5,
            'use_early_stopping': True,  # 콜백 사용 여부 설정
        },

        'ResNet': {
            'back_end': 'tensorflow',
            'data_shape': 'matrix',
            'undersampling': None,
            'ENN_parameter': {'sampling_strategy': 'auto', 'n_neighbors': 50},
            'oversampling': None,
            'SMOTE_parameter': {'sampling_strategy': 0.5, 'k_neighbors': 30},
            'TSSMOTE_parameter': {'sampling_strategy': 0.5, 'k_neighbors': 10},
            'model_parameter': {
                'dropout_rate': 0.5,
                'cnn_layers': [
                    {'filters': 32, 'kernel_size': (3, 3), 'padding': "same"},
                    {'filters': 64, 'kernel_size': (3, 3), 'padding': "same"},
                    {'filters': 128, 'kernel_size': (3, 3), 'padding': "same"},
                    {'filters': 256, 'kernel_size': (3, 3), 'padding': "same"}
                ],
                'output_layer': {'units': 1, 'activation': 'sigmoid'}
                },
            'compile_parameter': {
                'learning_rate': 0.001,
                'clipnorm': 1.0,
                'loss': 'binary_crossentropy',
                'metrics': ['accuracy', Recall()]
            },
            'fit_parameter': {
                'epochs': 10,
                'batch_size': 64,
                'class_weight': {0: 1, 1: 30},
                'shuffle': True
            },
            'threshold': 0.5
        },

        'MLP_Mixer': {
            'back_end': 'tensorflow',
            'data_shape': 'matrix',
            'undersampling': None,
            'ENN_parameter': {'sampling_strategy': 'auto', 'n_neighbors': 50},
            'oversampling': None,
            'SMOTE_parameter': {'sampling_strategy': 0.5, 'k_neighbors': 30},
            'TSSMOTE_parameter': {'sampling_strategy': 0.5, 'k_neighbors': 30},
            'model_parameter': {
                'dropout_rate': 0.5,
                'n_mixer_layers': 4,
                'hidden_dim': 32,
                'token_mlp_dim': 32,
                'channel_mlp_dim': 32,
                'output_layer': {'units': 1, 'activation': 'sigmoid'}
            },
            'compile_parameter': {
                'learning_rate': 0.001,
                'clipnorm': 1.0,
                'loss': 'binary_crossentropy',
                'metrics': ['accuracy']
            },
            'fit_parameter': {
                'epochs': 5,
                'batch_size': 64,
                'shuffle': True
            },
            'threshold': 0.5
        }
    }

    config.update(model_config[config['model_type']])

    config['dice_parameter'] = {
        # 실험 2: 변동가능 범위
        'percentage_change': 0.3,
        # 실험 1: 개월 수 6 -> 3
        'month_length': 4,
        'method': 'random',  # 설명 방식 ('random', 'genetic', 'kdtree')
        # 설명할 대상: 'predicted_positives', 'misclassified'
        'query_instance_mode': 'predicted_positives',
        'total_CFs': 1,  # 찾을 대안의 최대 개수
        'desired_class': 'opposite',  # 반대 클래스로 바뀌는 대안을 찾음
        'features_to_ban': ['기성매출', '종합평가', '경영 영향', '안전사고']  # 변경을 허용하지 않는 특성 리스트
    }

    # 학습된 모델 저장 여부 설정
    config['save_model'] = False  # 추가 구현 필요
    config['save_train_data'] = False
    config['save_validation_data'] = False
    config['save_test_data'] = False
    config['save_confusion_matrix'] = True  # 혼동 행렬 저장 여부
    config['save_loss_history'] = True

    # 결과 저장 폴더 설정 (타임스탬프 기반)
    config['ymd'] = time.strftime('%Y%m%d')
    config['hour'] = str(time.localtime().tm_hour)
    config['minute'] = str(time.localtime().tm_min)
    config['second'] = str(time.localtime().tm_sec)
    config["result_folder_path"] = '../results/{0}_{1}h_{2}m_{3}s'.format(
        config['ymd'], config['hour'], config['minute'], config['second'])

    # 결과 폴더 생성 (폴더가 없으면 생성, 상위 디렉토리도 함께 생성)
    if not os.path.exists(config["result_folder_path"]):
        os.makedirs(config["result_folder_path"], exist_ok=True)

    # 현재 설정 정보를 엑셀 파일로 저장
    config_df = pd.json_normalize(config, sep='_').transpose()
    config_df.to_excel(config['result_folder_path'] + '/configuration.xlsx', index=True)

    return config
