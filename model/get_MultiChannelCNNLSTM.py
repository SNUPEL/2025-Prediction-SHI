# PyTorch 기본 모듈: 텐서 연산과 GPU 가속을 위한 핵심 라이브러리
import torch
# 신경망 계층을 구성하기 위한 모듈
import torch.nn as nn
# 손실 함수 및 추가 연산을 위한 함수형 API
import torch.nn.functional as F
# 최적화 알고리즘 모음
import torch.optim as optim
# 텐서 기반 데이터셋/로더 유틸리티
from torch.utils.data import TensorDataset, DataLoader
# 학습 진행도를 시각적으로 확인하기 위한 프로그레스바
from tqdm.auto import tqdm
# 수치 연산 편의를 위한 NumPy
import numpy as np
# 표 형태 데이터 처리를 위한 pandas
import pandas as pd
# 모델 저장 경로 생성을 위해 os 사용
import os
# 정확도, 재현율 계산을 위한 scikit-learn 메트릭
from sklearn.metrics import accuracy_score, recall_score


class MultiChannelCNNLSTM(nn.Module):
    """
    MultiChannel CNN-LSTM 모델

    Args:
        n_channels: 채널 수 (시트 수)
        time_points: 시간 포인트 수 (데이터 기간)
        feature_dim: 각 채널의 피처 차원 (일반적으로 1)
        hidden_size: LSTM 은닉층 크기 (lstm_layers가 없을 때 사용)
        lstm_layers: LSTM 층별 설정 리스트 [{'hidden_size': int, 'dropout': float}, ...]
        cnn_filters: CNN 필터 수 (cnn_layers가 없을 때 사용)
        cnn_layers: CNN 층별 설정 리스트 [{'filters': int, 'kernel_size': int, 'dropout': float}, ...]
        kernel_size: CNN 커널 크기 (cnn_layers가 없을 때 사용)
        dropout: 드롭아웃 비율
        use_channel_attention: 채널 어텐션 사용 여부
        use_temporal_attention: 시간적 어텐션 사용 여부
    """

    def __init__(self, n_channels, time_points, feature_dim=1, hidden_size=128,
                 lstm_layers=None, cnn_filters=64, cnn_layers=None, kernel_size=3, dropout=0.5,
                 use_channel_attention=True, use_temporal_attention=True,
                 use_bidirectional=False, use_transformer=False, 
                 transformer_position='after_lstm', positional_encoding=False,
                 transformer_config=None, use_residual_connection=False, residual_weight=0.1,
                 use_cross_channel_transformer=False, cross_channel_transformer_config=None):
        super(MultiChannelCNNLSTM, self).__init__()

        # 사용자 설정이 없을 경우 기본 CNN 층 구성을 정의
        # CNN 층 설정 (없으면 기본값 사용)
        if cnn_layers is None:
            cnn_layers = [{'filters': cnn_filters, 'kernel_size': kernel_size, 'dropout': dropout}]
        
        # 사용자 설정이 없을 경우 기본 LSTM 층 구성을 정의
        # LSTM 층 설정 (없으면 기본값 사용)
        if lstm_layers is None:
            lstm_layers = [{'hidden_size': hidden_size, 'dropout': dropout}]
        
        # 하이퍼파라미터 구성을 인스턴스 상태로 보관
        self.cnn_layers_config = cnn_layers
        self.lstm_layers_config = lstm_layers
        # CNN 블록 마지막 층의 필터 수를 기억해 이후 계층 크기를 계산
        final_cnn_filters = cnn_layers[-1]['filters']  # 마지막 CNN 층의 필터 수
        # LSTM 블록 마지막 층의 hidden_size를 기억해 이후 연산에 활용
        final_hidden_size = lstm_layers[-1]['hidden_size']  # 마지막 LSTM 층의 hidden_size

        # 입력 데이터의 구조 정보를 저장
        self.n_channels = n_channels
        self.time_points = time_points
        self.feature_dim = feature_dim
        # 공통 드롭아웃 비율 관리
        self.dropout = dropout
        # 어텐션 사용 여부 플래그
        self.use_channel_attention = use_channel_attention
        self.use_temporal_attention = use_temporal_attention
        self.use_bidirectional = use_bidirectional
        self.use_transformer = use_transformer
        self.transformer_position = transformer_position
        self.positional_encoding = positional_encoding
        self.use_residual_connection = use_residual_connection
        self.residual_weight = residual_weight
        self.use_cross_channel_transformer = use_cross_channel_transformer

        # 채널별 CNN 블록을 저장할 리스트
        # 각 채널별 다층 CNN 처리
        self.channel_cnns = nn.ModuleList()
        for _ in range(n_channels):
            # 채널마다 동일 구조의 CNN 서브모델을 구성
            cnn_list = nn.ModuleList()
            # 첫 Conv1d에서 사용할 입력 채널 수 지정 (특징 차원)
            input_channels = feature_dim
            
            for i, layer_config in enumerate(cnn_layers):
                # 현재 CNN 층의 필터 수와 커널 크기를 읽어온다
                layer_filters = layer_config['filters']
                layer_kernel_size = layer_config['kernel_size']
                # 층별 드롭아웃 비율이 지정되지 않았으면 기본값 사용
                layer_dropout = layer_config.get('dropout', dropout)
                
                # CNN 층 + 활성화 + 정규화 + 드롭아웃
                cnn_block = nn.Sequential(
                    nn.Conv1d(input_channels, layer_filters, 
                             kernel_size=layer_kernel_size, 
                             padding=layer_kernel_size // 2),
                    nn.ReLU(),
                    nn.BatchNorm1d(layer_filters),
                    nn.Dropout(layer_dropout)
                )
                
                # 구성한 블록을 채널별 리스트에 추가
                cnn_list.append(cnn_block)
                # 다음 CNN 층의 입력 채널 수를 현재 필터 수로 갱신
                input_channels = layer_filters  # 다음 층의 입력 채널 수
            
            # 완성된 CNN 서브모델을 전체 채널 모듈리스트에 등록
            self.channel_cnns.append(cnn_list)

        # 채널별로 독립적인 LSTM 블록을 구성
        # 각 채널별 다층 LSTM 처리 (CNN 출력을 입력으로 받음)
        self.channel_lstms = nn.ModuleList()
        for _ in range(n_channels):
            # 채널마다 여러 층의 LSTM을 순차적으로 쌓는다
            lstm_list = nn.ModuleList()
            # 첫 LSTM 입력 크기는 CNN 최종 필터 수와 동일
            input_size = final_cnn_filters  # CNN의 마지막 층 출력을 입력으로 사용
            
            for i, layer_config in enumerate(lstm_layers):
                # 현재 LSTM 층의 hidden_size를 읽어온다
                layer_hidden_size = layer_config['hidden_size']
                
                # 단일 층 LSTM (bidirectional 옵션 추가)
                lstm_layer = nn.LSTM(input_size, layer_hidden_size, 
                                   batch_first=True, dropout=0.0,
                                   bidirectional=self.use_bidirectional)
                
                # 구성한 LSTM 레이어를 채널별 리스트에 추가
                lstm_list.append(lstm_layer)
                # Bidirectional LSTM은 출력이 2배가 되므로 다음 층 입력 크기 조정
                input_size = layer_hidden_size * (2 if self.use_bidirectional else 1)
            
            # 채널별 LSTM 레이어 묶음을 등록
            self.channel_lstms.append(lstm_list)

        # LSTM 층간 dropout 레이어들을 빌드하여 과적합 방지
        # LSTM 층간 dropout 레이어들
        self.lstm_dropouts = nn.ModuleList([
            nn.Dropout(layer_config.get('dropout', dropout)) 
            for layer_config in lstm_layers[:-1]  # 마지막 층 제외
        ])

        # 양방향 LSTM을 쓰면 출력 차원이 2배가 되므로 이를 저장
        # Bidirectional 사용시 출력 차원이 2배가 됨
        self.bidirectional_factor = 2 if use_bidirectional else 1
        adjusted_hidden_size = final_hidden_size * self.bidirectional_factor

        # 시계열 특징을 Transformer가 처리할지 여부에 따라 블록 구성
        # Transformer 블록 추가
        if use_transformer:
            # transformer_position에 따라 d_model 동적 설정
            if transformer_position == 'after_cnn':
                # CNN 출력 차원 사용
                transformer_d_model = final_cnn_filters
            else:  # after_lstm
                # LSTM 출력 차원 사용 (bidirectional 고려)
                transformer_d_model = adjusted_hidden_size
            
            if transformer_config is None:
                transformer_config = {}
            
            # config에서 제공된 값 또는 동적 계산된 값 사용
            transformer_config = {
                'num_heads': transformer_config.get('num_heads', 8),
                'num_encoder_layers': transformer_config.get('num_encoder_layers', 2),
                'd_model': transformer_d_model,  # 동적으로 설정
                'dim_feedforward': transformer_config.get('dim_feedforward', transformer_d_model * 4),
                'dropout': transformer_config.get('dropout', 0.1)
            }
            
            # d_model이 num_heads로 나누어떨어지는지 확인 및 조정
            if transformer_d_model % transformer_config['num_heads'] != 0:
                # num_heads를 조정
                transformer_config['num_heads'] = min(transformer_config['num_heads'], transformer_d_model)
                while transformer_d_model % transformer_config['num_heads'] != 0:
                    transformer_config['num_heads'] -= 1
                    if transformer_config['num_heads'] < 1:
                        transformer_config['num_heads'] = 1
                        break
            
            # 시간 축을 입력 시퀀스로 사용하는 Transformer 인코더 레이어 구성
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=transformer_d_model,
                nhead=transformer_config['num_heads'],
                dim_feedforward=transformer_config.get('dim_feedforward', transformer_d_model * 4),
                dropout=transformer_config.get('dropout', 0.1),
                batch_first=True
            )
            # 지정된 층 수만큼 Transformer 인코더를 쌓는다
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=transformer_config.get('num_encoder_layers', 2)
            )
            
            # Transformer 설정 저장 (디버깅용)
            self.transformer_config = transformer_config

        # 채널 간 가중치를 학습하기 위한 어텐션 모듈
        # 채널 어텐션 메커니즘 (adjusted_hidden_size 사용)
        if use_channel_attention:
            self.channel_attention = nn.Sequential(
                nn.Linear(adjusted_hidden_size, adjusted_hidden_size // 2),
                nn.ReLU(),
                nn.Linear(adjusted_hidden_size // 2, 1)
            )

        # 시간 축에서 중요한 시점을 강조하기 위한 어텐션 모듈
        # 시간적 어텐션 메커니즘 (adjusted_hidden_size 사용)
        if use_temporal_attention:
            self.temporal_attention = nn.Sequential(
                nn.Linear(adjusted_hidden_size, adjusted_hidden_size // 2),
                nn.ReLU(),
                nn.Linear(adjusted_hidden_size // 2, 1)
            )

        # 채널 간 상호작용을 학습하기 위한 Transformer 구성
        # Cross-Channel Transformer 추가 (채널 간 상호작용)
        if use_cross_channel_transformer:
            if cross_channel_transformer_config is None:
                cross_channel_transformer_config = {}
            
            # 채널 차원을 sequence로 사용하므로 d_model은 adjusted_hidden_size
            cross_channel_d_model = adjusted_hidden_size
            
            # config 설정
            cross_config = {
                'num_heads': cross_channel_transformer_config.get('num_heads', 4),
                'num_encoder_layers': cross_channel_transformer_config.get('num_encoder_layers', 1),
                'd_model': cross_channel_d_model,
                'dim_feedforward': cross_channel_transformer_config.get('dim_feedforward', cross_channel_d_model * 4),
                'dropout': cross_channel_transformer_config.get('dropout', 0.1)
            }
            
            # num_heads가 d_model로 나누어떨어지는지 확인
            if cross_channel_d_model % cross_config['num_heads'] != 0:
                cross_config['num_heads'] = min(cross_config['num_heads'], cross_channel_d_model)
                while cross_channel_d_model % cross_config['num_heads'] != 0:
                    cross_config['num_heads'] -= 1
                    if cross_config['num_heads'] < 1:
                        cross_config['num_heads'] = 1
                        break
            
            # 채널 축을 입력으로 삼는 Transformer 인코더 레이어 정의
            cross_encoder_layer = nn.TransformerEncoderLayer(
                d_model=cross_channel_d_model,
                nhead=cross_config['num_heads'],
                dim_feedforward=cross_config['dim_feedforward'],
                dropout=cross_config['dropout'],
                batch_first=True
            )
            # 여러 층을 중첩한 Cross-Channel Transformer 구성
            self.cross_channel_transformer = nn.TransformerEncoder(
                cross_encoder_layer,
                num_layers=cross_config['num_encoder_layers']
            )
            # 설정값을 보관해 디버깅 시 참고
            self.cross_channel_config = cross_config
        
        # LSTM/Transformer 결과를 기반으로 최종 이진 분류를 수행하는 MLP
        # 최종 분류기 (adjusted_hidden_size 사용)
        self.classifier = nn.Sequential(
            nn.Linear(adjusted_hidden_size, adjusted_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(adjusted_hidden_size // 2, 1)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: 입력 데이터 (batch_size, n_channels, time_points, feature_dim)

        Returns:
            output: 분류 결과 (batch_size, 1)
        """
        # 입력 텐서의 배치 크기를 기록해 이후 연산에서 재사용
        batch_size = x.size(0)

        # 각 채널별 독립적인 처리
        channel_outputs = []

        for i in range(self.n_channels):
            # 채널 i 데이터 추출: (batch_size, time_points, feature_dim)
            channel_data = x[:, i, :, :]

            # Conv1d를 위해 차원 변경: (batch_size, feature_dim, time_points)
            cnn_input = channel_data.transpose(1, 2)

            # 다층 CNN 처리
            for j, cnn_layer in enumerate(self.channel_cnns[i]):
                # 동일 채널에 대해 구성된 CNN 블록을 순차적으로 적용
                cnn_input = cnn_layer(cnn_input)

            # LSTM을 위해 차원 변경: (batch_size, time_points, final_cnn_filters)
            lstm_input = cnn_input.transpose(1, 2)
            
            # Transformer 적용 (after_cnn 위치)
            if self.use_transformer and self.transformer_position == 'after_cnn':
                # CNN 출력 시점에 Transformer를 적용하도록 설정된 경우
                lstm_input = self.transformer(lstm_input)

            # 다층 LSTM 처리
            # 잔차 연결이 필요하면 원본 CNN 출력을 복사하여 보관
            lstm_original = lstm_input.clone() if self.use_residual_connection else None
            
            for j, lstm_layer in enumerate(self.channel_lstms[i]):
                # LSTM 순전파: 시계열 특징을 순차적으로 요약
                lstm_out, (h_n, c_n) = lstm_layer(lstm_input)
                
                # Residual Connection 적용 (첫 번째 층과 차원이 같을 때만)
                if self.use_residual_connection and j == 0 and lstm_out.shape == lstm_original.shape:
                    # 첫 번째 LSTM 층에서만 동일 차원이라면 잔차 연결을 적용
                    lstm_out = lstm_out + lstm_original * self.residual_weight
                
                # 마지막 층이 아니면 dropout 적용
                if j < len(self.channel_lstms[i]) - 1:
                    # 중간층 출력에 드롭아웃을 적용하여 과적합을 줄인다
                    lstm_out = self.lstm_dropouts[j](lstm_out)
                
                lstm_input = lstm_out  # 다음 층의 입력으로 사용
            
            # Transformer 적용 (after_lstm 위치)
            if self.use_transformer and self.transformer_position == 'after_lstm':
                # LSTM 출력 직후 Transformer를 적용하도록 설정된 경우
                lstm_out = self.transformer(lstm_out)

            # 시간적 어텐션 적용
            if self.use_temporal_attention:
                # 어텐션 가중치 계산
                attention_weights = self.temporal_attention(lstm_out)  # (batch_size, time_points, 1)
                attention_weights = F.softmax(attention_weights, dim=1)

                # 가중 평균 계산
                channel_output = torch.sum(lstm_out * attention_weights, dim=1)  # (batch_size, final_hidden_size)
            else:
                # 마지막 시간 단계의 출력 사용
                channel_output = lstm_out[:, -1, :]  # (batch_size, final_hidden_size)

            # 계산된 채널별 표현을 리스트에 추가
            channel_outputs.append(channel_output)

        # 모든 채널 출력을 결합: (batch_size, n_channels, final_hidden_size)
        channel_outputs = torch.stack(channel_outputs, dim=1)
        
        # Cross-Channel Transformer 적용 (채널 간 상호작용 학습)
        if self.use_cross_channel_transformer:
            # 채널을 sequence로 취급하여 Transformer 적용
            # (batch_size, n_channels, hidden_size) -> Transformer -> (batch_size, n_channels, hidden_size)
            channel_outputs = self.cross_channel_transformer(channel_outputs)

        # 채널 어텐션 적용
        if self.use_channel_attention:
            # 어텐션 가중치 계산
            attention_weights = self.channel_attention(channel_outputs)  # (batch_size, n_channels, 1)
            attention_weights = F.softmax(attention_weights, dim=1)

            # 가중 평균 계산
            final_output = torch.sum(channel_outputs * attention_weights, dim=1)  # (batch_size, final_hidden_size)
        else:
            # 단순 평균
            final_output = torch.mean(channel_outputs, dim=1)  # (batch_size, final_hidden_size)

        # 최종 분류
        output = self.classifier(final_output)

        return output


def get_MultiChannelCNNLSTM(self):
    # 샘플링 여부에 따라 사용할 학습 데이터를 선택
    if self.config['undersampling'] or self.config['oversampling']:
        x_dict = self.data.df_x_train_matrix_dict_after_sampling
        y_dict = self.data.df_y_train_dict_after_sampling
    else:
        x_dict = self.data.df_x_train_matrix_dict
        y_dict = self.data.df_y_train_dict

    # 훈련 데이터: (샘플, 특성, 시간) 형태의 3D NumPy 배열로 변환
    X_train_3d = np.array([df.values for df in x_dict.values()])
    y_train = np.array(list(y_dict.values()))

    # 검증 데이터
    X_valid_3d = np.array([df.values for df in self.data.df_x_valid_matrix_dict.values()])
    y_valid = np.array(list(self.data.df_y_valid_dict.values()))

    # 테스트 데이터
    X_test_3d = np.array([df.values for df in self.data.df_x_test_matrix_dict.values()])

    # MultiChannel 모델을 위한 4D 형태로 변환 (샘플, 채널=특성, 시간, 1)
    X_train = np.expand_dims(X_train_3d, axis=-1)
    X_valid = np.expand_dims(X_valid_3d, axis=-1)
    X_test = np.expand_dims(X_test_3d, axis=-1)

    # --- 2. PyTorch 학습 설정 ---
    # GPU 사용 가능 시 CUDA, 아니면 CPU를 선택
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 재현성을 위해 시드 고정
    torch.manual_seed(self.config['random_state'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(self.config['random_state'])

    # DataLoader 생성
    # Numpy 배열을 PyTorch 텐서로 변환
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_valid_tensor = torch.FloatTensor(X_valid)
    y_valid_tensor = torch.FloatTensor(y_valid).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)

    # 텐서 쌍을 Dataset으로 감싸서 배치 로딩을 쉽게 한다
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)

    # 학습 데이터 로더: 셔플을 켜서 매 에포크 샘플 순서를 섞는다
    train_loader = DataLoader(
        train_dataset,
        batch_size=self.config['fit_parameter']['batch_size'],
        shuffle=True
    )
    # 검증 데이터 로더: 순서 고정을 위해 shuffle=False
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=self.config['fit_parameter']['batch_size'],
        shuffle=False
    )

    # 모델 구성 파라미터를 config에서 읽어와 초기화
    model_params = self.config['model_parameter']
    model = MultiChannelCNNLSTM(
        n_channels=X_train.shape[1],
        time_points=X_train.shape[2],
        feature_dim=X_train.shape[3],
        **model_params
    ).to(device)

    # 이 강력한 모델과 함께 손실 함수 가중치(pos_weight)를 사용하는 것이 매우 중요합니다.
    # pos_weight_value = torch.tensor([10.0]).to(device) # (거래중 샘플 수 / 경영악화 샘플 수)
    pos_weight_value = torch.tensor([50.0]).to(device) # (거래중 샘플 수 / 경영악화 샘플 수)

    # 클래스 불균형 보정을 위해 pos_weight를 적용한 BCEWithLogitsLoss 사용
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_value)
    # criterion = nn.BCEWithLogitsLoss()
    
    # 최적화기와 스케줄러 파라미터를 설정에서 불러와 초기화
    optimizer_params = self.config['optimizer_parameter']
    optimizer = optim.Adam(model.parameters(), **optimizer_params)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.config['scheduler_parameter'])

    # 학습 관련 하이퍼파라미터 묶음
    fit_params = self.config['fit_parameter']
    best_loss = float('inf')
    patience_counter = 0
    # 추후 시각화를 위해 손실과 메트릭을 누적할 기록 변수
    history = {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_recall': [], 'val_recall': []
    }

    def calculate_metrics(outputs, targets, threshold=0.5):
        """메트릭 계산 함수"""
        # 로짓을 시그모이드로 변환해 확률로 변환
        probs = torch.sigmoid(outputs)
        # 임계값을 기준으로 이진 예측 생성
        preds = (probs > threshold).int()
        
        # CPU로 이동해 넘파이 배열로 변환
        y_true = targets.cpu().numpy().flatten()
        y_pred = preds.cpu().numpy().flatten()
        
        # 정확도와 재현율을 계산
        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        return accuracy, recall

    for epoch in range(fit_params['epochs']):
        # === 훈련 단계 ===
        model.train()
        train_loss = 0.0
        train_outputs_list = []
        train_targets_list = []
        
        for batch_X, batch_y in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{fit_params["epochs"]}', leave=False):
            # 배치를 장치로 이동
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            # 이전 배치의 기울기 초기화
            optimizer.zero_grad()
            # 모델 순전파 수행
            outputs = model(batch_X)
            # 로짓 기반 BCE 손실 계산
            loss = criterion(outputs, batch_y)
            # 역전파로 기울기 계산
            loss.backward()
            # 옵티마이저로 파라미터 갱신
            optimizer.step()
            # 배치 손실을 누적
            train_loss += loss.item()
            
            # 메트릭 계산을 위해 저장
            train_outputs_list.append(outputs.detach())
            train_targets_list.append(batch_y.detach())

        avg_train_loss = train_loss / len(train_loader)
        
        # 훈련 메트릭 계산
        # 각 배치의 예측/정답을 하나로 연결해 전체 메트릭을 계산
        train_outputs_all = torch.cat(train_outputs_list, dim=0)
        train_targets_all = torch.cat(train_targets_list, dim=0)
        train_accuracy, train_recall = calculate_metrics(train_outputs_all, train_targets_all, self.config['threshold'])
        
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['train_recall'].append(train_recall)

        # === 검증 단계 ===
        model.eval()
        val_loss = 0.0
        val_outputs_list = []
        val_targets_list = []
        
        with torch.no_grad():
            for batch_X, batch_y in valid_loader:
                # 검증 배치 역시 장치로 이동
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                # 순전파만 수행하여 예측 획득
                outputs = model(batch_X)
                # 검증 손실 계산
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                # 메트릭 계산을 위해 저장
                val_outputs_list.append(outputs.detach())
                val_targets_list.append(batch_y.detach())
                
        avg_val_loss = val_loss / len(valid_loader)
        
        # 검증 메트릭 계산
        val_outputs_all = torch.cat(val_outputs_list, dim=0)
        val_targets_all = torch.cat(val_targets_list, dim=0)
        val_accuracy, val_recall = calculate_metrics(val_outputs_all, val_targets_all, self.config['threshold'])
        
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_recall'].append(val_recall)

        print(f"Epoch {epoch + 1}/{fit_params['epochs']} - "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, "
              f"Train Recall: {train_recall:.4f}, Val Recall: {val_recall:.4f}")

        # Plateau 스케줄러로 검증 손실을 모니터링하여 학습률 조정
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            if self.config['save_model']:
                torch.save(model.state_dict(), os.path.join(self.config['result_folder_path'], 'best_model.pth'))
        else:
            # 성능 향상이 없으면 인내심 카운터 증가
            patience_counter += 1

        if patience_counter >= fit_params['early_stopping_patience']:
            print(f"조기 종료: {fit_params['early_stopping_patience']} 에포크 동안 성능 개선 없음")
            break

    if self.config['save_model']:
        # 저장해둔 최고 성능 가중치를 불러와 최종 평가에 사용
        model.load_state_dict(torch.load(os.path.join(self.config['result_folder_path'], 'best_model.pth')))

    # === 테스트 예측 ===
    model.eval()
    with torch.no_grad():
        # 테스트 입력 전체를 장치로 올려 예측 수행
        outputs = model(X_test_tensor.to(device))
        # 로짓을 확률로 변환
        probabilities = torch.sigmoid(outputs)
        # 설정된 임계값을 기준으로 이진 결과를 얻는다
        predictions = (probabilities > self.config['threshold']).int()
        # 평가 지표 기록을 위해 CPU 넘파이 배열로 변환
        test_predictions = predictions.cpu().numpy().flatten()
        test_probabilities = probabilities.cpu().numpy().flatten()

    self.data.df_y_pred_proba = pd.DataFrame(
        test_probabilities,
        index=self.data.name_test,
        columns=['label']
    )
    self.data.df_y_pred = pd.DataFrame(
        test_predictions,
        index=self.data.name_test,
        columns=['label']
    )

    # 추후 재사용을 위해 학습된 모델 인스턴스를 저장
    self.model = model  # 나중에 모델 자체를 사용하기 위해 저장
    # 학습 곡선 및 메트릭 기록을 상위 객체에 전달
    self.history = history  # 학습 곡선 저장을 위해 history 저장
