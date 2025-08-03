import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import os
from multichannel_models import MultiChannelCNNLSTM


class MultiChannelCNNLSTM(nn.Module):
    """
    MultiChannel CNN-LSTM 모델

    Args:
        n_channels: 채널 수 (시트 수)
        time_points: 시간 포인트 수 (데이터 기간)
        feature_dim: 각 채널의 피처 차원 (일반적으로 1)
        hidden_size: LSTM 은닉층 크기
        cnn_filters: CNN 필터 수
        kernel_size: CNN 커널 크기
        dropout: 드롭아웃 비율
        use_channel_attention: 채널 어텐션 사용 여부
        use_temporal_attention: 시간적 어텐션 사용 여부
    """

    def __init__(self, n_channels, time_points, feature_dim=1, hidden_size=128,
                 cnn_filters=64, kernel_size=3, dropout=0.5,
                 use_channel_attention=True, use_temporal_attention=True):
        super(MultiChannelCNNLSTM, self).__init__()

        self.n_channels = n_channels
        self.time_points = time_points
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.cnn_filters = cnn_filters
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.use_channel_attention = use_channel_attention
        self.use_temporal_attention = use_temporal_attention

        # 각 채널별 독립적인 CNN 처리
        self.channel_cnns = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(feature_dim, cnn_filters, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.BatchNorm1d(cnn_filters),
                nn.Dropout(dropout),
                nn.Conv1d(cnn_filters, cnn_filters, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.BatchNorm1d(cnn_filters),
                nn.Dropout(dropout)
            ) for _ in range(n_channels)
        ])

        # 각 채널별 독립적인 LSTM 처리
        self.channel_lstms = nn.ModuleList([
            nn.LSTM(cnn_filters, hidden_size, batch_first=True, dropout=dropout)
            for _ in range(n_channels)
        ])

        # 채널 어텐션 메커니즘
        if use_channel_attention:
            self.channel_attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1)
            )

        # 시간적 어텐션 메커니즘
        if use_temporal_attention:
            self.temporal_attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1)
            )

        # 최종 분류기
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: 입력 데이터 (batch_size, n_channels, time_points, feature_dim)

        Returns:
            output: 분류 결과 (batch_size, 1)
        """
        batch_size = x.size(0)

        # 각 채널별 독립적인 처리
        channel_outputs = []

        for i in range(self.n_channels):
            # 채널 i 데이터 추출: (batch_size, time_points, feature_dim)
            channel_data = x[:, i, :, :]

            # Conv1d를 위해 차원 변경: (batch_size, feature_dim, time_points)
            channel_data = channel_data.transpose(1, 2)

            # CNN 처리
            cnn_out = self.channel_cnns[i](channel_data)

            # LSTM을 위해 차원 변경: (batch_size, time_points, cnn_filters)
            cnn_out = cnn_out.transpose(1, 2)

            # LSTM 처리
            lstm_out, (h_n, c_n) = self.channel_lstms[i](cnn_out)

            # 시간적 어텐션 적용
            if self.use_temporal_attention:
                # 어텐션 가중치 계산
                attention_weights = self.temporal_attention(lstm_out)  # (batch_size, time_points, 1)
                attention_weights = F.softmax(attention_weights, dim=1)

                # 가중 평균 계산
                channel_output = torch.sum(lstm_out * attention_weights, dim=1)  # (batch_size, hidden_size)
            else:
                # 마지막 시간 단계의 출력 사용
                channel_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)

            channel_outputs.append(channel_output)

        # 모든 채널 출력을 결합: (batch_size, n_channels, hidden_size)
        channel_outputs = torch.stack(channel_outputs, dim=1)

        # 채널 어텐션 적용
        if self.use_channel_attention:
            # 어텐션 가중치 계산
            attention_weights = self.channel_attention(channel_outputs)  # (batch_size, n_channels, 1)
            attention_weights = F.softmax(attention_weights, dim=1)

            # 가중 평균 계산
            final_output = torch.sum(channel_outputs * attention_weights, dim=1)  # (batch_size, hidden_size)
        else:
            # 단순 평균
            final_output = torch.mean(channel_outputs, dim=1)  # (batch_size, hidden_size)

        # 최종 분류
        output = self.classifier(final_output)

        return output


def get_MultiChannelCNNLSTM(self):
    if self.config['undersampling'] or self.config['oversampling']:
        x_dict = self.data.df_x_train_matrix_dict_after_sampling
        y_dict = self.data.df_y_train_dict_after_sampling
    else:
        x_dict = self.data.df_x_train_matrix_dict
        y_dict = self.data.df_y_train_dict

        # 훈련 데이터: (샘플, 특성, 시간) 형태의 3D NumPy 배열로 변환
    X_train_3d = np.array([df.values for df in x_dict.values()])
    y_train = np.array(list(y_dict.values()))

    # 테스트 데이터
    X_test_3d = np.array([df.values for df in self.data.df_x_test_matrix_dict.values()])

    # MultiChannel 모델을 위한 4D 형태로 변환 (샘플, 채널=특성, 시간, 1)
    X_train = np.expand_dims(X_train_3d, axis=-1)
    X_test = np.expand_dims(X_test_3d, axis=-1)

    print(f"최종 학습 데이터 형태: {X_train.shape}")

    # --- 2. PyTorch 학습 설정 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(self.config['random_state'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(self.config['random_state'])

    # DataLoader 생성
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=self.config['fit_parameter']['batch_size'],
        shuffle=True
    )

    model_params = self.config['model_parameter']
    model = MultiChannelCNNLSTM(
        n_channels=X_train.shape[1],
        time_points=X_train.shape[2],
        feature_dim=X_train.shape[3],
        **model_params
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_params = self.config['optimizer_parameter']
    optimizer = optim.Adam(model.parameters(), **optimizer_params)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.config['scheduler_parameter'])

    fit_params = self.config['fit_parameter']
    best_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(fit_params['epochs']):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{fit_params["epochs"]}', leave=False):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # 간단한 검증 (여기서는 테스트셋을 검증셋으로 활용)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            outputs = model(X_test_tensor.to(device))
            y_test_tensor = torch.FloatTensor(np.array(list(self.data.df_y_test_dict.values()))).unsqueeze(1).to(device)
            val_loss = criterion(outputs, y_test_tensor).item()
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch + 1}/{fit_params['epochs']} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            if self.config['save_model']:
                torch.save(model.state_dict(), os.path.join(self.config['result_folder_path'], 'best_model.pth'))
        else:
            patience_counter += 1

        if patience_counter >= fit_params['early_stopping_patience']:
            print(f"조기 종료: {fit_params['early_stopping_patience']} 에포크 동안 성능 개선 없음")
            break

    if self.config['save_model']:
        model.load_state_dict(torch.load(os.path.join(self.config['result_folder_path'], 'best_model.pth')))

    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor.to(device))
        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities > self.config['threshold']).int()
        test_predictions = predictions.cpu().numpy().flatten()

    self.data.df_y_pred = pd.DataFrame(
        test_predictions,
        index=self.data.name_test,
        columns=['label']
    )

    self.model = model  # 나중에 모델 자체를 사용하기 위해 저장
    self.history = history  # 학습 곡선 저장을 위해 history 저장
