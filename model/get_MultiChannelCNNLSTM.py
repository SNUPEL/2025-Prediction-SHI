import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, recall_score


class MultiChannelCNNLSTM(nn.Module):
    """
    Hierarchical CNN-BiLSTM + Transformer Integrator Model
    - 1단계: 각 채널을 독립적인 (CNN -> Bi-LSTM -> Temporal Attention)으로 처리하여 '요약 벡터' 생성
    - 2단계: 모든 채널의 '요약 벡터'들을 Transformer Encoder에 입력하여 상호 관계를 학습하고 최종 예측
    """

    def __init__(self, n_channels, time_points, feature_dim=1, hidden_size=128,
                 lstm_layers=None, cnn_filters=64, cnn_layers=None, kernel_size=3, dropout=0.5,
                 # Transformer 파라미터 추가
                 transformer_heads=4, transformer_layers=2):
        super(MultiChannelCNNLSTM, self).__init__()

        # --- 기본 설정 ---
        if cnn_layers is None: cnn_layers = [{'filters': cnn_filters, 'kernel_size': kernel_size, 'dropout': dropout}]
        if lstm_layers is None: lstm_layers = [{'hidden_size': hidden_size, 'dropout': dropout}]
        
        self.n_channels = n_channels
        final_cnn_filters = cnn_layers[-1]['filters']
        # Bi-LSTM의 출력은 히든 사이즈의 2배
        final_bilstm_hidden_size = lstm_layers[-1]['hidden_size'] * 2

        # --- 1단계: 채널별 독립 분석기 (CNN + Bi-LSTM) ---
        self.channel_cnns = nn.ModuleList()
        for _ in range(n_channels):
            cnn_list = nn.ModuleList()
            input_channels = feature_dim
            for config in cnn_layers:
                cnn_list.append(nn.Sequential(
                    nn.Conv1d(input_channels, config['filters'], kernel_size=config['kernel_size'], padding=config['kernel_size'] // 2),
                    nn.ReLU(), nn.BatchNorm1d(config['filters']), nn.Dropout(config.get('dropout', dropout))
                ))
                input_channels = config['filters']
            self.channel_cnns.append(cnn_list)

        self.channel_lstms = nn.ModuleList()
        for _ in range(n_channels):
            lstm_list = nn.ModuleList()
            input_size = final_cnn_filters
            for config in lstm_layers:
                lstm_list.append(nn.LSTM(input_size, config['hidden_size'], batch_first=True, bidirectional=True))
                input_size = config['hidden_size'] * 2 # 다음 LSTM의 입력은 Bi-LSTM 출력이므로 2배
            self.channel_lstms.append(lstm_list)

        # 각 Bi-LSTM의 시계열 출력에서 중요한 부분을 요약하는 Temporal Attention
        self.temporal_attention = nn.Sequential(
            nn.Linear(final_bilstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # --- 2단계: Transformer 종합 분석기 ---
        transformer_d_model = final_bilstm_hidden_size # Transformer의 입력 차원
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_heads,
            dim_feedforward=transformer_d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # --- 3단계: 최종 분류기 ---
        self.classifier = nn.Sequential(
            nn.Linear(transformer_d_model, transformer_d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_d_model // 2, 1)
        )

    def forward(self, x):
        # --- 1단계: 채널별 독립 분석 ---
        channel_summaries = []
        for i in range(self.n_channels):
            # CNN 처리
            cnn_input = x[:, i, :, :].transpose(1, 2)
            for cnn_layer in self.channel_cnns[i]:
                cnn_input = cnn_layer(cnn_input)
            
            # Bi-LSTM 처리
            lstm_input = cnn_input.transpose(1, 2)
            for lstm_layer in self.channel_lstms[i]:
                lstm_input, _ = lstm_layer(lstm_input)

            # Temporal Attention으로 각 채널의 '분석 요약' 벡터 생성
            attn_weights = F.softmax(self.temporal_attention(lstm_input), dim=1)
            summary_vector = torch.sum(lstm_input * attn_weights, dim=1)
            channel_summaries.append(summary_vector)

        # --- 2단계: Transformer 종합 분석 ---
        # (batch_size, n_channels, feature_dim) 형태로 텐서 결합
        transformer_input = torch.stack(channel_summaries, dim=1)
        
        # Transformer Encoder로 채널 간 상호 관계 학습
        transformer_output = self.transformer_encoder(transformer_input)

        # --- 3단계: 최종 결정 ---
        # Transformer 출력의 평균을 내어 최종 벡터 생성
        final_vector = transformer_output.mean(dim=1)
        
        output = self.classifier(final_vector)
        return output


def get_MultiChannelCNNLSTM(self):
    if self.config['undersampling'] or self.config['oversampling']:
        x_dict = self.data.df_x_train_matrix_dict_after_sampling
        y_dict = self.data.df_y_train_dict_after_sampling
    else:
        x_dict = self.data.df_x_train_matrix_dict
        y_dict = self.data.df_y_train_dict

    X_train_3d = np.array([df.values for df in x_dict.values()])
    y_train = np.array(list(y_dict.values()))

    X_valid_3d = np.array([df.values for df in self.data.df_x_valid_matrix_dict.values()])
    y_valid = np.array(list(self.data.df_y_valid_dict.values()))

    X_test_3d = np.array([df.values for df in self.data.df_x_test_matrix_dict.values()])

    X_train = np.expand_dims(X_train_3d, axis=-1)
    X_valid = np.expand_dims(X_valid_3d, axis=-1)
    X_test = np.expand_dims(X_test_3d, axis=-1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(self.config['random_state'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(self.config['random_state'])

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_valid_tensor = torch.FloatTensor(X_valid)
    y_valid_tensor = torch.FloatTensor(y_valid).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=self.config['fit_parameter']['batch_size'],
        shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=self.config['fit_parameter']['batch_size'],
        shuffle=False
    )

    model_params = self.config['model_parameter']
    # use_channel_attention, use_temporal_attention 파라미터는 더 이상 모델에서 직접 사용하지 않음
    model_params.pop('use_channel_attention', None)
    model_params.pop('use_temporal_attention', None)

    model = MultiChannelCNNLSTM(
        n_channels=X_train.shape[1],
        time_points=X_train.shape[2],
        feature_dim=X_train.shape[3],
        **model_params
    ).to(device)

    # 이 강력한 모델과 함께 손실 함수 가중치(pos_weight)를 사용하는 것이 매우 중요합니다.
    pos_weight_value = torch.tensor([10.0]).to(device) # (거래중 샘플 수 / 경영악화 샘플 수)
    # pos_weight_value = torch.tensor([50.0]).to(device) # (거래중 샘플 수 / 경영악화 샘플 수)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_value)
    # criterion = nn.BCEWithLogitsLoss()
    
    optimizer_params = self.config['optimizer_parameter']
    optimizer = optim.Adam(model.parameters(), **optimizer_params)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.config['scheduler_parameter'])

    fit_params = self.config['fit_parameter']
    best_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_recall': [], 'val_recall': []
    }

    def calculate_metrics(outputs, targets, threshold=0.5):
        probs = torch.sigmoid(outputs)
        preds = (probs > threshold).int()
        
        y_true = targets.cpu().numpy().flatten()
        y_pred = preds.cpu().numpy().flatten()
        
        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        return accuracy, recall

    for epoch in range(fit_params['epochs']):
        model.train()
        train_loss = 0.0
        train_outputs_list = []
        train_targets_list = []
        
        for batch_X, batch_y in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{fit_params["epochs"]}', leave=False):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            train_outputs_list.append(outputs.detach())
            train_targets_list.append(batch_y.detach())

        avg_train_loss = train_loss / len(train_loader)
        
        train_outputs_all = torch.cat(train_outputs_list, dim=0)
        train_targets_all = torch.cat(train_targets_list, dim=0)
        train_accuracy, train_recall = calculate_metrics(train_outputs_all, train_targets_all, self.config['threshold'])
        
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['train_recall'].append(train_recall)

        model.eval()
        val_loss = 0.0
        val_outputs_list = []
        val_targets_list = []
        
        with torch.no_grad():
            for batch_X, batch_y in valid_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                val_outputs_list.append(outputs.detach())
                val_targets_list.append(batch_y.detach())
                
        avg_val_loss = val_loss / len(valid_loader)
        
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

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
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
        test_probabilities = probabilities.cpu().numpy().flatten()

    self.data.df_y_pred = pd.DataFrame({
        'label': test_predictions,
        'probability': test_probabilities
    }, index=self.data.name_test)

    self.model = model
    self.history = history