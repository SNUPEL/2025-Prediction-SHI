"""
MultiChannel CNN-LSTM 모델 구조
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
                nn.Conv1d(feature_dim, cnn_filters, kernel_size=kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.BatchNorm1d(cnn_filters),
                nn.Dropout(dropout),
                nn.Conv1d(cnn_filters, cnn_filters, kernel_size=kernel_size, padding=kernel_size//2),
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
    
    def get_attention_weights(self, x):
        """
        어텐션 가중치 시각화를 위한 메서드
        
        Args:
            x: 입력 데이터 (batch_size, n_channels, time_points, feature_dim)
            
        Returns:
            channel_weights: 채널별 어텐션 가중치
            temporal_weights: 시간별 어텐션 가중치 (채널별)
        """
        batch_size = x.size(0)
        
        # 각 채널별 독립적인 처리
        channel_outputs = []
        temporal_weights_list = []
        
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
                temporal_weights_list.append(attention_weights.squeeze(-1))  # (batch_size, time_points)
                
                # 가중 평균 계산
                channel_output = torch.sum(lstm_out * attention_weights, dim=1)  # (batch_size, hidden_size)
            else:
                temporal_weights_list.append(None)
                channel_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
            
            channel_outputs.append(channel_output)
        
        # 모든 채널 출력을 결합: (batch_size, n_channels, hidden_size)
        channel_outputs = torch.stack(channel_outputs, dim=1)
        
        # 채널 어텐션 적용
        channel_weights = None
        if self.use_channel_attention:
            # 어텐션 가중치 계산
            attention_weights = self.channel_attention(channel_outputs)  # (batch_size, n_channels, 1)
            channel_weights = F.softmax(attention_weights, dim=1).squeeze(-1)  # (batch_size, n_channels)
        
        return channel_weights, temporal_weights_list 