"""
신경망 모델 클래스 모듈
MultiChannelCNNLSTM 모델 정의
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import USE_MASKING

# MultiChannel CNN LSTM 모델 클래스
class MultiChannelCNNLSTM(nn.Module):
    """
    다중 채널 CNN-LSTM 모델 with Temporal & Channel Attention
    각 채널(시트)별로 독립적인 CNN-LSTM 처리 후 어텐션 메커니즘으로 결합
    마스킹 기능 추가: 패딩된 데이터는 학습에 영향을 주지 않음
    """
    def __init__(self, input_size, hidden_size, num_channels, 
                 cnn_filters=64, kernel_size=3, dropout=0.5, use_masking=USE_MASKING):
        super(MultiChannelCNNLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_channels = num_channels
        self.input_size = input_size
        self.use_masking = use_masking
        
        if self.use_masking:
            # 마스킹 사용 시, CNN 입력 채널 +1 (마스크 채널 추가)
            self.cnn_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(input_size + 1, cnn_filters, kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(cnn_filters),
                    nn.ReLU(),
                    nn.Conv1d(cnn_filters, input_size, 1),  # 1x1 Conv로 원래 차원으로 복원
                    nn.BatchNorm1d(input_size)
                ) for _ in range(num_channels)
            ])
        else:
            # 기존 방식 (마스킹 없음)
            self.cnn_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(input_size, cnn_filters, kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(cnn_filters),
                    nn.ReLU(),
                    nn.Conv1d(cnn_filters, input_size, 1),  # 1x1 Conv로 원래 차원으로 복원
                    nn.BatchNorm1d(input_size)
                ) for _ in range(num_channels)
            ])
        
        # 채널별 LSTM 레이어
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size, hidden_size//2, batch_first=True, bidirectional=True) 
            for _ in range(num_channels)
        ])
        
        # 시간 어텐션 메커니즘 (각 채널별)
        self.temporal_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1)
            ) for _ in range(num_channels)
        ])
        
        # 채널 어텐션 메커니즘 (동적 채널 가중치)
        self.channel_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_channels),
            nn.Softmax(dim=-1)
        )
        
        # 기존 채널 가중치는 백업용으로 유지
        self.channel_weights = nn.Parameter(torch.randn(num_channels))
        
        # 출력 레이어
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def temporal_attention_forward(self, lstm_output, attention_layer, mask=None):
        """
        시간 어텐션 적용
        params:
            lstm_output: [batch_size, seq_length, hidden_size]
            attention_layer: 해당 채널의 어텐션 레이어
            mask: [batch_size, seq_length] 마스크 정보
        returns:
            attended_output: [batch_size, hidden_size]
            attention_weights: [batch_size, seq_length, 1]
        """
        # 어텐션 점수 계산
        attention_scores = attention_layer(lstm_output)  # [batch_size, seq_length, 1]
        
        if mask is not None:
            # LSTM 출력과 마스크의 길이를 맞춤
            lstm_seq_len = attention_scores.size(1)
            mask_seq_len = mask.size(1)
            
            if lstm_seq_len != mask_seq_len:
                # 길이가 다른 경우 짧은 쪽에 맞춤
                min_len = min(lstm_seq_len, mask_seq_len)
                attention_scores = attention_scores[:, :min_len, :]
                mask = mask[:, :min_len]
                lstm_output = lstm_output[:, :min_len, :]
            
            # 마스킹된 부분은 매우 작은 값으로 설정
            mask_expanded = mask.unsqueeze(-1)  # [batch_size, seq_length, 1]
            attention_scores = attention_scores.masked_fill(mask_expanded == 0, -1e9)
        
        # 소프트맥스로 정규화
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_length, 1]
        
        # 가중 평균으로 최종 출력 계산
        attended_output = torch.sum(lstm_output * attention_weights, dim=1)  # [batch_size, hidden_size]
        
        return attended_output, attention_weights
    
    def forward(self, x, masks=None):
        """
        순방향 전파 with Temporal & Channel Attention
        params:
            x: 입력 데이터 [batch_size, num_channels, seq_length, input_size]
            masks: 마스크 데이터 [batch_size, num_channels, seq_length], 없으면 None
        """
        batch_size = x.size(0)
        
        # 각 채널별 CNN+LSTM+시간어텐션 처리
        channel_outputs = []
        temporal_attention_weights = []
        
        for i in range(self.num_channels):
            # 채널 데이터 추출 [batch_size, seq_length, input_size]
            channel_data = x[:, i, :, :]
            
            if self.use_masking and masks is not None:
                # 해당 채널의 마스크 추출 [batch_size, seq_length]
                channel_mask = masks[:, i, :]
                
                # 마스크를 input_size 차원으로 확장 [batch_size, seq_length, input_size]
                channel_mask_expanded = channel_mask.unsqueeze(2).expand(-1, -1, self.input_size)
                
                # 마스크를 추가 채널로 생성 [batch_size, seq_length, 1]
                channel_mask_single = channel_mask.unsqueeze(2)
                
                # 마스크를 CNN 입력에 연결 [batch_size, seq_length, input_size+1]
                cnn_input_with_mask = torch.cat([channel_data, channel_mask_single], dim=2)
                
                # CNN 처리를 위한 차원 변환 [batch_size, input_size+1, seq_length]
                cnn_input = cnn_input_with_mask.transpose(1, 2)
                
                # CNN 통과
                cnn_output = self.cnn_layers[i](cnn_input)
                
                # 잔차 연결: 마스킹된 원본 데이터와 결합
                masked_channel_data = channel_data * channel_mask_expanded
                masked_channel_data_transposed = masked_channel_data.transpose(1, 2)  # [batch_size, input_size, seq_length]
                cnn_output = cnn_output + masked_channel_data_transposed
                
                # LSTM 입력 형태로 차원 변환 [batch_size, seq_length, input_size]
                lstm_input = cnn_output.transpose(1, 2)
                
                # 마스킹 적용 (0인 부분은 무시)
                lstm_input = lstm_input * channel_mask_expanded
                
                # 각 시퀀스의 실제 길이 계산 (마스크의 합)
                seq_lengths = torch.sum(channel_mask, dim=1).int()
                
                # 실제 길이가 0인 경우 최소 1로 설정 (에러 방지)
                seq_lengths = torch.clamp(seq_lengths, min=1)
                
                # PackedSequence 사용을 위해 길이 순으로 정렬
                seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
                lstm_input = lstm_input[perm_idx]
                
                # PackedSequence 생성
                packed_input = pack_padded_sequence(lstm_input, seq_lengths.cpu(), batch_first=True)
                
                # LSTM 통과
                packed_output, _ = self.lstm_layers[i](packed_input)
                
                # 패킹 해제
                output, _ = pad_packed_sequence(packed_output, batch_first=True)
                
                # 원래 순서로 복원
                _, unperm_idx = perm_idx.sort(0)
                output = output[unperm_idx]
                
                # 시간 어텐션 적용
                attended_output, attention_weight = self.temporal_attention_forward(
                    output, self.temporal_attention[i], channel_mask
                )
                
            else:
                # 기존 방식 (마스킹 없음)
                # CNN 처리를 위한 차원 변환 [batch_size, input_size, seq_length]
                cnn_input = channel_data.transpose(1, 2)
                
                # CNN 통과
                cnn_output = self.cnn_layers[i](cnn_input)
                
                # 잔차 연결: CNN 출력 + 원본 데이터
                cnn_output = cnn_output + cnn_input
                
                # LSTM 입력 형태로 다시 차원 변환 [batch_size, seq_length, input_size]
                lstm_input = cnn_output.transpose(1, 2)
                
                # LSTM 통과
                output, _ = self.lstm_layers[i](lstm_input)
                
                # 시간 어텐션 적용 (마스크 없음)
                attended_output, attention_weight = self.temporal_attention_forward(
                    output, self.temporal_attention[i], mask=None
                )
            
            channel_outputs.append(attended_output)
            temporal_attention_weights.append(attention_weight)
        
        # 채널별 출력을 스택으로 결합 [batch_size, num_channels, hidden_size]
        stacked_outputs = torch.stack(channel_outputs, dim=1)
        
        # 채널 어텐션을 위한 컨텍스트 계산 (모든 채널의 평균)
        context = torch.mean(stacked_outputs, dim=1)  # [batch_size, hidden_size]
        
        # 동적 채널 가중치 계산
        dynamic_channel_weights = self.channel_attention(context)  # [batch_size, num_channels]
        
        # 채널별 출력 가중 합산
        combined = torch.sum(stacked_outputs * dynamic_channel_weights.unsqueeze(-1), dim=1)  # [batch_size, hidden_size]
        
        # 최종 분류
        combined = self.dropout(combined)
        logits = self.fc(combined)  # [batch_size, 1] 형태
        
        # 어텐션 가중치 저장 (디버깅 및 해석용)
        self.last_temporal_attention_weights = temporal_attention_weights
        self.last_channel_attention_weights = dynamic_channel_weights
        
        return logits
    
    def get_channel_importance(self):
        """채널별 중요도 반환 (동적 어텐션 가중치)"""
        if hasattr(self, 'last_channel_attention_weights'):
            return self.last_channel_attention_weights.cpu().detach().numpy().mean(axis=0)
        else:
            # 백업: 기존 고정 가중치
            return F.softmax(self.channel_weights, dim=0).cpu().detach().numpy()
    
    def get_temporal_attention_weights(self):
        """시간별 어텐션 가중치 반환 (해석용)"""
        if hasattr(self, 'last_temporal_attention_weights'):
            return [weight.cpu().detach().numpy() for weight in self.last_temporal_attention_weights]
        else:
            return None 