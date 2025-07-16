"""
MultiChannel CNN-LSTM 모델 학습 및 평가
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from multichannel_models import MultiChannelCNNLSTM


def get_MultiChannelCNNLSTM(self):
    """
    MultiChannel CNN-LSTM 모델 학습 및 평가
    
    Args:
        self: model 클래스의 인스턴스 (config, data 속성 포함)
    """
    print("=" * 60)
    print("MultiChannel CNN-LSTM 모델 학습 시작")
    print("=" * 60)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 데이터 형태 확인
    print(f"훈련 데이터 형태: {self.data.X_train.shape}")
    print(f"테스트 데이터 형태: {self.data.X_test.shape}")
    
    # 모델 하이퍼파라미터 추출
    n_channels = self.data.X_train.shape[1]
    time_points = self.data.X_train.shape[2]
    feature_dim = self.data.X_train.shape[3]
    
    # 시트 이름 가져오기
    first_company_key = next(iter(self.data.company_dict))
    sheet_names = sorted(list(self.data.company_dict[first_company_key].data_dict.keys()))
    
    print(f"채널 수: {n_channels}")
    print(f"시간 포인트 수: {time_points}")
    print(f"피처 차원: {feature_dim}")
    print(f"채널 이름 (시트): {sheet_names}")
    
    # 모델 생성
    model = MultiChannelCNNLSTM(
        n_channels=n_channels,
        time_points=time_points,
        feature_dim=feature_dim,
        hidden_size=self.config.get('hidden_size', 128),
        cnn_filters=self.config.get('cnn_filters', 64),
        kernel_size=self.config.get('kernel_size', 3),
        dropout=self.config.get('dropout', 0.5),
        use_channel_attention=self.config.get('use_channel_attention', True),
        use_temporal_attention=self.config.get('use_temporal_attention', True)
    ).to(device)
    
    print(f"모델 생성 완료:")
    print(f"  - 은닉층 크기: {self.config.get('hidden_size', 128)}")
    print(f"  - CNN 필터 수: {self.config.get('cnn_filters', 64)}")
    print(f"  - 커널 크기: {self.config.get('kernel_size', 3)}")
    print(f"  - 드롭아웃: {self.config.get('dropout', 0.5)}")
    print(f"  - 채널 어텐션: {self.config.get('use_channel_attention', True)}")
    print(f"  - 시간적 어텐션: {self.config.get('use_temporal_attention', True)}")
    
    # 손실함수 및 옵티마이저 설정
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=self.config.get('learning_rate', 0.001),
        weight_decay=self.config.get('weight_decay', 1e-4)
    )
    
    # 학습률 스케줄러
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # 데이터 로더 준비
    X_train_tensor = torch.FloatTensor(self.data.X_train).to(device)
    y_train_tensor = torch.FloatTensor(self.data.y_train).unsqueeze(1).to(device)
    X_test_tensor = torch.FloatTensor(self.data.X_test).to(device)
    y_test_tensor = torch.FloatTensor(self.data.y_test).unsqueeze(1).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=self.config.get('batch_size', 32), 
        shuffle=True
    )
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=self.config.get('batch_size', 32), 
        shuffle=False
    )
    
    # 모델 학습
    print("\n모델 학습 시작...")
    epochs = self.config.get('epochs', 100)
    patience = self.config.get('patience', 15)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # 훈련 단계
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for batch_X, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            optimizer.zero_grad()
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 정확도 계산
            predictions = torch.sigmoid(outputs) > 0.5
            train_correct += (predictions == batch_y).sum().item()
        
        train_loss /= len(train_loader)
        train_accuracy = train_correct / len(train_dataset)
        
        # 검증 단계
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                
                predictions = torch.sigmoid(outputs) > 0.5
                val_correct += (predictions == batch_y).sum().item()
        
        val_loss /= len(test_loader)
        val_accuracy = val_correct / len(test_dataset)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        
        # 학습률 스케줄러 업데이트
        scheduler.step(val_loss)
        
        # 조기 종료 체크
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # 최고 성능 모델 저장
            if self.config.get('save_model', True):
                torch.save(model.state_dict(), 
                          os.path.join(self.config['result_folder_path'], 'best_model.pth'))
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'조기 종료: {patience} 에포크 동안 성능 개선 없음')
            break
    
    # 최고 성능 모델 로드
    if self.config.get('save_model', True):
        model.load_state_dict(torch.load(
            os.path.join(self.config['result_folder_path'], 'best_model.pth')
        ))
    
    # 최종 예측
    model.eval()
    train_predictions = []
    train_probabilities = []
    test_predictions = []
    test_probabilities = []
    
    with torch.no_grad():
        # 훈련 데이터 예측
        for batch_X, batch_y in DataLoader(train_dataset, batch_size=64, shuffle=False):
            outputs = model(batch_X)
            probabilities = torch.sigmoid(outputs)
            predictions = probabilities > 0.5
            
            train_predictions.extend(predictions.cpu().numpy().flatten())
            train_probabilities.extend(probabilities.cpu().numpy().flatten())
        
        # 테스트 데이터 예측
        for batch_X, batch_y in DataLoader(test_dataset, batch_size=64, shuffle=False):
            outputs = model(batch_X)
            probabilities = torch.sigmoid(outputs)
            predictions = probabilities > 0.5
            
            test_predictions.extend(predictions.cpu().numpy().flatten())
            test_probabilities.extend(probabilities.cpu().numpy().flatten())
    
    # 성능 평가
    train_accuracy = accuracy_score(self.data.y_train, train_predictions)
    train_precision = precision_score(self.data.y_train, train_predictions, zero_division=0)
    train_recall = recall_score(self.data.y_train, train_predictions, zero_division=0)
    train_f1 = f1_score(self.data.y_train, train_predictions, zero_division=0)
    
    test_accuracy = accuracy_score(self.data.y_test, test_predictions)
    test_precision = precision_score(self.data.y_test, test_predictions, zero_division=0)
    test_recall = recall_score(self.data.y_test, test_predictions, zero_division=0)
    test_f1 = f1_score(self.data.y_test, test_predictions, zero_division=0)
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("MultiChannel CNN-LSTM 모델 학습 완료")
    print("=" * 60)
    
    print(f"\n훈련 세트 성능:")
    print(f"  정확도: {train_accuracy:.4f}")
    print(f"  정밀도: {train_precision:.4f}")
    print(f"  재현율: {train_recall:.4f}")
    print(f"  F1 점수: {train_f1:.4f}")
    
    print(f"\n테스트 세트 성능:")
    print(f"  정확도: {test_accuracy:.4f}")
    print(f"  정밀도: {test_precision:.4f}")
    print(f"  재현율: {test_recall:.4f}")
    print(f"  F1 점수: {test_f1:.4f}")
    
    # 혼동 행렬 생성 및 저장
    if self.config.get('save_confusion_matrix', True):
        cm = confusion_matrix(self.data.y_test, test_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['거래중', '경영악화'], 
                    yticklabels=['거래중', '경영악화'])
        plt.title('MultiChannel CNN-LSTM 혼동 행렬')
        plt.ylabel('실제 라벨')
        plt.xlabel('예측 라벨')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['result_folder_path'], 'confusion_matrix.png'))
        plt.close()
    
    # 학습 곡선 저장
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('모델 손실')
    plt.xlabel('에포크')
    plt.ylabel('손실')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.title('학습 곡선')
    plt.xlabel('에포크')
    plt.ylabel('손실')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(self.config['result_folder_path'], 'learning_curves.png'))
    plt.close()
    
    # 예측 결과 저장
    if self.config.get('save_predictions', True):
        # 훈련 예측 결과
        train_results = pd.DataFrame({
            'company_id': self.data.name_train,
            'actual': self.data.y_train,
            'predicted': train_predictions,
            'probability': train_probabilities
        })
        train_results.to_csv(
            os.path.join(self.config['result_folder_path'], 'train_predictions.csv'),
            index=False, encoding='utf-8-sig'
        )
        
        # 테스트 예측 결과
        test_results = pd.DataFrame({
            'company_id': self.data.name_test,
            'actual': self.data.y_test,
            'predicted': test_predictions,
            'probability': test_probabilities
        })
        test_results.to_csv(
            os.path.join(self.config['result_folder_path'], 'test_predictions.csv'),
            index=False, encoding='utf-8-sig'
        )
    
    # 성능 지표 저장
    if self.config.get('save_metrics', True):
        metrics = {
            'model_type': 'MultiChannelCNNLSTM',
            'n_channels': n_channels,
            'time_points': time_points,
            'sheet_names': sheet_names,
            'train_accuracy': train_accuracy,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1': train_f1,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'epochs_trained': epoch + 1,
            'best_val_loss': best_val_loss
        }
        
        with open(os.path.join(self.config['result_folder_path'], 'metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)
        
        # Excel로도 저장
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_excel(
            os.path.join(self.config['result_folder_path'], 'metrics.xlsx'),
            index=False
        )
    
    # 모델 하이퍼파라미터 저장
    self.models_hyperparameters = {
        'model_type': 'MultiChannelCNNLSTM',
        'n_channels': n_channels,
        'time_points': time_points,
        'feature_dim': feature_dim,
        'sheet_names': sheet_names,
        'hidden_size': self.config.get('hidden_size', 128),
        'cnn_filters': self.config.get('cnn_filters', 64),
        'kernel_size': self.config.get('kernel_size', 3),
        'dropout': self.config.get('dropout', 0.5),
        'use_channel_attention': self.config.get('use_channel_attention', True),
        'use_temporal_attention': self.config.get('use_temporal_attention', True)
    }
    
    # 모델 저장
    self.model = model
    self.test_predictions = test_predictions
    self.test_probabilities = test_probabilities
    
    print("\n모든 결과가 저장되었습니다.")
    print(f"결과 폴더: {self.config['result_folder_path']}")
    
    return model, test_predictions, test_probabilities 