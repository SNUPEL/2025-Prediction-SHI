"""
모델 학습 및 평가 모듈
학습, 검증, 테스트 과정을 처리
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm.auto import tqdm

from config import OUTPUT_DIR, DEFAULT_MODEL_PARAMS, TRAIN_PARAMS, USE_MASKING, OVERSAMPLING_METHOD, USE_OSS, AUTO_THRESHOLD, FIXED_THRESHOLD
from utils import ensure_fold_dir, log_message, find_best_threshold_simple, plot_loss_curves, plot_roc_curve
from .models import MultiChannelCNNLSTM
from .sampling import apply_smote, apply_tssmote, apply_smote_with_masks, apply_tomek_links

# 모델 학습 및 평가 함수
def train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, test_ids, 
                           sheet_names, masks_train=None, masks_val=None, masks_test=None,
                           params=None, output_dir=OUTPUT_DIR, model_name="MultiChannel_CNNLSTM"):
    """
    모델 학습 및 평가 함수 (기업별 정규화 적용)
    
    params:
        X_train, y_train: 학습 데이터 및 레이블
        X_val, y_val: 검증 데이터 및 레이블
        X_test, y_test: 테스트 데이터 및 레이블
        test_ids: 테스트 기업 ID 목록
        sheet_names: 시트 이름 목록
        masks_train, masks_val, masks_test: 마스크 데이터 (USE_MASKING이 True인 경우 사용)
        params: 모델 하이퍼파라미터
        output_dir: 출력 디렉토리
        model_name: 모델 이름
        
    returns:
        model: 학습된 모델
        predictions: 예측 결과 딕셔너리 목록
        metrics: 성능 지표 딕셔너리
    """
    # 순환 임포트 방지를 위해 여기서 save_results 함수 임포트
    from data import save_results
    
    log_message("  모델 학습 시작...")
    
    # 기본 하이퍼파라미터 설정
    if params is None:
        params = DEFAULT_MODEL_PARAMS.copy()
    
    # 학습 파라미터 가져오기
    batch_size = TRAIN_PARAMS['batch_size']
    epochs = TRAIN_PARAMS['epochs']
    patience = TRAIN_PARAMS['patience']
    learning_rate = TRAIN_PARAMS['learning_rate']
    
    # 모델 결과 폴더 생성
    model_dir = ensure_fold_dir(output_dir, model_name)
    
    # 데이터 형태 확인 및 변환
    if len(X_train.shape) == 3:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    if len(X_val.shape) == 3:
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
    if len(X_test.shape) == 3:
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    
    # 마스킹 사용 여부 로깅
    if USE_MASKING:
        log_message("  마스킹 기능 활성화: 패딩된 데이터는 학습에 영향을 주지 않습니다.")
        
        # 마스크 데이터 확인
        if masks_train is None or masks_val is None or masks_test is None:
            log_message("  경고: 마스크 데이터가 누락되었습니다. 모든 데이터를 유효하게 처리합니다.")
            # 마스크 데이터가 없으면 모든 값을 1로 생성 (모든 데이터 유효)
            if masks_train is None:
                masks_train = np.ones((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
            if masks_val is None:
                masks_val = np.ones((X_val.shape[0], X_val.shape[1], X_val.shape[2]))
            if masks_test is None:
                masks_test = np.ones((X_test.shape[0], X_test.shape[1], X_test.shape[2]))
    else:
        log_message("  마스킹 기능 비활성화: 기존 방식으로 모델을 학습합니다.")
    
    # 오버샘플링 방법 선택 및 적용
    log_message(f"  클래스 불균형 처리 (언더샘플링: {'Tomek Links' if USE_OSS else '없음'}, 오버샘플링: {OVERSAMPLING_METHOD})...")
    
    # 1단계: Tomek Links 언더샘플링 적용 (설정 시)
    if USE_OSS and USE_MASKING:
        X_train_processed, masks_train_processed, y_train_processed = apply_tomek_links(
            X_train, masks_train, y_train
        )
    else:
        if USE_OSS:
            log_message("  경고: Tomek Links는 마스킹과 함께 사용해야 합니다. Tomek Links를 건너뜁니다.")
        X_train_processed = X_train
        masks_train_processed = masks_train
        y_train_processed = y_train
    
    # 2단계: 오버샘플링 적용
    if USE_MASKING:
        if OVERSAMPLING_METHOD == "SMOTE":
            # 마스킹과 함께 SMOTE 적용
            X_train_resampled, masks_train_resampled, y_train_resampled = apply_smote_with_masks(
                X_train_processed, masks_train_processed, y_train_processed
            )
        elif OVERSAMPLING_METHOD == "TSSMOTE":
            # 마스킹과 함께 TSSMOTE 적용
            X_train_resampled, masks_train_resampled, y_train_resampled = apply_tssmote(
                X_train_processed, masks_train_processed, y_train_processed
            )
        else:  # "NONE"
            # 오버샘플링 없음 - Tomek Links 처리된 데이터 사용
            log_message("  오버샘플링 비활성화: Tomek Links 처리된 데이터를 그대로 사용합니다.")
            X_train_resampled = X_train_processed
            masks_train_resampled = masks_train_processed
            y_train_resampled = y_train_processed
    else:
        # 마스킹 없이 일반 SMOTE만 적용 (기존 방식)
        if OVERSAMPLING_METHOD == "SMOTE":
            X_train_resampled, y_train_resampled = apply_smote(X_train_processed, y_train_processed)
        else:
            # 오버샘플링 없음
            X_train_resampled = X_train_processed
            y_train_resampled = y_train_processed
        masks_train_resampled = None
    
    # 모델 하이퍼파라미터
    hidden_size = params['hidden_size']
    dropout = params['dropout']
    weight_decay = params['weight_decay']
    
    # 데이터 차원 추출
    num_channels = X_train_resampled.shape[1]  # 채널 수
    input_size = X_train_resampled.shape[3]    # 특성 수
    
    # 텐서 변환
    X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.float32).view(-1, 1)
    
    if USE_MASKING:
        masks_train_tensor = torch.tensor(masks_train_resampled, dtype=torch.float32)
        # 데이터로더 (마스크 포함)
        train_dataset = TensorDataset(X_train_tensor, masks_train_tensor, y_train_tensor)
    else:
        # 기존 데이터로더
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 검증 데이터 텐서 변환
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32) if len(X_val) > 0 else None
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1) if len(y_val) > 0 else None
    
    if X_val_tensor is not None and y_val_tensor is not None:
        if USE_MASKING:
            masks_val_tensor = torch.tensor(masks_val, dtype=torch.float32)
            # 검증 데이터로더 (마스크 포함)
            val_dataset = TensorDataset(X_val_tensor, masks_val_tensor, y_val_tensor)
        else:
            # 기존 검증 데이터로더
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None
    
    # 테스트 데이터 텐서 변환
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32) if len(X_test) > 0 else None
    if USE_MASKING and X_test_tensor is not None:
        masks_test_tensor = torch.tensor(masks_test, dtype=torch.float32)
    
    # 모델 초기화
    model = MultiChannelCNNLSTM(input_size, hidden_size, num_channels, 
                               cnn_filters=64, kernel_size=3, dropout=dropout,
                               use_masking=USE_MASKING)
    
    # 클래스 가중치 계산
    pos_weight = np.sum(y_train_resampled == 0) / max(np.sum(y_train_resampled == 1), 1)
    pos_weight = min(pos_weight, 10.0)  # 최대값 제한
    
    # 오버샘플링을 하지 않는 경우, 클래스 가중치를 더 강화
    if OVERSAMPLING_METHOD == "NONE":
        pos_weight = pos_weight * 1.5  # 가중치 증가
        log_message(f"  오버샘플링 없음: 클래스 가중치 강화 (pos_weight={pos_weight:.2f})")
    
    # 손실 함수 및 옵티마이저
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 학습
    best_loss = float('inf')
    early_stop_counter = 0
    train_losses = []
    val_losses = []
    
    try:
        # 학습 루프
        for epoch in tqdm(range(epochs), desc="  학습 진행"):
            # 훈련 단계
            model.train()
            epoch_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                if USE_MASKING:
                    inputs, masks, labels = batch
                    outputs = model(inputs, masks)
                else:
                    inputs, labels = batch
                    outputs = model(inputs)
                
                # 차원 확인 및 디버깅
                if epoch == 0 and epoch_loss == 0:
                    log_message(f"    출력 형태: {outputs.shape}, 레이블 형태: {labels.shape}")
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 검증 단계
            if val_loader is not None:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        if USE_MASKING:
                            inputs, masks, labels = batch
                            outputs = model(inputs, masks)
                        else:
                            inputs, labels = batch
                            outputs = model(inputs)
                            
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                    
                    avg_val_loss = val_loss / len(val_loader)
                    val_losses.append(avg_val_loss)
            else:
                # 검증 데이터가 없으면 훈련 손실 사용
                avg_val_loss = avg_train_loss
                val_losses.append(avg_val_loss)
            
            # 로깅
            if epoch % 10 == 0:
                log_message(f"    에폭 {epoch}: 훈련 손실={avg_train_loss:.4f}, 검증 손실={avg_val_loss:.4f}")
            
            # 조기 종료 확인 (검증 손실 기반)
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                early_stop_counter = 0
                # 최적 모델 저장
                torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    log_message(f"    조기 종료: 에폭 {epoch+1}/{epochs}")
                    break
        
        # 손실 곡선 저장
        plot_loss_curves(train_losses, val_losses, os.path.join(model_dir, "loss_curve.png"))
        
        # 최적 모델 불러오기
        model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt")))
        model.eval()
        
        # 테스트 세트 평가
        metrics = {}
        predictions = []
        
        if X_test_tensor is not None and len(y_test) > 0:
            with torch.no_grad():
                if USE_MASKING:
                    test_outputs = model(X_test_tensor, masks_test_tensor)
                else:
                    test_outputs = model(X_test_tensor)
                    
                test_probs = torch.sigmoid(test_outputs).squeeze().cpu().numpy()
                
                # 최적 임계값 결정
                threshold = FIXED_THRESHOLD  # 기본값
                if AUTO_THRESHOLD and X_val_tensor is not None and len(y_val) > 0:
                    # 검증 데이터에서 최적 임계값 찾기
                    with torch.no_grad():
                        if USE_MASKING:
                            val_outputs = model(X_val_tensor, masks_val_tensor)
                        else:
                            val_outputs = model(X_val_tensor)
                        
                        val_probs = torch.sigmoid(val_outputs).squeeze().cpu().numpy()
                        threshold = find_best_threshold_simple(y_val, val_probs, metric='f1')
                        log_message(f"    자동 최적 임계값: {threshold:.3f}")
                else:
                    log_message(f"    고정 임계값 사용: {threshold:.3f}")
                
                test_preds = (test_probs >= threshold).astype(int)
            
            # 성능 지표 계산
            metrics = {
                'accuracy': accuracy_score(y_test, test_preds),
                'precision': precision_score(y_test, test_preds, zero_division=0),
                'recall': recall_score(y_test, test_preds, zero_division=0),
                'f1': f1_score(y_test, test_preds, zero_division=0),
                'threshold_used': threshold,
                'auto_threshold': AUTO_THRESHOLD
            }
            
            # ROC AUC 계산
            if len(np.unique(y_test)) > 1:
                metrics['auc'] = plot_roc_curve(y_test, test_probs, os.path.join(model_dir, "roc_curve.png"))
            else:
                metrics['auc'] = 0.0
            
            # 예측 결과 저장
            for i, company_id in enumerate(test_ids):
                predictions.append({
                    'No.': company_id,
                    '경영악화_확률': test_probs[i] if i < len(test_probs) else 0,
                    '예측': test_preds[i] if i < len(test_preds) else 0,
                    '실제': y_test[i],
                    '사용된_임계값': threshold
                })
            
            log_message(f"    테스트 성능 (임계값={threshold:.3f}): 정확도={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
        
        # 결과 저장
        predictions_df = pd.DataFrame(predictions)
        save_results(predictions_df, metrics, model_name=model_name)
        
        return model, predictions, metrics
        
    except Exception as e:
        import traceback
        log_message(f"    학습 중 오류 발생: {str(e)}")
        log_message(traceback.format_exc())
        # 임시 결과 반환
        return None, [], {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'auc': 0} 