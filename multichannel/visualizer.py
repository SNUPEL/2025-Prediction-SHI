"""
시각화 관련 함수들을 포함하는 모듈
특성 중요도 분석, 채널 중요도 시각화 등 기능 제공
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import seaborn as sns
import shap
from captum.attr import IntegratedGradients

from config import OUTPUT_DIR
from utils import ensure_dir, log_message, safe_korean_font

# 채널 중요도 시각화 함수
def visualize_channel_importance(model, sheet_names, output_dir):
    """
    각 채널(시트)의 중요도를 시각화하여 저장
    
    params:
        model: 학습된 MultiChannelCNNLSTM 모델
        sheet_names: 시트 이름 목록
        output_dir: 출력 디렉토리
    """
    try:
        # 채널 가중치 추출
        channel_weights = model.channel_weights.cpu().detach().numpy()
        # 소프트맥스 적용하여 확률로 변환
        channel_probs = F.softmax(model.channel_weights, dim=0).cpu().detach().numpy()
        
        # 시트 이름 준비 (필요한 경우 간략화)
        if len(sheet_names) > model.num_channels:
            sheet_names = sheet_names[:model.num_channels]
        elif len(sheet_names) < model.num_channels:
            sheet_names = sheet_names + [f"Channel_{i+1}" for i in range(len(sheet_names), model.num_channels)]
        
        # 짧은 표시 이름 생성 (20자 제한)
        display_names = [name[:20] + '...' if len(name) > 20 else name for name in sheet_names]
        
        # 중요도 내림차순으로 정렬
        indices = np.argsort(channel_probs)[::-1]
        sorted_weights = channel_probs[indices]
        sorted_names = [display_names[i] for i in indices]
        
        # 한글 폰트 문제 방지를 위한 이름 변환
        safe_names = [safe_korean_font(name) for name in sorted_names]
        
        # 시각화
        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(safe_names)))
        bars = plt.barh(range(len(safe_names)), sorted_weights, color=colors)
        
        # 각 막대에 값 표시
        for bar, weight in zip(bars, sorted_weights):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                     f'{weight:.2%}', va='center')
        
        plt.yticks(range(len(safe_names)), safe_names)
        plt.xlabel(safe_korean_font('상대적 중요도'))
        plt.title(safe_korean_font('시트별 중요도 분석'))
        plt.xlim(0, max(sorted_weights) * 1.1)  # x축 범위 조정
        plt.tight_layout()
        
        # 저장
        plt.savefig(os.path.join(output_dir, "channel_importance.png"), dpi=300)
        plt.close()
        
        # 중요도 데이터 CSV로도 저장
        importance_df = pd.DataFrame({
            '시트명': sheet_names,
            '중요도': channel_probs,
            '중요도(%)': channel_probs * 100
        })
        importance_df = importance_df.sort_values('중요도', ascending=False)
        importance_df.to_csv(os.path.join(output_dir, "channel_importance.csv"), index=False, encoding='utf-8-sig')
        
        log_message(f"  채널 중요도 분석 결과 저장 완료: {output_dir}")
        
        return importance_df
        
    except Exception as e:
        log_message(f"  채널 중요도 시각화 오류: {str(e)}")
        return None

# 특성 중요도 분석 함수
def analyze_feature_importance(model, X, y, sheet_names, company_ids, output_dir):
    """
    모델의 특성 중요도 분석 및 시각화 (어텐션 가중치 포함)
    
    params:
        model: 학습된 모델
        X: 입력 데이터
        y: 레이블
        sheet_names: 시트 이름 목록
        company_ids: 기업 ID 목록
        output_dir: 출력 디렉토리
    
    returns:
        results: 중요도 분석 결과
    """
    log_message("\n특성 중요도 분석 시작...")
    
    # 모델을 평가 모드로 전환
    model.eval()
    
    results = {}
    
    try:
        # 채널 중요도 시각화 수행
        importance_df = visualize_channel_importance(model, sheet_names, output_dir)
        
        # 채널 가중치 추출 (동적 어텐션 가중치)
        channel_weights = model.get_channel_importance()
        
        # 간소화된 시트 이름으로 변경 (한글 깨짐 문제 해결)
        simplified_names = [f"Sheet {i+1}" for i in range(len(sheet_names))]
        
        # 원래 시트명과 간소화된 시트명 매핑 저장
        sheet_mapping = pd.DataFrame({
            'Original_Sheet': sheet_names,
            'Simplified_Sheet': simplified_names,
            'Importance': channel_weights
        })
        sheet_mapping.to_csv(os.path.join(output_dir, "sheet_mapping.csv"), index=False, encoding='utf-8-sig')
        log_message(f"  시트 매핑 정보 저장 완료: {output_dir}")
        
        # 중요도 순으로 정렬
        sort_idx = np.argsort(-channel_weights)
        sorted_weights = channel_weights[sort_idx]
        sorted_names = [simplified_names[i] for i in sort_idx]
        
        # 채널 가중치 시각화
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(sorted_weights)), sorted_weights)
        plt.title('Channel Importance (Dynamic Attention)')
        plt.xlabel('Channel')
        plt.ylabel('Importance')
        
        # x축 레이블 설정 (간소화된 시트명)
        plt.xticks(range(len(sorted_weights)), sorted_names, rotation=45, ha='right')
        
        # 저장
        importance_file = os.path.join(output_dir, "channel_importance_simple.png")
        plt.tight_layout()
        plt.savefig(importance_file, dpi=300)
        plt.close()
        
        log_message(f"  채널 중요도 저장 완료: {importance_file}")
        
        # 시간 어텐션 가중치 분석 추가
        log_message("  시간 어텐션 가중치 분석 시작...")
        try:
            # 샘플 데이터로 모델 실행하여 어텐션 가중치 얻기
            num_samples = min(20, len(X))
            sample_indices = np.random.choice(len(X), num_samples, replace=False)
            X_sample = X[sample_indices]
            
            # 데이터 형태 확인 및 변환
            if len(X_sample.shape) == 3:
                X_sample = X_sample.reshape(X_sample.shape[0], X_sample.shape[1], X_sample.shape[2], 1)
            
            # 텐서로 변환
            X_tensor = torch.tensor(X_sample, dtype=torch.float32)
            
            # 마스킹이 활성화된 경우 마스크 생성
            if hasattr(model, 'use_masking') and model.use_masking:
                # 모든 데이터를 유효하게 처리하는 마스크 생성
                batch_size, num_channels, seq_len = X_tensor.shape[0], X_tensor.shape[1], X_tensor.shape[2]
                masks_tensor = torch.ones(batch_size, num_channels, seq_len)
                
                # 모델 실행 (어텐션 가중치 생성)
                with torch.no_grad():
                    _ = model(X_tensor, masks_tensor)
            else:
                # 마스킹 없이 모델 실행
                with torch.no_grad():
                    _ = model(X_tensor)
            
            # 시간 어텐션 가중치 추출
            temporal_weights = model.get_temporal_attention_weights()
            
            if temporal_weights is not None and len(temporal_weights) > 0:
                # 각 채널별 시간 어텐션 가중치 시각화
                for i, channel_idx in enumerate(sort_idx[:5]):  # 상위 5개 채널만
                    if channel_idx < len(temporal_weights):
                        channel_temporal_weights = temporal_weights[channel_idx]
                        
                        # 차원 확인 및 평균 계산
                        if len(channel_temporal_weights.shape) == 3:
                            # [samples, time_steps, 1] -> [time_steps] (평균)
                            avg_temporal_weights = np.mean(channel_temporal_weights, axis=0).squeeze()
                        elif len(channel_temporal_weights.shape) == 2:
                            # [samples, time_steps] -> [time_steps] (평균)
                            avg_temporal_weights = np.mean(channel_temporal_weights, axis=0)
                        else:
                            # [time_steps] (이미 1차원)
                            avg_temporal_weights = channel_temporal_weights
                        
                        # 1차원 배열로 변환
                        if avg_temporal_weights.ndim > 1:
                            avg_temporal_weights = avg_temporal_weights.flatten()
                        
                        # 시각화
                        plt.figure(figsize=(12, 6))
                        time_steps = range(len(avg_temporal_weights))
                        plt.plot(time_steps, avg_temporal_weights, 'b-', linewidth=2, marker='o', markersize=4)
                        plt.fill_between(time_steps, avg_temporal_weights, alpha=0.3)
                        plt.title(f'Temporal Attention Weights - {simplified_names[channel_idx]}')
                        plt.xlabel('Time Step')
                        plt.ylabel('Attention Weight')
                        plt.grid(True, alpha=0.3)
                        
                        # 최고점 표시
                        if len(avg_temporal_weights) > 0:
                            max_idx = np.argmax(avg_temporal_weights)
                            max_value = avg_temporal_weights[max_idx]
                            plt.scatter(max_idx, max_value, color='red', s=100, 
                                      label=f'Peak at step {max_idx} (value: {max_value:.3f})', zorder=5)
                            plt.legend()
                        
                        # 저장
                        temporal_file = os.path.join(output_dir, f"temporal_attention_{simplified_names[channel_idx].replace(' ', '_')}.png")
                        plt.tight_layout()
                        plt.savefig(temporal_file, dpi=300)
                        plt.close()
                
                log_message(f"  시간 어텐션 가중치 분석 완료: 상위 5개 채널 저장")
                
                # 시간 어텐션 가중치 CSV로 저장
                temporal_attention_data = []
                for i, channel_idx in enumerate(sort_idx[:5]):
                    if channel_idx < len(temporal_weights):
                        channel_temporal_weights = temporal_weights[channel_idx]
                        if len(channel_temporal_weights.shape) == 3:
                            avg_temporal_weights = np.mean(channel_temporal_weights, axis=0).squeeze()
                        elif len(channel_temporal_weights.shape) == 2:
                            avg_temporal_weights = np.mean(channel_temporal_weights, axis=0)
                        else:
                            avg_temporal_weights = channel_temporal_weights
                        
                        if avg_temporal_weights.ndim > 1:
                            avg_temporal_weights = avg_temporal_weights.flatten()
                        
                        for time_step, weight in enumerate(avg_temporal_weights):
                            temporal_attention_data.append({
                                'Channel': simplified_names[channel_idx],
                                'Time_Step': time_step,
                                'Attention_Weight': weight
                            })
                
                if temporal_attention_data:
                    temporal_df = pd.DataFrame(temporal_attention_data)
                    temporal_df.to_csv(os.path.join(output_dir, "temporal_attention_weights.csv"), index=False, encoding='utf-8-sig')
                    log_message(f"  시간 어텐션 가중치 CSV 저장 완료")
                
            else:
                log_message("  시간 어텐션 가중치를 찾을 수 없습니다.")
                
        except Exception as e:
            import traceback
            log_message(f"  시간 어텐션 분석 중 오류 발생: {str(e)}")
            log_message(traceback.format_exc())
        
        # Captum IntegratedGradients 분석 (마스킹 고려)
        try:
            log_message("  Captum IntegratedGradients 분석 시작...")
            
            # 래퍼 모델 사용
            class IGModelWrapper(torch.nn.Module):
                def __init__(self, original_model):
                    super().__init__()
                    self.model = original_model
                
                def forward(self, x):
                    if hasattr(self.model, 'use_masking') and self.model.use_masking:
                        batch_size, num_channels, seq_len = x.shape[0], x.shape[1], x.shape[2]
                        masks = torch.ones(batch_size, num_channels, seq_len, device=x.device)
                        return self.model(x, masks)
                    else:
                        return self.model(x)
            
            ig_wrapper = IGModelWrapper(model)
            
            # 샘플 데이터 준비
            num_samples = min(10, len(X))
            sample_indices = np.random.choice(len(X), num_samples, replace=False)
            X_sample = X[sample_indices]
            
            # 데이터 형태 확인 및 변환
            if len(X_sample.shape) == 3:
                X_sample = X_sample.reshape(X_sample.shape[0], X_sample.shape[1], X_sample.shape[2], 1)
            
            X_tensor = torch.tensor(X_sample, dtype=torch.float32)
            
            # IntegratedGradients 분석기 초기화
            ig = IntegratedGradients(ig_wrapper)
            
            # 배경 데이터로 0 텐서 사용
            baseline = torch.zeros_like(X_tensor)
            
            # 특성 중요도 계산
            attributions, _ = ig.attribute(X_tensor, baseline, target=0, return_convergence_delta=True)
            
            # 채널별 중요도 합산
            channel_attr = attributions.abs().mean(dim=(0, 2)).cpu().detach().numpy()
            
            # 특성 차원이 있는 경우 평균
            if len(channel_attr.shape) > 1:
                channel_attr = channel_attr.mean(axis=-1)
            
            # 중요도 시각화
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(channel_attr)), channel_attr)
            plt.title('Channel Importance (IntegratedGradients)')
            plt.xlabel('Channel')
            plt.ylabel('Attribution')
            plt.xticks(range(len(channel_attr)), [simplified_names[i] for i in range(len(channel_attr))], rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "channel_importance_ig.png"), dpi=300)
            plt.close()
            
            log_message(f"  IntegratedGradients 분석 저장 완료")
        except Exception as e:
            log_message(f"  IntegratedGradients 분석 중 오류 발생: {str(e)}")
        
        # 채널 중요도 정보 반환
        results['channel_importance'] = dict(zip(sheet_names, channel_weights))
        
        return results
        
    except Exception as e:
        import traceback
        log_message(f"  특성 중요도 분석 중 오류 발생: {str(e)}")
        log_message(traceback.format_exc())
        return None

# 예측 결과 시각화 함수
def visualize_predictions(predictions_df, output_dir, model_name="MultiChannel_CNNLSTM"):
    """
    예측 결과 시각화
    
    params:
        predictions_df: 예측 결과 DataFrame
        output_dir: 출력 디렉토리
        model_name: 모델 이름 (파일명에 사용)
    """
    try:
        ensure_dir(output_dir)
        
        # 정확도 매트릭스 (Confusion Matrix) 시각화
        if '실제' in predictions_df.columns and '예측' in predictions_df.columns:
            # 혼동 행렬 계산
            cm = pd.crosstab(predictions_df['실제'], predictions_df['예측'], 
                            rownames=['실제'], colnames=['예측'])
            
            # 시각화
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(safe_korean_font('예측 결과 혼동 행렬'))
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"), dpi=300)
            plt.close()
            
            log_message(f"  혼동 행렬 시각화 저장 완료: {os.path.join(output_dir, f'{model_name}_confusion_matrix.png')}")
        
        # 예측 확률 분포 시각화
        if '경영악화_확률' in predictions_df.columns and '실제' in predictions_df.columns:
            plt.figure(figsize=(10, 6))
            
            # 실제 클래스별로 구분하여 히스토그램 작성
            for label in [0, 1]:
                subset = predictions_df[predictions_df['실제'] == label]
                if len(subset) > 0:
                    plt.hist(subset['경영악화_확률'], bins=20, alpha=0.5, 
                            label=f"{'경영악화' if label==1 else '거래중'} (n={len(subset)})")
            
            plt.axvline(x=0.5, color='r', linestyle='--', label='결정 경계 (0.5)')
            plt.title(safe_korean_font('경영악화 예측 확률 분포'))
            plt.xlabel(safe_korean_font('경영악화 확률'))
            plt.ylabel(safe_korean_font('기업 수'))
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{model_name}_probability_dist.png"), dpi=300)
            plt.close()
            
            log_message(f"  확률 분포 시각화 저장 완료: {os.path.join(output_dir, f'{model_name}_probability_dist.png')}")
        
        # 개별 기업 예측 결과 시각화 (상위 기업만)
        if 'No.' in predictions_df.columns and '경영악화_확률' in predictions_df.columns:
            # 경영악화 확률 기준 정렬
            top_companies = predictions_df.sort_values('경영악화_확률', ascending=False).head(20)
            
            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(len(top_companies)), top_companies['경영악화_확률'])
            
            # 실제 상태에 따라 색상 설정
            if '실제' in top_companies.columns:
                colors = ['red' if label == 1 else 'blue' for label in top_companies['실제']]
                for i, bar in enumerate(bars):
                    bar.set_color(colors[i])
            
            plt.yticks(range(len(top_companies)), [f"기업 {cid}" for cid in top_companies['No.']])
            plt.xlabel(safe_korean_font('경영악화 확률'))
            plt.title(safe_korean_font('경영악화 확률 상위 기업'))
            plt.xlim(0, 1)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{model_name}_top_companies.png"), dpi=300)
            plt.close()
            
            log_message(f"  상위 기업 시각화 저장 완료: {os.path.join(output_dir, f'{model_name}_top_companies.png')}")
        
    except Exception as e:
        log_message(f"  예측 결과 시각화 중 오류 발생: {str(e)}")

# 시계열 데이터 시각화 함수
def visualize_timeseries(X, y, company_ids, sheet_names, output_dir, max_companies=5, max_sheets=3):
    """
    시계열 데이터 시각화
    
    params:
        X: 입력 데이터 [samples, channels, time_points]
        y: 레이블
        company_ids: 기업 ID 목록
        sheet_names: 시트 이름 목록
        output_dir: 출력 디렉토리
        max_companies: 시각화할 최대 기업 수
        max_sheets: 시각화할 최대 시트 수
    """
    try:
        # 디렉토리 생성
        ts_dir = os.path.join(output_dir, "timeseries")
        ensure_dir(ts_dir)
        
        # 각 클래스별로 샘플 선택
        class_0_idx = np.where(y == 0)[0]
        class_1_idx = np.where(y == 1)[0]
        
        # 시각화할 샘플 선택 (각 클래스별로 최대 샘플 수 제한)
        class_0_samples = np.random.choice(class_0_idx, min(max_companies, len(class_0_idx)), replace=False)
        class_1_samples = np.random.choice(class_1_idx, min(max_companies, len(class_1_idx)), replace=False)
        
        selected_samples = np.concatenate([class_0_samples, class_1_samples])
        
        # 시트 선택 (중요도 기준 상위 시트만)
        if len(sheet_names) > max_sheets:
            # 간단한 무작위 선택 (실제로는 중요도 기준으로 선택하는 것이 좋음)
            selected_sheets = np.random.choice(len(sheet_names), max_sheets, replace=False)
        else:
            selected_sheets = range(len(sheet_names))
        
        # 각 샘플별 시계열 데이터 시각화
        for i, sample_idx in enumerate(selected_samples):
            company_id = company_ids[sample_idx]
            label = y[sample_idx]
            sample_data = X[sample_idx]
            
            # 시각화 폴더 생성
            company_dir = os.path.join(ts_dir, f"company_{company_id}")
            ensure_dir(company_dir)
            
            # 선택된 시트만 시각화
            for sheet_idx in selected_sheets:
                if sheet_idx < len(sheet_names):
                    sheet_name = sheet_names[sheet_idx]
                    # Error Fix: sample_data is 2D [channels, time_steps]
                    # Access it with [sheet_idx, :]
                    sheet_data = sample_data[sheet_idx, :] 
                    
                    # 0이 아닌 데이터 포인트 찾기 (패딩 식별)
                    non_zero_mask = sheet_data != 0
                    
                    # 패딩이 아닌 실제 데이터 포인트만 사용
                    if np.any(non_zero_mask):
                        # 시각화
                        plt.figure(figsize=(10, 6))
                        plt.plot(sheet_data[non_zero_mask], marker='o')
                        plt.title(f"기업 {company_id} - {safe_korean_font(sheet_name)} 시계열")
                        plt.xlabel('시간점')
                        plt.ylabel('값')
                        plt.grid(True, alpha=0.3)
                        plt.figtext(0.02, 0.02, f"레이블: {'경영악화' if label == 1 else '거래중'}", ha='left')
                        plt.tight_layout()
                        
                        # 저장
                        safe_sheet_name = sheet_name.replace('/', '_').replace('\\', '_')
                        plt.savefig(os.path.join(company_dir, f"{safe_sheet_name}.png"), dpi=300)
                        plt.close()
        
        log_message(f"  시계열 데이터 시각화 완료: {ts_dir}")
        
    except Exception as e:
        log_message(f"  시계열 데이터 시각화 중 오류 발생: {str(e)}") 