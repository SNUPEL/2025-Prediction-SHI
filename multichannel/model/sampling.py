"""
클래스 불균형 처리 및 샘플링 모듈
SMOTE, TSSMOTE, Tomek Links 등의 오버/언더 샘플링 기법
"""
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

from utils import log_message

# 개선된 DTW 기반 TimeSeries SMOTE 함수
def apply_tssmote(X, masks, y, k=5):
    """
    다변량 DTW 거리 기반 TimeSeries SMOTE (개선된 버전)
    
    params:
        X: 입력 데이터 [samples, channels, time_points, features]
        masks: 마스크 데이터 [samples, channels, time_points]
        y: 레이블
        k: 이웃 수
    
    returns:
        오버샘플링된 데이터, 마스크, 레이블
    """
    # dtaidistance 라이브러리 사용 (다변량 DTW 지원)
    try:
        from dtaidistance import dtw
        log_message("  dtaidistance 라이브러리 사용: 다변량 DTW 지원")
    except ImportError:
        log_message("  dtaidistance 라이브러리를 찾을 수 없습니다. pip install dtaidistance 를 실행해주세요.")
        log_message("  일반 SMOTE로 대체합니다.")
        return apply_smote_with_masks(X, masks, y)
    
    log_message("  개선된 TSSMOTE(다변량 DTW 기반) 적용 시작...")
    
    # 소수 클래스 데이터 추출
    minority_indices = np.where(y == 1)[0]
    majority_indices = np.where(y == 0)[0]
    
    # 클래스 분포 확인
    class_counts = np.bincount(y.astype(int))
    log_message(f"    TSSMOTE 적용 전: {len(y)}개 샘플 (클래스 분포: 0={class_counts[0]}, 1={class_counts[1] if len(class_counts)>1 else 0})")
    
    # 소수 클래스가 충분히 있는지 확인
    if len(minority_indices) < 2 or len(class_counts) <= 1 or class_counts[1] < 2:
        log_message("    소수 클래스 샘플이 부족하여 TSSMOTE를 적용할 수 없습니다.")
        return X, masks, y
    
    # 필요한 샘플 수 계산
    n_samples_needed = max(0, len(majority_indices) - len(minority_indices))
    
    if n_samples_needed == 0:
        log_message("    클래스가 이미 균형 상태입니다. TSSMOTE를 적용하지 않습니다.")
        return X, masks, y
    
    # 소수 클래스 데이터만 추출
    X_minority = X[minority_indices]
    masks_minority = masks[minority_indices]
    
    # DTW 거리 행렬 계산 (다변량 DTW 사용)
    n_minority = len(minority_indices)
    dtw_matrix = np.zeros((n_minority, n_minority))
    
    # 진행 상황 표시
    log_message(f"    다변량 DTW 거리 행렬 계산 중... (총 {n_minority*(n_minority-1)//2}개 쌍)")
    
    for i in range(n_minority):
        for j in range(i+1, n_minority):
            # 각 채널별 DTW 거리의 평균 계산
            channel_distances = []
            for c in range(X_minority.shape[1]):  # 각 채널(시트)별 계산
                # 마스크를 이용해 유효한 시점(time point)만 추출
                mask1 = masks_minority[i, c].astype(bool)
                mask2 = masks_minority[j, c].astype(bool)
                
                # 모든 특성을 포함한 다변량 시계열 추출
                # shape: [time_points, features]
                seq1_multi = X_minority[i, c, mask1, :]  # 유효한 시점만 추출
                seq2_multi = X_minority[j, c, mask2, :]  # 유효한 시점만 추출
                
                if seq1_multi.shape[0] > 0 and seq2_multi.shape[0] > 0:
                    try:
                        # 다변량 DTW 거리 계산 (모든 특성 활용)
                        # dtaidistance는 다변량 배열을 직접 입력받을 수 있음
                        if seq1_multi.shape[1] == 1:
                            # 단변량인 경우 2D에서 1D로 변환
                            distance = dtw.distance(seq1_multi.flatten(), seq2_multi.flatten())
                        else:
                            # 다변량인 경우 그대로 사용
                            distance = dtw.distance(seq1_multi, seq2_multi)
                        
                        channel_distances.append(distance)
                    except Exception as e:
                        log_message(f"      채널 {c} DTW 계산 오류: {str(e)}")
                        # 오류 시 유클리드 거리로 대체
                        if seq1_multi.shape == seq2_multi.shape:
                            distance = np.linalg.norm(seq1_multi - seq2_multi)
                            channel_distances.append(distance)
            
            # 모든 채널의 평균 DTW 거리
            if channel_distances:
                avg_distance = np.mean(channel_distances)
                dtw_matrix[i, j] = avg_distance
                dtw_matrix[j, i] = avg_distance
            else:
                # 유효한 데이터가 없는 경우 최대값 설정
                dtw_matrix[i, j] = np.inf
                dtw_matrix[j, i] = np.inf
    
    # 생성된 샘플 저장용
    synthetic_X = []
    synthetic_masks = []
    
    # k 값 조정 (최대 소수 클래스 샘플 수 - 1)
    k = min(k, n_minority - 1)
    
    # 각 소수 클래스 샘플에 대해 합성 샘플 생성
    log_message(f"    합성 샘플 생성 중... (목표: {n_samples_needed}개, k={k})")
    for i in range(n_minority):
        # DTW 거리 기반으로 k개 최근접 이웃 찾기
        neighbors = np.argsort(dtw_matrix[i])[1:k+1]  # 자기 자신 제외
        
        # 이웃 중에 무한대 거리가 있는지 확인
        valid_neighbors = [n for n in neighbors if dtw_matrix[i, n] < np.inf]
        
        if not valid_neighbors:
            continue  # 유효한 이웃이 없으면 건너뜀
        
        # 필요한 만큼 합성 샘플 생성
        samples_per_minority = int(np.ceil(n_samples_needed / n_minority))
        for _ in range(samples_per_minority):
            if len(synthetic_X) >= n_samples_needed:
                break
            
            # 이웃 중 하나 선택
            nn_idx = np.random.choice(valid_neighbors)
            
            # 보간 비율 (0.2-0.8 사이로 제한하여 자연스러운 합성)
            alpha = 0.2 + 0.6 * np.random.random()
            
            # 새 샘플 생성 (선형 보간)
            new_X = X_minority[i] * alpha + X_minority[nn_idx] * (1-alpha)
            
            # 마스크는 OR 연산 (둘 중 하나라도 유효하면 유효)
            new_mask = np.logical_or(masks_minority[i], masks_minority[nn_idx]).astype(float)
            
            synthetic_X.append(new_X)
            synthetic_masks.append(new_mask)
    
    # 결과 합치기
    if synthetic_X:
        X_resampled = np.vstack([X, np.array(synthetic_X)])
        masks_resampled = np.vstack([masks, np.array(synthetic_masks)])
        y_resampled = np.concatenate([y, np.ones(len(synthetic_X))])
        
        # 오버샘플링 결과 로깅
        new_class_counts = np.bincount(y_resampled.astype(int))
        log_message(f"    개선된 TSSMOTE 적용 후: {len(y_resampled)}개 샘플 (클래스 분포: 0={new_class_counts[0]}, 1={new_class_counts[1]})")
        log_message(f"    생성된 합성 샘플: {len(synthetic_X)}개 (모든 특성 활용)")
    else:
        log_message("    유효한 합성 샘플을 생성할 수 없습니다. 원본 데이터를 반환합니다.")
        X_resampled, masks_resampled, y_resampled = X, masks, y
    
    return X_resampled, masks_resampled, y_resampled

# SMOTE 오버샘플링 함수 (마스크 처리 포함)
def apply_smote_with_masks(X, masks, y):
    """
    마스크 데이터를 포함한 SMOTE 오버샘플링
    
    params:
        X: 입력 데이터 [samples, channels, time_points, features]
        masks: 마스크 데이터 [samples, channels, time_points]
        y: 레이블
    
    returns:
        오버샘플링된 데이터, 마스크, 레이블
    """
    original_shape = X.shape
    X_2d = X.reshape(X.shape[0], -1)  # 2D로 변환
    masks_2d = masks.reshape(masks.shape[0], -1)  # 마스크도 2D로 변환

    # 클래스 분포 확인
    class_counts = np.bincount(y.astype(int))
    log_message(f"    SMOTE with masks 적용 전: {len(y)}개 샘플 (클래스 분포: 0={class_counts[0]}, 1={class_counts[1] if len(class_counts)>1 else 0})")

    # 소수 클래스가 충분히 있는 경우에만 SMOTE 적용
    if len(class_counts) > 1 and class_counts[1] >= 5:
        # 데이터와 마스크를 결합하여 SMOTE 적용
        combined = np.hstack([X_2d, masks_2d])
        
        # SMOTE 적용 
        smote = SMOTE(random_state=42)
        combined_resampled, y_resampled = smote.fit_resample(combined, y)
        
        # 결합된 데이터에서 X와 마스크 분리
        X_part_size = X_2d.shape[1]
        X_smote_2d = combined_resampled[:, :X_part_size]
        masks_smote_2d = combined_resampled[:, X_part_size:]
        
        # 원래 형태로 복원
        X_resampled = X_smote_2d.reshape(-1, original_shape[1], original_shape[2], original_shape[3])
        masks_resampled = masks_smote_2d.reshape(-1, original_shape[1], original_shape[2])
        
        # 오버샘플링 결과 로깅
        new_class_counts = np.bincount(y_resampled.astype(int))
        log_message(f"    SMOTE with masks 적용 후: {len(y_resampled)}개 샘플 (클래스 분포: 0={new_class_counts[0]}, 1={new_class_counts[1]})")
        
        return X_resampled, masks_resampled, y_resampled
    else:
        log_message("    소수 클래스 샘플 부족으로 SMOTE를 적용하지 않습니다.")
        return X, masks, y

# SMOTE 오버샘플링 함수
def apply_smote(X, y):
    """
    SMOTE를 적용하여 클래스 불균형 처리
    
    params:
        X: 입력 데이터
        y: 레이블
    
    returns:
        오버샘플링된 데이터와 레이블
    """
    original_shape = X.shape
    X_2d = X.reshape(X.shape[0], -1)  # 2D로 변환

    # 클래스 분포 확인
    class_counts = np.bincount(y.astype(int))
    log_message(f"    SMOTE 적용 전: {len(y)}개 샘플 (클래스 분포: 0={class_counts[0]}, 1={class_counts[1] if len(class_counts)>1 else 0})")

    # 소수 클래스가 충분히 있는 경우에만 SMOTE 적용
    if len(class_counts) > 1 and class_counts[1] >= 5:
        # SMOTE 적용 
        smote = SMOTE(random_state=42)
        X_smote_2d, y_smote = smote.fit_resample(X_2d, y)
        
        # 원래 형태로 복원
        X_resampled = X_smote_2d.reshape(-1, original_shape[1], original_shape[2], original_shape[3])
        y_resampled = y_smote
        
        # 오버샘플링 결과 로깅
        new_class_counts = np.bincount(y_resampled.astype(int))
        log_message(f"    SMOTE 적용 후: {len(y_resampled)}개 샘플 (클래스 분포: 0={new_class_counts[0]}, 1={new_class_counts[1]})")
        
        return X_resampled, y_resampled
    else:
        log_message("    소수 클래스 샘플 부족으로 SMOTE를 적용하지 않습니다.")
        return X, y

def apply_tomek_links(X, masks, y):
    """
    Tomek Links 언더샘플링 적용
    경계선에 있는 노이즈 샘플들을 제거하여 클래스 경계를 명확하게 함
    
    params:
        X: 입력 데이터 [samples, channels, time_points, features]
        masks: 마스크 데이터 [samples, channels, time_points] 
        y: 레이블
    
    returns:
        언더샘플링된 데이터, 마스크, 레이블
    """
    log_message("  Tomek Links 언더샘플링 적용 시작...")
    
    # 클래스 분포 확인
    class_counts = np.bincount(y.astype(int))
    log_message(f"    Tomek 적용 전: {len(y)}개 샘플 (클래스 분포: 0={class_counts[0]}, 1={class_counts[1] if len(class_counts)>1 else 0})")
    
    # 소수 클래스가 충분히 있는지 확인
    if len(class_counts) <= 1 or class_counts[1] < 2:
        log_message("    소수 클래스 샘플이 부족하여 Tomek Links를 적용할 수 없습니다.")
        return X, masks, y
    
    # 데이터를 2D로 변환 (Tomek Links 적용을 위해)
    original_shape = X.shape
    X_2d = X.reshape(X.shape[0], -1)
    masks_2d = masks.reshape(masks.shape[0], -1)
    
    # 데이터와 마스크를 결합
    combined = np.hstack([X_2d, masks_2d])
    
    try:
        # Tomek Links 적용
        tomek = TomekLinks(n_jobs=1)
        X_resampled_2d, y_resampled = tomek.fit_resample(combined, y)
        
        # 결합된 데이터에서 X와 마스크 분리
        X_part_size = X_2d.shape[1]
        X_tomek_2d = X_resampled_2d[:, :X_part_size]
        masks_tomek_2d = X_resampled_2d[:, X_part_size:]
        
        # 원래 형태로 복원
        X_resampled = X_tomek_2d.reshape(-1, original_shape[1], original_shape[2], original_shape[3])
        masks_resampled = masks_tomek_2d.reshape(-1, original_shape[1], original_shape[2])
        
        # 언더샘플링 결과 로깅
        new_class_counts = np.bincount(y_resampled.astype(int))
        removed_samples = len(y) - len(y_resampled)
        removed_majority = class_counts[0] - new_class_counts[0]
        removed_minority = class_counts[1] - new_class_counts[1] if len(new_class_counts) > 1 else 0
        
        log_message(f"    Tomek 적용 후: {len(y_resampled)}개 샘플 (클래스 분포: 0={new_class_counts[0]}, 1={new_class_counts[1] if len(new_class_counts)>1 else 0})")
        log_message(f"    제거된 샘플: 총 {removed_samples}개 (거래중기업: {removed_majority}개, 경영악화기업: {removed_minority}개)")
        
        # 불균형 비율 비교
        original_ratio = class_counts[0] / class_counts[1] if class_counts[1] > 0 else float('inf')
        new_ratio = new_class_counts[0] / new_class_counts[1] if new_class_counts[1] > 0 else float('inf')
        log_message(f"    클래스 불균형 비율: {original_ratio:.2f} → {new_ratio:.2f}")
        
        return X_resampled, masks_resampled, y_resampled
        
    except Exception as e:
        log_message(f"    Tomek Links 적용 중 오류 발생: {str(e)}")
        log_message("    원본 데이터를 반환합니다.")
        return X, masks, y 