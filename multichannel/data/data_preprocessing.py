"""
데이터 처리 관련 함수 모듈
시트 분석, 회사 분류, 데이터 추출, 전처리 등의 기능 제공
"""
import os
import re
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

from config import (
    OUTPUT_DIR, CUTOFF_DATE, DATA_DIRECTORIES, 
    COMPANY_GROUPS_PATH, COMPANY_GROUPS_ALT_PATH, MIN_TIME_POINTS,
    SPLIT_OUTPUT_DIR, REGENERATE_SPLIT_DATA, USE_EXTENDED_TEST_DATA
)
from utils import ensure_dir, log_message
from sklearn.preprocessing import StandardScaler

# 데이터 패딩 및 텐서 변환
def process_and_pad_data(combined_data, sheet_names_list, combined_masks=None):
    """
    데이터 패딩 및 3D 텐서 변환
    
    params:
        combined_data: 입력 데이터
        sheet_names_list: 시트 이름 목록
        combined_masks: 마스킹 데이터 (None이면 기본 마스킹 생성)
    
    returns:
        X: 패딩된 데이터 [samples, channels, time_points, features]
        masks: 마스크 데이터 [samples, channels, time_points] (1=유효, 0=패딩)
    """
    if not combined_data:
        return np.array([]), np.array([])
    
    num_companies = len(combined_data)
    num_sheets = len(sheet_names_list)
    
    log_message(f"  패딩 처리 시작: {num_companies}개 기업, {num_sheets}개 시트")
        
    # 각 시트별로 최대 길이 계산
    max_lengths = []
    for i in range(len(sheet_names_list)):
        # 모든 회사의 해당 시트 길이 중 최대값
        max_len = max([len(company_data[i]) for company_data in combined_data], default=0)
        max_lengths.append(max_len)
    
    # 전체 최대 길이
    global_max_length = max(max_lengths)
    log_message(f"  전체 최대 시계열 길이: {global_max_length}")
    
    # 패딩 처리
    padded_data = []
    mask_data = []  # 마스크 데이터 저장
    
    for company_idx, company_data in enumerate(combined_data):
        padded_company = []
        company_masks = []  # 해당 회사의 마스크
        
        for i, sheet_series in enumerate(company_data):
            # 마스킹 정보가 제공된 경우 사용, 없으면 기본 마스킹 생성
            if combined_masks is not None and len(combined_masks[company_idx][i]) > 0:
                # 제공된 마스킹 사용
                original_mask = combined_masks[company_idx][i]
                # 패딩 부분에 0 추가
                mask = original_mask + [0] * (global_max_length - len(original_mask))
            else:
                # 기본 마스킹 생성 (1=유효데이터, 0=패딩)
                mask = [1] * len(sheet_series) + [0] * (global_max_length - len(sheet_series))
            
            company_masks.append(mask)
            
            # 빈 시트 데이터는 0으로 채움
            if not sheet_series:
                padded_company.append([0.0] * global_max_length)
            else:
                # 패딩 (뒤쪽에 0 추가)
                padded = sheet_series + [0.0] * (global_max_length - len(sheet_series))
                padded_company.append(padded)
                
        padded_data.append(padded_company)
        mask_data.append(company_masks)
    
    # 균일한 텐서로 변환
    X = np.zeros((len(padded_data), len(sheet_names_list), global_max_length))
    masks = np.zeros((len(padded_data), len(sheet_names_list), global_max_length))
    
    for i, (company_data, company_masks) in enumerate(zip(padded_data, mask_data)):
        for j, (sheet_data, sheet_mask) in enumerate(zip(company_data, company_masks)):
            X[i, j, :] = sheet_data
            masks[i, j, :] = sheet_mask
    
    # 차원 확장 (features 차원 추가)
    X = np.expand_dims(X, axis=3)
    
    # 마스킹 디버깅 로그
    if combined_masks is not None:
        total_mask_1 = np.sum(masks == 1)
        total_mask_0 = np.sum(masks == 0)
        mask_ratio = total_mask_1 / (total_mask_1 + total_mask_0) * 100
        log_message(f"  마스킹 적용 완료: mask=1 {total_mask_1}개 ({mask_ratio:.1f}%), mask=0 {total_mask_0}개")
    
    return X, masks

# 전처리 데이터 저장 함수 추가
def save_preprocessing_step(X, y, ids, step_name, dataset_type, output_dir=SPLIT_OUTPUT_DIR):
    """
    전처리 단계별 데이터를 CSV로 저장
    
    params:
        X: 데이터
        y: 레이블
        ids: 기업 ID 목록
        step_name: 전처리 단계 이름 (normalization, padding, final)
        dataset_type: 데이터셋 유형 (train, test)
        output_dir: 출력 디렉토리
    """
    # 저장 디렉토리 생성
    preproc_dir = os.path.join(output_dir, dataset_type, "preprocessing")
    ensure_dir(preproc_dir)
    
    # 데이터 형태에 따른 처리
    if len(X.shape) == 4:  # [samples, channels, time_points, features]
        # 최대 5개 샘플만 저장
        for idx in range(min(5, len(X))):
            company_id = ids[idx]
            company_data = []
            
            # 각 채널(시트)의 첫 번째 특성만 저장
            for channel_idx in range(X.shape[1]):
                for time_idx in range(X.shape[2]):
                    row = {
                        'ID': company_id,
                        'Label': y[idx],
                        'Channel': channel_idx,
                        'TimePoint': time_idx,
                        'Value': X[idx, channel_idx, time_idx, 0]
                    }
                    company_data.append(row)
            
            # DataFrame으로 변환하여 CSV로 저장
            df = pd.DataFrame(company_data)
            file_path = os.path.join(preproc_dir, f"{step_name}_{dataset_type}_company_{company_id}.csv")
            df.to_csv(file_path, index=False)
    
    else:  # 다른 형태의 데이터
        # 간단한 요약 정보만 저장
        summary = {
            'Dataset': dataset_type,
            'Step': step_name,
            'Samples': len(X),
            'Shape': str(X.shape),
            'Class_Distribution': np.bincount(y.astype(int)).tolist(),
        }
        
        file_path = os.path.join(preproc_dir, f"{step_name}_{dataset_type}_summary.json")
        with open(file_path, 'w') as f:
            import json
            json.dump(summary, f, indent=2)
    
    log_message(f"  {dataset_type} 데이터의 {step_name} 단계 샘플 저장 완료: {preproc_dir}")

# 기업별 데이터 정규화 함수
def normalize_data_by_company(data):
    """
    각 기업의 데이터를 독립적으로 정규화
    
    params:
        data: 입력 데이터 [samples, channels, time_points, features]
    
    returns:
        정규화된 데이터
    """
    num_samples = data.shape[0]
    
    # 각 기업 샘플별로 정규화 적용
    for sample_idx in range(num_samples):
        # 해당 기업의 모든 채널, 모든 시간점 데이터 추출 (2D로 변환)
        sample_data = data[sample_idx].reshape(-1, data.shape[3])
        
        # 스케일러 학습 및 적용
        scaler = StandardScaler()
        sample_data = scaler.fit_transform(sample_data)
        
        # 원래 형태로 복원
        data[sample_idx] = sample_data.reshape(data.shape[1], data.shape[2], data.shape[3])
    
    return data 