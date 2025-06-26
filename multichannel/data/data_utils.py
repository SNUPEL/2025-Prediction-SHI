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

# split_data 폴더 경로 확인 함수
def check_data_directories():
    """필요한 데이터 디렉토리 존재 여부 확인"""
    for path in DATA_DIRECTORIES:
        if not os.path.exists(path):
            log_message(f"경고: 필요한 데이터 경로가 존재하지 않습니다: {path}")
            # 없는 경로 자동 생성
            try:
                ensure_dir(path)
                log_message(f"생성 완료: {path}")
            except Exception as e:
                log_message(f"경로 생성 실패: {path} - {str(e)}")
                return False
    return True

# 결과 저장 함수
def save_results(predictions, metrics, fold_metrics=None, model_name="MultiChannel_CNNLSTM"):
    """모델 결과를 파일로 저장"""
    try:
        # 결과 폴더 생성
        ensure_dir(OUTPUT_DIR)
        
        # 예측 결과 저장
        result_file = os.path.join(OUTPUT_DIR, f"{model_name}_예측결과.csv")
        predictions.to_csv(result_file, index=False, encoding='utf-8-sig')
        log_message(f"  예측 결과 저장 완료: {result_file}")
        
        # 모델 성능 지표 저장
        metrics_file = os.path.join(OUTPUT_DIR, f"{model_name}_성능지표.csv")
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(metrics_file, index=False, encoding='utf-8-sig')
        log_message(f"  성능 지표 저장 완료: {metrics_file}")
        
        # 폴드별 성능 지표 저장 (있는 경우에만)
        if fold_metrics is not None:
            fold_metrics_file = os.path.join(OUTPUT_DIR, f"{model_name}_폴드별성능.csv")
            fold_metrics.to_csv(fold_metrics_file, index=False, encoding='utf-8-sig')
            log_message(f"  폴드별 성능 저장 완료: {fold_metrics_file}")
        
    except Exception as e:
        log_message(f"  결과 저장 중 오류 발생: {str(e)}")

# 데이터 준비 함수
def prepare_data(file_path, exclude_sheets=None, min_time_points=6, min_test_time_points=1, add_data_path=None, 
                regenerate_split=REGENERATE_SPLIT_DATA, cutoff_date=CUTOFF_DATE, split_output_dir=SPLIT_OUTPUT_DIR, test_end_date=None, use_extended_test_data=USE_EXTENDED_TEST_DATA):
    """
    데이터 준비 함수 - 모든 데이터 처리 단계를 통합
    
    params:
        file_path: 입력 데이터 파일 경로
        exclude_sheets: 제외할 시트 목록
        min_time_points: 훈련 데이터 최소 시간점 개수
        min_test_time_points: 테스트 데이터 최소 시간점 개수
        add_data_path: 추가 데이터 파일 경로
        regenerate_split: 데이터 분할 재생성 여부 (True: 새로 생성, False: 기존 파일 사용)
        cutoff_date: 훈련/테스트 데이터 분리 기준 날짜
        split_output_dir: 분할 데이터 저장 경로
        test_end_date: 테스트 데이터 종료 날짜 (None이면 모든 데이터 사용)
        use_extended_test_data: 확장된 테스트 데이터 사용 여부
    """
    # 필요한 함수들 임포트
    from .data_split import split_and_save_data
    from .data_analysis import check_eligible_sheets
    from .data_extraction import extract_combined_timeseries
    from .data_preprocessing import save_preprocessing_step, normalize_data_by_company
    
    # 1. 데이터 분할 수행 (필요한 경우)
    if regenerate_split:
        log_message(f"\n데이터 분할 재생성 옵션이 활성화되었습니다.")
        log_message(f"기준 날짜: {cutoff_date.strftime('%Y-%m-%d')}")
        if test_end_date:
            log_message(f"테스트 데이터 종료 날짜: {test_end_date.strftime('%Y-%m-%d')}")
        
        # 분할 수행 - use_extended_test_data 파라미터 전달
        split_result = split_and_save_data(
            file_path=file_path,
            output_dir=split_output_dir,
            exclude_sheets=exclude_sheets,
            cutoff_date=cutoff_date,
            test_end_date=test_end_date,
            add_data_path=add_data_path,
            use_extended_test_data=use_extended_test_data
        )
        
        if split_result is None:
            log_message("오류: 데이터 분할에 실패했습니다.")
            return None
        
        log_message(f"데이터 분할이 완료되었습니다. 기존 CSV 파일이 재생성되었습니다.")
    else:
        log_message(f"\n기존 분할 데이터를 사용합니다.")
    
    # 2. 데이터 디렉토리 확인
    if not check_data_directories():
        log_message("경고: 필요한 데이터 디렉토리가 없습니다. 계속 진행합니다.")
    
    # 3. 엑셀 파일의 시트 확인
    eligible_sheets, sheet_info, sheet_names = check_eligible_sheets(file_path, exclude_sheets)
    
    # 분석에 실제 사용되는 시트 이름만 추출
    used_sheet_names = []
    for sheet in eligible_sheets:
        if isinstance(sheet, dict) and 'sheet_name' in sheet:
            used_sheet_names.append(sheet['sheet_name'])
        elif isinstance(sheet, str):
            used_sheet_names.append(sheet)
    
    log_message(f"  분석에 사용되는 시트: {', '.join(used_sheet_names)}")
    
    if not eligible_sheets:
        log_message("분석 가능한 시트가 없습니다.")
        return None
    
    log_message(f"\n총 {len(eligible_sheets)}개 시트에서 분석 진행 중...")
    
    # 4. 통합 데이터 로드 및 전처리
    log_message("\n다중 시트 시계열 데이터 추출 중...")
    train_X, train_masks, train_y, train_ids, test_X, test_masks, test_y, test_ids = extract_combined_timeseries(
        file_path, eligible_sheets, sheet_info, min_time_points, min_test_time_points, add_data_path, cutoff_date, test_end_date
    )
    
    # 데이터가 비어있는지 확인
    if len(train_y) == 0:
        log_message("오류: 분석할 유효한 학습 데이터가 없습니다.")
        return None
    
    # 4.1. 전처리 단계 1: 패딩 적용 후 데이터 저장
    save_preprocessing_step(train_X, train_y, train_ids, "padding", "train", split_output_dir)
    save_preprocessing_step(test_X, test_y, test_ids, "padding", "test", split_output_dir)
    
    # 4.2. 기업별 정규화 적용
    log_message("  기업별 데이터 정규화 적용...")
    train_X_norm = normalize_data_by_company(train_X.copy())
    test_X_norm = normalize_data_by_company(test_X.copy()) if len(test_X) > 0 else np.array([])
    
    # 4.3. 전처리 단계 2: 정규화 적용 후 데이터 저장
    save_preprocessing_step(train_X_norm, train_y, train_ids, "normalization", "train", split_output_dir)
    save_preprocessing_step(test_X_norm, test_y, test_ids, "normalization", "test", split_output_dir)
    
    # 5. 데이터 저장
    ensure_dir(os.path.join(OUTPUT_DIR, "data"))
    np.save(os.path.join(OUTPUT_DIR, "data/train_X.npy"), train_X_norm)
    np.save(os.path.join(OUTPUT_DIR, "data/train_masks.npy"), train_masks)
    np.save(os.path.join(OUTPUT_DIR, "data/train_y.npy"), train_y)
    np.save(os.path.join(OUTPUT_DIR, "data/test_X.npy"), test_X_norm)
    np.save(os.path.join(OUTPUT_DIR, "data/test_masks.npy"), test_masks)
    np.save(os.path.join(OUTPUT_DIR, "data/test_y.npy"), test_y)
    
    pd.DataFrame({'company_id': train_ids}).to_csv(
        os.path.join(OUTPUT_DIR, "data/train_ids.csv"), 
        index=False, encoding='utf-8-sig'
    )
    pd.DataFrame({'company_id': test_ids}).to_csv(
        os.path.join(OUTPUT_DIR, "data/test_ids.csv"), 
        index=False, encoding='utf-8-sig'
    )
    
    # 6. 검증 데이터 분할 (훈련 데이터의 20%)
    X_train, X_val, masks_train, masks_val, y_train, y_val, train_ids_split, val_ids = train_test_split(
        train_X_norm, train_masks, train_y, train_ids, test_size=0.2, random_state=42, stratify=train_y
    )
    
    # 6.1. 전처리 단계 3: 최종 데이터 저장
    save_preprocessing_step(X_train, y_train, train_ids_split, "final", "train", split_output_dir)
    save_preprocessing_step(X_val, y_val, val_ids, "final", "validation", split_output_dir)
    save_preprocessing_step(test_X_norm, test_y, test_ids, "final", "test", split_output_dir)
    
    return {
        'X_train': X_train, 
        'masks_train': masks_train,
        'y_train': y_train, 
        'X_val': X_val, 
        'masks_val': masks_val,
        'y_val': y_val,
        'X_test': test_X_norm, 
        'masks_test': test_masks,
        'y_test': test_y,
        'train_ids': train_ids_split,
        'val_ids': val_ids,
        'test_ids': test_ids,
        'used_sheet_names': used_sheet_names
    } 