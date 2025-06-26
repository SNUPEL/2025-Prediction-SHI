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

# 시트별 데이터 분할 함수
def split_sheet_data(file_path, sheet_name, company_groups, cutoff_date=CUTOFF_DATE, test_end_date=None):
    """시트별로 학습/테스트 데이터 분할"""
    log_message(f"\n시트 '{sheet_name}' 데이터 분할 중...")
    
    try:
        # 시트 데이터 로드
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # 기업 ID 열 확인
        id_col = 'No.'
        if id_col not in df.columns:
            log_message(f"  오류: 시트 '{sheet_name}'에 ID 열 '{id_col}'이 없습니다.")
            return None, None
        
        # 특별한 시트 처리
        if sheet_name == '4대보험 체납':
            log_message(f"  특별 처리: 4대보험 체납 시트 (3개월 단위)")
            
            # 기본 열 (날짜가 아닌 열)
            non_date_columns = []
            for col in df.columns:
                # 기본 열 분류
                if isinstance(col, str) and ('월' not in col or not any(c.isdigit() for c in col)):
                    non_date_columns.append(col)
            
            # 날짜 형식 열 분류 ('yy.m월' 형식)
            train_date_cols = []
            test_date_cols = []
            
            for col in df.columns:
                if col not in non_date_columns:
                    try:
                        # 'yy.m월' 형식 확인 (예: '20.3월', '24.9월')
                        if isinstance(col, str) and '월' in col:
                            parts = col.replace('월', '').split('.')
                            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                                year = int(parts[0])
                                # 연도가 두 자리(YY)인 경우 보정
                                if year < 100:
                                    year += 2000
                                
                                # cutoff_date 이후 날짜는 테스트 데이터로
                                month = int(parts[1])
                                col_date = pd.Timestamp(year=year, month=month, day=1)
                                
                                if col_date >= cutoff_date:
                                    # test_end_date가 설정된 경우 필터링
                                    if test_end_date is None or col_date <= test_end_date:
                                        test_date_cols.append(col)
                                else:
                                    train_date_cols.append(col)
                    except:
                        # 날짜 형식이 아닌 경우 비날짜 열로 처리
                        if col not in non_date_columns:
                            non_date_columns.append(col)
            
            # 날짜 열이 없으면 경고
            if not train_date_cols and not test_date_cols:
                log_message(f"  경고: '4대보험 체납' 시트에서 날짜 형식 열을 찾을 수 없습니다.")
                return None, None
                
        else:
            # 일반 시트 처리
            date_columns = []
            non_date_columns = []
            
            for col in df.columns:
                try:
                    if col in ['No.', '구분', '중구분', '소구분', '당사투입일', '계약종결일', '협력사구분', '계약종결사유', 'Unnamed: 8']:
                        non_date_columns.append(col)
                        continue
                    
                    # 열 이름이 날짜 형식인지 확인
                    date_obj = pd.to_datetime(col)
                    date_columns.append((col, date_obj))
                except:
                    non_date_columns.append(col)
            
            # 날짜 열을 날짜 순으로 정렬
            date_columns.sort(key=lambda x: x[1])
            
            # 날짜 열이 없으면 다음 시트로
            if not date_columns:
                log_message(f"  경고: 시트 '{sheet_name}'에 날짜 열이 없습니다.")
                return None, None
            
            # cutoff_date 기준으로 train과 test 분리
            train_date_cols = [col for col, date_obj in date_columns if date_obj < cutoff_date]
            
            # test_end_date 필터링 적용
            if test_end_date is None:
                test_date_cols = [col for col, date_obj in date_columns if date_obj >= cutoff_date]
            else:
                test_date_cols = [col for col, date_obj in date_columns if cutoff_date <= date_obj <= test_end_date]
        
        # 학습/테스트 데이터에 포함할 열
        train_columns = non_date_columns + train_date_cols
        test_columns = non_date_columns + test_date_cols
        
        # 각 회사의 데이터셋 정보 병합
        df = pd.merge(
            df, 
            company_groups[['No.', '데이터셋']], 
            on='No.', 
            how='left'
        )
        
        # 데이터셋 정보가 없는 회사는 학습 데이터로 간주
        df['데이터셋'] = df['데이터셋'].fillna('train')
        
        # 학습/테스트 데이터 분리
        train_data = df[df['데이터셋'] == 'train'][train_columns].copy() if len(train_columns) > 0 else pd.DataFrame()
        test_data = df[df['데이터셋'] == 'test'][test_columns].copy() if len(test_columns) > 0 else pd.DataFrame()
        
        log_message(f"  시트 '{sheet_name}' 분할 완료: 학습={len(train_data)}개 기업, 테스트={len(test_data)}개 기업")
        log_message(f"  학습 데이터 열: {len(train_columns)}개, 테스트 데이터 열: {len(test_columns)}개")
        
        return train_data, test_data
    except Exception as e:
        log_message(f"  오류: 시트 '{sheet_name}' 데이터 분할 중 예외 발생: {str(e)}")
        return None, None

# 통합 데이터 분할 및 저장 함수 (분할.py의 메인 기능)
def split_and_save_data(file_path, output_dir=SPLIT_OUTPUT_DIR, exclude_sheets=None, cutoff_date=CUTOFF_DATE, test_end_date=None, add_data_path=None, use_extended_test_data=USE_EXTENDED_TEST_DATA):
    """엑셀 파일을 읽고 데이터를 학습/테스트용으로 분할하여 저장"""
    
    # 필요한 함수들 임포트
    from .data_classification import classify_companies_for_split
    
    log_message(f"데이터 분할 작업 시작 - 기준 날짜: {cutoff_date.strftime('%Y-%m-%d')}")
    if test_end_date:
        log_message(f"테스트 데이터 종료 날짜: {test_end_date.strftime('%Y-%m-%d')}")
    if add_data_path:
        log_message(f"추가 데이터 파일 경로: {add_data_path}")
    
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "train"))
    ensure_dir(os.path.join(output_dir, "test"))
    
    # 엑셀 파일 읽기
    log_message(f"엑셀 파일 읽는 중: {file_path}")
    
    # 제외할 시트 목록 확인
    if exclude_sheets is None:
        exclude_sheets = []
    
    # 모든 시트 이름 가져오기
    xlsx = pd.ExcelFile(file_path)
    all_sheets = [sheet for sheet in xlsx.sheet_names if sheet not in exclude_sheets]
    
    # 회사 분류 수행 - use_extended_test_data 파라미터 전달
    company_groups, company_meta = classify_companies_for_split(file_path, cutoff_date, add_data_path, test_end_date, use_extended_test_data)
    if company_groups is None:
        log_message("오류: 회사 분류에 실패했습니다. 데이터 분할을 중단합니다.")
        return None
    
    # 회사 그룹 정보 저장
    company_groups_path = os.path.join(output_dir, 'company_groups.csv')
    company_groups.to_csv(company_groups_path, index=False, encoding='utf-8-sig')
    log_message(f"회사 그룹 정보 저장 완료: {company_groups_path}")
    
    # 각 시트별 처리
    processed_sheets = []
    for sheet_name in all_sheets:
        if sheet_name == '종합평가':
            continue  # 이미 처리함
            
        log_message(f"시트 '{sheet_name}' 처리 중...")
        
        # 시트 데이터 분할
        train_data, test_data = split_sheet_data(file_path, sheet_name, company_groups, cutoff_date, test_end_date)
        
        if train_data is None and test_data is None:
            log_message(f"  시트 '{sheet_name}' 처리 실패. 다음 시트로 진행합니다.")
            continue
        
        # CSV 파일로 저장
        train_file = os.path.join(output_dir, 'train', f"{sheet_name}.csv")
        test_file = os.path.join(output_dir, 'test', f"{sheet_name}.csv")
        
        # 데이터가 있는 경우에만 저장
        if len(train_data) > 0:
            train_data.to_csv(train_file, index=False, encoding='utf-8-sig')
            log_message(f"  훈련 데이터 저장 완료: {train_file} ({len(train_data)}개 행)")
        else:
            log_message(f"  훈련 데이터가 없습니다: {sheet_name}")
            
        if len(test_data) > 0:
            test_data.to_csv(test_file, index=False, encoding='utf-8-sig')
            log_message(f"  테스트 데이터 저장 완료: {test_file} ({len(test_data)}개 행)")
        else:
            log_message(f"  테스트 데이터가 없습니다: {sheet_name}")
        
        processed_sheets.append(sheet_name)
    
    log_message(f"데이터 분할 작업 완료. 처리된 시트: {len(processed_sheets)}개")
    return company_groups 