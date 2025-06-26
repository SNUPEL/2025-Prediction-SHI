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

# 시트별 시간 단위에 따른 컷오프 날짜 계산 함수
def get_cutoff_date_by_company(company_id, termination_date, months_to_remove, sheet_name=None):
    """
    기업의 계약종결일과 제거 개월 수를 기준으로 컷오프 날짜 계산
    시트별 시간 단위를 고려함
    """
    if pd.isna(termination_date) or months_to_remove <= 0:
        return None
    
    # 계약종결일로부터 제거할 개월 수만큼 이전 날짜 계산
    cutoff_date = termination_date - pd.DateOffset(months=months_to_remove)
    
    # 시트별 시간 단위 고려
    if sheet_name == '4대보험 체납':
        # 3개월 단위로 조정
        month_remainder = cutoff_date.month % 3
        if month_remainder > 0:
            cutoff_date = cutoff_date - pd.DateOffset(months=month_remainder)
    elif sheet_name == '종합평가':
        # 6개월 단위로 조정
        month_remainder = cutoff_date.month % 6
        if month_remainder > 0:
            cutoff_date = cutoff_date - pd.DateOffset(months=month_remainder)
    
    return cutoff_date

# 시간 열의 날짜 추출 함수
def get_date_from_time_col(col, sheet_name):
    """
    시간 열(컬럼명)에서 날짜 객체 추출
    """
    try:
        # 4대보험 체납 시트의 특수 날짜 형식 처리
        if sheet_name == '4대보험 체납' and isinstance(col, str) and '월' in col:
            parts = col.replace('월', '').split('.')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                year = int(parts[0]) + 2000 if int(parts[0]) < 100 else int(parts[0])
                month = int(parts[1])
                return pd.Timestamp(year=year, month=month, day=1)
        
        # 일반 날짜 형식 처리
        if isinstance(col, datetime):
            return pd.Timestamp(col)
        elif isinstance(col, str):
            if re.match(r'^\d{4}-\d{1,2}(?:-\d{1,2})?$', col):
                return pd.to_datetime(col)
            elif re.match(r'^\d{4}$', col):
                return pd.to_datetime(f"{col}-01-01")
    except:
        pass
    
    return None

# 통합 시계열 데이터 추출 함수
def extract_combined_timeseries(file_path, eligible_sheets, sheet_info, min_time_points=6, min_test_time_points=1, add_data_path=None, cutoff_date=CUTOFF_DATE, test_end_date=None):
    """
    여러 시트에서 동일 기업의 시계열 데이터를 추출하여 결합
    - company_groups.csv 기준으로 훈련/테스트 데이터 분리
    - 훈련 기업: cutoff_date 이전 데이터만 사용
    - 테스트 기업: cutoff_date 이후, test_end_date 이전 데이터만 사용 (test_end_date가 None이면 모든 데이터)
    - 경영악화 기업: 계약종결일 기준으로 일정 개월 수 데이터 제거 및 계약종결일 조정
    """
    # 필요한 함수들 임포트
    from .data_analysis import read_remove_months_info
    from .data_classification import get_company_groups
    from .data_preprocessing import process_and_pad_data
    
    try:
        log_message(f"통합 시계열 데이터 추출 시작... (기준 날짜: {cutoff_date.strftime('%Y-%m-%d')})")
        if test_end_date:
            log_message(f"테스트 데이터 종료 날짜: {test_end_date.strftime('%Y-%m-%d')}")
        log_message(f"훈련 데이터 최소 시간점: {min_time_points}, 테스트 데이터 최소 시간점: {min_test_time_points}")
        
        # 경영악화 기업 제거 개월 수 정보 로드
        remove_months_info = {}
        if add_data_path:
            remove_months_info = read_remove_months_info(add_data_path)
        
        # 계약종결일 및 투입일 정보 로드
        termination_dates = {}
        input_dates = {}
        adjusted_termination_dates = {}  # 조정된 계약종결일 저장용
        
        try:
            df_info = pd.read_excel(file_path, sheet_name='종합평가')
            
            # 1. 투입일과 계약종결일 정보 로드
            for _, row in df_info.iterrows():
                if 'No.' in df_info.columns:
                    company_id = row['No.']
                    
                    if pd.notna(company_id):
                        company_id = int(company_id)
                        
                        # 투입일 저장
                        if '당사투입일' in df_info.columns and pd.notna(row['당사투입일']):
                            try:
                                input_date_dt = pd.to_datetime(row['당사투입일'])
                                input_dates[company_id] = input_date_dt
                                log_message(f"  기업 {company_id}: 당사투입일 = {input_date_dt.strftime('%Y-%m-%d')}")
                            except Exception as e:
                                log_message(f"  기업 {company_id}: 당사투입일 변환 실패 - {row['당사투입일']} - {str(e)}")
                        
                        # 계약종결일 저장
                        if '계약종결일' in df_info.columns:
                            term_date = row['계약종결일']
                            
                            if pd.notna(term_date) and term_date != '-' and not (isinstance(term_date, str) and term_date.strip() == '-'):
                                try:
                                    term_date_dt = pd.to_datetime(term_date)
                                    termination_dates[company_id] = term_date_dt
                                    adjusted_termination_dates[company_id] = term_date_dt  # 초기값
                                    log_message(f"  기업 {company_id}: 계약종결일 = {term_date_dt.strftime('%Y-%m-%d')}")
                                except Exception as e:
                                    log_message(f"  기업 {company_id}: 계약종결일 변환 실패 - {term_date} - {str(e)}")
                            else:
                                log_message(f"  기업 {company_id}: 계약종결일 없음 (아직 거래중)")
            
            # 2. 계약종결일 조정 (add_data.xlsx 정보 기반)
            log_message("계약종결일 조정 시작...")
            for company_id, months_to_remove in remove_months_info.items():
                if company_id in termination_dates:
                    original_term_date = termination_dates[company_id]
                    
                    # 계약종결일 조정 (개월 수만큼 이전으로)
                    adjusted_term_date = original_term_date - pd.DateOffset(months=months_to_remove)
                    adjusted_termination_dates[company_id] = adjusted_term_date
                    
                    log_message(f"  기업 {company_id}: 계약종결일 조정 {original_term_date.strftime('%Y-%m-%d')} → {adjusted_term_date.strftime('%Y-%m-%d')} ({months_to_remove}개월 이전)")
                else:
                    log_message(f"  기업 {company_id}: 계약종결일 정보 없어 조정 불가 (개월 수: {months_to_remove})")
            
            log_message(f"계약종결일 정보 로드 및 조정 완료: {len(termination_dates)}개 기업, {len([k for k in adjusted_termination_dates if adjusted_termination_dates[k] != termination_dates.get(k, None)])}개 기업 조정됨")
        except Exception as e:
            log_message(f"계약종결일 정보 로드 오류: {str(e)}")
        
        # 3. 회사 분류 수행 (조정된 계약종결일 사용)
        company_sets = get_company_groups()
        
        # 테스트 기업 디버깅
        log_message(f"테스트 기업 목록: {company_sets['test']}")
        log_message(f"테스트 기업 수: {len(company_sets['test'])}")
        
        # 기업 데이터 저장 사전 (key: 기업ID, value: 시트별 시계열 데이터)
        company_data = {}
        
        # 제거된 데이터 로깅용 카운터
        removed_data_count = {sheet_name: 0 for sheet_name in eligible_sheets}
        
        # 시트별 처리
        for sheet_name in tqdm(eligible_sheets, desc="시트별 데이터 수집"):
            log_message(f"  {sheet_name} 시트 데이터 수집 중...")
            
            # 시트 데이터 로드
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # 시트 정보 가져오기
            info = sheet_info[sheet_name]
            status_col = info['status_col']
            id_col = info['id_col']
            
            # 훈련용과 테스트용 시간열 가져오기
            train_time_cols = info['train_time_cols']  # cutoff_date 이전
            test_time_cols = info['test_time_cols']    # cutoff_date 이후
            
            # 시간 열 정렬 함수 (날짜순) - 시트별 날짜 형식 고려
            def get_time_key(col):
                # 4대보험 체납 시트의 특수 날짜 형식 처리
                if sheet_name == '4대보험 체납' and isinstance(col, str) and '월' in col:
                    parts = col.replace('월', '').split('.')
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        year = int(parts[0]) + 2000 if int(parts[0]) < 100 else int(parts[0])
                        month = int(parts[1])
                        return pd.Timestamp(year=year, month=month, day=1)
                
                # 일반 날짜 형식 처리
                try:
                    if isinstance(col, datetime):
                        return pd.Timestamp(col)
                    elif isinstance(col, str):
                        if re.match(r'^\d{4}-\d{1,2}(?:-\d{1,2})?$', col):
                            return pd.to_datetime(col)
                        elif re.match(r'^\d{4}$', col):
                            return pd.to_datetime(f"{col}-01-01")
                except:
                    pass
                
                # 정렬에 사용할 기본값 반환
                return pd.Timestamp('1900-01-01')
            
            # 훈련용 시간 열 정렬 (cutoff_date 이전 데이터)
            sorted_train_time_cols = sorted(train_time_cols, key=get_time_key)
            
            # 테스트용 시간 열 정렬 (cutoff_date 이후 데이터)
            sorted_test_time_cols = sorted(test_time_cols, key=get_time_key)
            
            # 시간 열과 날짜 매핑 생성
            time_col_dates = {}
            for col in sorted_train_time_cols + sorted_test_time_cols:
                date_obj = get_date_from_time_col(col, sheet_name)
                if date_obj is not None:
                    time_col_dates[col] = date_obj
            
            # test_end_date가 설정된 경우, 테스트 시간열 필터링
            if test_end_date is not None:
                filtered_test_time_cols = []
                for col in sorted_test_time_cols:
                    if col in time_col_dates:
                        col_date = time_col_dates[col]
                        # cutoff_date 이후, test_end_date 이전 데이터만 포함
                        if col_date is not None and col_date <= test_end_date:
                            filtered_test_time_cols.append(col)
                
                # 필터링된 테스트 시간열로 업데이트
                sorted_test_time_cols = filtered_test_time_cols
                log_message(f"  {sheet_name} 시트: 테스트 종료 날짜 필터링 적용 후 시간 열 {len(sorted_test_time_cols)}개")
            
            # 훈련용/테스트용 시간열 크기 확인 로깅
            log_message(f"  {sheet_name} 시트: 훈련용 시간 열 {len(sorted_train_time_cols)}개, 테스트용 시간 열 {len(sorted_test_time_cols)}개 사용")
            
            # 각 기업별로 시계열 데이터 추출
            for idx, row in df.iterrows():
                # 기업 ID 가져오기
                company_id = row[id_col]
                
                # 상태 확인 (실제 상태)
                status = str(row[status_col]).strip().lower()
                if '경영악화' in status:
                    original_label = 1
                elif '거래중' in status:
                    original_label = 0
                else:
                    continue  # 상태가 명확하지 않으면 건너뜀
                
                # Both 회사들은 두 번 처리 (train용, test용)
                is_both_company = company_id in company_sets.get('both', [])
                is_test_only = company_id in company_sets['test'] and not is_both_company
                is_train_only = company_id in company_sets['train'] and not is_both_company
                
                # 처리할 케이스들 정의
                process_cases = []
                
                if is_train_only:
                    process_cases.append(('train_only', sorted_train_time_cols, False))
                elif is_test_only:
                    process_cases.append(('test_only', sorted_test_time_cols, True))
                elif is_both_company:
                    # Both 회사는 두 번 처리
                    process_cases.append(('both_train', sorted_train_time_cols, False))
                    process_cases.append(('both_test', sorted_test_time_cols, True))
                
                # 각 케이스별로 데이터 처리
                for case_type, time_cols_to_use, is_for_test in process_cases:
                    
                    # 경영악화 기업의 경우 망하기 직전 데이터 제거 (add_data.xlsx 기반, 동적 처리)
                    if company_id in remove_months_info and company_id in termination_dates:
                        months_to_remove = remove_months_info[company_id]
                        termination_date = termination_dates[company_id]
                        
                        # 컷오프 날짜 계산 (시트별 시간 단위 고려)
                        cutoff_date_company = get_cutoff_date_by_company(
                            company_id, termination_date, months_to_remove, sheet_name
                        )
                        
                        if cutoff_date_company is not None:
                            # 컷오프 날짜 이후 데이터 제거
                            original_count = len(time_cols_to_use)
                            filtered_cols = []
                            
                            for col in time_cols_to_use:
                                if col in time_col_dates:
                                    col_date = time_col_dates[col]
                                    # 컷오프 날짜 이전 데이터만 사용
                                    if col_date < cutoff_date_company:
                                        filtered_cols.append(col)
                            
                            # 필터링된 시간열 적용
                            removed_count = original_count - len(filtered_cols)
                            removed_data_count[sheet_name] += removed_count
                            
                            if removed_count > 0:
                                time_cols_to_use = filtered_cols
                                log_message(f"    기업 {company_id} ({case_type}, {sheet_name}): {months_to_remove}개월 제거, {removed_count}개 시간 열 제거됨 (남은 시간열: {len(filtered_cols)}개)")
                    
                    # 시계열 데이터 추출 (케이스 유형에 맞는 시간열 사용)
                    timeseries = []
                    timeseries_mask = []  # 마스킹 정보 추가
                    
                    # 기업 투입일/계약종결일 기반 마스킹 로직
                    company_input_date = input_dates.get(company_id)
                    company_term_date = adjusted_termination_dates.get(company_id)
                    
                    for col in time_cols_to_use:
                        if col in df.columns:
                            # 데이터 값 처리
                            if pd.notna(row[col]):
                                try:
                                    value = float(row[col])
                                    timeseries.append(value)
                                except:
                                    timeseries.append(0.0)  # 변환 실패 시 0으로 대체
                            else:
                                timeseries.append(0.0)  # NaN은 0으로 대체
                            
                            # 마스킹 로직: 계약 기간 내면 1, 외면 0
                            col_date = time_col_dates.get(col)
                            mask_value = 1  # 기본값
                            
                            if col_date and company_input_date and company_term_date:
                                # 투입일부터 계약종결일까지 기간 내인지 확인
                                if company_input_date <= col_date <= company_term_date:
                                    mask_value = 1  # 계약 기간 내
                                else:
                                    mask_value = 0  # 계약 기간 외
                            elif col_date and company_input_date and not company_term_date:
                                # 계약종결일이 없는 경우 (현재 거래중)
                                if col_date >= company_input_date:
                                    mask_value = 1  # 투입일 이후
                                else:
                                    mask_value = 0  # 투입일 이전
                            
                            timeseries_mask.append(mask_value)
                    
                    # 시계열 데이터가 너무 짧으면 건너뜀 (훈련/테스트 데이터에 다른 기준 적용)
                    required_time_points = min_test_time_points if is_for_test else min_time_points
                    if len(timeseries) < required_time_points:
                        continue
                    
                    # 회사가 사전에 없으면 추가 (케이스별로 고유 키 생성)
                    company_key = f"{company_id}_{case_type}"
                    if company_key not in company_data:
                        # 테스트 그룹이고 원래 라벨이 1(경영악화)인 경우, 훈련 라벨은 0으로 변경 (리크 방지)
                        train_label = 0 if (is_for_test and original_label == 1) else original_label

                        # 테스트 라벨은 항상 원래 상태 유지
                        test_label = original_label 
                        
                        # 계약종결일 정보 포함
                        termination_date_info = None
                        adjusted_termination_date_info = None
                        
                        if company_id in termination_dates:
                            termination_date_info = termination_dates[company_id]
                        
                        if company_id in adjusted_termination_dates:
                            adjusted_termination_date_info = adjusted_termination_dates[company_id]
                        
                        company_data[company_key] = {
                            'company_id': company_id,  # 원본 ID 저장
                            'case_type': case_type,
                            'original_label': original_label,
                            'train_label': train_label,
                            'test_label': test_label,
                            'is_test': is_for_test,
                            'termination_date': termination_date_info,
                            'adjusted_termination_date': adjusted_termination_date_info,
                            'input_date': company_input_date,
                            'sheets': {},
                            'masks': {}  # 마스킹 정보 저장
                        }
                    
                    # 해당 시트의 시계열 데이터 및 마스킹 저장
                    if timeseries:
                        company_data[company_key]['sheets'][sheet_name] = timeseries
                        company_data[company_key]['masks'][sheet_name] = timeseries_mask
                        
                        # 마스킹 디버깅 로그 (특정 기업만)
                        if company_id in [1, 2, 37, 38]:
                            mask_1_count = sum(timeseries_mask)
                            mask_0_count = len(timeseries_mask) - mask_1_count
                            input_str = company_input_date.strftime('%Y-%m-%d') if company_input_date else 'None'
                            term_str = company_term_date.strftime('%Y-%m-%d') if company_term_date else 'None'
                            log_message(f"    [마스킹] 기업 {company_id} ({case_type}, {sheet_name}): 투입일={input_str}, 종결일={term_str}, mask=1 {mask_1_count}개, mask=0 {mask_0_count}개")
        
        # 제거된 데이터 요약 로깅
        if any(removed_data_count.values()):
            log_message("\n=== 경영악화 기업 데이터 제거 요약 ===")
            for sheet_name, count in removed_data_count.items():
                if count > 0:
                    log_message(f"  * {sheet_name}: {count}개 시간 열 제거됨")
        
        # 계약종결일 조정 정보 요약 로깅
        log_message("\n=== 계약종결일 조정 요약 ===")
        adjusted_companies = 0
        for company_id, data in company_data.items():
            if data.get('termination_date') != data.get('adjusted_termination_date') and data.get('termination_date') is not None:
                adjusted_companies += 1
                original_date = data['termination_date'].strftime('%Y-%m-%d')
                adjusted_date = data['adjusted_termination_date'].strftime('%Y-%m-%d')
                log_message(f"  * 기업 {company_id}: {original_date} → {adjusted_date}")
        
        log_message(f"  총 {adjusted_companies}개 기업의 계약종결일 조정됨")
        
        # 2단계: 학습/테스트 데이터셋 생성
        train_combined_data = []
        train_combined_masks = []  # 마스킹 데이터 추가
        train_labels = []
        train_ids = []
        
        test_combined_data = []
        test_combined_masks = []  # 마스킹 데이터 추가
        test_labels = []
        test_ids = []
        
        sheet_names_list = eligible_sheets  # 시트 순서 고정
        
        # 각 회사 데이터 처리
        test_included = []
        test_excluded = []
        train_included = []
        
        for company_key, data in tqdm(company_data.items(), desc="데이터 구성"):
            company_id = data['company_id']
            case_type = data['case_type']
            
            if data['is_test']:
                # 테스트 데이터 처리 (test_only + both_test)
                sheets_count = len(data['sheets'])
                if sheets_count >= 1:
                    test_included.append(company_id)
                    sheet_data = []
                    sheet_masks = []
                    for sheet_name in sheet_names_list:
                        if sheet_name in data['sheets']:
                            sheet_data.append(data['sheets'][sheet_name])
                            sheet_masks.append(data['masks'][sheet_name])
                        else:
                            sheet_data.append([])  # 해당 시트 데이터 없음
                            sheet_masks.append([])  # 해당 시트 마스크 없음
                    
                    test_combined_data.append(sheet_data)
                    test_combined_masks.append(sheet_masks)
                    test_labels.append(data['test_label'])
                    test_ids.append(company_id)
                else:
                    test_excluded.append((company_id, sheets_count))
            else:
                # 훈련 데이터 처리 (train_only + both_train)
                # 학습 데이터셋에는 분석 가능한 시트 수가 절반 이상인 경우만 포함 (기존 로직 유지)
                if len(data['sheets']) >= len(eligible_sheets) // 2:
                    train_included.append(company_id)
                    sheet_data = []
                    sheet_masks = []
                    for sheet_name in sheet_names_list:
                        if sheet_name in data['sheets']:
                            sheet_data.append(data['sheets'][sheet_name])
                            sheet_masks.append(data['masks'][sheet_name])
                        else:
                            sheet_data.append([])  # 해당 시트 데이터 없음
                            sheet_masks.append([])  # 해당 시트 마스크 없음
                    
                    train_combined_data.append(sheet_data)
                    train_combined_masks.append(sheet_masks)
                    train_labels.append(data['train_label'])
                    train_ids.append(company_id)
        
        # 데이터셋 구성 요약
        test_only_companies = [cid for cid in test_ids if cid not in company_sets.get('both', [])]
        both_test_companies = [cid for cid in test_ids if cid in company_sets.get('both', [])]
        train_only_companies = [cid for cid in train_ids if cid not in company_sets.get('both', [])]
        both_train_companies = [cid for cid in train_ids if cid in company_sets.get('both', [])]
        
        log_message(f"  * 테스트 데이터 구성: Test Only {len(test_only_companies)}개 + Both Test {len(both_test_companies)}개 = 총 {len(test_ids)}개")
        log_message(f"  * 훈련 데이터 구성: Train Only {len(train_only_companies)}개 + Both Train {len(both_train_companies)}개 = 총 {len(train_ids)}개")
        
        # 테스트 데이터 포함/제외 정보 출력
        log_message(f"테스트 포함된 기업 (총 {len(test_included)}개): {test_included}")
        log_message(f"  → Test Only {len(test_only_companies)}개: {test_only_companies}")
        log_message(f"  → Both Test {len(both_test_companies)}개: {both_test_companies}")
        log_message(f"테스트 제외된 기업 (총 {len(test_excluded)}개): {test_excluded}")
        
        log_message(f"훈련 포함된 기업 (총 {len(train_included)}개): {train_included}")
        log_message(f"  → Train Only {len(train_only_companies)}개: {train_only_companies}")
        log_message(f"  → Both Train {len(both_train_companies)}개: {both_train_companies}")
        
        # 시간적 분리 관련 로깅 추가
        cutoff_str = cutoff_date.strftime('%Y-%m-%d')
        test_end_str = test_end_date.strftime('%Y-%m-%d') if test_end_date is not None else "마지막 시점"
        log_message("\n=== 개선된 시간적 분리 적용 상태 ===")
        log_message(f"  * 기준 날짜: {cutoff_str}")
        log_message(f"  * 테스트 종료 날짜: {test_end_str}")
        log_message(f"  * 훈련 기업: {cutoff_str} 이전 데이터만 사용")
        log_message(f"  * 테스트 기업: {cutoff_str} 이후 ~ {test_end_str} 데이터 사용")
        log_message(f"  * 경영악화 기업: 계약종결일 기준 직전 데이터 제거 및 계약종결일 조정 적용됨")
        log_message(f"  * 훈련 기업 수: {len(train_ids)}개")
        log_message(f"  * 테스트 기업 수: {len(test_ids)}개")
        
        # Both 회사들의 정보 추가
        both_in_test = [cid for cid in test_ids if cid in company_sets.get('both', [])]
        test_only_in_test = [cid for cid in test_ids if cid not in company_sets.get('both', [])]
        
        if both_in_test:
            log_message(f"  * 테스트 데이터 구성: Test Only {len(test_only_in_test)}개 + Both {len(both_in_test)}개")
        
        # 테스트 데이터 포함/제외 정보 출력  
        log_message(f"테스트 포함된 기업 (총 {len(test_included)}개): {test_included}")
        if both_in_test:
            log_message(f"  → 이 중 Both 회사 {len(both_in_test)}개: {both_in_test}")
        log_message(f"테스트 제외된 기업 (총 {len(test_excluded)}개): {test_excluded}")
        
        # 패딩 및 최종 형태 변환 (마스킹 정보 포함)
        train_X, train_masks = process_and_pad_data(train_combined_data, sheet_names_list, train_combined_masks)
        train_y = np.array(train_labels)
        
        test_X, test_masks = process_and_pad_data(test_combined_data, sheet_names_list, test_combined_masks)
        test_y = np.array(test_labels)
        
        log_message(f"  학습 데이터: {len(train_y)}개 기업, {train_X.shape[1]}개 시트, 최대 {train_X.shape[2]}개 시간점")
        log_message(f"  테스트 데이터: {len(test_y)}개 기업, {test_X.shape[1]}개 시트, 최대 {test_X.shape[2]}개 시간점")
        
        # 라벨 변경 정보 출력 로직 수정
        label_changes = []
        for company_id, data in company_data.items():
            # 테스트 그룹이고 원래 라벨이 1인데 훈련 라벨이 0으로 변경된 경우만 로깅
            if data['is_test'] and data['original_label'] == 1 and data['train_label'] == 0:
                label_changes.append({
                    'No.': company_id,
                    '원래라벨': data['original_label'],
                    '훈련라벨': data['train_label'],
                    '테스트라벨': data['test_label'] 
                })

        # 라벨 변경 정보 출력
        if label_changes:
            log_message("경영악화 기업 라벨 변경 정보:")
            for change in label_changes:
                log_message(f"  기업 {change['No.']}: 원래={change['원래라벨']}, 훈련={change['훈련라벨']}, 테스트={change['테스트라벨']}")
        
        return train_X, train_masks, train_y, train_ids, test_X, test_masks, test_y, test_ids
        
    except Exception as e:
        import traceback
        log_message(f"  통합 시계열 추출 중 오류 발생: {str(e)}")
        log_message(traceback.format_exc())
        return np.array([]), np.array([]), np.array([]), [], np.array([]), np.array([]), np.array([]), [] 