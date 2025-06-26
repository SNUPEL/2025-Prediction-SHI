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

# 회사 분류 함수
def classify_companies(file_path, cutoff_date=CUTOFF_DATE):
    """투입일과 계약종결일에 따라 회사 분류 (분할.py 방식 적용)"""
    log_message("\n회사 분류 중...")
    
    try:
        # 기본 정보가 포함된 '종합평가' 시트 로드
        df_info = pd.read_excel(file_path, sheet_name='종합평가')
        
        # 날짜 형식 변환
        for col in ['당사투입일', '계약종결일']:
            if col in df_info.columns:
                df_info[col] = pd.to_datetime(df_info[col], errors='coerce')
        
        # 회사 분류 (test/train 구분)
        company_sets = {
            'train': [],  # 훈련 데이터로 사용할 회사
            'test': []    # 테스트 데이터로 사용할 회사
        }
        
        # 회사별 그룹 저장
        for _, row in df_info.iterrows():
            company_id = row['No.']
            
            # 2024년 이후 투입 회사는 test로 분류
            if pd.notna(row['당사투입일']) and row['당사투입일'] >= cutoff_date:
                company_sets['test'].append(company_id)
                continue
                
            # 2024년에 계약 종결된 경영악화 회사는 test로 분류
            if (pd.notna(row['계약종결일']) and row['계약종결일'].year == 2024 and
                pd.notna(row['계약종결사유']) and '경영악화' in str(row['계약종결사유'])):
                company_sets['test'].append(company_id)
                continue
                
            # 나머지는 train으로 분류
            company_sets['train'].append(company_id)
        
        log_message(f"  회사 분류 완료: 훈련 {len(company_sets['train'])}개, 테스트 {len(company_sets['test'])}개")
        return company_sets, df_info
        
    except Exception as e:
        log_message(f"회사 분류 중 오류 발생: {str(e)}")
        return None, None

# company_groups.csv 사용 함수
def get_company_groups():
    """company_groups.csv에서 회사 분류 정보 가져오기"""
    csv_path = COMPANY_GROUPS_PATH
    
    # 파일 존재 확인
    if not os.path.exists(csv_path):
        log_message(f"오류: 회사 분류 파일이 없습니다: {csv_path}")
        # 파일이 없을 경우 대체 경로 시도
        alt_path = COMPANY_GROUPS_ALT_PATH
        if os.path.exists(alt_path):
            log_message(f"대체 경로에서 파일을 찾았습니다: {alt_path}")
            csv_path = alt_path
        else:
            log_message("오류: 회사 분류 파일을 찾을 수 없습니다. 데이터 분할이 불가능합니다.")
            raise FileNotFoundError(f"회사 분류 파일을 찾을 수 없습니다: {csv_path}")
    
    groups_df = pd.read_csv(csv_path)
    
    train_companies = groups_df[groups_df['데이터셋'] == 'train']['No.'].tolist()
    test_companies = groups_df[groups_df['데이터셋'] == 'test']['No.'].tolist()
    both_companies = groups_df[groups_df['데이터셋'] == 'both']['No.'].tolist()
    
    # both 회사들을 train과 test 모두에 포함
    train_companies_total = train_companies + both_companies
    test_companies_total = test_companies + both_companies
    
    log_message(f"회사 분류 완료: 훈련 {len(train_companies_total)}개 (train:{len(train_companies)}, both:{len(both_companies)}), 테스트 {len(test_companies_total)}개 (test:{len(test_companies)}, both:{len(both_companies)})")
    return {
        'train': train_companies_total, 
        'test': test_companies_total,
        'both': both_companies
    }

# 분할을 위한 회사 분류 함수 (분할.py와 동일 로직)
def classify_companies_for_split(file_path, cutoff_date=CUTOFF_DATE, add_data_path=None, test_end_date=None, use_extended_test_data=USE_EXTENDED_TEST_DATA):
    """투입일과 계약종결일에 따라 회사 분류 (분할.py 방식)"""
    log_message("\n데이터 분할을 위한 회사 분류 중...")
    
    # read_remove_months_info 함수를 임포트해야 함
    from .data_analysis import read_remove_months_info
    
    try:
        # 기본 정보가 포함된 '종합평가' 시트 로드
        df_info = pd.read_excel(file_path, sheet_name='종합평가')
        
        # 필요한 열 확인
        required_cols = ['No.', '구분', '당사투입일', '계약종결일', '계약종결사유']
        for col in required_cols:
            if col not in df_info.columns:
                log_message(f"오류: '종합평가' 시트에 필요한 열 '{col}'이 없습니다.")
                return None, None
        
        # 37번과 38번 회사 초기 상태 확인 (디버깅용)
        if 37 in df_info['No.'].values:
            company_37 = df_info[df_info['No.'] == 37].iloc[0]
            log_message(f"  [초기] 기업 37: 계약종결일={company_37['계약종결일'] if pd.notna(company_37['계약종결일']) else 'None'}, 계약종결사유={company_37['계약종결사유'] if pd.notna(company_37['계약종결사유']) else 'None'}")
        
        if 38 in df_info['No.'].values:
            company_38 = df_info[df_info['No.'] == 38].iloc[0]
            log_message(f"  [초기] 기업 38: 계약종결일={company_38['계약종결일'] if pd.notna(company_38['계약종결일']) else 'None'}, 계약종결사유={company_38['계약종결사유'] if pd.notna(company_38['계약종결사유']) else 'None'}")
        
        # 날짜 형식 확인 및 변환
        for col in ['당사투입일', '계약종결일']:
            if df_info[col].dtype != 'datetime64[ns]':
                # '-' 또는 빈 문자열을 NaT로 변환
                df_info[col] = df_info[col].apply(lambda x: None if isinstance(x, str) and (x.strip() == '-' or x.strip() == '') else x)
                df_info[col] = pd.to_datetime(df_info[col], errors='coerce')
        
        # 경영악화 기업 제거 개월 수 정보 로드 (add_data.xlsx에서)
        remove_months_info = {}
        if add_data_path and os.path.exists(add_data_path):
            remove_months_info = read_remove_months_info(add_data_path)
            log_message(f"  add_data.xlsx에서 {len(remove_months_info)}개 기업의 제거 개월 수 정보를 로드했습니다.")
            
            # 특정 기업 정보 확인 (디버깅용)
            for company_id in [37, 38]:
                if company_id in remove_months_info:
                    log_message(f"  기업 {company_id}의 제거 개월 수: {remove_months_info[company_id]}")
                else:
                    log_message(f"  기업 {company_id}는 제거 개월 수 정보가 없습니다.")
        
        # 계약종결일 조정
        adjusted_companies = 0
        for idx, row in df_info.iterrows():
            company_id = row['No.']
            if pd.notna(company_id) and int(company_id) in remove_months_info and pd.notna(row['계약종결일']):
                months_to_remove = remove_months_info[int(company_id)]
                
                # 계약종결일 조정 (개월 수만큼 이전으로)
                original_term_date = row['계약종결일']
                adjusted_term_date = original_term_date - pd.DateOffset(months=months_to_remove)
                
                # 조정된 계약종결일로 업데이트
                df_info.at[idx, '계약종결일'] = adjusted_term_date
                adjusted_companies += 1
                log_message(f"  기업 {company_id}: 계약종결일 조정 {original_term_date.strftime('%Y-%m-%d')} → {adjusted_term_date.strftime('%Y-%m-%d')} ({months_to_remove}개월 이전)")
            elif pd.notna(company_id) and (company_id == 37 or company_id == 38):
                # 37번과 38번 회사는 특별 로깅
                if pd.notna(row['계약종결일']):
                    log_message(f"  기업 {company_id}: 계약종결일={row['계약종결일'].strftime('%Y-%m-%d')}이지만 제거 개월 수 정보가 없어 조정하지 않음")
                else:
                    log_message(f"  기업 {company_id}: 계약종결일이 없어 조정하지 않음")
        
        log_message(f"  총 {adjusted_companies}개 기업의 계약종결일 조정됨")
        
        # 37번과 38번 회사 조정 후 상태 확인 (디버깅용)
        if 37 in df_info['No.'].values:
            company_37 = df_info[df_info['No.'] == 37].iloc[0]
            log_message(f"  [조정후] 기업 37: 계약종결일={company_37['계약종결일'].strftime('%Y-%m-%d') if pd.notna(company_37['계약종결일']) else 'None'}")
        
        if 38 in df_info['No.'].values:
            company_38 = df_info[df_info['No.'] == 38].iloc[0]
            log_message(f"  [조정후] 기업 38: 계약종결일={company_38['계약종결일'].strftime('%Y-%m-%d') if pd.notna(company_38['계약종결일']) else 'None'}")
        
        # 회사 분류 (test/train 구분)
        company_meta = df_info.copy()
        company_meta['데이터셋'] = 'train'  # 기본값은 train
        
        # 확장된 테스트 데이터 사용 시 - cutoff_date 이전부터 거래중인 회사들을 both로 분류
        if use_extended_test_data:
            log_message("  확장된 테스트 데이터 모드 활성화")
            
            # cutoff_date 이전부터 거래중인 회사 찾기
            mask_existing_active = (
                (company_meta['당사투입일'] < cutoff_date) &  # cutoff_date 이전부터 거래
                (company_meta['계약종결사유'].str.contains('거래중', na=False))  # 현재 거래중
            )
            company_meta.loc[mask_existing_active, '데이터셋'] = 'both'
            log_message(f"  기존 거래중 회사 분류: {sum(mask_existing_active)}개를 both(train+test)로 분류")
        
        # 2024년 이후 투입 회사는 test로 분류 (test_end_date 고려)
        if test_end_date is not None:
            mask_recent = ((company_meta['당사투입일'] >= cutoff_date) & 
                          (company_meta['당사투입일'] <= test_end_date))
            log_message(f"  test_end_date({test_end_date.strftime('%Y-%m-%d')})를 기준으로 투입 회사 필터링 적용")
        else:
            mask_recent = company_meta['당사투입일'] >= cutoff_date
            
        company_meta.loc[mask_recent, '데이터셋'] = 'test'
        log_message(f"  2024년 이후 투입 회사 분류: {sum(mask_recent)}개를 test로 분류")
        
        # 2024년에 계약 종결된 경영악화 회사는 test로 분류 (조정된 계약종결일 기준)
        # test_end_date도 고려하여 필터링
        if test_end_date is not None:
            mask_fail = ((company_meta['계약종결일'] >= cutoff_date) &
                        (company_meta['계약종결일'] <= test_end_date) &
                     (company_meta['계약종결사유'].str.contains('경영악화|부도', na=False)))
            log_message(f"  test_end_date({test_end_date.strftime('%Y-%m-%d')})를 기준으로 필터링 적용")
        else:
            mask_fail = ((company_meta['계약종결일'] >= cutoff_date) &
                        (company_meta['계약종결사유'].str.contains('경영악화|부도', na=False)))
            
        company_meta.loc[mask_fail, '데이터셋'] = 'test'
        log_message(f"  2024년에 계약 종결된 경영악화 회사 분류: {sum(mask_fail)}개를 test로 분류")
        
        # 37번과 38번 회사 분류 결과 확인 (디버깅용)
        for company_id in [37, 38]:
            if company_id in company_meta['No.'].values:
                company_data = company_meta[company_meta['No.'] == company_id].iloc[0]
                cutoff_cond = pd.notna(company_data['계약종결일']) and company_data['계약종결일'] >= cutoff_date
                end_date_cond = True if test_end_date is None else (pd.notna(company_data['계약종결일']) and company_data['계약종결일'] <= test_end_date)
                reason_cond = pd.notna(company_data['계약종결사유']) and '경영악화' in str(company_data['계약종결사유'])
                
                log_message(f"  기업 {company_id} 분류 결과:")
                log_message(f"    - 계약종결일: {company_data['계약종결일'].strftime('%Y-%m-%d') if pd.notna(company_data['계약종결일']) else 'None'}")
                log_message(f"    - 계약종결사유: {company_data['계약종결사유'] if pd.notna(company_data['계약종결사유']) else 'None'}")
                log_message(f"    - 2024년 이후 종결: {cutoff_cond}")
                log_message(f"    - test_end_date 이전 종결: {end_date_cond}")
                log_message(f"    - 경영악화 사유: {reason_cond}")
                log_message(f"    - 최종 데이터셋: {company_data['데이터셋']}")
        
        # 회사 그룹 정보 추출
        company_groups = company_meta[['No.', '데이터셋']].copy()
        
        # 분류 결과 로깅
        train_companies = company_groups[company_groups['데이터셋'] == 'train']['No.'].tolist()
        test_companies = company_groups[company_groups['데이터셋'] == 'test']['No.'].tolist()
        both_companies = company_groups[company_groups['데이터셋'] == 'both']['No.'].tolist()
        
        log_message(f"  데이터 분할용 회사 분류 완료: 훈련 {len(train_companies)}개, 테스트 {len(test_companies)}개, 공통 {len(both_companies)}개")
        
        return company_groups, df_info
        
    except Exception as e:
        log_message(f"회사 분류 중 오류 발생: {str(e)}")
        import traceback
        log_message(traceback.format_exc())
        return None, None 