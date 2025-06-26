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

    
# 분석 가능한 시트 확인 함수
def check_eligible_sheets(file_path, exclude_sheets=None, cutoff_date=CUTOFF_DATE):
    """
    분석 가능한 시트 확인 및 구조 파악 
    - 경영악화/거래중 상태를 포함한 시트 찾기
    - 날짜 열을 시간 기준으로 분류 (cutoff_date 기준)
    """
    log_message(f"\n파일 '{file_path}' 시트 분석 중...")
    
    # 모든 시트 확인
    excel = pd.ExcelFile(file_path)
    sheet_names = excel.sheet_names
    
    # 제외할 시트 목록 준비
    exclude_sheet_names = [] if exclude_sheets is None else exclude_sheets
    
    if exclude_sheet_names:
        log_message(f"  제외할 시트: {', '.join(exclude_sheet_names)}")
    
    # 분석 가능한 시트 목록
    eligible_sheets = []
    sheet_info = {}
    
    # 각 시트별로 분석
    for sheet_name in sheet_names:
        # 제외할 시트 목록에 있으면 건너뜀
        if sheet_name in exclude_sheet_names:
            log_message(f"  ✗ {sheet_name}: 제외 시트로 건너뜀")
            continue
            
        try:
            # 시트 데이터 로드
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # 상태 열 찾기 
            status_col = None
            
            for col in df.columns:
                # 열 이름으로 상태 열 후보 확인
                if isinstance(col, str) and ('상태' in col or '구분' in col):
                    # 값 확인
                    values = df[col].astype(str).str.lower().tolist()
                    if any('경영악화' in v for v in values) and any('거래중' in v for v in values):
                        status_col = col
                        break
            
            # 값으로 상태 열 찾기 (열 이름에서 못 찾은 경우)
            if status_col is None:
                for col in df.columns:
                    # 값 확인
                    values = df[col].astype(str).str.lower().tolist()
                    if any('경영악화' in v for v in values) and any('거래중' in v for v in values):
                        status_col = col
                        break
            
            # 상태 열이 없으면 다음 시트로
            if status_col is None:
                log_message(f"  ✗ {sheet_name}: 상태 열을 찾을 수 없음")
                continue
            
            # 시계열 열 찾기
            time_cols = []
            
            # 시트별 날짜 형식 처리
            if sheet_name == '4대보험 체납':
                # 'yy.m월' 형식 (예: '20.3월')
                for col in df.columns:
                    if isinstance(col, str) and '월' in col:
                        parts = col.replace('월', '').split('.')
                        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                            time_cols.append(col)
            else:
                # 일반 날짜 형식 (datetime 또는 문자열)
                for col in df.columns:
                    # datetime 객체인 경우
                    if isinstance(col, datetime):
                        time_cols.append(col)
                    # 문자열인 경우 패턴 확인
                    elif isinstance(col, str):
                        # YYYY-MM 또는 YYYY-MM-DD 패턴
                        if re.match(r'^\d{4}-\d{1,2}(?:-\d{1,2})?$', col):
                            time_cols.append(col)
                        # YYYY 패턴
                        elif re.match(r'^\d{4}$', col):
                            time_cols.append(col)
            
            # 시계열 열이 부족하면 다음 시트로
            if len(time_cols) < 6:
                log_message(f"  ✗ {sheet_name}: 시계열 열이 부족함 ({len(time_cols)} < 6)")
                continue
            
            # 기업 ID 열 찾기
            id_col = None
            for col in df.columns:
                if isinstance(col, str) and (col.strip() == 'No.' or 'ID' in col.upper()):
                    id_col = col
                    break
            
            # ID 열이 없으면 첫 번째 열 사용
            if id_col is None:
                id_col = df.columns[0]
                log_message(f"  ! {sheet_name}: ID 열을 찾을 수 없어 첫 번째 열({id_col})을 사용합니다.")
            
            # 경영악화와 거래중 기업 수 확인
            df_status = df[status_col].astype(str).str.lower()
            deteriorated = df[df_status.str.contains('경영악화')]
            active = df[df_status.str.contains('거래중')]
            
            # 분석 가능 여부 판단
            if len(deteriorated) > 0 and len(active) > 0:
                eligible_sheets.append(sheet_name)
                sheet_info[sheet_name] = {
                    'time_cols': time_cols,
                    'status_col': status_col,
                    'id_col': id_col
                }
                log_message(f"  ✓ {sheet_name}: 분석 가능 (경영악화={len(deteriorated)}, 거래중={len(active)})")
            else:
                log_message(f"  ✗ {sheet_name}: 경영악화/거래중 기업 부족")
        
        except Exception as e:
            log_message(f"  ✗ {sheet_name}: 처리 중 오류 발생 ({str(e)})")
    
    # 시트 정보에 시간 정보 추가
    for sheet_name in eligible_sheets:
        if sheet_name in sheet_info:
            # 시간 열을 cutoff_date 기준으로 분류
            time_cols = sheet_info[sheet_name]['time_cols']
            train_time_cols = []
            test_time_cols = []
            
            for col in time_cols:
                try:
                    # '4대보험 체납' 시트의 특수 날짜 형식 처리
                    if sheet_name == '4대보험 체납' and isinstance(col, str) and '월' in col:
                        parts = col.replace('월', '').split('.')
                        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                            year = int(parts[0]) + 2000 if int(parts[0]) < 100 else int(parts[0])
                            month = int(parts[1])
                            date_obj = pd.Timestamp(year=year, month=month, day=1)
                            
                            if date_obj < cutoff_date:
                                train_time_cols.append(col)
                            else:
                                test_time_cols.append(col)
                            continue
                
                    # 일반 날짜 형식 처리
                    date_obj = None
                    
                    # datetime 객체인 경우
                    if isinstance(col, datetime):
                        date_obj = pd.Timestamp(col)
                    # 문자열인 경우 패턴에 따라 처리
                    elif isinstance(col, str):
                        # YYYY-MM 또는 YYYY-MM-DD 패턴
                        if re.match(r'^\d{4}-\d{1,2}(?:-\d{1,2})?$', col):
                            date_obj = pd.to_datetime(col)
                        # YYYY 패턴
                        elif re.match(r'^\d{4}$', col):
                            date_obj = pd.to_datetime(f"{col}-01-01")
                    
                    if date_obj is not None:
                        if date_obj < cutoff_date:
                            train_time_cols.append(col)
                        else:
                            test_time_cols.append(col)
                except Exception as e:
                    # 날짜 해석 실패 시 학습 데이터로 처리
                    log_message(f"  ! 날짜 형식 해석 오류 ({col}): {str(e)}")
                    train_time_cols.append(col)
            
            # 시트 정보 업데이트
            sheet_info[sheet_name]['train_time_cols'] = train_time_cols
            sheet_info[sheet_name]['test_time_cols'] = test_time_cols
            
            log_message(f"  시트 '{sheet_name}' 시간 분할: 학습={len(train_time_cols)}개, 테스트={len(test_time_cols)}개 사용")
    
    return eligible_sheets, sheet_info, sheet_names

# 제거 개월 수 정보 로드 함수
def read_remove_months_info(add_data_path):
    """
    add_data.xlsx에서 기업별 제거해야 할 개월 수 정보를 읽어옴
    """
    try:
        # 추가정보 시트 로드
        log_message(f"제거 개월 수 정보 로드 중: {add_data_path}")
        
        # 엑셀 파일 존재 여부 확인
        if not os.path.exists(add_data_path):
            log_message(f"오류: 파일이 존재하지 않습니다: {add_data_path}")
            return {}
        
        # 시트 목록 확인
        try:
            xls = pd.ExcelFile(add_data_path)
            sheet_names = xls.sheet_names
            log_message(f"파일에 있는 시트 목록: {sheet_names}")
            
            if '추가정보' not in sheet_names:
                # 혹시 이름에 공백이 있는지 확인
                possible_sheet = [s for s in sheet_names if '추가' in s]
                if possible_sheet:
                    log_message(f"'추가정보' 시트를 찾을 수 없어 대체 시트를 사용합니다: {possible_sheet[0]}")
                    sheet_name = possible_sheet[0]
                else:
                    log_message(f"오류: '추가정보' 시트를 찾을 수 없습니다.")
                    return {}
            else:
                sheet_name = '추가정보'
        except Exception as e:
            log_message(f"시트 목록 확인 중 오류: {str(e)}")
            # 시트 목록을 확인할 수 없으면 기본 이름 사용
            sheet_name = '추가정보'
        
        # Data.xlsx와 동일한 방식으로 처리 - 원시 데이터 로드 후 직접 열 처리
        df_raw = pd.read_excel(add_data_path, sheet_name=sheet_name, header=None)
        log_message(f"원시 데이터 크기: {df_raw.shape}")
        
        # 첫 번째 행이 헤더인지 확인
        if '협력사별 데이터 제거 필요 개월(철수일 기준)' in str(df_raw.iloc[0]) or '철수일 기준' in str(df_raw.iloc[0]):
            # 첫 번째 행을 열 이름으로 사용
            log_message("첫 번째 행을 열 이름으로 사용")
            column_names = df_raw.iloc[0].tolist()
            df_info = df_raw.iloc[1:].copy()
            df_info.columns = column_names
        else:
            # 직접 열 이름 지정
            log_message("직접 열 이름 지정")
            column_names = [
                'No.', '구분', '대직종', '직종', '당사투입일', '당사철수일', 
                '협력사구분', '철수사유', 'Unnamed', '철수 당시/현 나이(만)', 
                '협력사별 데이터 제거 필요 개월(철수일 기준)', '경영악화 경/중 구분', '비고'
            ]
            
            # 실제 열 수에 맞춰 조정
            if len(column_names) > df_raw.shape[1]:
                column_names = column_names[:df_raw.shape[1]]
            elif len(column_names) < df_raw.shape[1]:
                for i in range(len(column_names), df_raw.shape[1]):
                    column_names.append(f'Column_{i}')
            
            df_info = df_raw.copy()
            df_info.columns = column_names
        
        log_message(f"열 이름: {df_info.columns.tolist()}")
        log_message(f"시트 '{sheet_name}'을 로드했습니다. 크기: {df_info.shape}")
        
        # 필요한 열 이름 찾기
        id_col = None
        months_col = None
        
        # No. 열 찾기
        if 'No.' in df_info.columns:
            id_col = 'No.'
        else:
            # 유사한 이름 찾기
            for col in df_info.columns:
                if isinstance(col, str) and ('no' in col.lower() or '번호' in str(col).lower()):
                    id_col = col
                    log_message(f"'No.' 대신 '{id_col}'을 사용합니다.")
                    break
        
        # 개월 수 열 찾기
        target_col_name = '협력사별 데이터 제거 필요 개월(철수일 기준)'
        if target_col_name in df_info.columns:
            months_col = target_col_name
        else:
            # 유사한 이름 찾기
            for col in df_info.columns:
                col_str = str(col).lower()
                if ('제거' in col_str and '개월' in col_str) or ('철수' in col_str and '개월' in col_str):
                    months_col = col
                    log_message(f"'{target_col_name}' 대신 '{months_col}'을 사용합니다.")
                    break
            
            # 그래도 못 찾으면 10번째 열 사용해보기
            if months_col is None and len(df_info.columns) > 10:
                months_col = df_info.columns[10]
                log_message(f"개월 수 열을 찾을 수 없어 10번째 열 '{months_col}'을 사용합니다.")
        
        if id_col is None:
            log_message("ID 열을 찾을 수 없어 첫 번째 열을 사용합니다.")
            id_col = df_info.columns[0]
            
        if months_col is None:
            log_message("오류: 개월 수 열을 찾을 수 없습니다.")
            # 열 데이터 확인
            for i, col in enumerate(df_info.columns):
                log_message(f"열 {i}: {col}")
                if i < 5:  # 처음 5개 행만 표시
                    log_message(f"  샘플 데이터: {df_info[col].head().tolist()}")
            return {}
        
        log_message(f"ID 열: {id_col}, 개월 수 열: {months_col}")
        
        # 기업별 제거 개월 수 정보 추출
        remove_months_info = {}
        for idx, row in df_info.iterrows():
            try:
                company_id = row[id_col]
                months_to_remove = row[months_col]
            
                if pd.notna(company_id) and pd.notna(months_to_remove):
                    try:
                        # 문자열인 경우 변환 전 처리
                        if isinstance(company_id, str):
                            company_id = company_id.strip()
                        if isinstance(months_to_remove, str):
                            months_to_remove = months_to_remove.strip()
                        
                        company_id_int = int(float(company_id))
                        months_to_remove_int = int(float(months_to_remove))
                        
                        # 유효한 값만 추가
                        if months_to_remove_int > 0:
                            remove_months_info[company_id_int] = months_to_remove_int
                            log_message(f"  기업 {company_id_int}: {months_to_remove_int}개월 제거 적용")
                    except ValueError:
                        log_message(f"  행 {idx}: 숫자 변환 실패 (ID={company_id}, 개월={months_to_remove})")
            except Exception as e:
                log_message(f"  행 {idx} 처리 중 오류: {str(e)}")
        
        log_message(f"제거 개월 수 정보 로드 완료: {len(remove_months_info)}개 기업")
        
        # 특정 기업 확인 (디버깅용)
        if 37 in remove_months_info:
            log_message(f"  기업 37의 제거 개월 수: {remove_months_info[37]}")
        if 38 in remove_months_info:
            log_message(f"  기업 38의 제거 개월 수: {remove_months_info[38]}")
        
        return remove_months_info
    except Exception as e:
        log_message(f"제거 개월 수 정보 로드 오류: {str(e)}")
        import traceback
        log_message(traceback.format_exc())
        return {}

# 엑셀 파일의 시트 분할 관련 함수
def check_eligible_sheets_for_split(file_path, exclude_sheets=None, cutoff_date=CUTOFF_DATE):
    """분할에 사용할 시트 확인"""
    log_message("엑셀 파일 시트 확인 중...")
    
    if exclude_sheets is None:
        exclude_sheets = []
    
    try:
        # 엑셀 파일의 모든 시트 이름 가져오기
        all_sheets = pd.ExcelFile(file_path).sheet_names
            
        # 제외할 시트 필터링
        eligible_sheets = [sheet for sheet in all_sheets if sheet not in exclude_sheets]
        
        # 시트 정보 수집
        sheet_info = {}
        sheet_names = []
        
        for sheet in eligible_sheets:
            try:
                # 시트의 첫 행만 읽어 구조 확인
                df = pd.read_excel(file_path, sheet_name=sheet, nrows=1)
                
                # 기업 ID 열과 날짜 열 확인
                if "No." in df.columns:
                    date_cols = [col for col in df.columns if isinstance(col, datetime) or 
                                (isinstance(col, str) and col.startswith('20'))]
                    
                    if date_cols:
                        sheet_info[sheet] = {
                            'id_col': 'No.',
                            'date_cols': date_cols,
                            'start_date': min(date_cols),
                            'end_date': max(date_cols)
                        }
                        sheet_names.append(sheet)
                        log_message(f"  시트 '{sheet}' 추가됨: 날짜 범위 {min(date_cols)} ~ {max(date_cols)}")
            except Exception as e:
                log_message(f"  시트 '{sheet}' 구조 확인 중 오류: {str(e)}")
        
        return eligible_sheets, sheet_info, sheet_names
        
    except Exception as e:
        log_message(f"엑셀 파일 시트 확인 중 오류: {str(e)}")
        return [], {}, [] 