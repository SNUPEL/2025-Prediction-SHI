"""
Data 패키지 - 데이터 처리 관련 모든 기능을 제공
"""
# 분할된 모듈들에서 모든 함수 임포트
from .data_analysis import (
    check_eligible_sheets, 
    read_remove_months_info, 
    check_eligible_sheets_for_split
)
from .data_classification import (
    classify_companies, 
    get_company_groups,
    classify_companies_for_split
)
from .data_extraction import (
    get_cutoff_date_by_company,
    get_date_from_time_col,
    extract_combined_timeseries
)
from .data_preprocessing import (
    process_and_pad_data,
    save_preprocessing_step,
    normalize_data_by_company
)
from .data_split import (
    split_sheet_data,
    split_and_save_data
)
from .data_utils import (
    check_data_directories,
    save_results,
    prepare_data
)

# 하위 호환성을 위한 재export - 모든 함수들을 포함
__all__ = [
    # 분석 함수들 (data_analysis.py)
    'check_eligible_sheets', 
    'read_remove_months_info', 
    'check_eligible_sheets_for_split',
    
    # 분류 함수들 (data_classification.py)
    'classify_companies', 
    'get_company_groups',
    'classify_companies_for_split',
    
    # 추출 함수들 (data_extraction.py)
    'get_cutoff_date_by_company',
    'get_date_from_time_col',
    'extract_combined_timeseries',
    
    # 전처리 함수들 (data_preprocessing.py)
    'process_and_pad_data',
    'save_preprocessing_step',
    'normalize_data_by_company',
    
    # 분할 함수들 (data_split.py)
    'split_sheet_data',
    'split_and_save_data',
    
    # 유틸리티 함수들 (data_utils.py)
    'check_data_directories',
    'save_results', 
    'prepare_data'
] 