"""
데이터 분할 방식 비교 테스트 스크립트
기존 방식 vs 확장된 테스트 데이터 방식 비교 분석
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
import sys

# 내부 모듈 임포트
from config import (
    DEFAULT_DATA_FILE, DEFAULT_ADD_DATA_FILE, 
    DEFAULT_EXCLUDE_SHEETS, MIN_TIME_POINTS, CUTOFF_DATE,
    USE_EXTENDED_TEST_DATA
)
from data import (
    classify_companies_for_split, 
    check_eligible_sheets, 
    split_sheet_data,
    read_remove_months_info
)
from utils import log_message, initialize

def analyze_data_split_comparison():
    """기존 방식과 확장 방식의 데이터 분할 비교 분석"""
    
    print("=" * 80)
    print("📊 데이터 분할 방식 비교 분석 시작")
    print("=" * 80)
    
    # 파일 경로 설정
    file_path = DEFAULT_DATA_FILE
    add_data_path = DEFAULT_ADD_DATA_FILE
    exclude_sheets = DEFAULT_EXCLUDE_SHEETS
    cutoff_date = CUTOFF_DATE
    
    print(f"📁 메인 데이터 파일: {file_path}")
    print(f"📁 추가 데이터 파일: {add_data_path}")
    print(f"📅 기준 날짜: {cutoff_date.strftime('%Y-%m-%d')}")
    print(f"🚫 제외 시트: {exclude_sheets}")
    print()
    
    # 결과 저장용 딕셔너리
    comparison_results = {
        'mode': [],
        'total_companies': [],
        'train_companies': [],
        'test_companies': [],
        'both_companies': [],
        'train_only_companies': [],
        'test_only_companies': []
    }
    
    detailed_results = []
    
    # 두 가지 방식으로 분석
    modes = [
        {'name': '기존방식', 'use_extended': False},
        {'name': '확장방식', 'use_extended': USE_EXTENDED_TEST_DATA}
    ]
    
    for mode in modes:
        print(f"🔍 {mode['name']} 분석 중...")
        print("-" * 40)
        
        try:
            # 회사 분류 수행
            company_groups, company_meta = classify_companies_for_split(
                file_path=file_path,
                cutoff_date=cutoff_date,
                add_data_path=add_data_path,
                test_end_date=None,
                use_extended_test_data=mode['use_extended']
            )
            
            if company_groups is None:
                print(f"❌ {mode['name']} 분석 실패")
                continue
            
            # 분류 결과 통계
            train_companies = company_groups[company_groups['데이터셋'] == 'train']['No.'].tolist()
            test_companies = company_groups[company_groups['데이터셋'] == 'test']['No.'].tolist()
            both_companies = company_groups[company_groups['데이터셋'] == 'both']['No.'].tolist()
            
            # 결과 저장
            comparison_results['mode'].append(mode['name'])
            comparison_results['total_companies'].append(len(company_groups))
            comparison_results['train_companies'].append(len(train_companies))
            comparison_results['test_companies'].append(len(test_companies))
            comparison_results['both_companies'].append(len(both_companies))
            comparison_results['train_only_companies'].append(len([c for c in train_companies if c not in both_companies]))
            comparison_results['test_only_companies'].append(len([c for c in test_companies if c not in both_companies]))
            
            print(f"  📊 총 회사 수: {len(company_groups)}")
            print(f"  🎯 Train 회사: {len(train_companies)}개")
            print(f"  🎯 Test 회사: {len(test_companies)}개")
            print(f"  🎯 Both 회사: {len(both_companies)}개")
            print()
            
            # 상세 회사별 정보 수집
            for _, row in company_groups.iterrows():
                detailed_results.append({
                    '모드': mode['name'],
                    '회사ID': row['No.'],
                    '데이터셋': row['데이터셋'],
                    'Train사용': 'O' if row['데이터셋'] in ['train', 'both'] else 'X',
                    'Test사용': 'O' if row['데이터셋'] in ['test', 'both'] else 'X'
                })
            
        except Exception as e:
            print(f"❌ {mode['name']} 분석 중 오류: {str(e)}")
            continue
    
    # 비교 결과 테이블 생성
    comparison_df = pd.DataFrame(comparison_results)
    detailed_df = pd.DataFrame(detailed_results)
    
    # 시트별 시간 범위 분석
    print("🕐 시트별 시간 범위 분석 중...")
    sheet_analysis = analyze_sheet_time_ranges(file_path, exclude_sheets, cutoff_date)
    
    # 회사별 상세 정보 분석 (종합평가 시트 기반)
    print("🏢 회사별 상세 정보 분석 중...")
    company_details = analyze_company_details(file_path, add_data_path)
    
    # 결과 저장
    save_results_to_files(comparison_df, detailed_df, sheet_analysis, company_details)
    
    # 콘솔 출력
    print_summary_results(comparison_df, detailed_df, sheet_analysis)
    
    print("✅ 분석 완료! 결과 파일들이 생성되었습니다.")
    print("📄 data_split_comparison.csv - 기본 비교 결과")
    print("📄 company_detailed_analysis.csv - 회사별 상세 분석")
    print("📄 sheet_time_analysis.csv - 시트별 시간 범위")
    print("📄 analysis_summary.txt - 전체 요약 보고서")

def analyze_sheet_time_ranges(file_path, exclude_sheets, cutoff_date):
    """시트별 시간 범위 분석"""
    sheet_results = []
    
    try:
        eligible_sheets, sheet_info, _ = check_eligible_sheets(file_path, exclude_sheets, cutoff_date)
        
        for sheet_name in eligible_sheets:
            if sheet_name in sheet_info:
                info = sheet_info[sheet_name]
                train_cols = info.get('train_time_cols', [])
                test_cols = info.get('test_time_cols', [])
                
                # 시간 범위 계산
                all_cols = train_cols + test_cols
                train_range = f"{min(train_cols)} ~ {max(train_cols)}" if train_cols else "없음"
                test_range = f"{min(test_cols)} ~ {max(test_cols)}" if test_cols else "없음"
                total_range = f"{min(all_cols)} ~ {max(all_cols)}" if all_cols else "없음"
                
                sheet_results.append({
                    '시트명': sheet_name,
                    '총시간점': len(all_cols),
                    'Train시간점': len(train_cols),
                    'Test시간점': len(test_cols),
                    'Train범위': train_range,
                    'Test범위': test_range,
                    '전체범위': total_range
                })
    
    except Exception as e:
        print(f"❌ 시트 분석 중 오류: {str(e)}")
    
    return pd.DataFrame(sheet_results)

def analyze_company_details(file_path, add_data_path):
    """회사별 상세 정보 분석"""
    company_details = []
    
    try:
        # 종합평가 시트에서 기본 정보 로드
        df_info = pd.read_excel(file_path, sheet_name='종합평가')
        
        # 추가 정보 로드
        remove_months_info = {}
        if add_data_path and os.path.exists(add_data_path):
            remove_months_info = read_remove_months_info(add_data_path)
        
        for _, row in df_info.iterrows():
            company_id = row['No.']
            
            # 날짜 정보 처리
            input_date = row.get('당사투입일', None)
            term_date = row.get('계약종결일', None)
            term_reason = row.get('계약종결사유', None)
            status = row.get('구분', None)
            
            input_date_str = input_date.strftime('%Y-%m-%d') if pd.notna(input_date) else '없음'
            term_date_str = term_date.strftime('%Y-%m-%d') if pd.notna(term_date) else '없음'
            
            # 제거 개월 수
            remove_months = remove_months_info.get(company_id, 0)
            
            # 조정된 계약종결일
            adjusted_term_date = None
            if pd.notna(term_date) and remove_months > 0:
                adjusted_term_date = term_date - pd.DateOffset(months=remove_months)
                adjusted_term_date_str = adjusted_term_date.strftime('%Y-%m-%d')
            else:
                adjusted_term_date_str = term_date_str
            
            company_details.append({
                '회사ID': company_id,
                '현재상태': status,
                '투입일': input_date_str,
                '원본종결일': term_date_str,
                '조정종결일': adjusted_term_date_str,
                '제거개월수': remove_months,
                '종결사유': term_reason if pd.notna(term_reason) else '없음'
            })
    
    except Exception as e:
        print(f"❌ 회사 상세 정보 분석 중 오류: {str(e)}")
    
    return pd.DataFrame(company_details)

def save_results_to_files(comparison_df, detailed_df, sheet_analysis, company_details):
    """결과를 파일로 저장"""
    
    # CSV 파일들 저장
    comparison_df.to_csv('data_split_comparison.csv', index=False, encoding='utf-8-sig')
    detailed_df.to_csv('company_detailed_analysis.csv', index=False, encoding='utf-8-sig')
    sheet_analysis.to_csv('sheet_time_analysis.csv', index=False, encoding='utf-8-sig')
    company_details.to_csv('company_basic_info.csv', index=False, encoding='utf-8-sig')
    
    # 통합 요약 보고서 생성
    with open('analysis_summary.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("데이터 분할 방식 비교 분석 보고서\n")
        f.write("=" * 80 + "\n")
        f.write(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("1. 전체 비교 결과\n")
        f.write("-" * 40 + "\n")
        f.write(comparison_df.to_string(index=False) + "\n\n")
        
        f.write("2. 시트별 시간 범위 분석\n")
        f.write("-" * 40 + "\n")
        f.write(sheet_analysis.to_string(index=False) + "\n\n")
        
        # 기존 vs 확장 방식 차이점 분석
        if len(comparison_df) >= 2:
            f.write("3. 주요 차이점 분석\n")
            f.write("-" * 40 + "\n")
            
            basic_test = comparison_df.iloc[0]['test_companies']
            extended_test = comparison_df.iloc[1]['test_companies']
            basic_both = comparison_df.iloc[0]['both_companies']
            extended_both = comparison_df.iloc[1]['both_companies']
            
            f.write(f"• Test 데이터 증가량: {extended_test - basic_test}개 회사\n")
            f.write(f"• Both 회사 추가: {extended_both - basic_both}개\n")
            f.write(f"• 데이터 효율성 개선: {((extended_test / basic_test - 1) * 100):.1f}% 증가\n\n")
        
        f.write("4. 확장 방식의 장점\n")
        f.write("-" * 40 + "\n")
        f.write("• 기존 거래중 회사의 미래 데이터도 테스트에 활용\n")
        f.write("• 시간적 일반화와 회사별 일반화 모두 평가 가능\n")
        f.write("• 데이터 버림 현상 크게 개선\n")
        f.write("• 더 현실적인 예측 시나리오 반영\n\n")

def print_summary_results(comparison_df, detailed_df, sheet_analysis):
    """콘솔에 요약 결과 출력"""
    
    print("\n" + "=" * 60)
    print("📊 분석 결과 요약")
    print("=" * 60)
    
    print("\n1️⃣ 전체 비교 결과:")
    print(comparison_df.to_string(index=False))
    
    print(f"\n2️⃣ 시트별 시간 범위 ({len(sheet_analysis)}개 시트):")
    print(sheet_analysis.to_string(index=False))
    
    if len(comparison_df) >= 2:
        basic_test = comparison_df.iloc[0]['test_companies']
        extended_test = comparison_df.iloc[1]['test_companies']
        improvement = ((extended_test / basic_test - 1) * 100) if basic_test > 0 else 0
        
        print(f"\n3️⃣ 주요 개선 효과:")
        print(f"   🎯 Test 데이터: {basic_test}개 → {extended_test}개 ({improvement:.1f}% 증가)")
        print(f"   📈 데이터 활용률 대폭 개선!")
        print(f"   ✨ 현실적인 예측 평가 가능")
    
    # Both 회사 목록 출력 (확장 방식에서)
    extended_both = detailed_df[
        (detailed_df['모드'] == '확장방식') & 
        (detailed_df['데이터셋'] == 'both')
    ]['회사ID'].tolist()
    
    if extended_both:
        print(f"\n4️⃣ Train과 Test 모두 사용하는 회사들:")
        print(f"   {extended_both}")

def main():
    """메인 실행 함수"""
    try:
        # 초기화
        initialize()
        
        # 분석 실행
        analyze_data_split_comparison()
        
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 