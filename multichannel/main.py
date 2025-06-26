"""
메인 실행 모듈
전체 실행 흐름을 제어하고 모든 모듈들을 통합
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# 내부 모듈 임포트
from config import (
    OUTPUT_DIR, DEFAULT_DATA_FILE, DEFAULT_ADD_DATA_FILE, 
    DEFAULT_EXCLUDE_SHEETS, MIN_TIME_POINTS, MIN_TIME_POINTS_TEST,
    DEFAULT_MODEL_PARAMS, CUTOFF_DATE, TEST_END_DATE, REGENERATE_SPLIT_DATA,
    SPLIT_OUTPUT_DIR, USE_MASKING, OVERSAMPLING_METHOD, USE_OSS,
    USE_EXTENDED_TEST_DATA
)
from utils import ensure_dir, log_message, initialize
from data import prepare_data, save_results, split_and_save_data
from model import train_and_evaluate_model
from visualizer import (
    analyze_feature_importance, 
    visualize_predictions,
    visualize_timeseries
)

def parse_arguments():
    """
    명령줄 인자 파싱
    """
    parser = argparse.ArgumentParser(description='기업 경영악화 예측 분석')
    
    # 파일 경로 인자
    parser.add_argument('--data_file', type=str, default=DEFAULT_DATA_FILE,
                        help='메인 데이터 파일 경로 (기본값: %(default)s)')
    parser.add_argument('--add_data_file', type=str, default=DEFAULT_ADD_DATA_FILE,
                        help='추가 데이터 파일 경로 (기본값: %(default)s)')
    
    # 데이터 처리 인자
    parser.add_argument('--min_time_points', type=int, default=MIN_TIME_POINTS,
                        help='훈련 데이터 최소 시간점 개수 (기본값: %(default)s)')
    parser.add_argument('--min_test_time_points', type=int, default=MIN_TIME_POINTS_TEST,
                        help='테스트 데이터 최소 시간점 개수 (기본값: %(default)s)')
    parser.add_argument('--exclude_sheets', type=str, nargs='*', default=DEFAULT_EXCLUDE_SHEETS,
                        help='제외할 시트 이름 목록')
    
    # 마스킹 옵션
    parser.add_argument('--use_masking', action='store_true', default=USE_MASKING,
                        help='패딩된 데이터 마스킹 사용 여부 (기본값: %(default)s)')
    parser.add_argument('--no_masking', action='store_false', dest='use_masking',
                        help='패딩된 데이터 마스킹 비활성화')
    
    # 오버샘플링 방법 옵션
    parser.add_argument('--oversampling', type=str, default=OVERSAMPLING_METHOD, choices=['SMOTE', 'TSSMOTE', 'NONE'],
                        help='오버샘플링 방법 선택 (SMOTE, TSSMOTE, NONE) (기본값: %(default)s)')
    
    # 언더샘플링 옵션 (Tomek Links)
    parser.add_argument('--use_oss', action='store_true', default=USE_OSS,
                        help='Tomek Links 언더샘플링 사용 여부 (기본값: %(default)s)')
    parser.add_argument('--no_oss', action='store_false', dest='use_oss',
                        help='Tomek Links 언더샘플링 비활성화')
    
    # 날짜 및 분할 설정
    parser.add_argument('--cutoff_date', type=str, default=CUTOFF_DATE.strftime('%Y-%m-%d'),
                        help='훈련/테스트 데이터 분리 기준 날짜 (YYYY-MM-DD 형식) (기본값: %(default)s)')
    parser.add_argument(
        '--test_end_date',
        type=str,
        default=(None if TEST_END_DATE is None else TEST_END_DATE.strftime('%Y-%m-%d')),
        help=(
            '테스트 데이터 종료 날짜 (YYYY-MM-DD 형식). '
            '지정하지 않으면 엑셀의 마지막 시점까지 사용 (기본값: None 또는 config.TEST_END_DATE)'
        )
    )
    parser.add_argument('--regenerate_split', action='store_true', default=REGENERATE_SPLIT_DATA,
                        help='데이터 분할 재생성 여부 (기본값: %(default)s)')
    parser.add_argument('--split_output_dir', type=str, default=SPLIT_OUTPUT_DIR,
                        help='분할 데이터 저장 경로 (기본값: %(default)s)')
    
    # 모델 하이퍼파라미터
    parser.add_argument('--hidden_size', type=int, default=DEFAULT_MODEL_PARAMS['hidden_size'],
                        help='LSTM 은닉층 크기 (기본값: %(default)s)')
    parser.add_argument('--dropout', type=float, default=DEFAULT_MODEL_PARAMS['dropout'],
                        help='드롭아웃 비율 (기본값: %(default)s)')
    parser.add_argument('--weight_decay', type=float, default=DEFAULT_MODEL_PARAMS['weight_decay'],
                        help='가중치 감쇠 (기본값: %(default)s)')
    
    # 결과 옵션
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='결과 저장 디렉토리 (기본값: %(default)s)')
    parser.add_argument('--model_name', type=str, default="MultiChannel_CNNLSTM",
                        help='모델 이름 (기본값: %(default)s)')
    
    # 실행 옵션
    parser.add_argument('--only_split', action='store_true',
                        help='데이터 분할만 수행하고 종료')
    parser.add_argument('--no_feature_importance', action='store_true',
                        help='특성 중요도 분석 건너뛰기')
    parser.add_argument('--no_timeseries_viz', action='store_true',
                        help='시계열 데이터 시각화 건너뛰기')

    # 확장된 테스트 데이터 옵션
    parser.add_argument('--use_extended_test', action='store_true', default=USE_EXTENDED_TEST_DATA,
                        help='확장된 테스트 데이터 사용 여부 (기존 거래중 회사의 cutoff_date 이후 데이터도 test에 포함) (기본값: %(default)s)')
    parser.add_argument('--no_extended_test', action='store_false', dest='use_extended_test',
                        help='확장된 테스트 데이터 비활성화')
        
    return parser.parse_args()

def run_analysis(args):
    """
    전체 분석 실행 함수
    """
    log_message("\n==== 기업 경영악화 예측 LSTM 분석 시작 ====")
    log_message(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 명령줄 인자에서 받은 설정 적용
    import config
    config.USE_MASKING = args.use_masking
    config.OVERSAMPLING_METHOD = args.oversampling
    config.USE_OSS = args.use_oss
    
    # 입력 파라미터 로깅
    log_message("\n=== 입력 파라미터 ===")
    log_message(f"데이터 파일: {args.data_file}")
    log_message(f"추가 데이터 파일: {args.add_data_file}")
    log_message(f"기준 날짜: {args.cutoff_date}")
    if args.test_end_date:
        log_message(f"테스트 종료 날짜: {args.test_end_date}")
    log_message(f"데이터 분할 재생성: {args.regenerate_split}")
    log_message(f"훈련 데이터 최소 시간점 개수: {args.min_time_points}")
    log_message(f"테스트 데이터 최소 시간점 개수: {args.min_test_time_points}")
    log_message(f"제외 시트: {args.exclude_sheets}")
    log_message(f"패딩 마스킹 사용: {args.use_masking}")
    log_message(f"오버샘플링 방법: {args.oversampling}")
    log_message(f"언더샘플링 (Tomek Links) 사용: {args.use_oss}")
    log_message(f"결과 저장 경로: {args.output_dir}")
    
    # 날짜 문자열을 datetime 객체로 변환
    try:
        cutoff_date = pd.to_datetime(args.cutoff_date)
        test_end_date = pd.to_datetime(args.test_end_date) if args.test_end_date else None
    except:
        log_message(f"오류: 잘못된 날짜 형식 '{args.cutoff_date}' 또는 '{args.test_end_date}'. 'YYYY-MM-DD' 형식을 사용해주세요.")
        return None
    
    # 데이터 분할만 수행하는 경우
    if args.only_split:
        log_message("\n=== 데이터 분할만 수행 ===")
        split_result = split_and_save_data(
            file_path=args.data_file,
            output_dir=args.split_output_dir,
            exclude_sheets=args.exclude_sheets,
            cutoff_date=cutoff_date,
            test_end_date=test_end_date,
            add_data_path=args.add_data_file,
            use_extended_test_data=args.use_extended_test
        )
        
        if split_result is not None:
            log_message("데이터 분할이 성공적으로 완료되었습니다.")
        else:
            log_message("오류: 데이터 분할에 실패했습니다.")
        
        return {'split_result': split_result}
    
    # 1. 데이터 준비
    log_message("\n=== 데이터 준비 단계 ===")
    data_result = prepare_data(
        file_path=args.data_file, 
        exclude_sheets=args.exclude_sheets,
        min_time_points=args.min_time_points, 
        min_test_time_points=args.min_test_time_points,
        add_data_path=args.add_data_file,
        regenerate_split=args.regenerate_split,
        cutoff_date=cutoff_date,
        split_output_dir=args.split_output_dir,
        test_end_date=test_end_date,
        use_extended_test_data=args.use_extended_test,        
    )
    
    # 데이터 준비 결과 확인
    if data_result is None:
        log_message("오류: 유효한 데이터를 준비할 수 없습니다. 분석을 중단합니다.")
        return None
    
    # 2. 모델 학습 및 평가
    log_message("\n=== 모델 학습 및 평가 단계 ===")
    
    # 모델 하이퍼파라미터 설정
    model_params = {
        'hidden_size': args.hidden_size,
        'dropout': args.dropout,
        'weight_decay': args.weight_decay
    }
    
    # 모델 학습 및 평가
    model, predictions, metrics = train_and_evaluate_model(
        data_result['X_train'],
        data_result['y_train'],
        data_result['X_val'],
        data_result['y_val'],
        data_result['X_test'],
        data_result['y_test'],
        data_result['test_ids'],
        data_result['used_sheet_names'],
        masks_train=data_result.get('masks_train'),
        masks_val=data_result.get('masks_val'),
        masks_test=data_result.get('masks_test'),
        params=model_params,
        output_dir=args.output_dir,
        model_name=args.model_name
    )
    
    # 모델 학습 실패 확인
    if model is None:
        log_message("오류: 모델 학습에 실패했습니다. 분석을 중단합니다.")
        return None
    
    # 3. 결과 시각화
    log_message("\n=== 결과 시각화 단계 ===")
    
    # 예측 결과 시각화
    if len(predictions) > 0:
        predictions_df = pd.DataFrame(predictions)
        visualize_predictions(
            predictions_df, 
            args.output_dir, 
            model_name=args.model_name
        )
    
    # 4. 특성 중요도 분석 (선택적)
    if not args.no_feature_importance:
        log_message("\n=== 특성 중요도 분석 단계 ===")
        importance_results = analyze_feature_importance(
            model, 
            data_result['X_train'], 
            data_result['y_train'],
            data_result['used_sheet_names'],
            data_result['train_ids'],
            args.output_dir
        )
    
    # 5. 시계열 데이터 시각화 (선택적)
    if not args.no_timeseries_viz:
        log_message("\n=== 시계열 데이터 시각화 단계 ===")
        visualize_timeseries(
            data_result['X_train'], 
            data_result['y_train'],
            data_result['train_ids'],
            data_result['used_sheet_names'],
            args.output_dir,
            max_companies=3,
            max_sheets=3
        )
    
    # 6. 결과 요약
    log_message("\n=== 분석 결과 요약 ===")
    if metrics:
        log_message(f"정확도: {metrics.get('accuracy', 0):.4f}")
        log_message(f"정밀도: {metrics.get('precision', 0):.4f}")
        log_message(f"재현율: {metrics.get('recall', 0):.4f}")
        log_message(f"F1 점수: {metrics.get('f1', 0):.4f}")
        log_message(f"AUC: {metrics.get('auc', 0):.4f}")
    
    log_message(f"\n==== 분석 완료 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ====")
    
    return {
        'model': model,
        'metrics': metrics,
        'predictions': predictions
    }

def main():
    """
    메인 함수
    """
    # 명령줄 인자 파싱
    args = parse_arguments()
    
    # 결과 디렉토리 생성 및 초기화
    ensure_dir(args.output_dir)
    font_path = initialize()
    
    # 분석 실행
    results = run_analysis(args)
    
    return results

if __name__ == "__main__":
    # 스크립트 직접 실행 시 메인 함수 호출
    results = main() 