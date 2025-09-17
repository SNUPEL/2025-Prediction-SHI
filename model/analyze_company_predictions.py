import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_and_analyze_data(file_path):
    """엑셀 파일을 읽고 데이터 구조를 분석"""
    df = pd.read_excel(file_path)
    print("데이터 구조:")
    print(df.head(10))
    print(f"\n데이터 형태: {df.shape}")
    print(f"\n컬럼: {df.columns.tolist()}")
    return df

def extract_company_and_date(df):
    """company_id_date에서 회사 ID와 날짜를 분리"""
    df['company_id'] = df['company_id_date'].str.split('_').str[0]
    df['date'] = pd.to_datetime(df['company_id_date'].str.split('_').str[1])
    return df

def create_prediction_matrix():
    """unnamed.png 이미지의 예측-라벨 매트릭스 생성"""
    # 이미지의 표를 그대로 구현
    matrix = {
        'TTT': {'TTT': 'y', 'FTT': 'y', 'FFT': 'n', 'FFF': 'n'},
        'TTF': {'TTT': 'y', 'FTT': 'n', 'FFT': 'n', 'FFF': 'n'},
        'TFT': {'TTT': 'y', 'FTT': 'n', 'FFT': 'y(철수)', 'FFF': 'n'},
        'TFF': {'TTT': 'n', 'FTT': 'n', 'FFT': 'n', 'FFF': 'y'},
        'FTT': {'TTT': 'y', 'FTT': 'y', 'FFT': 'y(철수)', 'FFF': 'n'},
        'FTF': {'TTT': 'n', 'FTT': 'n', 'FFT': 'n', 'FFF': 'y'},
        'FFT': {'TTT': 'n', 'FTT': 'y(철수)', 'FFT': 'y', 'FFF': 'y'},
        'FFF': {'TTT': 'n', 'FTT': 'n', 'FFT': 'y', 'FFF': 'y'}
    }
    return matrix

def get_company_prediction_pattern(group):
    """회사별 예측 패턴 생성 (TTT, TTF 등)"""
    predictions = group['predicted_label'].tolist()
    
    # 1=T(경영악화), 0=F(거래중)으로 변환
    pattern = ''.join(['T' if pred == 1 else 'F' for pred in predictions])
    return pattern

def get_company_true_pattern(group):
    """회사별 실제 라벨 패턴 생성 (TTT, TTF 등)"""
    true_labels = group['true_label'].tolist()
    
    # True=1, False=0으로 변환
    pattern = ''.join(['T' if label else 'F' for label in true_labels])
    return pattern

def get_company_final_prediction_case1(group):
    """Case 1: y로만 표시된 항목에 대해 예측 성공"""
    matrix = create_prediction_matrix()
    
    # 회사별 예측 패턴과 실제 패턴 생성
    pred_pattern = get_company_prediction_pattern(group)
    true_pattern = get_company_true_pattern(group)
    
    # 매트릭스에서 결과 확인
    if pred_pattern in matrix and true_pattern in matrix[pred_pattern]:
        result = matrix[pred_pattern][true_pattern]
        # Case 1: 'y'만 성공으로 간주
        success = result == 'y'
        print(f"    Case 1: {pred_pattern} vs {true_pattern} = {result} → {success}")
        return success
    else:
        # 패턴이 없으면 False (실패)
        print(f"    Case 1: {pred_pattern} vs {true_pattern} = 패턴 없음 → False")
        return False

def get_company_final_prediction_case2(group):
    """Case 2: Case1 + y(철수) 항목도 예측 성공으로 판단"""
    matrix = create_prediction_matrix()
    
    # 회사별 예측 패턴과 실제 패턴 생성
    pred_pattern = get_company_prediction_pattern(group)
    true_pattern = get_company_true_pattern(group)
    
    # 매트릭스에서 결과 확인
    if pred_pattern in matrix and true_pattern in matrix[pred_pattern]:
        result = matrix[pred_pattern][true_pattern]
        # Case 2: 'y'와 'y(철수)' 모두 성공으로 간주
        success = result in ['y', 'y(철수)']
        print(f"    Case 2: {pred_pattern} vs {true_pattern} = {result} → {success}")
        return success
    else:
        # 패턴이 없으면 False (실패)
        print(f"    Case 2: {pred_pattern} vs {true_pattern} = 패턴 없음 → False")
        return False

def get_company_true_label(group):
    """회사별 실제 라벨 패턴 (TTT, FTT, FFT, FFF)"""
    true_labels = group['true_label'].tolist()
    pattern = ''.join(['T' if label else 'F' for label in true_labels])
    return pattern

def create_company_level_predictions_case1(df):
    """Case 1: 엄격한 기준으로 회사 레벨의 예측 결과 생성"""
    company_results = []
    
    print("\n=== Case 1 (엄격한 기준) 회사별 분석 ===")
    
    for company_id in df['company_id'].unique():
        company_data = df[df['company_id'] == company_id].copy()
        
        # Case 1: 엄격한 기준으로 회사별 최종 예측
        final_prediction = get_company_final_prediction_case1(company_data)
        
        # 회사별 실제 라벨 (가장 마지막 날짜)
        true_label = get_company_true_label(company_data)
        
        # Case 1에서 True면 성공, False면 실패
        is_correct = final_prediction
        
        # 예측 패턴 생성 (개별 예측을 패턴으로 변환)
        predicted_pattern = get_company_prediction_pattern(company_data)
        
        # 디버깅: 경영악화 회사 확인
        if true_label in ['TTT', 'FTT', 'FFT']:  # 경영악화 회사
            print(f"🏢 회사 {company_id}: {true_label} (경영악화)")
            print(f"   예측 패턴: {predicted_pattern}")
            print(f"   성공 여부: {final_prediction} ({'성공' if final_prediction else '실패'})")
            print(f"   개별 예측: {company_data['predicted_label'].tolist()}")
            print(f"   개별 실제: {company_data['true_label'].tolist()}")
            
            # 매트릭스 결과 확인
            matrix = create_prediction_matrix()
            if predicted_pattern in matrix and true_label in matrix[predicted_pattern]:
                result = matrix[predicted_pattern][true_label]
                print(f"   매트릭스: {predicted_pattern} vs {true_label} = {result}")
            print()
        
        company_results.append({
            'company_id': company_id,
            'company_true_label': true_label,
            'company_predicted_label': predicted_pattern,  # 패턴으로 변경
            'company_is_correct': is_correct,
            'data_count': len(company_data)
        })
    
    return pd.DataFrame(company_results)

def create_company_level_predictions_case2(df):
    """Case 2: 관대한 기준으로 회사 레벨의 예측 결과 생성"""
    company_results = []
    
    print("\n=== Case 2 (관대한 기준) 회사별 분석 ===")
    
    for company_id in df['company_id'].unique():
        company_data = df[df['company_id'] == company_id].copy()
        
        # Case 2: 관대한 기준으로 회사별 최종 예측
        final_prediction = get_company_final_prediction_case2(company_data)
        
        # 회사별 실제 라벨 (가장 마지막 날짜)
        true_label = get_company_true_label(company_data)
        
        # Case 2에서 True면 성공, False면 실패
        is_correct = final_prediction
        
        # 예측 패턴 생성 (개별 예측을 패턴으로 변환)
        predicted_pattern = get_company_prediction_pattern(company_data)
        
        # 디버깅: 경영악화 회사 확인
        if true_label in ['TTT', 'FTT', 'FFT']:  # 경영악화 회사
            print(f"🏢 회사 {company_id}: {true_label} (경영악화)")
            print(f"   예측 패턴: {predicted_pattern}")
            print(f"   성공 여부: {final_prediction} ({'성공' if final_prediction else '실패'})")
            print(f"   개별 예측: {company_data['predicted_label'].tolist()}")
            print(f"   개별 실제: {company_data['true_label'].tolist()}")
            
            # 매트릭스 결과 확인
            matrix = create_prediction_matrix()
            if predicted_pattern in matrix and true_label in matrix[predicted_pattern]:
                result = matrix[predicted_pattern][true_label]
                print(f"   매트릭스: {predicted_pattern} vs {true_label} = {result}")
            print()
        
        company_results.append({
            'company_id': company_id,
            'company_true_label': true_label,
            'company_predicted_label': predicted_pattern,  # 패턴으로 변경
            'company_is_correct': is_correct,
            'data_count': len(company_data)
        })
    
    return pd.DataFrame(company_results)

def create_confusion_matrices(df, company_df):
    """기존과 새로운 confusion matrix 생성"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 기존 confusion matrix (개별 예측) - 데이터 타입 변환
    y_true_orig = [int(bool(x)) for x in df['true_label']]
    y_pred_orig = [int(bool(x)) for x in df['predicted_label']]
    cm1 = confusion_matrix(y_true_orig, y_pred_orig, labels=[1, 0])
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['경영악화(1)', '거래중(0)'],
                yticklabels=['경영악화(1)', '거래중(0)'])
    ax1.set_title('기존 Confusion Matrix\n(개별 예측)')
    ax1.set_xlabel('예측된 라벨')
    ax1.set_ylabel('실제 라벨')
    
    # 새로운 confusion matrix (회사별 예측) - 패턴 기반
    # 라벨: TTT, FTT = 경영악화(1), FFT, FFF = 거래중(0)
    def pattern_to_binary_label(pattern):
        if pattern in ['TTT', 'FTT']:
            return 1  # 경영악화
        else:  # FFT, FFF
            return 0  # 거래중
    
    # 실제 라벨 변환 (TTT, FTT = 경영악화, FFT, FFF = 거래중)
    y_true_comp = [pattern_to_binary_label(x) for x in company_df['company_true_label']]
    
    # 예측 결과: 성공(True) = 거래중(0), 실패(False) = 경영악화(1)
    y_pred_comp = [0 if x else 1 for x in company_df['company_is_correct']]
    cm2 = confusion_matrix(y_true_comp, y_pred_comp, labels=[1, 0])
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['경영악화(1)', '거래중(0)'],
                yticklabels=['경영악화(1)', '거래중(0)'])
    ax2.set_title('새로운 Confusion Matrix\n(회사별 예측)')
    ax2.set_xlabel('예측된 라벨')
    ax2.set_ylabel('실제 라벨')
    
    plt.tight_layout()
    plt.savefig('../results/20250915_10h_53m_49s/comparison_confusion_matrix.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm1, cm2

def create_company_confusion_matrix(company_df, title_suffix=""):
    """회사별 예측 방식의 독립적인 confusion matrix 생성"""
    plt.figure(figsize=(8, 6))
    
    # 라벨 변환 (TTT, FTT = 경영악화, FFT, FFF = 거래중)
    def pattern_to_binary_label(pattern):
        if pattern in ['TTT', 'FTT']:
            return 1  # 경영악화
        else:  # FFT, FFF
            return 0  # 거래중
    
    # 실제 라벨 변환
    y_true = [pattern_to_binary_label(x) for x in company_df['company_true_label']]
    
    # 예측 결과: 성공(True) = 거래중(0), 실패(False) = 경영악화(1)
    y_pred = [0 if x else 1 for x in company_df['company_is_correct']]
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['경영악화(1)', '거래중(0)'],
                yticklabels=['경영악화(1)', '거래중(0)'])
    plt.title(f'회사별 예측 방식 Confusion Matrix {title_suffix}', fontsize=16, pad=20)
    plt.xlabel('예측된 라벨', fontsize=12)
    plt.ylabel('실제 라벨', fontsize=12)
    
    # 파일명에 Case 정보 추가
    filename = f'company_prediction_confusion_matrix_{title_suffix.replace(" ", "_").replace("(", "").replace(")", "")}.png'
    plt.tight_layout()
    plt.savefig(f'../results/20250915_10h_53m_49s/{filename}', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm

def create_original_confusion_matrix(df):
    """기존 방식(개별 예측)의 독립적인 confusion matrix 생성"""
    plt.figure(figsize=(8, 6))
    
    # 데이터 타입 변환
    y_true = [int(bool(x)) for x in df['true_label']]
    y_pred = [int(bool(x)) for x in df['predicted_label']]
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['경영악화(1)', '거래중(0)'],
                yticklabels=['경영악화(1)', '거래중(0)'])
    plt.title('기존 방식 Confusion Matrix\n(개별 예측)', fontsize=16, pad=20)
    plt.xlabel('예측된 라벨', fontsize=12)
    plt.ylabel('실제 라벨', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('../results/20250915_10h_53m_49s/original_prediction_confusion_matrix.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm

def calculate_metrics(y_true, y_pred, title):
    """성능 지표 계산"""
    # 데이터 타입 확인 및 변환
    print(f"\n=== 디버깅 정보 ===")
    print(f"y_true 타입: {type(y_true)}, 샘플: {list(y_true)[:10]}")
    print(f"y_pred 타입: {type(y_pred)}, 샘플: {list(y_pred)[:10]}")
    print(f"y_true 유니크 값: {sorted(set(y_true))}")
    print(f"y_pred 유니크 값: {sorted(set(y_pred))}")
    
    # Boolean을 int로 변환 (더 강력한 변환)
    if hasattr(y_true, 'dtype') and y_true.dtype == bool:
        y_true = y_true.astype(int)
    elif isinstance(y_true, (list, tuple, pd.Series)) and len(y_true) > 0:
        y_true = [int(bool(x)) for x in y_true]
        if isinstance(y_true, pd.Series):
            y_true = pd.Series(y_true)
    
    if hasattr(y_pred, 'dtype') and y_pred.dtype == bool:
        y_pred = y_pred.astype(int)
    elif isinstance(y_pred, (list, tuple, pd.Series)) and len(y_pred) > 0:
        # Boolean과 int가 섞인 경우 처리
        y_pred = [int(bool(x)) for x in y_pred]
        if isinstance(y_pred, pd.Series):
            y_pred = pd.Series(y_pred)
    
    print(f"변환 후 y_true 유니크 값: {sorted(set(y_true))}")
    print(f"변환 후 y_pred 유니크 값: {sorted(set(y_pred))}")
    
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    # 수동으로 재현율 계산해서 확인
    print(f"Confusion Matrix 확인:")
    print(cm)
    if cm.shape == (2, 2):
        recall_0_manual = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
        recall_1_manual = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
        print(f"수동 계산 재현율 - 클래스 0: {recall_0_manual:.9f}, 클래스 1: {recall_1_manual:.9f}")
        print(f"수동 계산 Macro 재현율: {(recall_0_manual + recall_1_manual) / 2:.9f}")
    
    # Macro average (전체 평균)
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    # 각 클래스별 지표
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    print(f"\n{title}")
    print("="*50)
    print(f"정확도: {accuracy:.9f}")
    print(f"정밀도 (경영악화): {precision_per_class[1]:.9f}")
    print(f"재현율 (경영악화): {recall_per_class[1]:.9f}")
    print(f"F1-점수 (경영악화): {f1_per_class[1]:.9f}")
    print(f"\n클래스별 세부 지표:")
    print(f"  거래중(0) - 정밀도: {precision_per_class[0]:.9f}, 재현율: {recall_per_class[0]:.9f}, F1: {f1_per_class[0]:.9f}")
    print(f"  경영악화(1) - 정밀도: {precision_per_class[1]:.9f}, 재현율: {recall_per_class[1]:.9f}, F1: {f1_per_class[1]:.9f}")
    print(f"Confusion Matrix:")
    print(cm)
    print(f"Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['거래중(0)', '경영악화(1)']))
    
    return {
        'accuracy': accuracy,
        'precision': precision_per_class[1],  # 경영악화(1) 클래스만
        'recall': recall_per_class[1],        # 경영악화(1) 클래스만
        'f1_score': f1_per_class[1],          # 경영악화(1) 클래스만
        'confusion_matrix': cm,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class
    }

def create_three_way_comparison_plot(original_metrics, case1_metrics, case2_metrics):
    """기존 vs Case 1 vs Case 2 성능 비교 플롯 생성"""
    metrics = ['정확도', '정밀도(경영악화)', '재현율(경영악화)', 'F1 Score(경영악화)']
    original_values = [original_metrics['accuracy'], original_metrics['precision'], 
                      original_metrics['recall'], original_metrics['f1_score']]
    case1_values = [case1_metrics['accuracy'], case1_metrics['precision'], 
                   case1_metrics['recall'], case1_metrics['f1_score']]
    case2_values = [case2_metrics['accuracy'], case2_metrics['precision'], 
                   case2_metrics['recall'], case2_metrics['f1_score']]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bars1 = ax.bar(x - width, original_values, width, label='기존 방식 (개별 예측)', alpha=0.8, color='lightgreen')
    bars2 = ax.bar(x, case1_values, width, label='Case 1 (엄격한 기준)', alpha=0.8, color='lightcoral')
    bars3 = ax.bar(x + width, case2_values, width, label='Case 2 (관대한 기준)', alpha=0.8, color='lightblue')
    
    ax.set_xlabel('성능 지표', fontsize=12)
    ax.set_ylabel('점수', fontsize=12)
    ax.set_title('기존 vs Case 1 vs Case 2 성능 비교\n(회사별 예측 결정 방식)', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # 값 표시
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('../results/20250915_10h_53m_49s/three_way_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()



def main():
    file_path = "../results/20250915_10h_53m_49s/predict_result.xlsx"
    
    # 1. 데이터 로드 및 분석
    print("1. 데이터 로드 중...")
    df = load_and_analyze_data(file_path)
    
    # 2. 회사 ID와 날짜 분리
    print("\n2. 회사 ID와 날짜 분리...")
    df = extract_company_and_date(df)
    
    # 3. Case 1: 엄격한 기준으로 회사별 예측 결과 생성
    print("\n3. Case 1: 엄격한 기준으로 회사별 예측 결과 생성...")
    company_df_case1 = create_company_level_predictions_case1(df)
    
    # 4. Case 2: 관대한 기준으로 회사별 예측 결과 생성
    print("\n4. Case 2: 관대한 기준으로 회사별 예측 결과 생성...")
    company_df_case2 = create_company_level_predictions_case2(df)
    
    # 5. 원본 데이터에 Case 1, Case 2 결과 병합
    print("\n5. 원본 데이터에 Case 1, Case 2 결과 병합...")
    df_merged_case1 = df.merge(company_df_case1[['company_id', 'company_true_label', 'company_predicted_label', 'company_is_correct']], 
                              on='company_id', how='left', suffixes=('', '_case1'))
    df_merged_case2 = df.merge(company_df_case2[['company_id', 'company_true_label', 'company_predicted_label', 'company_is_correct']], 
                              on='company_id', how='left', suffixes=('', '_case2'))
    
    # 6. 결과 저장
    print("\n6. 결과 저장...")
    output_file = "../results/20250915_10h_53m_49s/predict_result_with_case1_case2_analysis.xlsx"
    with pd.ExcelWriter(output_file) as writer:
        df_merged_case1.to_excel(writer, sheet_name='원본_데이터_with_Case1', index=False)
        df_merged_case2.to_excel(writer, sheet_name='원본_데이터_with_Case2', index=False)
        company_df_case1.to_excel(writer, sheet_name='Case1_회사별_최종결과', index=False)
        company_df_case2.to_excel(writer, sheet_name='Case2_회사별_최종결과', index=False)
    
    print(f"결과가 저장되었습니다: {output_file}")
    
    # Case별 CSV 파일 저장
    print("\n6-1. Case별 CSV 파일 저장...")
    case1_csv_file = "../results/20250915_10h_53m_49s/case1_analysis.csv"
    case2_csv_file = "../results/20250915_10h_53m_49s/case2_analysis.csv"
    
    df_merged_case1.to_csv(case1_csv_file, index=False, encoding='utf-8-sig')
    df_merged_case2.to_csv(case2_csv_file, index=False, encoding='utf-8-sig')
    
    print(f"Case 1 CSV 저장: {case1_csv_file}")
    print(f"Case 2 CSV 저장: {case2_csv_file}")
    
    # 7. 성능 비교 (기존 방식 vs Case 1 vs Case 2)
    print("\n7. 성능 비교...")
    original_metrics = calculate_metrics(df['true_label'], df['predicted_label'], "기존 방식 (개별 예측)")
    
    # Case 1, Case 2: 패턴을 이진값으로 변환하여 성능 계산
    def pattern_to_binary(pattern):
        if pattern in ['TTT', 'FTT']:
            return 1  # 경영악화
        else:  # FFT, FFF
            return 0  # 거래중
    
    # Case 1, Case 2: 패턴을 이진값으로 변환하여 성능 계산
    def convert_prediction_to_pattern(pred_value, true_pattern):
        if isinstance(pred_value, bool):
            # Boolean 값이면 True=성공(경영악화), False=실패(거래중)로 해석
            return 1 if pred_value else 0
        else:
            # 이미 패턴이면 그대로 변환
            return pattern_to_binary(pred_value)
    
    # 디버깅: Case 1 결과 확인
    print("\n=== Case 1 디버깅 ===")
    print(f"총 회사 수: {len(company_df_case1)}")
    print(f"경영악화 회사 (TTT, FTT): {len(company_df_case2[company_df_case2['company_true_label'].isin(['TTT', 'FTT', 'FFT'])])}")
    print(f"거래중 회사 (FFT, FFF): {len(company_df_case2[company_df_case2['company_true_label'].isin(['FFF'])])}")
    print(f"예측 패턴 분포:")
    print(f"  TTT: {len(company_df_case1[company_df_case1['company_predicted_label'] == 'TTT'])}개")
    print(f"  TTF: {len(company_df_case1[company_df_case1['company_predicted_label'] == 'TTF'])}개")
    print(f"  TFT: {len(company_df_case1[company_df_case1['company_predicted_label'] == 'TFT'])}개")
    print(f"  TFF: {len(company_df_case1[company_df_case1['company_predicted_label'] == 'TFF'])}개")
    print(f"  FTT: {len(company_df_case1[company_df_case1['company_predicted_label'] == 'FTT'])}개")
    print(f"  FTF: {len(company_df_case1[company_df_case1['company_predicted_label'] == 'FTF'])}개")
    print(f"  FFT: {len(company_df_case1[company_df_case1['company_predicted_label'] == 'FFT'])}개")
    print(f"  FFF: {len(company_df_case1[company_df_case1['company_predicted_label'] == 'FFF'])}개")
    
    # 각 회사별 성공/실패 확인
    print("\n=== 각 회사별 성공/실패 확인 ===")
    matrix = create_prediction_matrix()
    success_cases = []
    fail_cases = []
    y_cheolsu_cases = []
    
    for _, row in company_df_case1.iterrows():
        company_id = row['company_id']
        pred_pattern = row['company_predicted_label']
        true_pattern = row['company_true_label']
        is_correct = row['company_is_correct']
        
        if pred_pattern in matrix and true_pattern in matrix[pred_pattern]:
            result = matrix[pred_pattern][true_pattern]
            if result == 'y(철수)':
                y_cheolsu_cases.append(f"회사 {company_id}: {pred_pattern} vs {true_pattern} = {result}")
            
            if is_correct:
                success_cases.append(f"회사 {company_id}: {pred_pattern} vs {true_pattern} = {result} → 성공")
            else:
                fail_cases.append(f"회사 {company_id}: {pred_pattern} vs {true_pattern} = {result} → 실패")
    
    print(f"성공 케이스 ({len(success_cases)}개):")
    for case in success_cases:
        print(f"  {case}")
    
    print(f"\n실패 케이스 ({len(fail_cases)}개):")
    for case in fail_cases:
        print(f"  {case}")
    
    if y_cheolsu_cases:
        print(f"\ny(철수) 케이스 ({len(y_cheolsu_cases)}개):")
        for case in y_cheolsu_cases:
            print(f"  {case}")
    else:
        print("\ny(철수) 케이스 없음 - Case 1과 Case 2가 동일한 이유!")
    
    # 디버깅: Case 2 결과 확인
    print("\n=== Case 2 디버깅 ===")
    print(f"총 회사 수: {len(company_df_case2)}")
    print(f"경영악화 회사 (TTT, FTT): {len(company_df_case2[company_df_case2['company_true_label'].isin(['TTT', 'FTT', 'FFT'])])}")
    print(f"거래중 회사 (FFT, FFF): {len(company_df_case2[company_df_case2['company_true_label'].isin(['FFF'])])}")
    print(f"예측 패턴 분포:")
    print(f"  TTT: {len(company_df_case2[company_df_case2['company_predicted_label'] == 'TTT'])}개")
    print(f"  TTF: {len(company_df_case2[company_df_case2['company_predicted_label'] == 'TTF'])}개")
    print(f"  TFT: {len(company_df_case2[company_df_case2['company_predicted_label'] == 'TFT'])}개")
    print(f"  TFF: {len(company_df_case2[company_df_case2['company_predicted_label'] == 'TFF'])}개")
    print(f"  FTT: {len(company_df_case2[company_df_case2['company_predicted_label'] == 'FTT'])}개")
    print(f"  FTF: {len(company_df_case2[company_df_case2['company_predicted_label'] == 'FTF'])}개")
    print(f"  FFT: {len(company_df_case2[company_df_case2['company_predicted_label'] == 'FFT'])}개")
    print(f"  FFF: {len(company_df_case2[company_df_case2['company_predicted_label'] == 'FFF'])}개")
    
    # Case 1: 실제 라벨 변환 (TTT, FTT = 경영악화, FFT, FFF = 거래중)
    def pattern_to_binary_label(pattern):
        if pattern in ['TTT', 'FTT']:
            return 1  # 경영악화
        else:  # FFT, FFF
            return 0  # 거래중
    
    case1_true_binary = [pattern_to_binary_label(x) for x in company_df_case1['company_true_label']]
    # Case 1: 예측 결과 - 성공(True) = 거래중(0), 실패(False) = 경영악화(1)
    case1_pred_binary = [0 if x else 1 for x in company_df_case1['company_is_correct']]
    
    # 디버깅: Case 1 변환 결과 확인
    print("\n=== Case 1 변환 결과 ===")
    print(f"실제 라벨 (이진): {case1_true_binary[:10]}...")  # 처음 10개만
    print(f"예측 라벨 (이진): {case1_pred_binary[:10]}...")  # 처음 10개만
    print(f"경영악화 실제: {sum(case1_true_binary)}개")
    print(f"경영악화 예측: {sum(case1_pred_binary)}개")
    
    case1_metrics = calculate_metrics(case1_true_binary, case1_pred_binary, "Case 1 (엄격한 기준)")
    
    # Case 2: 실제 라벨 변환 (TTT, FTT = 경영악화, FFT, FFF = 거래중)
    case2_true_binary = [pattern_to_binary_label(x) for x in company_df_case2['company_true_label']]
    # Case 2: 예측 결과 - 성공(True) = 거래중(0), 실패(False) = 경영악화(1)
    case2_pred_binary = [0 if x else 1 for x in company_df_case2['company_is_correct']]
    
    # 디버깅: Case 2 변환 결과 확인
    print("\n=== Case 2 변환 결과 ===")
    print(f"실제 라벨 (이진): {case2_true_binary[:10]}...")  # 처음 10개만
    print(f"예측 라벨 (이진): {case2_pred_binary[:10]}...")  # 처음 10개만
    print(f"경영악화 실제: {sum(case2_true_binary)}개")
    print(f"경영악화 예측: {sum(case2_pred_binary)}개")
    
    case2_metrics = calculate_metrics(case2_true_binary, case2_pred_binary, "Case 2 (관대한 기준)")
    
    # 8. 성능 지표 저장
    print("\n8. 성능 지표 저장...")
    metrics_df = pd.DataFrame({
        '방식': ['기존_방식(개별예측)', 'Case1_엄격한기준', 'Case2_관대한기준'],
        'accuracy': [original_metrics['accuracy'], case1_metrics['accuracy'], case2_metrics['accuracy']],
        'precision': [original_metrics['precision'], case1_metrics['precision'], case2_metrics['precision']],
        'recall': [original_metrics['recall'], case1_metrics['recall'], case2_metrics['recall']],
        'f1_score': [original_metrics['f1_score'], case1_metrics['f1_score'], case2_metrics['f1_score']]
    })
    
    metrics_output_file = "../results/20250915_10h_53m_49s/performance_metrics_case1_case2.xlsx"
    metrics_df.to_excel(metrics_output_file, index=False)
    print(f"성능 지표가 저장되었습니다: {metrics_output_file}")
    
    # CSV로도 저장 (소수점 9자리까지)
    metrics_csv_file = "../results/20250915_10h_53m_49s/performance_metrics_case1_case2.csv"
    metrics_df.to_csv(metrics_csv_file, index=False, float_format='%.9f')
    print(f"성능 지표 CSV가 저장되었습니다: {metrics_csv_file}")
    

    # 10. 세 가지 방식 비교 플롯 생성
    print("\n10. 기존 vs Case 1 vs Case 2 비교 플롯 생성...")
    create_three_way_comparison_plot(original_metrics, case1_metrics, case2_metrics)
    
    # 11. Confusion Matrix 비교
    print("\n11. Confusion Matrix 비교 생성...")
    cm1, cm2 = create_confusion_matrices(df, company_df_case1)
    
    # 12. Case 1 독립 Confusion Matrix 생성
    print("\n12. Case 1 독립 Confusion Matrix 생성...")
    case1_cm = create_company_confusion_matrix(company_df_case1, "Case 1 (엄격한 기준)")
    
    # 13. Case 2 독립 Confusion Matrix 생성
    print("\n13. Case 2 독립 Confusion Matrix 생성...")
    case2_cm = create_company_confusion_matrix(company_df_case2, "Case 2 (관대한 기준)")
    
    # 14. 회사별 데이터 분포 확인
    print("\n14. 회사별 데이터 분포:")
    print(f"총 회사 수: {len(company_df_case1)}")
    print(f"회사별 데이터 개수 분포:")
    print(company_df_case1['data_count'].value_counts().sort_index())
    
    # 15. Case 1 vs Case 2 예측 차이 분석
    print("\n15. Case 1 vs Case 2 예측 차이 분석:")
    prediction_diff = (company_df_case1['company_predicted_label'] != company_df_case2['company_predicted_label']).sum()
    print(f"예측이 다른 회사 수: {prediction_diff}/{len(company_df_case1)} ({prediction_diff/len(company_df_case1)*100:.1f}%)")
    
    # Case 1이 False, Case 2가 True인 경우 (철수 케이스)
    case1_false_case2_true = ((company_df_case1['company_predicted_label'] == False) & 
                             (company_df_case2['company_predicted_label'] == True)).sum()
    print(f"Case 1: False, Case 2: True (철수 케이스): {case1_false_case2_true}개")
    
    return df_merged_case1, df_merged_case2, company_df_case1, company_df_case2, metrics_df

if __name__ == "__main__":
    df_merged_case1, df_merged_case2, company_df_case1, company_df_case2, metrics_df = main()
