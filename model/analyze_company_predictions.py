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

def get_company_final_prediction(group):
    """회사별 최종 예측 결정"""
    predictions = group['predicted_label'].tolist()
    probabilities = group['probability'].tolist()
    
    # 예측값 개수 세기
    pred_counts = Counter(predictions)
    
    # 가장 많은 예측값들 찾기
    max_count = max(pred_counts.values())
    most_common_preds = [pred for pred, count in pred_counts.items() if count == max_count]
    
    if len(most_common_preds) == 1:
        # 명확한 다수결
        final_prediction = most_common_preds[0]
    else:
        # 동점인 경우 확률이 높은 것 선택
        tied_indices = [i for i, pred in enumerate(predictions) if pred in most_common_preds]
        max_prob_idx = tied_indices[np.argmax([probabilities[i] for i in tied_indices])]
        final_prediction = predictions[max_prob_idx]
    
    return final_prediction

def get_company_true_label(group):
    """회사별 실제 라벨 (가장 마지막 날짜)"""
    latest_date_idx = group['date'].idxmax()
    return group.loc[latest_date_idx, 'true_label']

def create_company_level_predictions(df):
    """회사 레벨의 예측 결과 생성"""
    company_results = []
    
    for company_id in df['company_id'].unique():
        company_data = df[df['company_id'] == company_id].copy()
        
        # 회사별 최종 예측
        final_prediction = get_company_final_prediction(company_data)
        
        # 회사별 실제 라벨 (가장 마지막 날짜)
        true_label = get_company_true_label(company_data)
        
        # 정확도
        is_correct = (final_prediction == true_label)
        
        company_results.append({
            'company_id': company_id,
            'company_true_label': true_label,
            'company_predicted_label': final_prediction,
            'company_is_correct': is_correct,
            'data_count': len(company_data)
        })
    
    return pd.DataFrame(company_results)

def create_confusion_matrices(df, company_df):
    """기존과 새로운 confusion matrix 생성"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 기존 confusion matrix (개별 예측)
    cm1 = confusion_matrix(df['true_label'], df['predicted_label'])
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['거래중(0)', '경영악화(1)'],
                yticklabels=['거래중(0)', '경영악화(1)'])
    ax1.set_title('기존 Confusion Matrix\n(개별 예측)')
    ax1.set_xlabel('예측된 라벨')
    ax1.set_ylabel('실제 라벨')
    
    # 새로운 confusion matrix (회사별 예측)
    cm2 = confusion_matrix(company_df['company_true_label'], company_df['company_predicted_label'])
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['거래중(0)', '경영악화(1)'],
                yticklabels=['거래중(0)', '경영악화(1)'])
    ax2.set_title('새로운 Confusion Matrix\n(회사별 예측)')
    ax2.set_xlabel('예측된 라벨')
    ax2.set_ylabel('실제 라벨')
    
    plt.tight_layout()
    plt.savefig('C:/Users/User/Desktop/real/results/2022_02_01/comparison_confusion_matrix.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm1, cm2

def create_company_confusion_matrix(company_df):
    """회사별 예측 방식의 독립적인 confusion matrix 생성"""
    plt.figure(figsize=(8, 6))
    
    cm = confusion_matrix(company_df['company_true_label'], company_df['company_predicted_label'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['거래중(0)', '경영악화(1)'],
                yticklabels=['거래중(0)', '경영악화(1)'])
    plt.title('회사별 예측 방식 Confusion Matrix', fontsize=16, pad=20)
    plt.xlabel('예측된 라벨', fontsize=12)
    plt.ylabel('실제 라벨', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('C:/Users/User/Desktop/real/results/2022_02_01/company_prediction_confusion_matrix.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm

def create_original_confusion_matrix(df):
    """기존 방식(개별 예측)의 독립적인 confusion matrix 생성"""
    plt.figure(figsize=(8, 6))
    
    cm = confusion_matrix(df['true_label'], df['predicted_label'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['거래중(0)', '경영악화(1)'],
                yticklabels=['거래중(0)', '경영악화(1)'])
    plt.title('기존 방식 Confusion Matrix\n(개별 예측)', fontsize=16, pad=20)
    plt.xlabel('예측된 라벨', fontsize=12)
    plt.ylabel('실제 라벨', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('C:/Users/User/Desktop/real/results/2022_02_01/original_prediction_confusion_matrix.png', 
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
    
    # Boolean을 int로 변환
    if hasattr(y_true, 'dtype') and y_true.dtype == bool:
        y_true = y_true.astype(int)
    elif isinstance(y_true, (list, tuple)) and len(y_true) > 0 and isinstance(y_true[0], bool):
        y_true = [int(x) for x in y_true]
    
    if hasattr(y_pred, 'dtype') and y_pred.dtype == bool:
        y_pred = y_pred.astype(int)
    elif isinstance(y_pred, (list, tuple)) and len(y_pred) > 0 and isinstance(y_pred[0], bool):
        y_pred = [int(x) for x in y_pred]
    
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
    print(f"정밀도 (Macro): {precision_macro:.9f}")
    print(f"재현율 (Macro): {recall_macro:.9f}")
    print(f"F1-점수 (Macro): {f1_macro:.9f}")
    print(f"\n클래스별 세부 지표:")
    print(f"  거래중(0) - 정밀도: {precision_per_class[0]:.9f}, 재현율: {recall_per_class[0]:.9f}, F1: {f1_per_class[0]:.9f}")
    print(f"  경영악화(1) - 정밀도: {precision_per_class[1]:.9f}, 재현율: {recall_per_class[1]:.9f}, F1: {f1_per_class[1]:.9f}")
    print(f"Confusion Matrix:")
    print(cm)
    print(f"Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['거래중(0)', '경영악화(1)']))
    
    return {
        'accuracy': accuracy,
        'precision': precision_macro,
        'recall': recall_macro,
        'f1_score': f1_macro,
        'confusion_matrix': cm,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class
    }

def main():
    file_path = "C:/Users/User/Desktop/real/results/2022_02_01/predict_result.xlsx"
    
    # 1. 데이터 로드 및 분석
    print("1. 데이터 로드 중...")
    df = load_and_analyze_data(file_path)
    
    # 2. 회사 ID와 날짜 분리
    print("\n2. 회사 ID와 날짜 분리...")
    df = extract_company_and_date(df)
    
    # 3. 회사별 예측 결과 생성
    print("\n3. 회사별 예측 결과 생성...")
    company_df = create_company_level_predictions(df)
    
    # 4. 원본 데이터에 회사별 결과 병합
    print("\n4. 원본 데이터에 회사별 결과 병합...")
    df_merged = df.merge(company_df[['company_id', 'company_true_label', 'company_predicted_label', 'company_is_correct']], 
                        on='company_id', how='left')
    
    # 5. 결과 저장
    print("\n5. 결과 저장...")
    output_file = "C:/Users/User/Desktop/real/results/2022_02_01/predict_result_with_company_analysis.xlsx"
    with pd.ExcelWriter(output_file) as writer:
        df_merged.to_excel(writer, sheet_name='원본_데이터_with_회사분석', index=False)
        company_df.to_excel(writer, sheet_name='회사별_최종결과', index=False)
    
    print(f"결과가 저장되었습니다: {output_file}")
    
    # 6. 성능 비교
    print("\n6. 성능 비교...")
    original_metrics = calculate_metrics(df['true_label'], df['predicted_label'], "기존 방식 (개별 예측)")
    company_metrics = calculate_metrics(company_df['company_true_label'], company_df['company_predicted_label'], "새로운 방식 (회사별 예측)")
    
    # 7. 성능 지표 저장
    print("\n7. 성능 지표 저장...")
    metrics_df = pd.DataFrame({
        '방식': ['기존_방식(개별예측)', '새로운_방식(회사별예측)'],
        'accuracy': [original_metrics['accuracy'], company_metrics['accuracy']],
        'precision': [original_metrics['precision'], company_metrics['precision']],
        'recall': [original_metrics['recall'], company_metrics['recall']],
        'f1_score': [original_metrics['f1_score'], company_metrics['f1_score']]
    })
    
    metrics_output_file = "C:/Users/User/Desktop/real/results/2022_02_01/performance_metrics.xlsx"
    metrics_df.to_excel(metrics_output_file, index=False)
    print(f"성능 지표가 저장되었습니다: {metrics_output_file}")
    
    # CSV로도 저장 (소수점 9자리까지)
    metrics_csv_file = "C:/Users/User/Desktop/real/results/2022_02_01/performance_metrics.csv"
    metrics_df.to_csv(metrics_csv_file, index=False, float_format='%.9f')
    print(f"성능 지표 CSV가 저장되었습니다: {metrics_csv_file}")
    
    # 8. Confusion Matrix 비교
    print("\n8. Confusion Matrix 비교 생성...")
    cm1, cm2 = create_confusion_matrices(df, company_df)
    
    # 9. 기존 방식 독립 Confusion Matrix 생성
    print("\n9. 기존 방식 독립 Confusion Matrix 생성...")
    original_cm = create_original_confusion_matrix(df)
    
    # 10. 회사별 예측 방식 독립 Confusion Matrix 생성
    print("\n10. 회사별 예측 방식 독립 Confusion Matrix 생성...")
    company_cm = create_company_confusion_matrix(company_df)
    
    # 11. 회사별 데이터 분포 확인
    print("\n11. 회사별 데이터 분포:")
    print(f"총 회사 수: {len(company_df)}")
    print(f"회사별 데이터 개수 분포:")
    print(company_df['data_count'].value_counts().sort_index())
    
    return df_merged, company_df, metrics_df

if __name__ == "__main__":
    df_merged, company_df, metrics_df = main()
