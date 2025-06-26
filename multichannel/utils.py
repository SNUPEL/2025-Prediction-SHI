"""
유틸리티 함수 모듈 - 로깅, 디렉토리 관리, 한글 폰트 처리 등
"""
import os
import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from config import OUTPUT_DIR, LOG_FILE

# 디렉토리 생성 함수
def ensure_dir(directory):
    """디렉토리가 없으면 생성"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

# 로그 출력 함수
def log_message(message, also_print=True):
    """메시지를 로그 파일에 기록하고 필요시 콘솔에 출력"""
    ensure_dir(os.path.dirname(LOG_FILE))
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{message}\n")
    if also_print:
        print(message)

# 한글 폰트 설정
def setup_korean_font():
    """운영체제별 한글 폰트 설정"""
    # 운영체제별 기본 폰트 설정
    system_name = platform.system()
    if system_name == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우의 '맑은 고딕'
    elif system_name == 'Darwin':  # macOS
        plt.rcParams['font.family'] = 'AppleGothic'  # macOS의 애플고딕
    else:  # Linux 등
        plt.rcParams['font.family'] = 'NanumGothic'  # 나눔고딕

    # 폰트 경로 설정 (필요시 사용)
    font_path = None
    for font_name in ['Malgun Gothic', 'NanumGothic', 'NanumGothicOTF', 'AppleGothic']:
        try:
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if font_path and not font_path.endswith('DejaVuSans.ttf'):
                plt.rcParams['font.family'] = font_name
                break
        except:
            continue
    
    return font_path

# 한글 폰트 에러 방지 함수
def safe_korean_font(text, font_path=None):
    """한글 텍스트를 안전하게 처리 (폰트 문제 방지)"""
    if font_path and not font_path.endswith('DejaVuSans.ttf'):
        return text  # 한글 폰트가 있으면 그대로 사용
    else:
        # 한글을 로마자로 변환 (또는 간단한 영문 대체)
        hangul_map = {
            '중요도': 'Importance',
            '시트명': 'Sheet',
            '채널': 'Channel',
            '상대적': 'Relative',
            '시트별': 'By Sheet',
            '분석': 'Analysis',
            '폴드': 'Fold',
            '시간': 'Time',
            '시간대': 'Time Period',
            '특성': 'Feature'
        }
        
        # 기본 변환
        for kor, eng in hangul_map.items():
            text = text.replace(kor, eng)
            
        # 그래도 한글이 남아있다면 transliteration (더 복잡한 경우)
        has_hangul = any('\uAC00' <= char <= '\uD7A3' for char in text)
        if has_hangul:
            # 간단히 '시트N'을 'SheetN'으로 변환
            for i in range(1, 20):
                text = text.replace(f'시트{i}', f'Sheet{i}')
            # 그래도 한글이 있다면 영문 대체
            has_hangul = any('\uAC00' <= char <= '\uD7A3' for char in text)
            if has_hangul:
                text = ''.join(c if c.isascii() else '_' for c in text)
        
        return text

# 폴드별 디렉토리 생성 함수
def ensure_fold_dir(base_dir, model_name, fold=None):
    """모델 및 폴드별 결과 디렉토리 생성"""
    # 기본 모델 디렉토리
    model_dir = os.path.join(base_dir, model_name)
    ensure_dir(model_dir)
    
    # 폴드별 디렉토리 (지정된 경우)
    if fold is not None:
        fold_dir = os.path.join(model_dir, f"fold_{fold}")
        ensure_dir(fold_dir)
        return fold_dir
    
    return model_dir

# 초기 설정 수행
def initialize():
    """초기 설정 함수 - 필요한 디렉토리 생성 및 폰트 설정"""
    ensure_dir(OUTPUT_DIR)
    font_path = setup_korean_font()
    return font_path 

# 간단한 최적 임계값 찾기 함수
def find_best_threshold_simple(y_true, y_probs, metric='f1'):
    """
    간단한 최적 임계값 찾기 (기존 코드에 최소 침입적)
    
    params:
        y_true: 실제 레이블
        y_probs: 예측 확률
        metric: 최적화할 지표 ('f1', 'precision', 'recall')
    
    returns:
        best_threshold: 최적 임계값
    """
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    thresholds = np.arange(0.1, 0.9, 0.05)  # 0.1~0.9, 0.05 간격
    best_score = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)  # 기본값
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold

# 학습 손실 곡선 시각화 함수
def plot_loss_curves(train_losses, val_losses, save_path):
    """
    학습 및 검증 손실 곡선 시각화
    
    params:
        train_losses: 학습 손실 목록
        val_losses: 검증 손실 목록
        save_path: 저장 경로
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# ROC 곡선 시각화 함수
def plot_roc_curve(y_true, y_prob, save_path):
    """
    ROC 곡선 시각화
    
    params:
        y_true: 실제 레이블
        y_prob: 예측 확률
        save_path: 저장 경로
    """
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        
        return auc
    except Exception as e:
        log_message(f"    ROC 곡선 생성 오류: {str(e)}")
        return 0.0 