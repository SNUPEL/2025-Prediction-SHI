import time
import warnings
import os
import random
import numpy as np
import tensorflow as tf
import traceback
from config import create_config
from data import Data
from model import Model

def set_seeds(seed_value):
    """
    프로그램의 모든 무작위성을 고정하여 실행할 때마다 동일한 결과를 얻도록 합니다.
    """
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    print(f" 모든 난수 시드를 {seed_value} (으)로 고정")

if __name__ == '__main__':
    start_time = time.time()
    warnings.filterwarnings('ignore')  # 불필요한 경고 메시지 무시

    # 설정 파일 로딩 및 시드 고정
    config = create_config()
    set_seeds(config['random_state'])

    try:
        print(f"--- 모델 실행 시작 ---")

        # 1. 데이터 준비
        data = Data(config)
        data.prepare_data()

        # 2. 모델 생성, 학습 및 평가
        model = Model(config, data)
        model.make_model()
        model.evaluate_model()

        # 3. 결과 저장
        model.save_result()

        print("\n" + "=" * 24 + " 최종 모델 성능 요약 " + "=" * 24)
        res = model.result
        if 'best_threshold' in res:
            print(f" (최적 임계값: {res.get('best_threshold', 0):.2f} 에서의 성능)")
        print("-" * 68)
        print(f"  - 정확도 (Accuracy) : {res.get('accuracy', 0):.4f}")
        print(f"  - 정밀도 (Precision): {res.get('precision', 0):.4f}")
        print(f"  - 재현율 (Recall)   : {res.get('recall', 0):.4f}")
        print(f"  - F1-Score         : {res.get('f1_score', 0):.4f}")
        print("-" * 68)
        if 'TP' in res:  # 상세 지표가 있는 경우에만 출력
            print(" [혼동 행렬 상세]")
            print(f"  - TP (진짜 양성): {res.get('TP', 0):>4d} (실제 '철수'를 '철수'로 정확히 예측)")
            print(f"  - FN (가짜 음성): {res.get('FN', 0):>4d} (실제 '철수'를 '정상'으로 잘못 예측) 🚨 놓친 위험!")
            print(f"  - FP (가짜 양성): {res.get('FP', 0):>4d} (실제 '정상'을 '철수'로 잘못 예측) ⚠️ 거짓 경보!")
            print(f"  - TN (진짜 음성): {res.get('TN', 0):>4d} (실제 '정상'을 '정상'으로 정확히 예측)")
        print("=" * 68)

    except Exception as e:
        print(f"\n 프로그램 실행 중 심각한 오류가 발생했습니다: {e}")
        traceback.print_exc()

    finally:
        print(f"\n--- 모든 작업 완료 ---")
        if 'config' in locals() and os.path.exists(config['result_folder_path']):
            print(f"결과는 '{config['result_folder_path']}' 폴더에 저장되었습니다.")
        print(f"총 실행 시간: {time.time() - start_time:.2f}초")