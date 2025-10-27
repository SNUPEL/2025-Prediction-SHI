import warnings
# pandas의 모든 FutureWarning 완전 차단
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
warnings.filterwarnings('ignore', message='.*fillna.*deprecated.*')
warnings.filterwarnings('ignore', message='.*Dtype inference.*deprecated.*')
# 추가적인 pandas 경고들도 차단
warnings.filterwarnings('ignore', category=FutureWarning)
import time
import random
from config import *
from data import *
from model import *
from XAI import *

if __name__ == '__main__':
    start_time = time.time()

    config = create_config()
    print("\n==== 설정 정보 ====")
    print(f"데이터 파일 경로: {config['data_file_path']}")
    print(f"입력 데이터 길이: {config['data_duration']}")
    print(f"라벨 판단 시점: {config['label_date']}")
    print(f"라벨 판단 길이: {config['label_duration']}")
    print(f"사용하지 않는 시트: {config['sheet_ban_list']}")
    print(f"정규화 기법: {config['scaler']}")
    print(f"Undersampling 기법: {config['undersampling']}")
    print(f"Oversampling 기법: {config['oversampling']}")
    print(f"선택된 모델: {config['model_type']}")
    print("=================\n")

    # 랜덤 상태 정의
    os.environ['PYTHONHASHSEED'] = str(config['random_state'])
    random.seed(config['random_state'])
    np.random.seed(config['random_state'])
    tf.random.set_seed(config['random_state'])
    torch.manual_seed(config['random_state'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['random_state'])

    data = Data(config)
    print("\n==== 데이터 로딩 시작 ====")
    data.load_data()
    print("==== 데이터 로딩 완료 =====\n")

    print("\n====== 데이터 전처리 시작======")
    data.preprocess_data()
    print("====== 데이터 전처리 완료 =======\n")

    model = Model(config, data)
    print(f"\n==== {config['model_type']} 모델 학습 시작 ====")
    model.make_model()
    print(f"==== {config['model_type']} 모델 학습 완료 ====\n")

    print("\n==== test data 평가 시작 ====")
    model.evaluate_model()
    print("==== test data 평가 완료 =====\n")

    print("\n==== 결과 저장 시작 ====")
    model.save_result()
    print("==== 결과 저장 완료 =====\n")

    if config['DiCE']:
        print("\n==== DiCE ====")
        get_dice(config, data, model)
        print("\n==== DiCE 종료 ====")

    print('Run Time: ', time.time() - start_time)
