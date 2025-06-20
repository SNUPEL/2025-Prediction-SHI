from config import *
from data import *
from model import *

if __name__ == '__main__':
    start_time = time.time()

    config = create_config()
    print("\n==== 설정 정보 ====")
    print(f"데이터 파일 경로: {config['data_file_path']}")
    print(f"입력 데이터 길이: {config['data_duration']}")
    print(f"라벨 판단 시점: {config['label_date']}")
    print(f"라벨 판단 길이: {config['label_duration']}")
    print(f"사용하지 않는 시트: {config['label_duration']}")
    print(f"정규화 기법: {config['scaler']}")
    print(f"Undersampling 기법: {config['undersampling']}")
    print(f"Oversampling 기법: {config['oversampling']}")
    print(f"선택된 모델: {config['model_type']}")
    print("=================\n")

    data = Data(config)
    print("\n==== 데이터 로딩 시작 ====")
    data.load_data()
    print("==== 데이터 로딩 완료 =====\n")

    print("\n==== 데이터 전처리 시작====")
    data.preprocess_data()
    print("==== 데이터 전처리 완료 =====\n")

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

    print('Run Time: ', time.time() - start_time)
