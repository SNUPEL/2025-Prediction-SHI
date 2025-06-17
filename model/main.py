from config import *
from data import *
from model import *

if __name__ == '__main__':
    start_time = time.time()

    config = create_config()
    print("\n==== 설정 정보 ====")
    print(f"데이터 파일 경로: {config['data_file_path']}")
    print(f"Undersampling 사용 여부: {config['undersampling']}")
    print(f"Oversampling 사용 여부: {config['oversampling']}")
    print(f"선택된 모델: {config['model_type']}")
    print("=================\n")

    data = Data(config)
    data.load_data()

    print("\n==== 데이터 분할 ====")
    data.preprocess_data()
    print("=================================\n")

    model = Model(config, data)
    print("\n==== 모델 학습 시작 ====")
    model.make_model()
    print("==== 모델 학습 완료 ====\n")
    model.evaluate_model()
    model.save_result()

    print('Run Time: ', time.time() - start_time)


