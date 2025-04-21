from config import *
from data import *
from model import *

if __name__ == '__main__':
    start_time = time.time()

    config = create_config()
    print("\n==== 설정 정보 ====")
    print(f"데이터 파일 경로: {config['data_file_path']}")
    print(f"SMOTE 사용 여부: {config['SMOTE']}")
    print(f"선택된 모델: {config['model_type']}")
    print("=================\n")

    data = Data(config)
    data.load_data()
    data.preprocess_data()
    
    print("\n==== 데이터 분할 및 SMOTE 적용 ====")
    data.split_data()
    print("=================================\n")

    model = Model(config, data)
    model.make_model()
    model.evaluate_model()
    model.save_result()

    print('Run Time: ', time.time() - start_time)


