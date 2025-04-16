from config import *
from data import *
from model import *

if __name__ == '__main__':
    start_time = time.time()

    config = create_config()
    data = Data(config)
    data.load_data()
    data.preprocess_data()
    data.split_data()

    model = Model(config, data)
    model.make_model()
    model.evaluate_model()
    model.save_result()

    print('Run Time: ', time.time() - start_time)


