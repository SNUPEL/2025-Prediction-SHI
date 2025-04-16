from config import *
from data import *

if __name__ == '__main__':
    start_time = time.time()

    config = create_config()
    data = Data(config)
    data.load_data()

