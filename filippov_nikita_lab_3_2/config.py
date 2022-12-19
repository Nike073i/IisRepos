# Параметры таблицы данных
import sys

FILE_PATH = 'region25.csv'
DATA_N_ROWS = 50000
INDEX_COLUMN = None
TARGET_COLUMN = ['price']
FEATURE_COLUMNS = ['year', 'mileage', 'fuelType', 'brand']
PRICE_CLASSES = dict(zip([1, 2, 3], [500000, 2000000, sys.maxsize]))

# Параметры дерева решений
DTC_RANDOM_STATE = None

# Параметры тестовой выборки
TRAIN_TEST_SIZE = 0.05
TRAIN_RANDOM_STATE = None
