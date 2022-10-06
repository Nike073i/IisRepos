# Параметры таблицы данных
FILE_PATH = 'region25.csv'
INDEX_COLUMN = None
TARGET_COLUMN = ['price']
FEATURE_COLUMNS = ['year', 'mileage', 'fuelType', 'brand']

# Параметры дерева решений
DTC_RANDOM_STATE = None

# Параметры тестовой выборки
TRAIN_TEST_SIZE = 0.01
TRAIN_RANDOM_STATE = None

# Параметры кластеризации
CLUSTERING_N_ROW = 25
CLUSTERING_FEATURES = ['year']
CLUSTERING_METHOD = 'ward'
