# Параметры таблицы данных
FILE_PATH = 'region25.csv'
DATA_N_ROWS = 50000
FEATURE_COLUMNS = ['year', 'mileage']
TARGET_COLUMN = 'price'
TARGET_LABEL = 'price_label'

# Параметры нейронной сети
MLP_RS = 0
HIDDEN_LAYER_SIZES = [100, 100]
TRAIN_TS = 0.05
TRAIN_RS = 0
MAX_ITER = 200
