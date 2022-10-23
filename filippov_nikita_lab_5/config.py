# Параметры таблицы данных
import sys

FILE_PATH = 'region25.csv'
DATA_N_ROWS = 50000
FEATURE_COLUMNS = ['year', 'mileage']
TARGET_COLUMN = 'price'
TARGET_CLASSES = dict(zip([0, 1], [1000000, sys.maxsize]))
TARGET_LABEL = 'price_label'


# Параметры логистической регрессии
REGRESSION_RS = 0
TRAIN_TS = 0.01
TRAIN_RS = 0
