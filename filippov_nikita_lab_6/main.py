
# Это же задание идет в 6 лабу.
# Завтрашний план:
# 5:
#     1) Прочитать необходимые данные (price, power, brand)
#     2) Обработать данные (brand, и power. brand по числовому, а power в 2 класса)
#     3) Взять данные, разбить на тестовые и тренировочные
#     4) Взять лог.рес, обучить на трен, оценить на тест.
#     5) Взять рок кривую, создать по ней график
#     6) Вывести отчет по року
# 6:
#     1) Взять пункты 1-3 из 5.
#     2) Взять млпклассифаер, обучить нейронку и проверить

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from config import *

pd.options.mode.chained_assignment = None


def read_data(file_path, required_columns, n_rows):
    data = pd.read_csv(file_path, nrows=n_rows)
    required_data = data[required_columns]
    return required_data.dropna()


def prepare_data(data):
    # Генерация меток для цены
    labels = []

    for price in data[TARGET_COLUMN]:
        for label, max_price in TARGET_CLASSES.items():
            if price < max_price:
                labels.append(label)
                break

    data[TARGET_LABEL] = labels

    return data


def mlp_classifier(X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=TRAIN_TS, random_state=TRAIN_RS)
    model = MLPClassifier(random_state=MLP_RS)
    model.fit(x_train, y_train)
    model_score = model.score(x_test, y_test)
    print(model_score)


if __name__ == "__main__":
    cars_data = read_data(FILE_PATH, FEATURE_COLUMNS + [TARGET_COLUMN], DATA_N_ROWS)
    prepared_data = prepare_data(cars_data)
    mlp_classifier(prepared_data[FEATURE_COLUMNS], prepared_data[TARGET_LABEL])


# Задача прогнозирования класса стоимости а/м-ля (Премиум, бюджетные) по пробегу и году выпуска
# Определение категории страхования автомобиля
