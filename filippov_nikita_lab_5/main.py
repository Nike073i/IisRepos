import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from config import *


def read_data(file_path, required_columns, n_rows):
    data = pd.read_csv(file_path, nrows=n_rows)
    required_data = data[required_columns]
    return required_data.dropna()


def prepare_data_new(data):
    # надо еще в отчет добавить по среднему значению, и константному значению в 1кк
    # Рассчет медианного значения цены автомобиля

    median_price = data[TARGET_COLUMN].median()
    price_classes = dict(zip([0, 1], [median_price, sys.maxsize]))

    # Генерация меток для года
    year_labels = []

    year_classes = dict(zip([0, 1, 2, 3, 4], [1990, 2000, 2010, 2020, 2022]))

    for year in data['year']:
        for label, max_year in year_classes.items():
            if year < max_year:
                year_labels.append(label)
                break

    data['year_label'] = year_labels

    # Генерация меток для цены
    labels = []

    for price in data[TARGET_COLUMN]:
        for label, max_price in price_classes.items():
            if price < max_price:
                labels.append(label)
                break

    data[TARGET_LABEL] = labels

    # Нормализация значений пробега
    minmax = MinMaxScaler()
    data[['mileage']] = minmax.fit_transform(data[['mileage']])

    return data


def logic_regression(X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=TRAIN_TS, random_state=TRAIN_RS)
    model = LogisticRegression(random_state=REGRESSION_RS)
    model.fit(x_train, y_train)
    model_score = model.score(x_test, y_test)

    logic_roc_auc = roc_auc_score(y_train, model.predict(x_train))
    fpr, tpr, thresholds = roc_curve(y_train, model.predict_proba(x_train)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='Точность регрессии = %0.2f' % logic_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate (FPR)')
    plt.ylabel('True positive rate (TPR)')
    plt.title('Оценка бинарной классификации с помощью ROC-кривой')
    plt.legend(loc='lower right')
    plt.show()

    print("Оценка модели (средняя точность) : %0.2f" % model_score)
    print("Площадь под кривой ROC (AUC ROC) : %0.2f" % logic_roc_auc)

    print("Отчеты по качеству классификации с помощью логистической регресии")
    print("Тестовые данные:")
    print(classification_report(y_test, model.predict(x_test)))
    print("Тренировочные данные:")
    print(classification_report(y_train, model.predict(x_train)))


if __name__ == "__main__":
    cars_data = read_data(FILE_PATH, FEATURE_COLUMNS + [TARGET_COLUMN], DATA_N_ROWS)
    prepared_data = prepare_data_new(cars_data)
    logic_regression(prepared_data[['mileage', 'year_label']], prepared_data[TARGET_LABEL])


# Задача прогнозирования класса стоимости а/м-ля (Премиум, бюджетные) по пробегу и году выпуска
# Определение категории страхования автомобиля
