import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
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


def logic_regressio(X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=TRAIN_TS, random_state=TRAIN_RS)
    model = LogisticRegression(random_state=REGRESSIO_RS)
    model.fit(x_train, y_train)
    model_score = model.score(x_test, y_test)
    print(model_score)
    logic_roc_auc = roc_auc_score(y_train, model.predict(x_train))
    fpr, tpr, thresholds = roc_curve(y_train, model.predict_proba(x_train)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='Точность регрессии = %0.2f' % logic_roc_auc)
    plt.show()
    print(logic_roc_auc)


if __name__ == "__main__":
    cars_data = read_data(FILE_PATH, FEATURE_COLUMNS + [TARGET_COLUMN], DATA_N_ROWS)
    prepared_data = prepare_data(cars_data)
    logic_regressio(prepared_data[FEATURE_COLUMNS], prepared_data[TARGET_LABEL])


# Задача прогнозирования класса стоимости а/м-ля (Премиум, бюджетные) по пробегу и году выпуска
# Определение категории страхования автомобиля
