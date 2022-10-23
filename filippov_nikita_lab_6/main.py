import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

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
    model = MLPClassifier(random_state=MLP_RS, hidden_layer_sizes=HIDDEN_LAYER_SIZES, max_iter=MAX_ITER)
    model.fit(x_train, y_train)
    model_score = model.score(x_test, y_test)
    print(model_score)


if __name__ == "__main__":
    cars_data = read_data(FILE_PATH, FEATURE_COLUMNS + [TARGET_COLUMN], DATA_N_ROWS)
    prepared_data = prepare_data(cars_data)
    mlp_classifier(prepared_data[FEATURE_COLUMNS], prepared_data[TARGET_LABEL])

# Задача прогнозирования класса стоимости а/м-ля (Премиум, бюджетные) по пробегу и году выпуска
# Определение категории страхования автомобиля
