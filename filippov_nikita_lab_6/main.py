import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
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
    # Рассчет медианного значения цены автомобиля
    cut_of_price = data['price'].mean()
    # cut_of_price = data['price'].median()
    # cut_of_price = 1000000
    price_classes = dict(zip([0, 1], [cut_of_price, sys.maxsize]))

    labels = []

    for price in data['price']:
        for label, max_price in price_classes.items():
            if price < max_price:
                labels.append(label)
                break

    data['price_label'] = labels

    print(labels.count(0))
    print(labels.count(1))

    return data


def best_prepare_data(data):
    prepare_data(data)

    # Генерация меток для года
    year_labels = []

    year_classes = dict(zip([0, 1, 2, 3, 4], [1990, 2000, 2010, 2020, 2022]))

    for year in data['year']:
        for label, max_year in year_classes.items():
            if year < max_year:
                year_labels.append(label)
                break

    data['year_label'] = year_labels

    # Нормализация значений пробега
    minmax = MinMaxScaler()
    data[['mileage']] = minmax.fit_transform(data[['mileage']])

    return data


def mlp_classifier(X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=TRAIN_TS, random_state=TRAIN_RS)
    model = MLPClassifier(random_state=MLP_RS, hidden_layer_sizes=HIDDEN_LAYER_SIZES, max_iter=MAX_ITER)
    model.fit(x_train, y_train)
    model_score = model.score(x_test, y_test)
    # accuracy = accuracy_score(y_test, model.predict(x_test)) для потверждения, что оценка через score верна
    print(model_score)
    # print(accuracy)


if __name__ == "__main__":
    cars_data = read_data(FILE_PATH, FEATURE_COLUMNS + [TARGET_COLUMN], DATA_N_ROWS)

    # prepared_data = prepare_data(cars_data)
    # feature_columns = ['year', 'mileage']

    prepared_data = best_prepare_data(cars_data)
    feature_columns = ['year_label', 'mileage']

    mlp_classifier(prepared_data[feature_columns], prepared_data[TARGET_LABEL])

# Задача прогнозирования класса стоимости а/м-ля (Премиум, бюджетные) по пробегу и году выпуска
# Определение категории страхования автомобиля
