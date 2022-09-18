from flask import Flask
import pandas as pd
from config import *
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None
app = Flask(__name__)
cars_data = None


def read_data(file_path, required_columns, index_col):
    data = pd.read_csv(file_path, index_col=index_col)
    required_data = data[required_columns]
    return required_data


def prepare_data(data):
    # Выбрасываем строки с пустыми значениями
    m_data = data.dropna()

    # # Числовое кодирование для трансмисии
    le = LabelEncoder()
    le.fit(m_data['transmission'])
    m_data['transmission'] = le.transform(m_data['transmission'])

    # OHE для трансмисии
    # m_data = pd.get_dummies(m_data, columns=['transmission'], prefix='transmission')

    # Убираем ЕИ у значений рабочего объема двигателя
    m_data['engineDisplacement'] = m_data['engineDisplacement'].map(lambda ed: ed[:-4])

    return m_data


def print_classifier_info(feature_importances):
    feature_names = FEATURE_COLUMNS
    scores = map(lambda score: round(score, 2), feature_importances)
    print(dict(zip(feature_names, scores)))


@app.route("/")
def home():
    return "Hello" if cars_data is None else str(cars_data.head(5))


if __name__ == "__main__":
    cars_data = read_data(FILE_PATH, FEATURE_COLUMNS + TARGET_COLUMN, INDEX_COLUMN)
    cars_data = prepare_data(cars_data)

    classifier_tree = DecisionTreeClassifier(random_state=DTC_RANDOM_STATE)

    x = cars_data[['price', 'power', 'engineDisplacement', 'mileage']]
    y = cars_data.drop(['price', 'power', 'engineDisplacement', 'mileage'], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TRAIN_TEST_SIZE, random_state=TRAIN_RANDOM_STATE)

    classifier_tree.fit(x_train, y_train)
    print(classifier_tree.score(x_test, y_test))
    print_classifier_info(classifier_tree.feature_importances_)

    app.run(debug=False)

# Вынести все в конфиг
# Сделать работу веб-сервиса
# Выводить по запросу html с таблицей результатов и оценкой модели.
# Мб сделать выбор параметров через веб-сервис

# Задачи по курсовой работе:
# 1. (Нейронка и дерево решений) Выявление наиболее важного признака, влияющего на тип коробки передач в автомобиле с помощью дерева решений.
# transmission от price, power и engineDisplacement (mileage для теста результата)
