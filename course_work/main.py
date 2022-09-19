from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from config import *
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None
app = Flask(__name__, template_folder='templates')
cars_data = None
classifier_tree = None
x = []
y = []


def read_data(file_path, required_columns, index_col):
    data = pd.read_csv(file_path, index_col=index_col)
    required_data = data[required_columns]
    return required_data


def prepare_data(data):
    # Выбрасываем строки с пустыми значениями
    m_data = data.dropna()

    # Числовое кодирование для трансмисии
    le = LabelEncoder()
    le.fit(m_data['transmission'])
    m_data['transmission'] = le.transform(m_data['transmission'])

    # OHE для трансмисии
    # m_data = pd.get_dummies(m_data, columns=['transmission'], prefix='transmission')

    # Убираем ЕИ у значений рабочего объема двигателя
    m_data['engineDisplacement'] = m_data['engineDisplacement'].map(lambda ed: ed[:-4])

    return m_data


def classification(test_size, precision):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size,
                                                        random_state=TRAIN_RANDOM_STATE)
    classifier_tree.fit(x_train, y_train)
    test_score = classifier_tree.score(x_test, y_test)
    feature_scores = map(lambda score: round(score, precision), classifier_tree.feature_importances_)
    return test_score, dict(zip(FEATURE_COLUMNS, feature_scores))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/error')
def error():
    return render_template('error.html')


@app.route('/get_classified')
def get_classified():
    try:
        test_size = float(request.args.get('size'))
        precision = int(request.args.get('precision'))
    except ValueError:
        return redirect(url_for('error'))

    # Проверка входных условий
    score, feature_importances = classification(test_size, precision)
    return render_template('classified.html', data=dict(score=score, feature_importances=feature_importances))


if __name__ == "__main__":
    cars_data = read_data(FILE_PATH, FEATURE_COLUMNS + TARGET_COLUMN, INDEX_COLUMN)
    cars_data = prepare_data(cars_data)

    classifier_tree = DecisionTreeClassifier(random_state=DTC_RANDOM_STATE)

    x = cars_data[FEATURE_COLUMNS]
    y = cars_data.drop(FEATURE_COLUMNS, axis=1)

    app.run(debug=False)

# Задачи по курсовой работе:
# 1. (Нейронка и дерево решений) Выявление наиболее важного признака, влияющего на тип коробки передач в автомобиле с помощью дерева решений.
# transmission от price, power и engineDisplacement (mileage для теста результата)