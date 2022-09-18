import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from config import *
pd.options.mode.chained_assignment = None


def read_data(file_path, required_columns, index_col):
    data = pd.read_csv(file_path, index_col=index_col)
    required_data = data[required_columns]
    return required_data


def prepare_data(data):
    # Выбрасываем все строки, где Cabin отсутствует.(В таблице их 600 строк. Любое заполнение этих пробелов неккоректно)
    m_data = data[data['Cabin'].isnull() == False]

    # Заполняем age медианами, а точки отправлениях (их 2 неуказанных) рандомными.
    m_data['Age'] = m_data['Age'].fillna(data.loc[:, 'Age'].median())
    m_data['Embarked'] = m_data['Embarked'].fillna(np.random.choice(['S', 'C', 'Q']))

    # Числовое кодирование для кабин
    le = LabelEncoder()
    le.fit(m_data['Cabin'])
    m_data['Cabin'] = le.transform(m_data['Cabin'])

    # OHE для точки отправления
    prepared_data = pd.get_dummies(m_data, columns=['Embarked'], prefix='embarked')
    return prepared_data


def print_classifier_info(feature_importances):
    feature_names = ['Age', 'Cabin', 'Embarked']
    embarked_score = feature_importances[-3:].sum()
    scores = np.append(feature_importances[:2], embarked_score)
    scores = map(lambda score: round(score, 2), scores)
    print(dict(zip(feature_names, scores)))


if __name__ == '__main__':
    titanic_data = read_data(FILE_PATH, REQUIRED_COLUMNS, INDEX_COLUMN)
    processed_data = prepare_data(titanic_data)

    classifier_tree = DecisionTreeClassifier()
    x = processed_data.drop(TARGET_COLUMN, axis=1)
    y = processed_data[TARGET_COLUMN]

    classifier_tree.fit(x, y)
    print_classifier_info(classifier_tree.feature_importances_)
