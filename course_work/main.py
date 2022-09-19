from config import *
from sklearn.preprocessing import LabelEncoder
import classification
from web_service import app
import pandas as pd
pd.options.mode.chained_assignment = None


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


if __name__ == "__main__":
    cars_data = read_data(FILE_PATH, FEATURE_COLUMNS + TARGET_COLUMN, INDEX_COLUMN)
    cars_data = prepare_data(cars_data)

    classification.x = cars_data[FEATURE_COLUMNS]
    classification.y = cars_data.drop(FEATURE_COLUMNS, axis=1)

    app.run(debug=False)

# Задачи по курсовой работе:
# 1. (Нейронка и дерево решений) Выявление наиболее важного признака, влияющего на тип коробки передач в автомобиле с помощью дерева решений.
# transmission от price, power и engineDisplacement (mileage для теста результата)