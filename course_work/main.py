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
    print("До : ", len(data))
    # Выбрасываем строки с пустыми значениями
    m_data = data.dropna()
    print("После : ", len(m_data))

    # Числовое кодирование для бренда
    le = LabelEncoder()
    le.fit(m_data['brand'])
    m_data['brand'] = le.transform(m_data['brand'])

    # Числовое кодирование для типа топлива
    le = LabelEncoder()
    le.fit(m_data['fuelType'])
    m_data['fuelType'] = le.transform(m_data['fuelType'])

    return m_data


if __name__ == "__main__":
    cars_data = read_data(FILE_PATH, FEATURE_COLUMNS + TARGET_COLUMN, INDEX_COLUMN)
    cars_data = prepare_data(cars_data)

    classification.x = cars_data[FEATURE_COLUMNS]
    classification.y = cars_data.drop(FEATURE_COLUMNS, axis=1)

    app.run(debug=False)


# Задачи по курсовой работе:
# Выявление наиболее важного признака, влияющего на стоимость автомобиля,
# на основе которого можно формировать стоимость страховки автомобиля.
