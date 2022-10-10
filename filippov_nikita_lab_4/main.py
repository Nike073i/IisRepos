from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from clustering import *

pd.options.mode.chained_assignment = None


def read_data(file_path, required_columns, n_rows):
    data = pd.read_csv(file_path, nrows=n_rows)
    required_data = data[required_columns]
    return required_data.dropna()


def prepare_data(data):
    minmax = MinMaxScaler()
    data = minmax.fit_transform(data)
    return data


if __name__ == "__main__":
    cars_data = read_data(FILE_PATH, CLUSTERING_FEATURES, DATA_N_ROWS)
    prepared_data = prepare_data(cars_data)
    lineage_matrix = cluster_analysis(prepared_data, CLUSTERING_METHOD)
    print_dendrogram(lineage_matrix)
    plt.show()


# Задача анализа количества страховых предложений, основываясь на годе производства автообиля и его пробеге
# (Выявление оптимального числа страховых предложений)