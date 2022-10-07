from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from config import *


def cluster_analysis(data, method):
    length = len(data)
    if CLUSTERING_N_ROW_MIN > length or length > CLUSTERING_N_ROW_MAX:
        raise ValueError("Размер данных не соответствует ограничениям")
    return linkage(data, method=method)


def get_flat_cluster(linkage_matrix, cluster_count, criterion="maxclust"):
    return fcluster(linkage_matrix, t=cluster_count, criterion=criterion)


def print_dendrogram(linkage_matrix):
    plt.figure(figsize=(DENDROGRAM_WIDTH,DENDROGRAM_HEIGHT))
    dendrogram(linkage_matrix, leaf_rotation=90, no_labels=True)

