from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
from config import *

data = pd.read_csv(FILE_PATH, nrows=CLUSTERING_N_ROW)

samples = data[CLUSTERING_FEATURES].dropna()

print(samples)

minmax = MinMaxScaler()
samples = minmax.fit_transform(samples)

print(samples)

hierarchical_clustering = linkage(samples, method=CLUSTERING_METHOD)
print(hierarchical_clustering)
plt.figure(figsize=(15, 15))
dendrogram(hierarchical_clustering,
           leaf_font_size=6,
           leaf_rotation=90)

plt.show()
