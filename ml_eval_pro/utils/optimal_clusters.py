from sklearn.cluster import KMeans
import numpy as np


def calculate_optimal_bins(dataset: np.ndarray):
    """
    Calculating the optimal number of bins of a dataset using elbow method for k-means clustering.
    :param dataset: the whole dataset without the target value.
    :return: The optimal number of clusters.
    """
    wcss = np.zeros((10,))

    if len(dataset.shape) == 1:
        dataset = dataset.reshape(-1, 1)

    for i in range(1, 11):
        k_means = KMeans(n_clusters=i, random_state=42)
        k_means.fit(dataset)
        wcss[i - 1] = k_means.inertia_

    min_max_line = np.linspace(wcss[0], wcss[9], 10)
    return np.argmax(np.abs(min_max_line - wcss)) + 1
