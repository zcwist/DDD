"""
=========================================================
Comparing different clustering algorithms on toy datasets
=========================================================

This example aims at showing characteristics of different
clustering algorithms on datasets that are "interesting"
but still in 2D. The last dataset is an example of a 'null'
situation for clustering: the data is homogeneous, and
there is no good clustering.

While these examples give some intuition about the algorithms,
this intuition might not apply to very high dimensional data.

The results could be improved by tweaking the parameters for
each clustering strategy, for instance setting the number of
clusters for the methods that needs this parameter
specified. Note that affinity propagation has a tendency to
create many clusters. Thus in this example its two parameters
(damping and per-point preference) were set to mitigate this
behavior.
"""
print(__doc__)

import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler


def get_clusters(datapts, method='spectral', n_clusters=2):

    assert method in ['spectral', 'kmeans']

    np.random.seed(0)

    X, y = datapts
    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)
    
    # connectivity matrix for structured Ward
    # connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
    # make connectivity symmetric
    # connectivity = 0.5 * (connectivity + connectivity.T)

    # create clustering estimators
    methods = dict()
    # bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)
    # ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # ward = cluster.AgglomerativeClustering(n_clusters=2, linkage='ward',
                                           # connectivity=connectivity)
    methods['kmeans'] = cluster.MiniBatchKMeans(n_clusters)
    methods['spectral'] = cluster.SpectralClustering(n_clusters,
                                          eigen_solver='arpack',
                                          affinity="nearest_neighbors")
    # dbscan = cluster.DBSCAN(eps=.2)
    # affinity_propagation = cluster.AffinityPropagation(damping=.9,
                                                       # preference=-200)

    # average_linkage = cluster.AgglomerativeClustering(
        # linkage="average", affinity="cityblock", n_clusters=2,
        # connectivity=connectivity)

    # birch = cluster.Birch(n_clusters=2)
    # clustering_algorithms = [two_means, spectral]
        # two_means, affinity_propagation, ms, spectral, ward, average_linkage, dbscan, birch]

    algorithm = methods[method]
    
    # predict cluster memberships
    algorithm.fit(X)
    if hasattr(algorithm, 'labels_'):   y_pred = algorithm.labels_.astype(np.int)
    else:                               y_pred = algorithm.predict(X)

    return y_pred

if __name__ == '__main__':
    # import pdb; pdb.set_trace()
    # main(datasets)
    noisy_moons = datasets.make_moons(n_samples=1500, noise=.05)
    y_pred = get_clusters(noisy_moons, "spectral", 2)

    print(noisy_moons)
    print(y_pred)

    X, y = noisy_moons
    X = StandardScaler().fit_transform(X)

    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    plt.figure(figsize=(2, 3))
    plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)
    plt.show()

    