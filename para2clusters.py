import get_clusters
import numpy as np
import os
from ConceptManager import ConceptManager as CM

import matplotlib.plt as plt

nclusters = 13
emb = np.load(os.path.join("save", "em_005_020000.npy"))
data = [emb, []]
y_pred = get_clusters.get_clusters(data, "spectral", nclusters)

cm = CM(80)
y = [cm.getCateIndex(cpt.getCategory()) for cpt in cm.conceptList]


intensity = np.zeros([ncluster, 13])

for yp, yl in zip(y_pred, y):
    intensity(yp, yl) += 1


import pdb; pdb.set_trace()

    # print(noisy_moons)
    # print(y_pred)

    # X, y = noisy_moons
    # X = StandardScaler().fit_transform(X)

    # colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    # colors = np.hstack([colors] * 20)
    # plt.figure(figsize=(2, 3))
    # plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)
    # plt.show()

