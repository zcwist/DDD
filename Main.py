from Plot import Plot
from ConceptManager import ConceptManager as CM

def main():
	cm = CM(40)
	cm.dimRed('tsne')
	Plot(cm).drawWithTag()

def test():
	from sklearn.cluster import KMeans
	import numpy as np
	X = np.array([[1, 2], [1, 4], [1, 0],
                  [4, 2], [4, 4], [4, 0]])
	kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
	print kmeans.labels_


if __name__ == '__main__':
	test()
	