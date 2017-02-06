import matplotlib.pyplot as plt
class Plot(object):
	"""docstring for Plot"""
	def __init__(self):
		super(Plot, self).__init__()

	def dendrogram(self,dist_matrix,labels,show=True):
		from scipy.cluster.hierarchy import dendrogram, linkage
		Z = linkage(dist_matrix,'average',metric='cosine')

		plt.figure(figsize=(8,8))
		dendrogram(Z,
			labels=labels,
			orientation='right',
			count_sort='descendent',
			leaf_font_size=10)
		fig = plt.gcf()
		fig.subplots_adjust(left=0.25)
		if show:
			plt.show()

	def heatmap(self,dist_matrix,labels):
		import seaborn as sns
		sns.set(font_scale=0.9)
		
		sns.clustermap(dist_matrix,
							method='average',
							metric='cosine',
							yticklabels=labels,
							xticklabels=labels)
		fig = plt.gcf()
		fig.subplots_adjust(left=0.1,right=0.75,bottom=0.25)

		plt.show()

if __name__ == '__main__':
	import numpy as np
	uniform_data = np.random.rand(10, 10)
	labels = np.random.rand(10,1)

	Plot().heatmap(uniform_data,labels)

		