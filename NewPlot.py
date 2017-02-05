import matplotlib.pyplot as plt
class Plot(object):
	"""docstring for Plot"""
	def __init__(self):
		super(Plot, self).__init__()

	def dendrogram(self,dist_matrix,labels):
		from scipy.cluster.hierarchy import dendrogram, linkage
		Z = linkage(dist_matrix,'average',metric='cosine')

		plt.figure(figsize=(9,9))
		dendrogram(Z,
			labels=labels,
			orientation='right',
			count_sort='descendent',
			leaf_font_size=15)
		plt.show() 
		