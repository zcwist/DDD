import matplotlib.pyplot as plt
class Plot(object):
	"""docstring for Plot"""
	def __init__(self):
		super(Plot, self).__init__()

	def dendrogram(self,dist_matrix,labels,show=True):
		from scipy.cluster.hierarchy import dendrogram, linkage
		Z = linkage(dist_matrix,'average',metric='cosine')

		plt.figure(figsize=(9,9))
		dendrogram(Z,
			labels=labels,
			orientation='right',
			count_sort='descendent',
			leaf_font_size=10)
		fig = plt.gcf()
		fig.subplots_adjust(left=0.25)
		if show:
			plt.show() 
		