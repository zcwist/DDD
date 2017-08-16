from cluster import Cluster

class RandomCluster(Cluster):
	"""docstring for RandomCluster"""
	def __init__(self, conceptManager, k = None):
		super(RandomCluster, self).__init__(conceptManager)
		self.k = k
		if k is None:
			self.k = len(conceptManager.concept_category_list)

	def doCluster(self):
		pass
