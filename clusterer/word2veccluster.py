from cluster import Cluster

class Word2VecCluster(Cluster):
	"""docstring for Word2VecCluster"""
	def __init__(self, conceptManager, k = None):
		super(Word2VecCluster, self).__init__(conceptManager)
		self.k = k
		if k is None:
			k = len(conceptManager.concept_category_list)

	def doCluster(self):
		pass

		