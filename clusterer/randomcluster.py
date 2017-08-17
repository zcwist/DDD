from cluster import Cluster
from random import randint
class RandomCluster(Cluster):
	"""docstring for RandomCluster"""
	def __init__(self, conceptManager, k = None):
		super(RandomCluster, self).__init__(conceptManager)
		self.k = k
		if k is None:
			self.k = len(conceptManager.concept_category_list)

	def doCluster(self):
		list = self.conceptManager.conceptList
		for i in list:
			randIndex = randint(0, self.k)
			if randIndex in self.cluster:
				self.cluster[randIndex].append(i)
			else:
				self.cluster[randIndex] = [i]
