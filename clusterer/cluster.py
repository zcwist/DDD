class Cluster(object):
	"""docstring for Cluster"""
	def __init__(self, conceptManager):
		super(Cluster, self).__init__()
		self.conceptManager = conceptManager
		self.cluster = {}
		self.mapping = {}
		self.type = "cluster"
		# self.doCluster()

	def doCluster(self):
		pass
		# return {"cluster":[ConceptItem]}

	def getCluster(self):
		return self.cluster

	def getMapping(self):
		return self.mapping

	def getType(self):
		return self.type
if __name__ == '__main__':
	from os import sys, path
	sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
	from concept.ConceptManager import ConceptManager as CM
	CM(10)