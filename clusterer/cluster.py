class Cluster(object):
	"""docstring for Cluster"""
	def __init__(self, conceptManager, **kwarg):
		super(Cluster, self).__init__()
		self.conceptManager = conceptManager
		self.kwarg = kwarg

	def doCluster(self):
		pass

if __name__ == '__main__':
	from os import sys, path
	sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
	from concept.ConceptManager import ConceptManager as CM
	CM(10)